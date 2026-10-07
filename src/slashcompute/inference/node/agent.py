"""Inference node agent: hosts llama.cpp layers for the pool, next to the MLX training agent.

It only makes outbound connections to the coordinator (register, heartbeat, long-poll commands,
stream tokens back). In direct mode heads reach workers' rpc-server on the LAN; in relay mode the
RPC bytes go through the coordinator over WebSockets. The agent enforces its memory commitment,
steps aside while the MLX agent is training, and drains before leaving.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import errno
import hashlib
import json
import logging
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Optional

import httpx

from slashcompute.inference.node.config import Commitment, NodeConfig
from slashcompute.inference.node.engine import Engine, EngineError
from slashcompute.inference.node.relay import HeadProxy, bridge_stream

log = logging.getLogger("slashcompute.inference.node")
GIB = 1024 ** 3
TOKEN_HEADER = "X-Inference-Token"


class CommitmentError(RuntimeError):
    pass


def in_hours(hours, now: Optional[datetime] = None) -> bool:
    h = (now or datetime.now(timezone.utc)).hour
    return any((a <= h < b) if a <= b else (h >= a or h < b) for a, b in hours)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def scan_models(dirs: list[str]) -> list[dict]:
    """GGUFs on disk (first shard of split models), with parsed headers for the coordinator."""
    from slashcompute.inference.gguf import is_first_shard_or_single, read_header_file, shard_paths
    out, seen = [], set()
    for d in dirs:
        root = Path(d).expanduser()
        for f in sorted(root.glob("*.gguf")) if root.is_dir() else []:
            if f.name in seen or f.name.startswith(".") or not is_first_shard_or_single(f.name):
                continue
            try:
                parts = shard_paths(f)
                headers = [read_header_file(p).to_json() for p in parts]
            except Exception as e:  # noqa: BLE001
                log.warning("skipping %s: %s", f.name, e)
                continue
            seen.add(f.name)
            out.append({"name": f.name, "size": sum(p.stat().st_size for p in parts), "headers": headers})
    return out


def move_to_trash(path: Path) -> None:
    """Move a file to the macOS Trash, as Finder does (Put Back works). Raises OSError when it can't;
    a file in someone's own models folder is never deleted instead."""
    try:
        import objc
        from Foundation import NSURL, NSFileManager
    except ImportError as e:
        raise OSError(errno.ENOTSUP, "the Trash is not available on this Mac") from e
    with objc.autorelease_pool():
        url = NSURL.fileURLWithPath_(str(path))
        ok, _, err = NSFileManager.defaultManager().trashItemAtURL_resultingItemURL_error_(url, None, None)
        if not ok:
            raise OSError(errno.EIO, str(err.localizedDescription()) if err is not None else "not moved to the Trash")


def _same_dir(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve() or (a.is_dir() and b.is_dir() and os.path.samefile(a, b))
    except OSError:
        return False


def remove_model_files(name: str, models_dir: str, download_dir: str, trash: bool) -> dict:
    """Remove every shard of a model from this Mac: the app's copies in ``download_dir`` (and a partial
    download) are deleted; copies in the user's own ``models_dir`` go to the Trash when ``trash``, else stay.
    When both are one folder it is the user's: Trash, never delete. Tries every file, then raises one
    OSError naming the files that failed (names only: the coordinator shows it to the pool)."""
    from slashcompute.inference.gguf import plain_gguf, shard_paths

    if not plain_gguf(name):
        raise ValueError(f"not a model file name: {name!r}")
    user, app = Path(models_dir).expanduser(), Path(download_dir).expanduser()
    out: dict[str, list[str]] = {"trashed": [], "deleted": [], "kept": []}
    failed = []

    def why(f: Path, e: OSError) -> str:
        return f"{f.name}: {(e.strerror or type(e).__name__).replace(str(f.parent) + os.sep, '')}"

    if not _same_dir(user, app):
        for f in [*shard_paths(app / name), app / f".{name}.part"]:
            try:
                f.unlink()
                out["deleted"].append(f.name)
            except FileNotFoundError:
                pass
            except OSError as e:
                failed.append(why(f, e))
    for f in shard_paths(user / name):
        if not f.is_file():
            continue
        if not trash:
            out["kept"].append(f.name)
            continue
        try:
            move_to_trash(f)
            out["trashed"].append(f.name)
        except OSError as e:
            failed.append(why(f, e))
    if failed:
        raise OSError("; ".join(failed))   # no errno: str() is then exactly this text
    return out


def training_busy(status_file: str) -> bool:
    """Whether the MLX agent on this Mac is running a training stage right now.

    Reads the agent's ``status.json``; a status left behind by an agent that is no longer running
    (no live pid in ``agent.pid`` next to it) does not count."""
    path = Path(status_file).expanduser()
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    if data.get("status") != "running":
        return False
    try:
        os.kill(int((path.parent / "agent.pid").read_text().strip()), 0)
    except (OSError, ValueError):
        return False
    return True


class Agent:
    def __init__(self, cfg: NodeConfig, engine: Engine, *, info: dict, build: str, ip: str,
                 gguf_files: list[dict], latency_fn: Callable[[list[dict]], Awaitable[dict]],
                 client: Optional[httpx.AsyncClient] = None, state: Optional[dict] = None,
                 heartbeat_seconds: Optional[float] = None,
                 probe_fn: Optional[Callable[[str, int], Awaitable]] = None,
                 busy_fn: Optional[Callable[[], bool]] = None):
        self.cfg = cfg
        self.engine = engine
        self.info = info
        self.build = build
        self.ip = ip
        self.gguf_files = gguf_files
        self.latency_fn = latency_fn
        self.client = client or httpx.AsyncClient(base_url=cfg.coordinator_url, timeout=httpx.Timeout(60, connect=10))
        self.state = state if state is not None else {}
        self.node_id: Optional[str] = self.state.get("node_id")
        self.token: Optional[str] = self.state.get("token")
        self.heartbeat_seconds = heartbeat_seconds or 5.0
        self.in_use: dict[str, int] = {}   # pipeline id -> bytes committed to it
        self.transport = "direct"          # set by the coordinator at registration and on heartbeats
        self.pending_transport: Optional[str] = None
        self.head_approved = False
        self.local_rpc: dict[str, str] = {}               # pipeline -> local rpc-server endpoint (worker)
        self.proxies: dict[str, list[HeadProxy]] = {}     # pipeline -> relay proxies (head)
        self.rtt_ms: Optional[float] = None
        self.paused = False
        self.probe_fn = probe_fn                          # direct mode: start the latency probe listener
        self.probe = None
        self.probe_port = cfg.probe_port
        self.busy_fn = busy_fn or (lambda: training_busy(cfg.agent_status_file))
        self.downloads: dict[str, float] = {}             # filename -> fraction done
        self.jobs: dict[str, asyncio.Task] = {}           # job id -> run_job task (head)
        self.last_available: Optional[bool] = None
        self.last_reason = ""
        self.last_error = ""
        self._tasks: set[asyncio.Task] = set()
        self.stopped = asyncio.Event()

    # ------------------------------------------------------------ plumbing

    @property
    def headers(self) -> dict:
        h = {"Authorization": f"Bearer {self.token}"}
        if self.cfg.inference_token:
            h[TOKEN_HEADER] = self.cfg.inference_token
        return h

    def _spawn(self, coro) -> asyncio.Task:
        t = asyncio.create_task(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)
        return t

    async def _post(self, path: str, body: dict) -> dict:
        r = await self.client.post(path, json=body, headers=self.headers)
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------ lifecycle

    async def register(self) -> None:
        body = {
            "name": self.cfg.name, "os": self.info.get("os"), "chip": self.info.get("chip"),
            "total_mem_bytes": self.info.get("total_mem_bytes"), "tailscale_ip": self.ip,
            "probe_port": self.probe_port or None, "llama_build": self.build,
            "commitment": self.cfg.commitment.to_json(), "gguf_files": self.gguf_files,
            "node_id": self.node_id, "token": self.token, "coordinator_rtt_ms": self.rtt_ms,
            "session_token": self.cfg.session_token or None,
        }
        headers = {TOKEN_HEADER: self.cfg.inference_token} if self.cfg.inference_token else {}
        r = await self.client.post("/nodes/register", json=body, headers=headers)
        if r.status_code in (401, 403, 409):
            raise SystemExit(f"coordinator refused this node: {r.json().get('detail')}")
        r.raise_for_status()
        data = r.json()
        self.node_id, self.token = data["node_id"], data["token"]
        self.heartbeat_seconds = data.get("heartbeat_seconds", self.heartbeat_seconds)
        self.head_approved = bool(data.get("head_approved"))
        await self._set_transport(data.get("transport", "direct"))
        self.state.update(node_id=self.node_id, token=self.token)
        log.info("registered as %s (%s transport)", self.node_id, self.transport)

    async def start(self, benchmark: Optional[dict] = None) -> None:
        await self.register()
        bench = benchmark or await self.engine.benchmark()
        await self._post("/nodes/benchmark", bench)
        self._spawn(self.heartbeat_loop())
        self._spawn(self.command_loop())

    async def leave(self) -> None:
        """Tell the coordinator we're going away so it drains our pipelines now, not after a timeout."""
        with contextlib.suppress(httpx.HTTPError):
            await self.client.post("/nodes/heartbeat", headers=self.headers, timeout=5,
                                   json={"available": False, "llama_build": self.build, "leaving": True})

    async def stop(self) -> None:
        for t in list(self._tasks):
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.engine.stop_all()
        for pid in list(self.proxies):
            await self._close_proxies(pid)
        if self.probe is not None:
            self.probe.close()
            self.probe = None
        self.stopped.set()

    def availability(self) -> tuple[bool, str]:
        """Whether this Mac takes inference work right now, and why not."""
        if self.paused:
            return False, "paused"
        if not in_hours(self.cfg.commitment.hours):
            return False, "outside allowed hours"
        if self.busy_fn():
            return False, "training on this Mac"
        return True, "available"

    async def _set_transport(self, transport: str) -> None:
        """Switch direct <-> relay. Only while holding no pipelines; otherwise after they end."""
        if transport == self.transport and self.pending_transport is None:
            return
        if self.in_use:
            self.pending_transport = transport
            return
        self.pending_transport = None
        self.transport = transport
        if hasattr(self.engine, "bind_ip"):
            # relay: the RPC server never listens beyond this machine
            self.engine.bind_ip = "127.0.0.1" if transport == "relay" else (self.ip or "127.0.0.1")
        if transport == "direct" and self.probe is None and self.probe_fn and self.ip:
            self.probe = await self.probe_fn(self.ip, self.probe_port)
            with contextlib.suppress(AttributeError, IndexError):
                self.probe_port = self.probe.sockets[0].getsockname()[1]
        elif transport == "relay" and self.probe is not None:
            self.probe.close()
            self.probe = None

    async def heartbeat(self) -> None:
        available, reason = self.availability()
        self.last_available, self.last_reason = available, reason
        r = await self.client.post("/nodes/heartbeat", headers=self.headers, json={
            "available": available, "reason": reason, "llama_build": self.build,
            "pipelines": sorted(self.in_use), "downloads": self.downloads})
        if r.status_code == 401:
            await self.register()
            return
        with contextlib.suppress(ValueError):
            t = r.json().get("transport")
            if t:
                await self._set_transport(t)
        self.write_status()

    async def heartbeat_loop(self) -> None:
        while True:
            try:
                await self.heartbeat()
            except httpx.HTTPError as e:
                self.last_error = f"heartbeat failed: {e}"
                log.warning("heartbeat failed: %s", e)
                self.write_status()
            await asyncio.sleep(self.heartbeat_seconds)

    def write_status(self) -> None:
        """What the /compute shell shows for this Mac's inference node."""
        path = Path(self.cfg.status_file).expanduser()
        data = {
            "pid": os.getpid(), "node_id": self.node_id, "transport": self.transport, "available": self.last_available,
            "reason": self.last_reason, "pipelines": sorted(self.in_use), "models": [f["name"] for f in self.gguf_files],
            "downloads": self.downloads, "last_error": self.last_error, "updated_at": time.time(),
        }
        with contextlib.suppress(OSError):
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data))
            os.replace(tmp, path)

    async def set_paused(self, paused: bool) -> None:
        self.paused = paused
        with contextlib.suppress(httpx.HTTPError):
            await self.heartbeat()

    async def command_loop(self) -> None:
        while True:
            try:
                r = await self.client.get("/agent/commands", params={"wait": 20}, headers=self.headers,
                                          timeout=httpx.Timeout(40, connect=10))
                if r.status_code == 401:
                    await self.register()
                    continue
                r.raise_for_status()
                for cmd in r.json()["commands"]:
                    if cmd["kind"].startswith("stop_"):
                        await self.handle(cmd)  # free memory before any start that follows it
                    else:
                        self._spawn(self.handle(cmd))
            except httpx.HTTPError as e:
                log.warning("command poll failed: %s", e)
                await asyncio.sleep(1)

    async def handle(self, cmd: dict) -> None:
        try:
            result = await self.dispatch(cmd["kind"], cmd["payload"])
            body = {"ok": True, "result": result}
        except Exception as e:  # noqa: BLE001 - report every failure to the coordinator
            log.warning("command %s failed: %s", cmd["kind"], e)
            self.last_error = f"{cmd['kind']}: {e}"
            body = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        with contextlib.suppress(httpx.HTTPError):
            await self._post(f"/agent/commands/{cmd['id']}/result", body)

    # ------------------------------------------------------------ commitment

    def _enforce(self, pipeline_id: str, mem_bytes: int) -> None:
        c = self.cfg.commitment
        available, why = self.availability()
        if not available:
            raise CommitmentError(f"not taking work: {why}")
        committed = int(c.memory_gb * GIB)
        used = sum(v for k, v in self.in_use.items() if k != pipeline_id)
        if used + mem_bytes > committed:
            raise CommitmentError(f"needs {mem_bytes / 1e9:.1f} GB but only {(committed - used) / 1e9:.1f} GB "
                                  f"of the commitment is free")

    async def update_commitment(self, new: Commitment) -> None:
        """New commitment: the coordinator drains affected pipelines (finish the current job,
        then leave) and plans with the new values from now on."""
        log.info("commitment changed: %s", new)
        self.cfg = dataclasses.replace(self.cfg, commitment=new)
        await self._post("/nodes/commitment", new.to_json())

    # ------------------------------------------------------------ commands

    async def measure_rtt(self, samples: int = 5) -> float:
        """Round trip to the coordinator (relay mode: all RPC goes through it)."""
        rtts = []
        for _ in range(samples):
            t0 = time.perf_counter()
            r = await self.client.get("/ping")
            r.raise_for_status()
            rtts.append((time.perf_counter() - t0) * 1000)
        self.rtt_ms = statistics.median(rtts)
        return self.rtt_ms

    async def dispatch(self, kind: str, p: dict) -> dict:
        pid = p.get("pipeline_id")
        if kind == "start_worker":
            self._enforce(pid, p["mem_bytes"])
            self.in_use[pid] = p["mem_bytes"]
            try:
                res = await self.engine.start_worker(pid, {**p, "device": self.cfg.commitment.device})
            except Exception:
                self.in_use.pop(pid, None)
                raise
            if pid not in self.in_use:   # stopped while starting: the stop found nothing to stop yet
                await self.engine.stop_worker(pid)
                raise EngineError("stopped while loading", pipeline_broken=False)
            if self.transport == "relay":
                self.local_rpc[pid] = res["endpoint"]
                return {"endpoint": f"relay:{self.node_id}"}
            return res
        if kind == "start_head":
            self._enforce(pid, p["mem_bytes"])
            self.in_use[pid] = p["mem_bytes"]
            try:
                if self.transport == "relay":
                    p = await self._relay_head_spec(p)
                res = await self.engine.start_head(pid, p)
            except Exception:
                self.in_use.pop(pid, None)
                await self._close_proxies(pid)
                raise
            if pid not in self.in_use:   # stopped while loading (Unload, Stop serving): don't leave it running
                await self.engine.stop_head(pid)
                await self._close_proxies(pid)
                raise EngineError("stopped while loading", pipeline_broken=False)
            return res
        if kind in ("stop_worker", "stop_head"):
            await (self.engine.stop_worker if kind == "stop_worker" else self.engine.stop_head)(pid)
            self.in_use.pop(pid, None)
            self.local_rpc.pop(pid, None)
            await self._close_proxies(pid)
            if self.pending_transport and not self.in_use:
                await self._set_transport(self.pending_transport)
            return {}
        if kind == "open_stream":
            local = self.local_rpc.get(pid)
            if local is None:
                raise ValueError(f"no RPC server for pipeline {pid} on this node")
            self._spawn(bridge_stream(self.cfg.coordinator_url, self.token, p["stream_id"], local))
            return {}
        if kind == "run_job":
            job_id = p["job_id"]
            self.jobs[job_id] = self._spawn(self.run_job(p))
            self.jobs[job_id].add_done_callback(lambda _: self.jobs.pop(job_id, None))
            return {"accepted": True}
        if kind == "cancel_job":  # the requester went away
            task = self.jobs.get(p["job_id"])
            if task is not None:
                task.cancel()
            return {"cancelled": task is not None}
        if kind == "measure_latency":
            if p.get("mode") == "relay":
                return {"coordinator_rtt_ms": await self.measure_rtt()}
            return {"results": await self.latency_fn(p["peers"])}
        if kind == "download_model":
            return await self.download_model(p)
        if kind == "remove_model":
            return await self.remove_model(p)
        raise ValueError(f"unknown command {kind}")

    async def _relay_head_spec(self, p: dict) -> dict:
        """Point llama-server's --rpc at local proxies that tunnel to each worker via the relay."""
        proxies, workers = [], []
        for w in p["workers"]:
            proxy = HeadProxy(self.cfg.coordinator_url, self.token, p["pipeline_id"], w["node_id"])
            workers.append({**w, "endpoint": await proxy.start()})
            proxies.append(proxy)
        self.proxies[p["pipeline_id"]] = proxies
        return {**p, "workers": workers}

    async def _close_proxies(self, pipeline_id: str) -> None:
        for proxy in self.proxies.pop(pipeline_id, []):
            await proxy.close()

    async def run_job(self, p: dict) -> None:
        """Run a request on the local head and relay engine events over one streamed POST."""
        async def events():
            try:
                async with contextlib.aclosing(self.engine.complete(p["pipeline_id"], p["body"])) as evs:
                    async for ev in evs:
                        yield (json.dumps(ev) + "\n").encode()
            except EngineError as e:
                yield (json.dumps({"type": "error", "error": str(e), "retryable": e.status is None,
                                   "pipeline_broken": e.pipeline_broken, "status": e.status}) + "\n").encode()
            except Exception as e:  # noqa: BLE001
                yield (json.dumps({"type": "error", "error": f"{type(e).__name__}: {e}", "retryable": True,
                                   "pipeline_broken": False}) + "\n").encode()

        body = events()
        try:
            r = await self.client.post(f"/agent/jobs/{p['job_id']}/stream", content=body, headers=self.headers,
                                       timeout=httpx.Timeout(None, connect=10))
            r.raise_for_status()
        except httpx.HTTPError as e:
            log.warning("stream for %s failed: %s", p["job_id"], e)
        finally:
            await body.aclose()  # cancel_job: stop the engine (llama-server) generating for nobody

    async def remove_model(self, p: dict) -> dict:
        """The model was removed from the pool: delete the app's copy, and move one in this Mac's models
        folder to the Trash only when the pool asked for it and this Mac allows it (--trash-removed)."""
        trash = bool(p.get("trash")) and self.cfg.trash_removed
        try:
            res = await asyncio.to_thread(remove_model_files, p.get("filename"), self.cfg.models_dir,
                                          self.cfg.download_dir, trash)
            log.info("removed %s: %s", p.get("filename"), res)
            return res
        finally:   # report what is on disk now, whatever happened: a stale list would keep the model listed
            self.gguf_files = await asyncio.to_thread(scan_models, self.cfg.model_dirs)
            try:
                await self._post("/nodes/models", {"files": self.gguf_files})
            except httpx.HTTPError as e:
                log.warning("could not report models after removing %s: %s", p.get("filename"), e)
            self.write_status()

    def _existing(self, filename: str, size: Optional[int], sha256: Optional[str]) -> Optional[Path]:
        for d in self.cfg.model_dirs:
            f = Path(d).expanduser() / filename
            if f.is_file() and (size is None or f.stat().st_size == size) and (not sha256 or file_sha256(f) == sha256):
                return f
        return None

    async def download_model(self, p: dict) -> dict:
        """Fetch an uploaded GGUF from the coordinator (or ``url``), verify it, then report it."""
        name = Path(p["filename"]).name
        size, sha = p.get("size"), p.get("sha256")
        found = await asyncio.to_thread(self._existing, name, size, sha)
        if found is None:
            dest_dir = Path(self.cfg.download_dir).expanduser()
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest, tmp = dest_dir / name, dest_dir / f".{name}.part"
            h = hashlib.sha256()
            self.downloads[name] = 0.0
            own = None if p.get("path") else httpx.AsyncClient(follow_redirects=True,
                                                                timeout=httpx.Timeout(None, connect=10))
            try:
                if own is None:   # relative to the coordinator: same auth, prefix and relay as everything else
                    req = self.client.stream("GET", p["path"], headers=self.headers,
                                             timeout=httpx.Timeout(None, connect=10))
                else:
                    req = own.stream("GET", p["url"])
                async with req as r:
                    r.raise_for_status()
                    total = int(r.headers.get("content-length") or size or 0)
                    done = 0
                    with tmp.open("wb") as fh:
                        async for chunk in r.aiter_bytes(1 << 20):
                            fh.write(chunk)
                            h.update(chunk)
                            done += len(chunk)
                            if total:
                                self.downloads[name] = round(done / total, 3)
                if sha and h.hexdigest() != sha:
                    raise ValueError("sha256 mismatch")
                os.replace(tmp, dest)
                found = dest
            finally:
                if own is not None:
                    await own.aclose()
                tmp.unlink(missing_ok=True)
                self.downloads.pop(name, None)
        self.gguf_files = await asyncio.to_thread(scan_models, self.cfg.model_dirs)
        await self._post("/nodes/models", {"files": self.gguf_files})
        self.write_status()
        return {"path": str(found), "sha256": sha or ""}


# ------------------------------------------------------------ real-node startup

def load_state(path: str) -> dict:
    p = Path(path).expanduser()
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except ValueError:
        return {}


def save_state(path: str, state: dict) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2))
    p.chmod(0o600)
