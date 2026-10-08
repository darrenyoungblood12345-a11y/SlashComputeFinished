"""Inference node agent: hosts llama.cpp layers for the pool, next to the MLX training agent.

It only makes outbound connections to the coordinator (register, heartbeat, long-poll commands,
stream tokens back). In direct mode heads reach workers' rpc-server on the LAN; in relay mode the
RPC bytes go through the coordinator over WebSockets. The agent enforces its memory commitment,
steps aside while the MLX agent is training, and drains before leaving.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import dataclasses
import errno
import hashlib
import json
import logging
import os
import statistics
import time
import uuid
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
STOPPED_EARLY_MAX = 256   # pipelines whose stop overtook their start, remembered (ids are never reused)


class CommitmentError(RuntimeError):
    pass


class StoppedWhileLoading(EngineError):
    """The pool stopped a pipeline before its start finished (Unload, Stop serving, Remove): expected, not a
    fault, so it is not shown as this Mac's last error. The coordinator still gets the failed start."""

    def __init__(self, message: str = "stopped while loading"):
        super().__init__(message, pipeline_broken=False)


def in_hours(hours, now: Optional[datetime] = None) -> bool:
    h = (now or datetime.now(timezone.utc)).hour
    return any((a <= h < b) if a <= b else (h >= a or h < b) for a, b in hours)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _missing_bytes(path: Path, header) -> int:
    """How much of the tensor data its header promises is not in the file yet (a copy still under way)."""
    end = header.data_start + max((t.offset + t.nbytes for t in header.tensors), default=0)
    return max(0, end - path.stat().st_size)


def scan_models(dirs: list[str]) -> list[dict]:
    """GGUFs on disk (first shard of split models), with parsed headers for the coordinator. A file still
    being copied in (shorter than its header says), or a split model with parts still to come, is left out
    until it is whole: the coordinator keeps the first layer table it gets."""
    from slashcompute.inference.gguf import SHARD_RE, is_first_shard_or_single, read_header_file, shard_paths
    out, seen = [], set()
    for d in dirs:
        root = Path(d).expanduser()
        for f in sorted(root.glob("*.gguf")) if root.is_dir() else []:
            if f.name in seen or f.name.startswith(".") or not is_first_shard_or_single(f.name):
                continue
            try:
                parts = shard_paths(f)
                if (m := SHARD_RE.match(f.name)) and len(parts) != int(m.group(3)):
                    log.info("skipping %s for now: %d of %s parts on disk", f.name, len(parts), int(m.group(3)))
                    continue
                headers = [read_header_file(p) for p in parts]
                missing = sum(_missing_bytes(p, h) for p, h in zip(parts, headers))
            except Exception as e:  # noqa: BLE001
                log.warning("skipping %s: %s", f.name, e)
                continue
            if missing:
                log.info("skipping %s for now: %.1f GB of it is not on disk yet (still being copied?)",
                         f.name, missing / 1e9)
                continue
            seen.add(f.name)
            out.append({"name": f.name, "size": sum(p.stat().st_size for p in parts),
                        "headers": [h.to_json() for h in headers]})
    return out


def model_folders(dirs: list[str]) -> tuple:
    """A cheap fingerprint of the model folders: each one's visible GGUFs with size and mtime (None: no such
    folder). It changes when a file is dropped in, moved to the Trash, put back, or still growing (a copy)."""
    out = []
    for d in dirs:
        try:
            with os.scandir(Path(d).expanduser()) as it:
                entries = [e for e in it if e.name.endswith(".gguf") and not e.name.startswith(".")]
            files = []
            for e in entries:
                with contextlib.suppress(OSError):   # gone between the listing and the stat
                    st = e.stat()
                    files.append((e.name, st.st_size, st.st_mtime_ns))
            out.append(tuple(sorted(files)))
        except OSError:
            out.append(None)
    return tuple(out)


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


def _listing(d: Path) -> set[str]:
    try:
        return set(os.listdir(d))
    except OSError:
        return set()


def remove_model_files(name: str, models_dir: str, download_dir: str, trash: bool,
                       deletable: frozenset[str] | set[str] = frozenset()) -> dict:
    """Remove every shard of a model from this Mac. In ``download_dir`` the shards named in ``deletable``
    (the ones the asking pool sent here) and their partial download are deleted. Every other copy there,
    and every copy in the user's own ``models_dir``, goes to the Trash when ``trash``, else stays: ``kept``
    lists each one with its folder, "models" (``models_dir``) or "app" (``download_dir``). When both are one
    folder it is the user's: Trash, never delete. Only exact names from a folder's listing count: APFS
    would open 'Model.gguf' for 'model.gguf', a different model. Tries every file, then raises one
    OSError naming the files that failed (names only: the coordinator shows it to the pool)."""
    from slashcompute.inference.gguf import plain_gguf, shard_paths

    if not plain_gguf(name):
        raise ValueError(f"not a model file name: {name!r}")
    user, app = Path(models_dir).expanduser(), Path(download_dir).expanduser()
    out: dict[str, list] = {"trashed": [], "deleted": [], "kept": []}
    failed = []

    def why(f: Path, e: OSError) -> str:
        return f"{f.name}: {(e.strerror or type(e).__name__).replace(str(f.parent) + os.sep, '')}"

    def delete(f: Path) -> None:
        try:
            f.unlink()
            out["deleted"].append(f.name)
        except FileNotFoundError:
            pass
        except OSError as e:
            failed.append(why(f, e))

    folders = [(user, "models", frozenset())] if _same_dir(user, app) else \
        [(app, "app", frozenset(deletable)), (user, "models", frozenset())]
    for d, label, ours in folders:
        listed = _listing(d)
        for f in shard_paths(d / name):
            if f.name not in listed or not f.is_file():
                continue
            if f.name in ours:
                delete(f)
            elif not trash:
                out["kept"].append({"name": f.name, "folder": label})
            else:
                try:
                    move_to_trash(f)
                    out["trashed"].append(f.name)
                except OSError as e:
                    failed.append(why(f, e))
        if name in ours and f".{name}.part" in listed:
            delete(d / f".{name}.part")
    if failed:
        raise OSError("; ".join(failed))   # no errno: str() is then exactly this text
    return out


def still_ours(download_dir: str, sent: dict[str, dict]) -> set[str]:
    """The files a coordinator sent into ``download_dir`` (``sent``: name -> {size, mtime_ns} as downloaded)
    that are still the file it sent: one written over since (this Mac's own upload when it hosts a pool,
    say) is not."""
    app = Path(download_dir).expanduser()
    out = set()
    for name, rec in sent.items():
        with contextlib.suppress(OSError):
            st = (app / name).stat()
            if (st.st_size, st.st_mtime_ns) == (rec.get("size"), rec.get("mtime_ns")):
                out.add(name)
    return out


def save_json(path: Path, data) -> None:
    """Write JSON atomically: a crash mid-write leaves the previous file, never half of one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:6]}.tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


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
                 busy_fn: Optional[Callable[[], bool]] = None, folders: Optional[tuple] = None):
        self.cfg = cfg
        self.engine = engine
        self.info = info
        self.build = build
        self.ip = ip
        self.gguf_files = gguf_files
        # the model folders as of that scan (take it before scanning, so a change during the scan is seen)
        self.folders = folders if folders is not None else model_folders(cfg.model_dirs)
        self.reported: Optional[list] = None            # the files the coordinator has last been told about
        self.report_due = False                           # the coordinator asked for them again
        self.models_kick = asyncio.Event()                # wakes models_loop before its next pass
        self._scan_lock = asyncio.Lock()                  # scan + report one at a time: the latest scan wins
        # files this node downloaded, per coordinator: only those may that coordinator delete again
        self.downloaded_file = Path(cfg.state_file).expanduser().with_name("downloaded.json")
        self.downloaded: dict[str, dict[str, dict]] = _load_downloaded(self.downloaded_file)
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
        self.stopping: set[str] = set()                   # pipelines whose stop is running now
        # pipelines stopped before their start ran (both in one poll): the start is refused when it comes
        self.stopped_early: collections.OrderedDict[str, None] = collections.OrderedDict()
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
        self.reported = _listing_key(self.gguf_files)
        log.info("registered as %s (%s transport)", self.node_id, self.transport)

    async def start(self, benchmark: Optional[dict] = None) -> None:
        await self.register()
        bench = benchmark or await self.engine.benchmark()
        await self._post("/nodes/benchmark", bench)
        self._spawn(self.heartbeat_loop())
        self._spawn(self.models_loop())
        self._spawn(self.command_loop())

    async def leave(self) -> None:
        """Tell the coordinator we're going away so it drains our pipelines now, not after a timeout."""
        with contextlib.suppress(httpx.HTTPError):
            await self.client.post("/nodes/heartbeat", headers=self.headers, timeout=5,
                                   json={"available": False, "llama_build": self.build, "leaving": True})

    async def cancel_tasks(self) -> None:
        """Cancel every task of this agent and wait until each has ended. A cancel that lands while httpx
        opens a connection can be swallowed: anyio cancels its other connect attempts once one succeeds and
        takes the two cancels for one of its own. So a task still running a moment later is cancelled again,
        and so is any task one of them spawned meanwhile (a command handled after the list was taken)."""
        while pending := {t for t in self._tasks if not t.done()}:
            for t in pending:
                t.cancel()
            await asyncio.wait(pending, timeout=1.0)

    async def stop(self) -> None:
        await self.cancel_tasks()
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
            data = r.json()
            if data.get("report_models"):   # back from offline, or the pool forgot this Mac's files meanwhile
                self.report_due = True
                self.models_kick.set()
            if data.get("transport"):
                await self._set_transport(data["transport"])
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

    async def models_loop(self) -> None:
        """Keep the pool's list of this Mac's models current: a file dropped in, moved to the Trash or put
        back by hand shows without a restart. Its own task, so a slow scan (a big folder on a slow disk)
        or a slow report never holds up a heartbeat."""
        while True:
            try:
                await self.sync_models()
            except Exception as e:  # noqa: BLE001 - tried again next pass
                log.warning("could not report this Mac's models: %s", e)
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self.models_kick.wait(), self.heartbeat_seconds)
            self.models_kick.clear()

    async def sync_models(self, rescan: bool = False) -> None:
        """Tell the coordinator which GGUFs this Mac has. Rescans when ``rescan`` (after a download or a
        removal) or when the folders changed since the last scan, and reports when ``rescan``, when the
        coordinator asked for it, or when the list differs from the last one reported (a report that failed
        is retried). One at a time, so the coordinator always ends up with the latest scan."""
        async with self._scan_lock:
            folders = await asyncio.to_thread(model_folders, self.cfg.model_dirs)   # before the scan it covers
            if rescan or folders != self.folders:
                self.gguf_files = await asyncio.to_thread(scan_models, self.cfg.model_dirs)
                self.folders = folders
            listing = _listing_key(self.gguf_files)
            if rescan or self.report_due or listing != self.reported:
                if listing != self.reported and self.reported is not None and not rescan:
                    log.info("models on disk changed: %s", [f["name"] for f in self.gguf_files])
                self.report_due = False   # a request that arrives during the POST asks again
                self.reported = None      # unknown until the coordinator answers
                await self._post("/nodes/models", {"files": self.gguf_files})
                self.reported = listing
                self.write_status()

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
        except StoppedWhileLoading as e:   # Unload / Stop serving while it loaded: nothing went wrong here
            log.info("command %s: %s", cmd["kind"], e)
            body = {"ok": False, "error": f"{type(e).__name__}: {e}"}
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

    def _refuse_if_stopped(self, pid: str) -> None:
        """A start whose stop already ran (the coordinator queued both before this Mac polled; stops run
        first) must not load anything."""
        if pid in self.stopped_early:
            del self.stopped_early[pid]
            raise StoppedWhileLoading("stopped before loading")

    def _stopped_meanwhile(self, pid: str) -> bool:
        """Whether a failed start failed because its pipeline was stopped (llama.cpp killed mid-load)."""
        return pid in self.stopping or pid not in self.in_use

    async def dispatch(self, kind: str, p: dict) -> dict:
        pid = p.get("pipeline_id")
        if kind == "start_worker":
            self._refuse_if_stopped(pid)
            self._enforce(pid, p["mem_bytes"])
            self.in_use[pid] = p["mem_bytes"]
            try:
                res = await self.engine.start_worker(pid, {**p, "device": self.cfg.commitment.device})
            except Exception as e:
                stopped = self._stopped_meanwhile(pid)
                self.in_use.pop(pid, None)
                if stopped:
                    raise StoppedWhileLoading() from e
                raise
            if pid not in self.in_use:   # stopped while starting: the stop found nothing to stop yet
                await self.engine.stop_worker(pid)
                raise StoppedWhileLoading()
            if self.transport == "relay":
                self.local_rpc[pid] = res["endpoint"]
                return {"endpoint": f"relay:{self.node_id}"}
            return res
        if kind == "start_head":
            self._refuse_if_stopped(pid)
            self._enforce(pid, p["mem_bytes"])
            self.in_use[pid] = p["mem_bytes"]
            try:
                if self.transport == "relay":
                    p = await self._relay_head_spec(p)
                res = await self.engine.start_head(pid, p)
            except Exception as e:
                stopped = self._stopped_meanwhile(pid)
                self.in_use.pop(pid, None)
                await self._close_proxies(pid)
                if stopped:
                    raise StoppedWhileLoading() from e
                raise
            if pid not in self.in_use:   # stopped while loading (Unload, Stop serving): don't leave it running
                await self.engine.stop_head(pid)
                await self._close_proxies(pid)
                raise StoppedWhileLoading()
            return res
        if kind in ("stop_worker", "stop_head"):
            if pid not in self.in_use:   # its start has not run yet (same poll): refuse it when it does
                self.stopped_early[pid] = None
                while len(self.stopped_early) > STOPPED_EARLY_MAX:
                    self.stopped_early.popitem(last=False)
            self.stopping.add(pid)
            try:
                await (self.engine.stop_worker if kind == "stop_worker" else self.engine.stop_head)(pid)
            finally:
                self.stopping.discard(pid)
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
        """The model was removed from the pool: delete the copy this pool sent here, and move any other copy
        (one in this Mac's models folder, its own uploads when it hosts a pool, one another pool sent) to the
        Trash only when the pool asked for it and this Mac allows it (--trash-removed); else keep it."""
        trash = bool(p.get("trash")) and self.cfg.trash_removed
        try:
            sent = await asyncio.to_thread(still_ours, self.cfg.download_dir,
                                           dict(self.downloaded.get(self.cfg.coordinator_url, {})))
            res = await asyncio.to_thread(remove_model_files, p.get("filename"), self.cfg.models_dir,
                                          self.cfg.download_dir, trash, sent)
            log.info("removed %s: %s", p.get("filename"), res)
            self._forget_downloads(res["deleted"])
            return res
        finally:   # report what is on disk now, whatever happened: a stale list would keep the model listed
            try:
                await self.sync_models(rescan=True)
            except httpx.HTTPError as e:
                log.warning("could not report models after removing %s: %s", p.get("filename"), e)
                self.write_status()

    # ------------------------------------------------------------ what each pool sent this Mac

    def _remember_download(self, path: Path) -> None:
        try:
            st = path.stat()
        except OSError as e:
            log.warning("could not record the download of %s: %s", path.name, e)
            return
        self.downloaded.setdefault(self.cfg.coordinator_url, {})[path.name] = \
            {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
        self._save_downloads()

    def _forget_downloads(self, names) -> None:
        """Deleted: no pool may claim a file of that name later (it would be someone else's)."""
        names = set(names)
        if not any(names & set(files) for files in self.downloaded.values()):
            return
        self.downloaded = {url: {n: rec for n, rec in files.items() if n not in names}
                           for url, files in self.downloaded.items()}
        self._save_downloads()

    def _save_downloads(self) -> None:
        try:
            save_json(self.downloaded_file, self.downloaded)
        except OSError as e:   # then this pool can't delete it later, only Trash or keep it: safe
            log.warning("could not record the downloaded models: %s", e)

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
                self._remember_download(dest)
            finally:
                if own is not None:
                    await own.aclose()
                tmp.unlink(missing_ok=True)
                self.downloads.pop(name, None)
        await self.sync_models(rescan=True)
        return {"path": str(found), "sha256": sha or ""}


def _listing_key(files: list[dict]) -> list:
    return [(f["name"], f.get("size")) for f in files]


def _load_downloaded(path: Path) -> dict[str, dict[str, dict]]:
    """``downloaded.json``: coordinator URL -> file name -> {size, mtime_ns} as downloaded. Unreadable: none
    (then no pool deletes anything here; copies go to the Trash or stay)."""
    try:
        data = load_state(str(path))
    except OSError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {url: {n: rec for n, rec in files.items() if isinstance(rec, dict)}
            for url, files in data.items() if isinstance(files, dict)}


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
