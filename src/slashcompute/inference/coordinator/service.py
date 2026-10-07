"""Inference coordinator: routes the main coordinator mounts (under ``/inference``, plus ``/v1/*``).

  /inference/nodes/*     inference nodes register, heartbeat, report benchmarks/models (bearer token)
  /inference/agent/*     command long-poll + token stream from the head (outbound from the node)
  /inference/relay/*     llama.cpp RPC relayed over WebSockets (TRANSPORT=relay)
  /inference/models/*    GGUF upload, download by heads; unload, Serve switch and remove (LLMs tab)
  /inference/status      snapshot for the GUI
  /v1/models, /v1/chat/completions   OpenAI-compatible API (credits via :class:`Accounting`)
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from slashcompute.common.jsonbool import body_bool
from slashcompute.inference import PREFIX
from slashcompute.inference import flops as fl
from slashcompute.inference.accounting import Accounting, AccountingError, NullAccounting
from slashcompute.inference.config import GIB, InferenceSettings
from slashcompute.inference.coordinator import nodes, registry, status
from slashcompute.inference.coordinator.bus import CommandBus, CommandFailed, JobStreams
from slashcompute.inference.coordinator.db import connect, tx
from slashcompute.inference.coordinator.layers import build_layout
from slashcompute.inference.coordinator.pipelines import PipelineManager, prompt_token_bound, serve
from slashcompute.inference.coordinator.planner import NoPlan, build_matches
from slashcompute.inference.coordinator.relay import Relay
from slashcompute.inference.gguf import read_header_file

log = logging.getLogger(__name__)
TOKEN_HEADER = "x-inference-token"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+\.gguf$")
DOWNLOAD_TIMEOUT_S = 6 * 3600
REMOVE_TIMEOUT_S = 120
DEFAULT_MAX_TOKENS = 256


def completion_limit(body: dict) -> int:
    """The one completion budget a request gets: sizes the context, the credit reservation and the engine's cap."""
    raw = next((body[k] for k in ("max_tokens", "max_completion_tokens", "n_predict") if body.get(k) is not None),
               DEFAULT_MAX_TOKENS)
    try:
        n = int(raw)
    except (TypeError, ValueError, OverflowError):
        n = 0
    if n <= 0 or isinstance(raw, bool):
        raise HTTPException(400, "max_tokens must be a positive integer.")
    return n


def max_ctx_for(row, s: InferenceSettings) -> int:
    """The longest context a pipeline for this model gets: its trained length (GGUF), else MAX_CTX."""
    return registry.model_layout(row).max_ctx or s.MAX_CTX


def ctx_for(body: dict, s: InferenceSettings, max_ctx: int) -> int:
    """Context to plan for: room for an upper bound on the prompt plus the whole completion budget, capped at
    the model's context length (a prompt that really doesn't fit is then rejected by the engine)."""
    limit = completion_limit(body)
    if limit >= max_ctx:
        raise HTTPException(400, f"max_tokens ({limit}) must be below the model's context length ({max_ctx} tokens).")
    need = prompt_token_bound(body) + limit + 64
    ctx = s.DEFAULT_CTX
    while ctx < min(need, max_ctx):
        ctx *= 2
    return min(ctx, max_ctx)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _bearer(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(401, "missing bearer token")
    return auth[7:].strip()


async def _json_body(request: Request) -> dict:
    """An admin route's JSON body (read after check_admin, so a stranger gets 401, not a 400)."""
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(400, "send a JSON body") from None
    if not isinstance(body, dict):
        raise HTTPException(400, "send a JSON object")
    return body


def _model_field(body: dict) -> str:
    name = body.get("model")
    if not isinstance(name, str) or not name:
        raise HTTPException(400, "model must be a model file name")
    return name


def _removal_error(name: str, error: str) -> str:
    """A Mac's failed removal as the LLMs tab shows it (the node's text names files, never folders)."""
    if "unknown command" in error:   # nodes from before remove_model
        return f"runs an older /compute: update it, or delete {name} from its models folder by hand"
    if "timed out" in error:
        return "did not answer in time"
    return re.sub(r"^\w+Error: ", "", error)


class InferenceService:
    """State and background loops of the inference coordinator."""

    def __init__(self, settings: InferenceSettings, accounting: Optional[Accounting] = None) -> None:
        self.s = settings
        self.accounting = accounting or NullAccounting()
        self.conn = connect(settings.DB_PATH)
        self.bus, self.streams = CommandBus(), JobStreams()
        self.mgr = PipelineManager(self.conn, settings, self.bus, self.streams, self.accounting)
        self.relay = Relay(self.bus, self.mgr, self.node_by_token)
        self.mgr.teardown_hooks.append(self.relay.close_pipeline)
        self.online: set[str] = set()
        self.tasks: set[asyncio.Task] = set()
        self.pushing: set[tuple[str, str]] = set()
        # model -> node id -> {"removing", "error", "kept"} while a Mac removes its copy, or after it failed
        # or kept it (an entry that finished cleanly is dropped). In memory only: Remove again redoes it.
        self.removals: dict[str, dict[str, dict]] = {}
        self.latency_kick: Optional[asyncio.Event] = None
        if settings.MODELS_DIR:
            Path(settings.MODELS_DIR).expanduser().mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ lifecycle

    def spawn(self, coro) -> asyncio.Task:
        t = asyncio.create_task(coro)
        self.tasks.add(t)
        t.add_done_callback(self.tasks.discard)
        return t

    def start(self) -> None:
        self.latency_kick = asyncio.Event()
        self.bus.closing = asyncio.Event()  # bound to this run's event loop
        self.mgr.recover()
        if self.s.BACKGROUND_TASKS:
            for loop in (self.monitor_loop, self.tick_loop, self.latency_loop):
                self.spawn(loop())

    async def stop(self) -> None:
        for t in list(self.tasks):
            t.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.mgr.shutdown()

    def kick_latency(self) -> None:
        if self.latency_kick is not None:
            self.latency_kick.set()

    def node_state(self, node_id: str) -> Optional[dict]:
        """Liveness of an inference node for its owner's node list (None: not an inference node)."""
        row = self.conn.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()
        if row is None:
            return None
        if not nodes.is_online(row, self.s):
            return {"online": False, "status": None}
        busy = node_id in nodes.reserved_bytes(self.conn)  # a member of a live pipeline
        return {"online": True, "status": "draining" if row["draining"] else "running" if busy else "idle"}

    # ------------------------------------------------------------ auth

    def node_by_token(self, token: str):
        return self.conn.execute("SELECT * FROM nodes WHERE token_hash=?", (hash_token(token),)).fetchone()

    def node_from(self, request: Request):
        row = self.node_by_token(_bearer(request))
        if row is None:
            raise HTTPException(401, "unknown node token")
        return row

    def check_token(self, request: Request) -> None:
        """When a shared secret is set (internet-facing pools), require it."""
        if self.s.TOKEN and not secrets.compare_digest(request.headers.get(TOKEN_HEADER, ""), self.s.TOKEN):
            raise HTTPException(401, "missing or wrong inference token")

    def check_admin(self, request: Request) -> None:
        """Model and pipeline management: the shared secret, plus an admin session on public pools."""
        self.check_token(request)
        try:
            self.accounting.require_admin(request)
        except AccountingError as e:
            raise HTTPException(e.status, str(e)) from e

    # ------------------------------------------------------------ background loops

    async def monitor_loop(self) -> None:
        while True:
            now = time.time()
            for row in self.conn.execute("SELECT * FROM nodes").fetchall():
                if nodes.is_online(row, self.s, now):
                    self.online.add(row["id"])
                elif row["id"] in self.online:
                    self.online.discard(row["id"])
                    log.warning("node %s offline (no heartbeat for %.1fs)", row["name"], self.s.OFFLINE_AFTER_SECONDS)
                    self.bus.fail_node(row["id"], f"node {row['name']} went offline")
                    await self.mgr.on_node_offline(row["id"], row["name"])
            await asyncio.sleep(min(1.0, self.s.HEARTBEAT_SECONDS / 2))

    async def tick_loop(self) -> None:
        while True:
            await asyncio.sleep(self.s.TICK_SECONDS)
            try:
                await self.mgr.tick()
            except Exception:  # noqa: BLE001
                log.exception("tick failed")

    async def latency_round(self) -> None:
        rows = [r for r in self.conn.execute("SELECT * FROM nodes").fetchall() if r["id"] in self.online]
        if self.s.TRANSPORT == "relay":
            # all RPC goes head -> coordinator -> worker: measure each node's RTT to us
            for r in rows:
                try:
                    res = await self.bus.send(r["id"], "measure_latency", {"mode": "relay"}, timeout=30)
                    self.store_rtt(r["id"], res.get("coordinator_rtt_ms"))
                except CommandFailed as e:
                    log.info("latency from %s failed: %s", r["name"], e)
            return
        for r in rows:
            peers = [{"node_id": p["id"], "ip": p["tailscale_ip"], "port": p["probe_port"]}
                     for p in rows if p["id"] != r["id"] and p["tailscale_ip"] and p["probe_port"]]
            if not peers:
                continue
            try:
                res = await self.bus.send(r["id"], "measure_latency", {"peers": peers}, timeout=30)
            except CommandFailed as e:
                log.info("latency from %s failed: %s", r["name"], e)
                continue
            self.store_latency(r["id"], res.get("results", {}))

    async def latency_loop(self) -> None:
        while True:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self.latency_kick.wait(), self.s.LATENCY_INTERVAL_SECONDS)
            self.latency_kick.clear()
            await asyncio.sleep(0.2)
            try:
                await self.latency_round()
            except Exception:  # noqa: BLE001
                log.exception("latency round failed")

    def store_rtt(self, node_id: str, rtt_ms) -> None:
        """Relay mode: one-way delay a -> b = a -> relay + relay -> b = (rtt_a + rtt_b) / 2."""
        if rtt_ms is None:
            return
        conn = self.conn
        conn.execute("UPDATE nodes SET rtt_ms=? WHERE id=?", (float(rtt_ms), node_id))
        rtts = {r["id"]: r["rtt_ms"] for r in conn.execute("SELECT id, rtt_ms FROM nodes WHERE rtt_ms IS NOT NULL")}
        now = time.time()
        with tx(conn):
            for other, ms in rtts.items():
                if other == node_id:
                    continue
                one_way = (float(rtt_ms) + ms) / 2
                for a, b in ((node_id, other), (other, node_id)):
                    conn.execute("INSERT OR REPLACE INTO latency (a, b, one_way_ms, measured_at) VALUES (?,?,?,?)",
                                 (a, b, one_way, now))

    def store_latency(self, node_id: str, results: dict) -> None:
        now = time.time()
        with tx(self.conn):
            for peer, ms in results.items():
                if ms is not None:
                    self.conn.execute("INSERT OR REPLACE INTO latency (a, b, one_way_ms, measured_at) VALUES (?,?,?,?)",
                                      (node_id, peer, float(ms), now))

    # ------------------------------------------------------------ models

    def accept_headers(self, node_id: str, files: list[dict]) -> None:
        for f in files:
            if f.get("headers") and registry.accept_node_header(
                    self.conn, self.s, node_id, f["name"], int(f["size"]), f["headers"]):
                log.info("model %s: layer table from node %s", f["name"], node_id)

    def models_dir(self) -> Path:
        if not self.s.MODELS_DIR:
            raise HTTPException(404, "model uploads are not enabled on this coordinator")
        return Path(self.s.MODELS_DIR).expanduser()

    def push_models(self, node_id: Optional[str] = None) -> int:
        """Send ``download_model`` for every uploaded GGUF to each online head-capable node lacking it."""
        if not self.s.MODELS_DIR:
            return 0
        files = self.conn.execute("SELECT * FROM model_files").fetchall()
        rows = self.conn.execute("SELECT * FROM nodes WHERE can_head=1").fetchall()
        sent = 0
        for n in rows:
            if (node_id and n["id"] != node_id) or not nodes.is_online(n, self.s):
                continue
            have = set(json.loads(n["gguf_files_json"] or "[]"))
            for f in files:
                key = (n["id"], f["name"])
                if f["name"] in have or key in self.pushing:
                    continue
                self.pushing.add(key)
                self.spawn(self._push(n["id"], n["name"], f["name"], f["sha256"], f["size"]))
                sent += 1
        return sent

    async def _push(self, node_id: str, name: str, filename: str, sha256: str, size: int) -> None:
        try:
            log.info("sending %s to %s", filename, name)
            await self.bus.send(node_id, "download_model", {
                "filename": filename, "path": f"/models/files/{filename}", "sha256": sha256, "size": size,
            }, timeout=DOWNLOAD_TIMEOUT_S)
        except CommandFailed as e:
            log.warning("%s could not fetch %s: %s", name, filename, e)
        finally:
            self.pushing.discard((node_id, filename))

    def removing(self, name: str) -> bool:
        return any(e["removing"] for e in self.removals.get(name, {}).values())

    def _refuse_while_removing(self, name: str) -> None:
        if self.removing(name):
            raise HTTPException(409, f"{name} is still being removed from the pool's Macs; try again in a moment")

    async def receive_upload(self, request: Request, name: str) -> dict:
        if not _SAFE_NAME.match(name or ""):
            raise HTTPException(400, "file name must look like model-name.gguf (letters, digits, . _ -)")
        root = self.models_dir()
        self._refuse_while_removing(name)   # a Mac still deleting it would delete the new copy too
        dest = root / name
        tmp = root / f".{name}.{uuid.uuid4().hex[:6]}.part"
        h = hashlib.sha256()
        size = 0
        try:
            with tmp.open("wb") as fh:
                async for chunk in request.stream():
                    fh.write(chunk)
                    h.update(chunk)
                    size += len(chunk)
            if size == 0:
                raise HTTPException(400, "empty upload")
            try:
                header = read_header_file(tmp)
                layout = build_layout(name, [header], size, self.s.KV_BYTES_PER_ELEMENT)
                if layout.n_layers == 0:
                    raise ValueError("no transformer layers found")
                mflops = fl.ModelFlops.from_headers([header])
            except Exception as e:  # noqa: BLE001
                raise HTTPException(422, f"not a usable GGUF model: {e}") from e
            self._refuse_while_removing(name)
            os.replace(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)
        digest = h.hexdigest()
        row = registry.model_row(self.conn, name)
        # a stopped model stays stopped; uploading a removed one again is how it comes back
        keep = {"status": "disabled", "status_reason": row["status_reason"]} \
            if row is not None and row["status"] == "disabled" else {}
        with tx(self.conn):
            self.conn.execute("INSERT OR REPLACE INTO model_files (name, size, sha256, uploaded_at) VALUES (?,?,?,?)",
                              (name, size, digest, time.time()))
            registry.store_layout(self.conn, name, layout, "upload", mflops, size_bytes=size, sha256=digest, **keep)
        self.removals.pop(name, None)
        pushed = self.push_models()
        log.info("uploaded %s (%.2f GB); pushing to %d head(s)", name, size / 1e9, pushed)
        return {"name": name, "size": size, "sha256": digest, "layers": layout.n_layers, "pushed": pushed}

    # ------------------------------------------------------------ chat

    def model_or_404(self, model_id: str):
        row = registry.model_row(self.conn, model_id)
        why = registry.unavailable(row, model_id)
        if why:
            raise HTTPException(404, why)
        return row

    # ------------------------------------------------------------ unload, Serve switch, remove (LLMs tab)

    def known_or_404(self, name: str):
        row = registry.model_row(self.conn, name)
        if row is None:
            raise HTTPException(404, f"no model {name!r} in this pool")
        return row

    async def unload(self, name: str) -> dict:
        """Stop the model's pipelines everywhere (the reply in progress finishes). The next chat loads it again."""
        self.known_or_404(name)
        return {"ok": True, "pipelines": await self.mgr.unload(name, "unloaded from the app")}

    async def set_serving(self, name: str, on: bool) -> dict:
        """The pool-wide Serve switch. Off unloads the model and keeps it from loading until switched on;
        the files stay on disk and the setting survives restarts and node reports."""
        row = self.known_or_404(name)
        status = row["status"]
        if on:
            if status == "ready":
                return {"ok": True, "serving": True, "pipelines": 0}
            if status not in ("disabled", "removed") or not row["layout_json"]:
                raise HTTPException(409, f"{name} has no usable layer table: upload it again")
            if status == "removed":
                self._refuse_while_removing(name)
            with tx(self.conn):
                self.conn.execute("UPDATE models SET status='ready', status_reason=NULL, updated_at=? WHERE id=?",
                                  (time.time(), name))
            self.removals.pop(name, None)
            log.info("model %s: serving again", name)
            return {"ok": True, "serving": True, "pipelines": 0}
        if status not in ("ready", "disabled"):
            raise HTTPException(409, f"{name} is {status}: there is nothing to stop serving")
        # the status first (synchronously): a chat arriving while the pipelines drain is already refused
        with tx(self.conn):
            self.conn.execute("UPDATE models SET status='disabled', status_reason=?, updated_at=? WHERE id=?",
                              ("stopped in the LLMs tab", time.time(), name))
        n = await self.mgr.unload(name, "stopped serving from the app")
        log.info("model %s: stopped serving (%d pipeline(s) stopped)", name, n)
        return {"ok": True, "serving": False, "pipelines": n}

    async def remove_model(self, name: str, trash: bool) -> dict:
        """Drop a model from the pool: unload it, delete the uploaded copy and ask every online Mac that
        has it to delete the app's copy (and, with ``trash``, move one in its own models folder to the Trash).
        The row stays as a tombstone so node reports don't bring the model back."""
        if not registry.plain_gguf(name):
            raise HTTPException(400, "model must be a .gguf file name")
        row = registry.model_row(self.conn, name)
        uploaded = self.conn.execute("SELECT 1 FROM model_files WHERE name=?", (name,)).fetchone() is not None
        holders = [r for r in self.conn.execute("SELECT * FROM nodes ORDER BY created_at").fetchall()
                   if name in json.loads(r["gguf_files_json"] or "[]")]
        if not ((row is not None and row["status"] != "removed") or uploaded or holders):
            raise HTTPException(404, f"no model {name!r} to remove")
        now = time.time()
        with tx(self.conn):
            self.conn.execute("INSERT OR IGNORE INTO models (id, created_at) VALUES (?, ?)", (name, now))
            self.conn.execute("UPDATE models SET status='removed', status_reason=?, updated_at=? WHERE id=?",
                              ("removed in the LLMs tab", now, name))
            self.conn.execute("DELETE FROM model_files WHERE name=?", (name,))  # stops pushes and downloads
        if uploaded and self.s.MODELS_DIR:
            try:
                (Path(self.s.MODELS_DIR).expanduser() / name).unlink(missing_ok=True)
            except OSError as e:
                log.warning("could not delete the uploaded %s: %s", name, e.strerror or e)
        await self.mgr.unload(name, "removed from the app")
        macs, offline = [], []
        for r in holders:
            if not nodes.is_online(r, self.s):
                offline.append(r["name"])
                continue
            macs.append(r["name"])
            entry = {"removing": True, "error": None, "kept": False}
            self.removals.setdefault(name, {})[r["id"]] = entry
            self.spawn(self._remove_on(r["id"], name, trash, entry))
        log.info("model %s removed (uploaded copy: %s); asking %s", name, uploaded, macs or "no Mac")
        return {"ok": True, "model": name, "macs": macs, "offline": offline, "uploaded": uploaded}

    async def _remove_on(self, node_id: str, name: str, trash: bool, entry: dict) -> None:
        try:
            res = await self.bus.send(node_id, "remove_model", {"filename": name, "trash": trash}, REMOVE_TIMEOUT_S)
        except CommandFailed as e:
            log.warning("removing %s on %s failed: %s", name, node_id, e)
            entry.update(removing=False, error=_removal_error(name, str(e)))
            return
        entry.update(removing=False, kept=bool(res.get("kept")))
        per_node = self.removals.get(name, {})
        if not entry["kept"] and per_node.get(node_id) is entry:   # done: nothing left to show for this Mac
            del per_node[node_id]
            if not per_node:
                self.removals.pop(name, None)

    def members_view(self, members) -> list[dict]:
        names = nodes.node_names(self.conn)
        return [{"node": names.get(m.node_id, m.node_id), "role": m.role,
                 "layers": [m.layer_start, m.layer_end - 1] if m.n_layers else [],
                 "share": round(m.share, 4)} for m in members]


def _commitment_fields(c: dict) -> dict:
    return {
        "committed_bytes": int(float(c.get("memory_gb", 0)) * GIB),
        "hours_json": json.dumps(c.get("hours") or [[0, 24]]),
        "allowed_models_json": json.dumps(c["allowed_models"]) if c.get("allowed_models") else None,
        "can_head": int(bool(c.get("may_be_head", False))),
        "device": c.get("device"),
    }


def make_router(svc: InferenceService) -> APIRouter:
    """Node, agent, relay, model and status routes (mounted under ``PREFIX``)."""
    r = APIRouter()
    s, conn, bus, mgr = svc.s, svc.conn, svc.bus, svc.mgr

    @r.websocket("/relay/head/{pipeline_id}/{worker_id}")
    async def relay_head(ws: WebSocket, pipeline_id: str, worker_id: str):
        await svc.relay.head(ws, pipeline_id, worker_id)

    @r.websocket("/relay/worker/{stream_id}")
    async def relay_worker(ws: WebSocket, stream_id: str):
        await svc.relay.worker(ws, stream_id)

    @r.get("/ping")
    async def ping():
        return {"ok": True, "t": time.time()}

    @r.get("/health")
    async def health():
        return {"ok": True, "nodes_online": len(svc.online), "pipelines": len(mgr.live()), "transport": s.TRANSPORT}

    @r.get("/status")
    async def status_view():
        return status.snapshot(conn, s, svc.removals)

    # ------------------------------------------------------------ nodes

    @r.post("/nodes/register")
    async def register_node(body: dict, request: Request):
        svc.check_token(request)
        try:
            svc.accounting.admit_node(body.get("session_token"))
        except AccountingError as e:
            raise HTTPException(e.status, str(e)) from e
        build = body.get("llama_build", "")
        if not build_matches(build, s.PINNED_LLAMA_BUILD):
            raise HTTPException(409, f"llama.cpp build {build!r} does not match the pinned build "
                                     f"{s.PINNED_LLAMA_BUILD!r}; RPC breaks across builds")
        now = time.time()
        fields = {
            "name": body["name"], "os": body.get("os"), "chip": body.get("chip"),
            "total_mem_bytes": body.get("total_mem_bytes"), "tailscale_ip": body.get("tailscale_ip"),
            "probe_port": body.get("probe_port"), "llama_build": build,
            "gguf_files_json": json.dumps(sorted({f["name"] for f in body.get("gguf_files", [])})),
            "available": 1, "draining": 0,
            **_commitment_fields(body.get("commitment", {})),
        }
        existing = None
        if body.get("node_id") and body.get("token"):
            existing = conn.execute("SELECT * FROM nodes WHERE id=? AND token_hash=?",
                                    (body["node_id"], hash_token(body["token"]))).fetchone()
        with tx(conn):
            if existing:
                node_id, token = existing["id"], body["token"]
                conn.execute(f"UPDATE nodes SET {', '.join(f'{k}=?' for k in fields)}, last_heartbeat=? WHERE id=?",
                             (*fields.values(), now, node_id))
            else:
                node_id, token = "n-" + secrets.token_hex(4), secrets.token_urlsafe(24)
                fields.update(id=node_id, token_hash=hash_token(token), created_at=now, last_heartbeat=now)
                conn.execute(f"INSERT INTO nodes ({', '.join(fields)}) VALUES ({', '.join('?' * len(fields))})",
                             tuple(fields.values()))
        svc.accept_headers(node_id, body.get("gguf_files", []))
        if body.get("session_token"):
            try:
                svc.accounting.bind_node(node_id, body["session_token"])
            except Exception:  # noqa: BLE001 - a stale session must not keep the node out
                log.exception("could not bind node %s to its owner", node_id)
        if body.get("coordinator_rtt_ms") is not None:
            svc.store_rtt(node_id, body["coordinator_rtt_ms"])
        svc.online.add(node_id)
        svc.kick_latency()
        if fields["can_head"]:
            svc.push_models(node_id)
        log.info("inference node %s (%s) registered: %s, commits %.0f GiB", body["name"], node_id, body.get("chip"),
                 fields["committed_bytes"] / GIB)
        approved = conn.execute("SELECT approved_head FROM nodes WHERE id=?", (node_id,)).fetchone()["approved_head"]
        return {"node_id": node_id, "token": token, "heartbeat_seconds": s.HEARTBEAT_SECONDS,
                "pinned_build": s.PINNED_LLAMA_BUILD, "transport": s.TRANSPORT,
                "head_approved": bool(approved) or not s.REQUIRE_HEAD_APPROVAL}

    @r.post("/nodes/heartbeat")
    async def heartbeat(body: dict, request: Request):
        row = svc.node_from(request)
        available = int(bool(body.get("available", True)))
        if body.get("llama_build") and not build_matches(body["llama_build"], s.PINNED_LLAMA_BUILD):
            available = 0
        conn.execute("UPDATE nodes SET last_heartbeat=?, available=?, downloads_json=? WHERE id=?",
                     (time.time(), available, json.dumps(body.get("downloads") or {}), row["id"]))
        svc.online.add(row["id"])
        mgr.reconcile(row["id"], body.get("pipelines") or [])
        if row["available"] and not available:
            # paused, training took the Mac, or shutting down: finish the current job, then leave pipelines
            reason = body.get("reason") or ("node shutting down" if body.get("leaving") else "node unavailable")
            await mgr.drain_node(row["id"], reason)
        if available and not row["available"] and row["can_head"]:
            svc.push_models(row["id"])
        return {"ok": True, "transport": s.TRANSPORT}

    @r.post("/nodes/commitment")
    async def commitment(body: dict, request: Request):
        row = svc.node_from(request)
        new = _commitment_fields(body)
        lowered = (new["committed_bytes"] < (row["committed_bytes"] or 0) or (row["can_head"] and not new["can_head"])
                   or new["hours_json"] != row["hours_json"] or new["allowed_models_json"] != row["allowed_models_json"])
        with tx(conn):
            conn.execute(f"UPDATE nodes SET {', '.join(f'{k}=?' for k in new)} WHERE id=?", (*new.values(), row["id"]))
        if lowered:
            await mgr.drain_node(row["id"], "commitment changed")
        return {"ok": True, "draining": bool(lowered)}

    @r.post("/nodes/benchmark")
    async def benchmark(body: dict, request: Request):
        row = svc.node_from(request)
        with tx(conn):
            conn.execute("UPDATE nodes SET prompt_score=?, gen_score=? WHERE id=?",
                         (body["prompt_score"], body["gen_score"], row["id"]))
            conn.execute("INSERT INTO node_benchmarks (node_id, at, prompt_score, gen_score, raw_json) "
                         "VALUES (?,?,?,?,?)",
                         (row["id"], time.time(), body["prompt_score"], body["gen_score"], json.dumps(body)))
        return {"ok": True}

    @r.post("/nodes/models")
    async def node_models(body: dict, request: Request):
        row = svc.node_from(request)
        files = body.get("files", [])
        conn.execute("UPDATE nodes SET gguf_files_json=? WHERE id=?",
                     (json.dumps(sorted({f["name"] for f in files})), row["id"]))
        svc.accept_headers(row["id"], files)
        return {"ok": True}

    @r.post("/nodes/latency")
    async def node_latency(body: dict, request: Request):
        row = svc.node_from(request)
        if "coordinator_rtt_ms" in body:
            svc.store_rtt(row["id"], body["coordinator_rtt_ms"])
        svc.store_latency(row["id"], body.get("results", {}))
        return {"ok": True}

    # ------------------------------------------------------------ agent channel

    @r.get("/agent/commands")
    async def agent_commands(request: Request, wait: float = 25.0):
        row = svc.node_from(request)
        conn.execute("UPDATE nodes SET last_heartbeat=? WHERE id=?", (time.time(), row["id"]))
        commands = await bus.poll(row["id"], min(wait, s.COMMAND_LONG_POLL_SECONDS))
        if not commands and bus.closing.is_set():  # an empty 200 would have agents re-poll in a hot loop
            raise HTTPException(503, "coordinator shutting down", headers={"Retry-After": "1"})
        return {"commands": commands}

    @r.post("/agent/commands/{cid}/result")
    async def agent_result(cid: str, body: dict, request: Request):
        row = svc.node_from(request)
        bus.resolve(row["id"], cid, bool(body.get("ok")), body.get("result"), body.get("error"))
        return {"ok": True}

    @r.post("/agent/jobs/{job_id}/stream")
    async def agent_stream(job_id: str, request: Request):
        row = svc.node_from(request)
        job = conn.execute("SELECT j.id, p.head_node_id FROM jobs j JOIN pipelines p ON p.id = j.pipeline_id "
                           "WHERE j.id=?", (job_id,)).fetchone()
        if job is None or job["head_node_id"] != row["id"]:
            raise HTTPException(403, "not the head of this job")
        buf = b""
        async for part in request.stream():
            buf += part
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if line.strip():
                    svc.streams.push(job_id, json.loads(line))
        if buf.strip():
            svc.streams.push(job_id, json.loads(buf))
        return {"ok": True}

    # ------------------------------------------------------------ models + pipelines

    @r.post("/models/upload")
    async def upload_model(request: Request, name: str = ""):
        svc.check_admin(request)
        return await svc.receive_upload(request, name or request.headers.get("x-filename", ""))

    @r.get("/models/files/{name}")
    async def model_file(name: str, request: Request):
        svc.node_from(request)
        root = svc.models_dir()
        row = conn.execute("SELECT name FROM model_files WHERE name=?", (name,)).fetchone()
        if row is None or not (root / name).is_file():
            raise HTTPException(404, "no such uploaded model")
        return FileResponse(root / name, media_type="application/octet-stream")

    # The model travels in the JSON body: the shell's proxy rebuilds decoded paths, and names may have spaces.
    @r.post("/models/unload")
    async def unload_model(request: Request):
        svc.check_admin(request)
        return await svc.unload(_model_field(await _json_body(request)))

    @r.post("/models/serving")
    async def model_serving(request: Request):
        svc.check_admin(request)
        body = await _json_body(request)
        name = _model_field(body)
        if body.get("on") is None:   # body_bool would read a missing value as False: Stop serving by accident
            raise HTTPException(400, "on must be true or false.")
        return await svc.set_serving(name, body_bool(body, "on"))

    @r.post("/models/remove")
    async def remove_model(request: Request):
        svc.check_admin(request)
        return await svc.remove_model(_model_field(await _json_body(request)), trash=True)

    @r.delete("/models/{name}")
    async def delete_model(name: str, request: Request):
        """The older UI's Remove (no confirm): the same removal, but never moves anyone's files to the Trash."""
        svc.check_admin(request)
        return await svc.remove_model(name, trash=False)

    @r.get("/plan")
    async def plan_view(model: str, max_tokens: int = DEFAULT_MAX_TOKENS):
        row = svc.model_or_404(model)
        body = {"model": model, "max_tokens": max_tokens}
        ctx = ctx_for(body, s, max_ctx_for(row, s))
        active = next((rt for rt in mgr.live(row["id"]) if rt.state == "active" and rt.ctx >= ctx), None)
        if active:
            view = {"source": "active pipeline", "pipeline_id": active.id,
                    "est_tok_s": conn.execute("SELECT est_tok_s FROM pipelines WHERE id=?",
                                              (active.id,)).fetchone()["est_tok_s"],
                    "members": svc.members_view(active.members)}
        else:
            p = mgr.plan_for(row["id"], ctx)
            view = {"source": "no plan", "reason": p.reason} if isinstance(p, NoPlan) else {
                "source": "planner dry run", "est_tok_s": p.est_tok_s, "explanation": p.explanation,
                "tensor_split": list(p.tensor_split), "members": svc.members_view(p.members)}
        return {"model": row["id"], "ctx": ctx, "gen_weight": mgr.gen_weight_for(row["id"]), "plan": view}

    @r.post("/pipelines/{pipeline_id}/stop")
    async def stop_pipeline(pipeline_id: str, request: Request):
        svc.check_admin(request)
        if pipeline_id not in mgr.runtimes:
            raise HTTPException(404, "unknown pipeline")
        svc.spawn(mgr.stop_pipeline(pipeline_id, "unloaded from the app"))
        return {"ok": True}

    return r


def make_v1_router(svc: InferenceService) -> APIRouter:
    """OpenAI-compatible ``/v1/models`` and ``/v1/chat/completions``."""
    r = APIRouter()
    s, conn, mgr = svc.s, svc.conn, svc.mgr

    @r.get("/v1/models")
    async def list_models(request: Request):
        svc.check_token(request)
        data = [{"id": row["id"], "object": "model", "owned_by": "slashcompute", "arch": row["arch"],
                 "moe": bool(row["moe"]), "size_gb": round((row["size_bytes"] or 0) / 1e9, 2),
                 "heads": [h["name"] for h in registry.holders(conn, s, row["id"])]}
                for row in registry.servable_models(conn, s)]
        return {"object": "list", "data": data}

    @r.post("/v1/chat/completions")
    async def chat(body: dict, request: Request):
        svc.check_token(request)
        try:
            user_id = svc.accounting.requester(request)
        except AccountingError as e:
            raise HTTPException(e.status, str(e)) from e
        row = svc.model_or_404(body.get("model", ""))
        max_tokens = completion_limit(body)
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages or not all(isinstance(m, dict) for m in messages):
            raise HTTPException(400, "messages must be a non-empty list of message objects.")
        stream = body_bool(body, "stream")
        ctx = ctx_for(body, s, max_ctx_for(row, s))
        # the engine gets exactly the budget we reserved for (llama-server alone would run to the end of the ctx)
        engine_body = {k: v for k, v in body.items()
                       if k not in ("stream", "stream_options", "model", "max_completion_tokens", "n_predict")}
        engine_body["max_tokens"] = max_tokens
        account_id = "inf-" + uuid.uuid4().hex[:12]
        reserved = False
        if user_id:
            # a true upper bound (the charge is capped at it); settle() refunds the rest
            est = fl.estimate_flops(registry.model_flops(row), prompt_token_bound(body), max_tokens, s.GEN_WEIGHT_MAX)
            try:
                svc.accounting.reserve(user_id, account_id, est)
            except AccountingError as e:
                raise HTTPException(e.status, str(e)) from e
            reserved = True

        def settle() -> None:
            if reserved:
                try:
                    svc.accounting.settle(account_id)
                except Exception:  # noqa: BLE001
                    log.exception("settling %s failed", account_id)

        events = serve(mgr, row["id"], engine_body, ctx, user_id or "anonymous", stream, account_id)

        if not stream:
            text, reasoning = [], []

            async def collect() -> Optional[dict]:
                """The 'final' or 'error' event, accumulating the reply text on the way."""
                async for ev in events:
                    if ev["type"] == "reset":
                        text.clear()
                        reasoning.clear()
                    elif ev["type"] == "chunk":
                        delta = ev["data"]["choices"][0]["delta"]
                        text.append(delta.get("content") or "")
                        # thinking models (Qwen3 etc.) stream their reasoning separately from the answer
                        reasoning.append(delta.get("reasoning_content") or "")
                    elif ev["type"] in ("error", "final"):
                        return ev
                return None

            async def hung_up() -> None:
                # the body is read, so the next ASGI message is the disconnect (request.is_disconnected()
                # never sees it behind the main coordinator's BaseHTTPMiddleware)
                while (await request.receive())["type"] != "http.disconnect":
                    pass

            # nothing is written until the reply is done, so watch for a hang-up (curl -m, a closed tab)
            # and cancel serve() then: the head stops and only the partial work is charged
            collector, watcher = asyncio.ensure_future(collect()), asyncio.ensure_future(hung_up())
            try:
                await asyncio.wait({collector, watcher}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                watcher.cancel()
                collector.cancel()  # a no-op once it has finished
                await asyncio.wait({collector})
                # like sse(): serve() has cancelled the job on the head and recorded the tokens so far
                # *before* settle() releases the reservation
                await events.aclose()
                settle()
            if collector.cancelled():
                return JSONResponse({"error": {"message": "client disconnected", "retryable": True,
                                               "job_id": None}}, status_code=499)
            final = collector.result()
            if final is None or final["type"] == "error":
                err = final or {"error": "no response", "status": 503, "retryable": True, "job_id": None}
                return JSONResponse({"error": {"message": err["error"], "retryable": err["retryable"],
                                               "job_id": err["job_id"]}}, status_code=err["status"])
            summ = final["summary"]
            message = {"role": "assistant", "content": "".join(text)}
            if any(reasoning):
                message["reasoning_content"] = "".join(reasoning)
            return {
                "id": summ["job_id"], "object": "chat.completion", "created": int(time.time()), "model": row["id"],
                "choices": [{"index": 0, "message": message,
                             "finish_reason": final.get("finish_reason") or "stop"}],
                "usage": {"prompt_tokens": summ["prompt_n"] + summ["cache_n"], "completion_tokens": summ["predicted_n"],
                          "total_tokens": summ["prompt_n"] + summ["cache_n"] + summ["predicted_n"]},
                "network": summ,
            }

        # streaming: hold the response until the first event so routing errors get a real status code
        first = None
        async for ev in events:
            if ev["type"] != "reset":
                first = ev
                break
        if first is None or first["type"] == "error":
            settle()
            err = first or {"error": "no response", "status": 503, "retryable": True, "job_id": None}
            return JSONResponse({"error": {"message": err["error"], "retryable": err["retryable"],
                                           "job_id": err["job_id"]}}, status_code=err["status"])

        async def sse():
            try:
                ev = first
                while True:
                    if ev["type"] == "chunk":
                        yield f"data: {json.dumps({**ev['data'], 'model': row['id']})}\n\n"
                    elif ev["type"] == "final":
                        summ = ev["summary"]
                        yield "data: " + json.dumps({
                            "object": "chat.completion.chunk", "model": row["id"], "choices": [],
                            "usage": {"prompt_tokens": summ["prompt_n"] + summ["cache_n"],
                                      "completion_tokens": summ["predicted_n"]},
                            "network": summ}) + "\n\n"
                        break
                    elif ev["type"] == "error":
                        yield "data: " + json.dumps({"error": {"message": ev["error"],
                                                               "retryable": ev["retryable"]}}) + "\n\n"
                        break
                    ev = await anext(events, None)
                    if ev is None:
                        break
                yield "data: [DONE]\n\n"
            finally:
                # a disconnect closes us here: close serve() too, so it cancels the job on the head and
                # records the tokens generated so far *before* settle() releases the reservation
                await events.aclose()
                settle()

        return StreamingResponse(sse(), media_type="text/event-stream")

    return r


def mount(app: FastAPI, svc: InferenceService) -> None:
    app.include_router(make_router(svc), prefix=PREFIX)
    app.include_router(make_v1_router(svc))


def create_inference_app(settings: Optional[InferenceSettings] = None,
                         accounting: Optional[Accounting] = None) -> FastAPI:
    """Standalone inference coordinator (tests, debugging)."""
    svc = InferenceService(settings or InferenceSettings.from_env(), accounting)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        svc.start()
        yield
        await svc.stop()

    app = FastAPI(title="slashcompute inference", lifespan=lifespan)
    app.state.inference = svc
    mount(app, svc)
    return app
