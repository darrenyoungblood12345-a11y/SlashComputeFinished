"""HTTP + WebSocket front end for the coordinator."""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import ValidationError
from sqlmodel import select

from slashcompute.community.credits import CreditError
from slashcompute.community.http import _token, mount_community
from slashcompute.common.config import EngineConfig
from slashcompute.common.protocol import Register, dump, parse_agent_message
from slashcompute.coordinator.core import MAX_DATASET_BYTES, Coordinator
from slashcompute.coordinator.db import Verification
from slashcompute.coordinator.inference_accounting import CoreAccounting
from slashcompute.inference.config import InferenceSettings
from slashcompute.inference.coordinator.service import InferenceService, mount
from slashcompute.jobs import parse_spec

log = logging.getLogger(__name__)


class AgentConnection:
    """One agent WebSocket. Sends go into a queue that a writer task drains, so the
    coordinator's loops never wait on a slow socket. A socket that stalls or backs up
    is closed instead; an agent with a reliable session reconnects and gets the
    messages it missed replayed."""

    SEND_TIMEOUT_S = 30.0
    MAX_QUEUED = 10_000

    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self._queue: asyncio.Queue = asyncio.Queue()
        self._closed = False
        self._writer = asyncio.create_task(self._write())

    async def send(self, msg) -> None:
        if self._closed:
            return
        if self._queue.qsize() >= self.MAX_QUEUED:
            log.warning("agent socket has %d unsent messages; closing it", self._queue.qsize())
            await self.close()
            return
        self._queue.put_nowait(msg)

    async def _write(self) -> None:
        try:
            while True:
                msg = await self._queue.get()
                await asyncio.wait_for(self.ws.send_text(dump(msg)), self.SEND_TIMEOUT_S)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.warning("agent socket send failed (%r); closing it", e)
            await self.close()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self.ws.close()
        except Exception:
            pass  # already closed

    def stop(self) -> None:
        self._closed = True
        self._writer.cancel()


def create_app(cfg: EngineConfig, advertise: bool = False,
               inference: Optional[InferenceSettings] = None) -> FastAPI:
    core = Coordinator(cfg)
    inf = inference or InferenceSettings.from_env(
        DB_PATH=str(cfg.home / "inference.sqlite3"), MODELS_DIR=str(cfg.home / "models"))
    svc = InferenceService(inf, CoreAccounting(core))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        core.start()
        svc.start()
        adv = None
        if advertise:
            try:
                from slashcompute.common.discovery import Advertiser

                adv = Advertiser(cfg.coordinator_port)
                await adv.start()
                log.info("advertising on the LAN as %s", adv.name)
            except Exception as e:
                adv = None
                log.warning("mDNS advertising unavailable: %r", e)
        app.state.advertiser = adv
        yield
        if adv:
            await adv.close()
        await svc.stop()
        await core.stop()

    app = FastAPI(title="/compute coordinator", lifespan=lifespan)
    app.state.core = core
    app.state.inference = svc
    mount(app, svc)

    @app.middleware("http")
    async def session_cookie(request: Request, call_next):
        response = await call_next(request)
        token = getattr(request.state, "session_token", None)
        if token:
            response.set_cookie("slashcompute_session", token, httponly=True, samesite="lax")
        if request.url.path.rstrip("/").endswith("/auth/logout"):
            response.delete_cookie("slashcompute_session")
        return response

    mount_community(app, core)

    def _user(request: Request, authorization: Optional[str] = None):
        user = core.auth.session_user(_token(request, authorization))
        if user is not None and user.banned:
            raise HTTPException(403, "This account is banned.")
        return user

    def _reserve(user, job, max_flops: Optional[float]):
        if user is None:
            return
        def fail(status: int, msg: str):
            core.abandon_job(job, msg)
            raise HTTPException(status, msg)
        if user.accepted_terms_at is None:
            fail(403, "Accept the terms before taking from the pool.")
        if max_flops is None:
            fail(400, "Set a FLOP budget (max_flops) to take.")
        try:
            budget = float(max_flops)
        except (TypeError, ValueError):
            fail(400, "max_flops must be a number.")
        try:
            core.credits.reserve_job(user.id, job.id, budget)
        except CreditError as e:
            core.abandon_job(job, str(e))
            raise HTTPException(e.status, str(e)) from e
        except Exception:
            core.abandon_job(job, "Could not reserve credits.")
            raise

    # ------------------------------------------------------------ agents

    @app.websocket("/ws/agent")
    async def agent_ws(ws: WebSocket):
        await ws.accept()
        conn = AgentConnection(ws)
        send = conn.send  # one object: the stale-session check below compares by identity

        node_id: Optional[str] = None
        try:
            first = parse_agent_message(await ws.receive_text())
            if not isinstance(first, Register):
                await ws.close(code=4000, reason="first message must be register")
                return
            node_id = first.node_id
            try:
                await core.on_register(first, send, close=conn.close)
            except PermissionError as e:
                await ws.close(code=4003, reason=str(e)[:123] or "banned")
                return
            while True:
                raw = await ws.receive_text()
                try:
                    msg = parse_agent_message(raw)
                except ValidationError as e:
                    log.warning("bad message from %s: %s", node_id[:8], e)
                    continue
                await core.handle(node_id, msg)
        except WebSocketDisconnect:
            pass
        except Exception:
            log.exception("agent session error")
        finally:
            conn.stop()
            node = core.registry.get(node_id) if node_id else None
            if node is not None and node.send is send:
                await core.on_disconnect(node_id)

    # ------------------------------------------------------------ jobs

    def _job(job_id: str):
        job = core.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return job

    def _public_user(request: Request):
        user = _user(request, request.headers.get("authorization"))
        if user is None:
            raise HTTPException(401, "Sign in first.")
        return user

    def _assigned_user(job, user_id: str, stage: Optional[int] = None,
                       epoch: Optional[int] = None) -> bool:
        current = job.current
        if current is None or current.closed or (epoch is not None and current.epoch != epoch):
            return False
        return any(
            node.user_id == user_id and node.assignment is not None
            and node.assignment.job_id == job.id
            and node.assignment.epoch == current.epoch
            and (stage is None or node.assignment.stage_idx == stage)
            for node in core.registry.nodes.values()
        )

    def _job_access(job, request: Request, *, assigned: bool = False) -> None:
        if not core.cfg.public_pool:
            return
        user = _public_user(request)
        account = core.credits.job_account(job.id)
        if user.admin or (account is not None and account.user_id == user.id):
            return
        if assigned and _assigned_user(job, user.id):
            return
        raise HTTPException(403, "You do not have access to this job.")

    def _verification_access(vid: str, request: Request, role: str) -> None:
        if not core.cfg.public_pool:
            return
        user = _public_user(request)
        row = core.db.get(Verification, vid)
        if row is None:
            raise HTTPException(404)
        node = core.registry.get(getattr(row, role))
        active = row.status == "fetching" if role == "target_node_id" else (
            row.status == "running" and node is not None and node.verifying == vid
        )
        if not active or node is None or node.user_id != user.id:
            raise HTTPException(403, "You are not assigned to this verification.")

    @app.post("/jobs")
    async def submit_job(body: dict, request: Request,
                         authorization: Optional[str] = Header(default=None)):
        user = _user(request, authorization)
        if core.cfg.public_pool and user is None:
            raise HTTPException(401, "Sign in first.")
        if _token(request, authorization) and user is None:
            raise HTTPException(401, "Sign in first.")
        if core.cfg.public_pool and not user.admin:
            # A caller-selected server path could copy another user's private dataset.
            raise HTTPException(403, "Upload your dataset using /jobs/upload.")
        try:
            spec = parse_spec(body)
            job = core.submit(spec)
        except (ValidationError, ValueError, FileNotFoundError) as e:
            raise HTTPException(400, str(e))
        _reserve(user, job, body.get("max_flops"))
        return core.job_view(job)

    @app.post("/jobs/upload")
    async def upload_job(
        request: Request,
        dataset: UploadFile = File(...),
        model: str = Form("mlx-community/Qwen2.5-0.5B-Instruct-4bit"),
        steps: int = Form(10),
        min_stages: int = Form(1),
        batch_size: int = Form(4),
        microbatches: int = Form(2),
        max_flops: Optional[float] = Form(None),
        authorization: Optional[str] = Header(default=None),
    ):
        user = _user(request, authorization)
        if user is None and (core.cfg.public_pool or _token(request, authorization)):
            raise HTTPException(401, "Sign in first.")
        uploads = cfg.home / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        raw = (dataset.filename or "train.jsonl").replace("\\", "/").split("/")[-1]
        name = re.sub(r"[^A-Za-z0-9._-]", "", raw) or "train.jsonl"
        if not name.lower().endswith(".jsonl"):
            name = "train.jsonl"
        dest = uploads / f"{uuid.uuid4().hex}_{name}"
        buf = bytearray()
        while True:
            chunk = await dataset.read(1024 * 1024)
            if not chunk:
                break
            if len(buf) + len(chunk) > MAX_DATASET_BYTES:
                raise HTTPException(400, "dataset is too large.")
            buf.extend(chunk)
        if core.cfg.public_pool:
            min_stages = 1
        try:
            dest.write_bytes(bytes(buf))
            try:
                spec = parse_spec({
                    "kind": "lora_finetune", "model": model, "dataset_path": str(dest),
                    "steps": steps, "min_stages": min_stages,
                    "batch_size": batch_size, "microbatches": microbatches,
                    **({"max_stages": 1} if core.cfg.public_pool else {}),
                })
                job = core.submit(spec)
            except (ValidationError, ValueError, FileNotFoundError) as e:
                raise HTTPException(400, str(e))
            _reserve(user, job, max_flops)
            return core.job_view(job)
        finally:
            # submit() keeps its own copy; failed submissions must not retain uploads.
            dest.unlink(missing_ok=True)

    @app.get("/jobs")
    async def list_jobs(request: Request, mine: int = 0,
                        authorization: Optional[str] = Header(default=None)):
        jobs = sorted(core.jobs.values(), key=lambda j: j.row.submitted_at)
        if mine:
            user = _user(request, authorization)
            if user is None:
                raise HTTPException(401, "Sign in first.")
            jobs = [j for j in jobs
                    if (acct := core.credits.job_account(j.id)) and acct.user_id == user.id]
        return [core.job_view(j) for j in jobs]

    @app.get("/jobs/waitlist")
    async def list_waitlist():
        return core.waitlist()

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str):
        return core.job_view(_job(job_id))

    @app.post("/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str, request: Request):
        job = _job(job_id)
        _job_access(job, request)
        await core.cancel_job(job)
        return core.job_view(job)

    @app.get("/jobs/{job_id}/usage")
    async def job_usage(job_id: str):
        _job(job_id)
        return [r.model_dump() for r in core.ledger.job_records(job_id)]

    @app.get("/jobs/{job_id}/dataset")
    async def get_dataset(job_id: str, request: Request):
        _job_access(_job(job_id), request, assigned=True)
        return FileResponse(core.dataset_path(job_id))

    @app.get("/jobs/{job_id}/checkpoints/{step}")
    async def get_checkpoint(job_id: str, step: int, request: Request):
        _job_access(_job(job_id), request, assigned=True)
        path = core.checkpoints.merged_path(job_id, step)
        if not path.exists():
            raise HTTPException(404, "checkpoint not found")
        return FileResponse(path)

    @app.post("/jobs/{job_id}/checkpoints/{step}")
    async def put_checkpoint(job_id: str, step: int, epoch: int, stage: int, request: Request):
        if core.cfg.public_pool:
            user = _public_user(request)
            if not _assigned_user(_job(job_id), user.id, stage=stage, epoch=epoch):
                raise HTTPException(403, "You are not assigned to this stage.")
        data = await request.body()
        try:
            await core.on_checkpoint_upload(job_id, epoch, stage, step, data)
        except ValueError as e:
            raise HTTPException(409, str(e))
        return {"ok": True}

    @app.get("/jobs/{job_id}/adapter/{name}")
    async def get_adapter(job_id: str, name: str, request: Request):
        _job_access(_job(job_id), request)
        if name not in ("adapters.safetensors", "adapter_config.json"):
            raise HTTPException(404)
        path = core.checkpoints.job_dir(job_id) / "adapter" / name
        if not path.exists():
            raise HTTPException(404, "adapter not ready")
        return FileResponse(path)

    # ------------------------------------------------------------ nodes, ledger, verification

    @app.get("/nodes")
    async def nodes():
        return core.node_view()

    @app.get("/ledger")
    async def ledger():
        return core.ledger.summary()

    @app.get("/verifications")
    async def verifications():
        with core.db.session() as s:
            rows = s.exec(select(Verification).order_by(Verification.created_at)).all()
        return [r.model_dump() for r in rows]

    @app.post("/verify/{vid}/bundle")
    async def put_bundle(vid: str, request: Request):
        _verification_access(vid, request, "target_node_id")
        try:
            core.verification.store_bundle(vid, await request.body())
        except KeyError:
            raise HTTPException(404, "no verification awaiting a bundle")
        return {"ok": True}

    @app.get("/verify/{vid}/bundle")
    async def get_bundle(vid: str, request: Request):
        _verification_access(vid, request, "verifier_node_id")
        if core.db.get(Verification, vid) is None:
            raise HTTPException(404)
        path = core.verification.bundle_path(vid)
        if not path.exists():
            raise HTTPException(404)
        return FileResponse(path)

    @app.post("/verify/{vid}/result")
    async def put_result(vid: str, request: Request):
        _verification_access(vid, request, "verifier_node_id")
        try:
            core.verification.store_result(vid, await request.body())
        except KeyError:
            raise HTTPException(404, "no verification awaiting a result")
        return {"ok": True}

    @app.get("/health")
    async def health():
        return {"ok": True, "nodes": len(core.registry.nodes), "jobs": len(core.jobs),
                "inference_nodes": len(svc.online), "inference_transport": svc.s.TRANSPORT}

    return app
