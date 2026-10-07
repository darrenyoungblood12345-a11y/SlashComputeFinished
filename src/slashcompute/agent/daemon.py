"""Coordinator WebSocket session: register, heartbeat, run stages.

Several tasks share the daemon (the receive loop, heartbeats, the running
stage and its stdout pump, signal-driven shutdown), so the stage it runs is one
``_Stage`` handle: lifecycle changes happen under ``_stage_lock`` and act on the
handle they captured, and a stage's own callbacks only reset state that still
belongs to it.

With a coordinator that supports it, the session is reliable (``common.reliable``):
messages are numbered and kept until acknowledged, a dropped connection is not the
end of the running stage, and after reconnecting within the coordinator's grace
window both sides replay what the other missed.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import socket
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Coroutine, Optional

import httpx
import websockets
from pydantic import BaseModel, ValidationError

from slashcompute.agent.benchmark import benchmark
from slashcompute.agent.http import CoordHTTP
from slashcompute.agent.paths import AgentPaths, claim_data_port, resolve_data_host
from slashcompute.agent.sandbox import sandbox_enabled, wrap_command
from slashcompute.agent.verify import run_canary, run_replay
from slashcompute.agent.worker import StageSession, WorkerContext, run_stage
from slashcompute.common.config import EngineConfig
from slashcompute.common.discovery import discover
from slashcompute.common.protocol import (
    Ack, CancelStage, Drain, DrainNotice, Heartbeat, Register, StageAssignment, StageFinished,
    StageReady, VerifyBundleReady, VerifyFetch, VerifyRequest, VerifyResult, Welcome, dump,
    parse_agent_message, parse_coordinator_message,
)
from slashcompute.common.reliable import Inbox, Outbox, is_sequenced

log = logging.getLogger(__name__)
RECONNECT_MAX_S = 15.0
FETCH_STOP_GRACE_S = 2.0        # a stopped download gets this long to exit before SIGKILL
FETCH_LOG_EVERY_S = 30.0


def _rejection(e: Exception) -> str:
    """Why the coordinator refused us for good (application close codes 4000-4999), else ""."""
    rcvd = getattr(e, "rcvd", None)
    if rcvd is not None and 4000 <= rcvd.code < 5000:
        return rcvd.reason or f"close code {rcvd.code}"
    return ""


def resolve_coordinator(url: Optional[str]) -> str:
    if url:
        return url.rstrip("/")
    found = discover()
    if not found:
        raise SystemExit("No coordinator found on the LAN. Pass --url or start one.")
    return found.rstrip("/")


def _ws_url(http_url: str) -> str:
    base = http_url.replace("https://", "wss://").replace("http://", "ws://").rstrip("/")
    return base + "/ws/agent"


def public_transport(url: str, public_pool: bool = False) -> bool:
    """HTTPS or a public-pool flag: do not open a WAN pipeline."""
    return bool(public_pool) or (url or "").lower().startswith("https://")


def peered_assignment(asg: StageAssignment) -> bool:
    return asg.prev_peer is not None or asg.next_peer is not None


def fetch_command(model: str) -> list[str]:
    """The model download, as a process a cancel can stop (see ``agent.fetch``)."""
    return [sys.executable, "-m", "slashcompute.agent.fetch", model]


async def _terminate(proc: asyncio.subprocess.Process, grace: float) -> None:
    if proc.returncode is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), grace)
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()


def _gb(n: Optional[int]) -> str:
    return "?" if n is None else f"{n / 1e9:.1f}"


def _alive(pid: int) -> bool:
    try:
        os_kill = __import__("os").kill
        os_kill(pid, 0)
        return True
    except OSError:
        return False


class AgentOptions:
    def __init__(
        self,
        url: Optional[str] = None,
        home: Optional[Path] = None,
        gpu_percent: int = 50,
        data_port: int = 9700,
        max_memory_gb: Optional[float] = None,
        localhost: bool = False,
        sandbox: Optional[bool] = None,
        name: Optional[str] = None,
        session_token: Optional[str] = None,
    ) -> None:
        self.cfg = EngineConfig.from_env(home=home)
        if home is not None:
            self.cfg.home = Path(home)
        self.paths = AgentPaths(self.cfg.home)
        self.gpu_percent = max(1, min(100, int(gpu_percent)))
        wanted = int(data_port)
        self.data_port = claim_data_port(wanted)
        if self.data_port != wanted:
            log.warning("data port %s in use; advertising %s", wanted, self.data_port)
        self.max_memory_bytes = int(max_memory_gb * 1024**3) if max_memory_gb else None
        self.localhost = localhost
        self.sandbox = sandbox_enabled(sandbox, self.cfg.sandbox)
        self.name = name or socket.gethostname().split(".")[0]
        self.coordinator = resolve_coordinator(url)
        self.session_token = session_token or __import__("os").environ.get("SLASHCOMPUTE_SESSION")
        self.http = CoordHTTP(self.coordinator, session_token=self.session_token)
        self.node_id = self.paths.node_id()
        self.data_host = resolve_data_host(localhost)
        self.data_bind = "0.0.0.0"


@dataclass(eq=False)
class _Stage:
    """The stage this agent runs: in-process (``session``) or a sandboxed worker
    process (``proc``, whose stdout ``pump`` forwards). ``starter`` first downloads the
    model in ``fetch_proc``, then starts it."""

    asg: StageAssignment
    session: Optional[StageSession] = None
    proc: Optional[asyncio.subprocess.Process] = None
    pump: Optional[asyncio.Task] = None
    starter: Optional[asyncio.Task] = None
    fetch_proc: Optional[asyncio.subprocess.Process] = None
    fetch_done: Optional[int] = None             # bytes of the model on disk so far
    fetch_total: Optional[int] = None
    ready: bool = False                          # StageReady went out
    reported: bool = False                       # its StageFinished went out: never send another

    def matches(self, job_id: str, epoch: int) -> bool:
        return self.asg.job_id == job_id and self.asg.epoch == epoch

    def runs(self, asg: StageAssignment) -> bool:
        return self.matches(asg.job_id, asg.epoch) and self.asg.stage_idx == asg.stage_idx


class Daemon:
    def __init__(self, opt: AgentOptions) -> None:
        self.opt = opt
        self.status = "idle"
        self.job_id: Optional[str] = None
        self.epoch: Optional[int] = None
        self._ws = None                             # the connection sends go to (after Welcome)
        self._conn = None                           # the open connection, from its first moment
        self._send_lock = asyncio.Lock()
        self._stage: Optional[_Stage] = None
        self._stage_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()
        self._stop = asyncio.Event()
        self._stopping = False                      # shutdown started; no new stages
        self._draining = False
        self._welcomed = False
        # Reliable session state; it outlives a connection so a reconnect can resume.
        self._session_id = uuid.uuid4().hex
        self._outbox = Outbox()
        self._inbox = Inbox()
        self._reliable = False                      # the coordinator agreed to a session
        self._grace = 0.0                           # how long it holds our place when we drop
        self._disconnected_at: Optional[float] = None

    # Read-only views of the running stage.
    @property
    def _session(self) -> Optional[StageSession]:
        return self._stage.session if self._stage else None

    @property
    def _proc(self) -> Optional[asyncio.subprocess.Process]:
        return self._stage.proc if self._stage else None

    @property
    def _pump(self) -> Optional[asyncio.Task]:
        return self._stage.pump if self._stage else None

    def _spawn(self, coro: Coroutine) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def _set_status(self, status: str, job_id: Optional[str], epoch: Optional[int]) -> None:
        self.status, self.job_id, self.epoch = status, job_id, epoch
        self._write_status()

    def _release(self, stage: _Stage) -> None:
        """``stage`` ended; forget it unless a newer stage has already replaced it."""
        if self._stage is stage:
            self._stage = None
            self._set_status("idle", None, None)

    def _fetching(self) -> Optional[_Stage]:
        """The current stage while its model downloads."""
        stage = self._stage
        return stage if stage is not None and stage.fetch_proc is not None else None

    def _write_status(self) -> None:
        fetch = self._fetching()
        self.opt.paths.write_status(
            pid=__import__("os").getpid(), node_id=self.opt.node_id, name=self.opt.name,
            status=self.status, draining=self._draining, job_id=self.job_id, epoch=self.epoch,
            coordinator=self.opt.coordinator, data_addr=f"{self.opt.data_host}:{self.opt.data_port}",
            gpu_percent=self.opt.gpu_percent,
            fetch_done_bytes=fetch.fetch_done if fetch else None,
            fetch_total_bytes=fetch.fetch_total if fetch else None,
        )

    async def send(self, msg: BaseModel) -> None:
        if self._reliable and is_sequenced(msg):
            msg = self._outbox.stamp(msg)  # kept until acknowledged, replayed after a reconnect
        async with self._send_lock:
            ws = self._ws  # read under the lock: a reconnect may have replaced or cleared it
            if ws is None:
                return
            try:
                await ws.send(dump(msg))
            except (websockets.exceptions.ConnectionClosed, OSError) as e:
                # The receive loop notices the closed socket and reconnects.
                log.info("could not send %s: %s", type(msg).__name__, e)

    async def run(self) -> None:
        opt = self.opt
        opt.paths.write_pid()
        self._write_status()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: self._spawn(self.shutdown()))
            except NotImplementedError:
                signal.signal(sig, lambda *_: self._spawn(self.shutdown()))

        try:
            log.info("benchmarking device…")
            device = benchmark(opt.max_memory_bytes)
            log.info("chip=%s lending %.1f GB  %.1f TFLOPS  gpu %d%%",
                     device.chip, device.memory_contrib_bytes / 1e9, device.matmul_tflops,
                     opt.gpu_percent)

            ws_url = _ws_url(opt.coordinator)
            delay = 1.0
            while not self._stop.is_set():
                log.info("connecting to %s as %s (%s)", ws_url, opt.node_id[:8], opt.name)
                self._welcomed = False
                if self._stage is None:          # a stage kept across a reconnect keeps its status
                    self.status = "connecting"   # shown in the app; never heartbeated (not welcomed yet)
                    self._write_status()
                try:
                    await self._connect_once(ws_url, device)
                except (OSError, asyncio.TimeoutError, websockets.exceptions.WebSocketException,
                        httpx.HTTPError) as e:
                    rejected = _rejection(e)
                    if rejected:
                        raise SystemExit(f"coordinator refused this agent: {rejected}") from e
                    log.warning("coordinator unreachable (%s)", e)
                if self._welcomed:
                    delay = 1.0                  # we were registered: a fresh outage starts a fresh backoff
                if self._stop.is_set():
                    break
                if self._stage is not None and not self._may_keep_stage():
                    await self._release_orphaned_stage()
                log.info("reconnecting in %.0fs", delay)
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), delay)
                delay = min(delay * 2, RECONNECT_MAX_S)
        finally:
            opt.paths.clear_pid()
            self.status = "stopped"
            self._write_status()

    def _may_keep_stage(self) -> bool:
        """While disconnected: the coordinator still holds our place, so keep the stage."""
        return (self._reliable and self._disconnected_at is not None
                and time.monotonic() - self._disconnected_at < self._grace)

    async def _release_orphaned_stage(self) -> None:
        """Stop a stage the coordinator no longer counts on (it restarted, or we were away
        longer than its grace window), and say so in case it is still listening."""
        if self._stage is None:
            return
        log.info("lost the coordinator mid-stage; releasing it (the job will be rescheduled)")
        await self._cancel_stage(detail="lost the coordinator")

    async def _connect_once(self, ws_url: str, device) -> None:
        """One coordinator connection: register, then handle messages until it closes."""
        opt = self.opt
        async with websockets.connect(ws_url, max_size=64 * 1024 * 1024, ping_interval=20,
                                      open_timeout=10) as ws:
            self._conn = ws
            hb = None
            try:
                await ws.send(dump(Register(
                    node_id=opt.node_id, name=opt.name, device=device,
                    data_host=opt.data_host, data_port=opt.data_port, gpu_percent=opt.gpu_percent,
                    session_token=opt.session_token, session_id=self._session_id,
                    last_seq=self._inbox.last,
                )))
                welcome = parse_coordinator_message(await ws.recv())
                if not isinstance(welcome, Welcome):
                    raise SystemExit(f"expected welcome, got {type(welcome).__name__}")
                self._welcomed = True
                await self._on_welcome(ws, welcome)
                hb = asyncio.create_task(self._heartbeats(welcome.heartbeat_interval_s))
                async for raw in ws:
                    if self._stop.is_set():
                        break
                    try:
                        msg = parse_coordinator_message(raw)
                    except ValidationError as e:
                        log.warning("bad coordinator message: %s", e)
                        continue
                    await self._receive(msg)
            finally:
                if hb is not None:
                    hb.cancel()
                self._conn = None
                async with self._send_lock:
                    if self._ws is ws:
                        self._ws = None
                        self._disconnected_at = time.monotonic()

    async def _on_welcome(self, ws, welcome: Welcome) -> None:
        if not welcome.resumed:
            # A fresh session: the coordinator knows nothing of a stage we kept running
            # or of messages we were holding for it.
            if self._stage is not None:
                await self._release_orphaned_stage()
            self._outbox, self._inbox = Outbox(), Inbox()
        self._reliable, self._grace = welcome.session, welcome.reconnect_grace_s
        async with self._send_lock:
            # Replay before anything new can be sent, so the coordinator sees our
            # messages in order.
            if welcome.resumed:
                self._outbox.ack(welcome.last_seq)
                pending = self._outbox.pending()
                for m in pending:
                    await ws.send(dump(m))
                log.info("resumed the coordinator session (%d message(s) replayed)", len(pending))
            self._ws = ws
            self._disconnected_at = None
        if self._stage is None:
            self._set_status("idle", None, None)

    async def _receive(self, msg) -> None:
        if isinstance(msg, Ack):
            self._outbox.ack(msg.upto)
            return
        seq = msg.seq if self._reliable else None
        if seq is not None and not self._inbox.accept(seq):
            await self.send(Ack(upto=self._inbox.last))  # a replay we already handled
            return
        try:
            await self._handle(msg)
        except Exception:  # one bad message must not take the agent down
            log.exception("handling %s failed", type(msg).__name__)
        if seq is not None:
            await self.send(Ack(upto=seq))

    def _heartbeat_msg(self) -> Heartbeat:
        fetch = self._fetching()
        return Heartbeat(
            node_id=self.opt.node_id, status=self.status if not self._draining else "draining",
            job_id=self.job_id, epoch=self.epoch,
            phase="fetching" if fetch else None,
            fetch_done_bytes=fetch.fetch_done if fetch else None,
            fetch_total_bytes=fetch.fetch_total if fetch else None,
        )

    async def _heartbeats(self, interval: float) -> None:
        try:
            while not self._stop.is_set():
                try:
                    await self.send(self._heartbeat_msg())
                except Exception:  # a silently dead heartbeat loop gets the node evicted
                    log.exception("heartbeat failed")
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            return

    async def _handle(self, msg) -> None:
        if isinstance(msg, StageAssignment):
            await self._start_stage(msg)
        elif isinstance(msg, Drain):
            log.info("drain requested for job %s", msg.job_id)
            await self._request_drain(msg.job_id, msg.epoch)
        elif isinstance(msg, CancelStage):
            log.info("cancel stage job %s epoch %d", msg.job_id, msg.epoch)
            await self._cancel_stage(msg.job_id, msg.epoch)
        elif isinstance(msg, VerifyFetch):
            await self._on_fetch(msg)
        elif isinstance(msg, VerifyRequest):
            self._spawn(self._on_verify(msg))  # a replay must not hold up CancelStage
        else:
            log.warning("unhandled coordinator message %s", type(msg).__name__)

    async def _start_stage(self, asg: StageAssignment) -> Optional[_Stage]:
        """Take on ``asg`` and return at once: ``stage.starter`` downloads the model and starts
        the worker, so this message loop stays free for a CancelStage, a newer assignment or a
        shutdown while a model downloads (that used to take 10-40 minutes, unread)."""
        if public_transport(self.opt.coordinator, self.opt.cfg.public_pool) and peered_assignment(asg):
            log.warning("refusing multi-peer assignment on public/https coordinator")
            return None
        async with self._stage_lock:
            if self._stopping:
                log.warning("ignoring assignment for job %s: shutting down", asg.job_id)
                return None
            cur = self._stage
            if cur is not None and cur.runs(asg):
                log.info("already running job %s epoch %d stage %d", asg.job_id, asg.epoch,
                         asg.stage_idx)
                return cur
            if cur is not None:
                log.warning("assignment while a stage is running; cancelling the old one")
                await self._stop_stage(cur, "replaced by a newer assignment")
            # Claimed before anything can yield: a CancelStage right behind this message
            # finds this stage and stops its starter, even before the starter has run.
            stage = self._stage = _Stage(asg)
            self._set_status("loading", asg.job_id, asg.epoch)
            stage.starter = self._spawn(self._launch(stage))
        return stage

    async def _launch(self, stage: _Stage) -> None:
        asg = stage.asg
        try:
            await self._fetch_model(stage)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("model fetch for job %s failed: %s", asg.job_id, e)
            await self._fail_start(stage, f"model fetch failed: {e}")
            return
        job_dir = self.opt.paths.job_dir(asg.job_id, asg.epoch)
        if self.opt.sandbox:
            await self._start_sandboxed(stage, job_dir)
            return
        async with self._stage_lock:
            if self._stage is stage and not self._stopping:
                self._start_in_process(stage, job_dir)

    async def _fetch_model(self, stage: _Stage) -> None:
        """Download the stage's model into the HF cache (the sandboxed worker can't write it,
        and the in-process one would block this loop) in a child process ``_stop_stage`` can
        kill. Its progress goes out in heartbeats and status.json."""
        asg = stage.asg
        model = asg.spec.model
        if Path(model).expanduser().is_dir():
            return
        proc = await asyncio.create_subprocess_exec(
            *fetch_command(model), stdout=asyncio.subprocess.PIPE, stdin=asyncio.subprocess.DEVNULL)
        stage.fetch_proc = proc            # no await in between: a stop always sees it
        if self._stage is stage:
            self._write_status()
        log.info("job %s: fetching %s", asg.job_id, model)
        error, logged = "", 0.0
        try:
            assert proc.stdout is not None
            while line := await proc.stdout.readline():
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if msg.get("type") == "progress":
                    stage.fetch_done, stage.fetch_total = msg.get("done"), msg.get("total")
                    if self._stage is stage:
                        self._write_status()
                    if time.monotonic() - logged >= FETCH_LOG_EVERY_S:
                        logged = time.monotonic()
                        log.info("job %s: fetched %s of %s GB of %s", asg.job_id,
                                 _gb(stage.fetch_done), _gb(stage.fetch_total), model)
                elif msg.get("type") == "error":
                    error = str(msg.get("detail") or "")
            code = await proc.wait()
        finally:
            await _terminate(proc, FETCH_STOP_GRACE_S)
            stage.fetch_proc = None
        if code != 0:
            raise RuntimeError(error or f"the download exited with code {code}")
        log.info("job %s: %s is ready", asg.job_id, model)

    def _start_in_process(self, stage: _Stage, job_dir: Path) -> None:
        asg = stage.asg
        session = stage.session = StageSession()
        session.assignment = asg
        ctx = WorkerContext(
            assignment=asg, http=self.opt.http, job_dir=job_dir,
            data_bind=self.opt.data_bind, data_port=self.opt.data_port,
            gpu_percent=self.opt.gpu_percent, node_id=self.opt.node_id, session=session,
        )

        async def emit(m) -> None:
            if isinstance(m, StageReady):
                stage.ready = True
                if self._stage is stage:
                    self._set_status("running", asg.job_id, asg.epoch)
            if isinstance(m, StageFinished):
                if stage.reported:
                    return                   # already reported stopped
                stage.reported = True
                self._release(stage)
            await self.send(m)

        async def _run() -> None:
            try:
                await run_stage(ctx, emit)
            except asyncio.CancelledError:
                pass
            except Exception:
                log.exception("worker crashed")
            finally:
                self._release(stage)

        session.task = asyncio.create_task(_run())

    async def _start_sandboxed(self, stage: _Stage, job_dir: Path) -> None:
        # The sandbox can't write the HF cache (~/.cache/huggingface: locks, refs, blobs), so
        # _launch fetched the model unsandboxed, and the worker runs offline against that cache.
        asg = stage.asg
        try:
            spec_path = job_dir / "assignment.json"
            spec_path.touch(mode=0o600)
            spec_path.chmod(0o600)
            spec_path.write_text(json.dumps({
                "assignment": json.loads(asg.model_dump_json()),
                "coordinator_url": self.opt.coordinator,
                "session_token": self.opt.session_token,
                "job_dir": str(job_dir),
                "data_bind": self.opt.data_bind,
                "data_port": self.opt.data_port,
                "gpu_percent": self.opt.gpu_percent,
                "node_id": self.opt.node_id,
            }))
            cmd = wrap_command(
                [sys.executable, "-m", "slashcompute.agent.worker", "--assignment", str(spec_path)],
                job_dir, self.opt.paths.root,
            )
            async with self._stage_lock:
                if self._stage is not stage or self._stopping:
                    log.info("job %s epoch %d was cancelled before its worker started",
                             asg.job_id, asg.epoch)
                    return
                log.info("starting sandboxed worker: %s", " ".join(cmd))
                stage.proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stdin=asyncio.subprocess.PIPE,
                    env={**os.environ, "HF_HUB_OFFLINE": "1"},
                )
                stage.pump = self._spawn(self._pump_worker_stdout(stage))
        except Exception as e:
            log.exception("could not start the sandboxed worker")
            await self._fail_start(stage, f"could not start the worker: {e}")

    async def _fail_start(self, stage: _Stage, detail: str) -> None:
        """Report a stage that never started now, rather than leave the coordinator
        waiting out its start timeout."""
        if self._stage is not stage or stage.reported:
            return  # cancelled meanwhile; that was reported
        stage.reported = True
        self._release(stage)
        asg = stage.asg
        await self.send(StageFinished(
            job_id=asg.job_id, epoch=asg.epoch, stage_idx=asg.stage_idx, reason="error",
            last_step=asg.resume_step, detail=detail,
        ))

    async def _pump_worker_stdout(self, stage: _Stage) -> None:
        proc = stage.proc
        if proc is None or proc.stdout is None:
            return
        asg, finished = stage.asg, False
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                raw = line.decode().strip()
                if not raw:
                    continue
                try:
                    msg = parse_agent_message(raw)
                except ValidationError:
                    log.warning("worker stdout: %s", raw[:200])
                    continue
                if isinstance(msg, StageReady):
                    stage.ready = True
                    if self._stage is stage:
                        self._set_status("running", asg.job_id, asg.epoch)
                if isinstance(msg, StageFinished):
                    if stage.reported:
                        continue                 # already reported stopped
                    stage.reported = finished = True
                    self._release(stage)
                await self.send(msg)
        finally:
            if proc.returncode is None:
                await proc.wait()
            if not finished and self._stage is stage and not stage.reported:
                # The worker died without reporting (crash, OOM kill): say so now rather
                # than leave the coordinator waiting for a stall timeout.
                stage.reported = True
                self._release(stage)
                await self.send(StageFinished(
                    job_id=asg.job_id, epoch=asg.epoch, stage_idx=asg.stage_idx, reason="error",
                    last_step=asg.resume_step, detail=f"worker exited with code {proc.returncode}",
                ))
            self._release(stage)

    async def _request_drain(self, job_id: Optional[str] = None, epoch: Optional[int] = None) -> None:
        stage = self._stage
        if stage is None or (job_id is not None and not stage.matches(job_id, epoch)):
            return
        if stage.session:
            stage.session.request_drain()
        proc = stage.proc
        if proc and proc.stdin:
            try:
                proc.stdin.write(b'{"type":"drain"}\n')
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass                             # the worker already exited

    async def _cancel_stage(self, job_id: Optional[str] = None, epoch: Optional[int] = None,
                            detail: str = "cancelled by the coordinator") -> None:
        """Stop the running stage; with ``job_id``/``epoch``, only if it is that one (a
        stale CancelStage must not kill a newer stage)."""
        async with self._stage_lock:
            stage = self._stage
            if stage is None:
                return
            if job_id is not None and not stage.matches(job_id, epoch):
                log.info("ignoring cancel for job %s epoch %s: running job %s epoch %d",
                         job_id, epoch, stage.asg.job_id, stage.asg.epoch)
                return
            await self._stop_stage(stage, detail)

    async def _stop_stage(self, stage: _Stage, detail: str = "cancelled by the coordinator") -> None:
        """Stop ``stage`` wherever it is (downloading, starting, running), then tell the
        coordinator it stopped, once: until it hears that, it keeps this Mac out of new work.
        The caller holds ``_stage_lock``."""
        self._release(stage)  # first, so the stage's own exit paths leave state alone
        starter, me = stage.starter, asyncio.current_task()
        live = starter is not None and starter is not me and not starter.done()
        if live:
            starter.cancel()
        if stage.fetch_proc is not None:
            await _terminate(stage.fetch_proc, FETCH_STOP_GRACE_S)
        if live:
            await asyncio.wait({starter}, timeout=5)   # it can't be holding our lock
        if stage.session is not None:
            await stage.session.cancel()
        proc = stage.proc
        if proc is not None and getattr(proc, "returncode", 0) is None:
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
        if stage.pump is not None and not stage.pump.done():
            await asyncio.wait({stage.pump}, timeout=2)   # forward the worker's last words first
        if not stage.reported:
            stage.reported = True
            asg = stage.asg
            log.info("job %s epoch %d stage %d stopped: %s", asg.job_id, asg.epoch, asg.stage_idx,
                     detail)
            await self.send(StageFinished(
                job_id=asg.job_id, epoch=asg.epoch, stage_idx=asg.stage_idx, reason="cancelled",
                last_step=asg.resume_step, detail=detail,
            ))

    async def _on_fetch(self, msg: VerifyFetch) -> None:
        dest = self.opt.paths.job_dir(msg.job_id, msg.epoch) / f"bundle_{msg.step}.safetensors"
        stage = self._stage
        # Only the stage that ran the step holds its bundle; another job's ring must not answer.
        session = stage.session if stage is not None and stage.matches(msg.job_id, msg.epoch) else None
        error = None
        try:
            if session and session.save_bundle(msg.step, dest):
                await asyncio.to_thread(self._upload, f"/verify/{msg.verify_id}/bundle", dest)
            else:
                error = "bundle no longer held"
        except Exception as e:           # the coordinator must hear back, and the session must survive
            log.exception("verify bundle upload failed")
            error = f"bundle upload failed: {e}"
        await self.send(VerifyBundleReady(
            verify_id=msg.verify_id, job_id=msg.job_id, stage_idx=msg.stage_idx,
            step=msg.step, path=None if error else str(dest), error=error,
        ))

    async def _on_verify(self, msg: VerifyRequest) -> None:
        # Downloads, replays and uploads run off the event loop so heartbeats keep flowing.
        try:
            if msg.kind == "canary":
                stats = await asyncio.to_thread(
                    run_canary, msg.seed or 0, msg.size or self.opt.cfg.canary_size)
                await self.send(VerifyResult(verify_id=msg.verify_id, kind="canary", stats=stats))
                return
            work = self.opt.paths.root / "verify" / msg.verify_id
            work.mkdir(parents=True, exist_ok=True)
            bundle = await asyncio.to_thread(
                self.opt.http.get_file, msg.bundle_url, work / "bundle.safetensors")
            dest = work / "replay.safetensors"
            stats = await asyncio.to_thread(run_replay, msg, bundle, dest)
            await asyncio.to_thread(self._upload, f"/verify/{msg.verify_id}/result", dest)
            await self.send(VerifyResult(
                verify_id=msg.verify_id, kind="replay", stats=stats, output_path=str(dest),
            ))
        except Exception as e:
            log.exception("verify %s failed", msg.kind)
            await self.send(VerifyResult(verify_id=msg.verify_id, kind=msg.kind, error=str(e)))

    def _upload(self, path: str, src: Path) -> None:
        self.opt.http.put_bytes(path, src.read_bytes())

    async def shutdown(self) -> None:
        if self._stopping:  # a second SIGTERM while draining
            return
        self._stopping = True
        log.info("shutting down")
        self._draining = True
        self._write_status()
        try:
            await self.send(DrainNotice(node_id=self.opt.node_id))
        except Exception:
            pass
        stage = self._stage
        if stage is not None and stage.ready:
            # A running stage drains to a checkpoint: in-process via its runner, a sandboxed
            # worker on a stdin line, with the pump forwarding its StageFinished. One still
            # downloading or loading has nothing to save and just stops below.
            await self._request_drain()
            ended = stage.session.task if stage.session else stage.pump
            if ended is not None:
                with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                    await asyncio.wait_for(asyncio.shield(ended), timeout=self.opt.cfg.grace_period_s)
        # Whatever is current now, including a stage assigned while we drained.
        await self._cancel_stage(detail="agent shutting down")
        self._stop.set()
        conn = self._conn or self._ws  # close it even mid-handshake, or run() never returns
        if conn is not None:
            try:
                await conn.close()
            except Exception:
                pass


async def start_daemon(opt: AgentOptions) -> None:
    await Daemon(opt).run()


def request_stop(paths: AgentPaths) -> bool:
    pid = paths.read_pid()
    if pid is None or not _alive(pid):
        paths.clear_pid()
        return False
    __import__("os").kill(pid, signal.SIGTERM)
    return True
