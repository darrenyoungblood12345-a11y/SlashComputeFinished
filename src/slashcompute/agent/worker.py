"""Run one pipeline stage: load shard, connect peers, train, report."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional

import httpx

from slashcompute.agent.http import CoordHTTP
from slashcompute.agent.throttle import make_pace
from slashcompute.common.protocol import StageAssignment, StageFinished, StageReady, StepMetrics
from slashcompute.metering.recorder import UsageRecorder
from slashcompute.pipeline.data import load_examples
from slashcompute.pipeline.local import build_compute
from slashcompute.pipeline.model_profile import profile_model
from slashcompute.pipeline.schedule import StageResult, StageRunner, StepStats
from slashcompute.transport import Link, LinkServer, connect

log = logging.getLogger(__name__)

UPLOAD_RETRY_DELAYS_S = (2.0, 4.0, 8.0, 16.0)

OnMessage = Callable[[object], Awaitable[None]]


class StageSession:
    """In-process stage so the daemon can drain / save verify bundles."""

    def __init__(self) -> None:
        self.runner: Optional[StageRunner] = None
        self.compute = None
        self.task: Optional[asyncio.Task] = None
        self.assignment: Optional[StageAssignment] = None

    def request_drain(self) -> None:
        if self.runner is not None:
            self.runner.drain.set()

    def save_bundle(self, step: int, dest: Path) -> bool:
        if self.compute is None:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        return self.compute.save_bundle(step, dest)

    async def cancel(self) -> None:
        if self.task is not None and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except (asyncio.CancelledError, Exception):
                pass


# A neighbour may still be downloading its share of the model (many GB), so a stage waits for
# it as long as the coordinator lets the epoch start; the coordinator's progress-aware start
# timeout, not this handshake, decides that a start is stuck (and then cancels the stage).
PEER_HANDSHAKE_TIMEOUT_S = 24 * 3600.0


@dataclass
class WorkerContext:
    assignment: StageAssignment
    http: CoordHTTP
    job_dir: Path
    data_bind: str
    data_port: int
    gpu_percent: int
    node_id: str
    session: Optional[StageSession] = None


async def _peer_links(ctx: WorkerContext) -> tuple[Optional[Link], Optional[Link], Optional[LinkServer]]:
    asg = ctx.assignment
    hello = {"job_id": asg.job_id, "epoch": asg.epoch}
    server = None
    prev = nxt = None
    # Downstream listens; upstream dials (see transport.peer).
    if asg.stage_idx > 0:
        server = await LinkServer(ctx.data_bind, ctx.data_port, hello,
                                  send_timeout=asg.peer_timeout_s,
                                  resume_window=asg.peer_timeout_s).start()
        log.info("listening for upstream on %s:%s", ctx.data_bind, server.port)
    if asg.next_peer is not None:
        log.info("dialing next stage %s:%s", asg.next_peer.host, asg.next_peer.port)
        nxt = await connect(asg.next_peer.host, asg.next_peer.port, hello,
                            timeout=PEER_HANDSHAKE_TIMEOUT_S,
                            send_timeout=asg.peer_timeout_s, resume_window=asg.peer_timeout_s)
    if server is not None:
        prev = await server.accept(timeout=None)
        log.info("upstream connected from %s", getattr(prev, "peername", "?"))
    return prev, nxt, server


class DatasetError(ValueError):
    """The job's dataset is malformed, so every retry would fail the same way."""


def _load_dataset(path: Path, spec) -> list:
    try:
        return load_examples(path, spec.model, spec.max_seq_len)
    except (ValueError, TypeError, KeyError) as e:
        raise DatasetError(f"bad dataset: {e}") from e


async def upload_checkpoint(http: CoordHTTP, url: str, path: Path, params: dict) -> None:
    """Upload off the event loop, so peer links keep flowing, and retry connection
    failures: a blip that briefly cuts the coordinator off must not fail the stage.
    An HTTP error (a stale epoch's 409, say) is final."""
    data = await asyncio.to_thread(path.read_bytes)
    for delay in (*UPLOAD_RETRY_DELAYS_S, None):
        try:
            await asyncio.to_thread(http.put_bytes, url, data, params)
            return
        except httpx.TransportError as e:
            if delay is None:
                raise
            log.warning("checkpoint upload failed (%s); retrying in %.0fs", e, delay)
            await asyncio.sleep(delay)


async def run_stage(ctx: WorkerContext, emit: OnMessage) -> StageResult:
    asg = ctx.assignment
    spec = asg.spec
    prev = nxt = server = compute = None
    # Setup is inside the try so a failure still reports StageFinished(error)
    # rather than leaving the coordinator waiting on a stage that never starts.
    try:
        ckdir = ctx.job_dir / "checkpoints"
        ckdir.mkdir(parents=True, exist_ok=True)

        dataset_path = None
        if asg.dataset_url:
            dataset_path = ctx.http.get_file(asg.dataset_url, ctx.job_dir / "dataset.jsonl")
        resume_from = None
        if asg.resume_step > 0 and asg.checkpoint_url:
            resume_from = ctx.http.get_file(asg.checkpoint_url, ctx.job_dir / "resume.safetensors")

        compute = build_compute(spec, asg.layer_start, asg.layer_end, asg.num_layers,
                                ring_size=asg.verify_ring_size)
        if resume_from is not None:
            compute.load_checkpoint(resume_from)

        profile = profile_model(spec.model)
        recorder = UsageRecorder(profile, asg.layer_start, asg.layer_end)
        examples = _load_dataset(dataset_path, spec) if dataset_path else None

        prev, nxt, server = await _peer_links(ctx)
        await emit(StageReady(job_id=asg.job_id, epoch=asg.epoch, stage_idx=asg.stage_idx))

        async def on_step(s: StepStats) -> None:
            await emit(StepMetrics(
                job_id=asg.job_id, epoch=asg.epoch, stage_idx=asg.stage_idx, step=s.step,
                loss=s.loss, in_digest=s.in_digest, out_digest=s.out_digest,
                usage=recorder.sample(s),
            ))

        async def on_checkpoint(step: int, path: Path) -> None:
            await upload_checkpoint(ctx.http, f"/jobs/{asg.job_id}/checkpoints/{step}", path,
                                    {"epoch": asg.epoch, "stage": asg.stage_idx})

        runner = StageRunner(
            compute=compute, total_steps=spec.steps, microbatches=spec.microbatches,
            microbatch_size=spec.microbatch_size, checkpoint_every=asg.checkpoint_every,
            checkpoint_dir=ckdir, prev=prev, next=nxt,
            examples=examples, batch_size=spec.batch_size, seed=spec.seed,
            start_step=asg.resume_step, on_step=on_step, on_checkpoint=on_checkpoint,
            pace=make_pace(ctx.gpu_percent), peer_timeout=asg.peer_timeout_s,
        )
        ctx_session = getattr(ctx, "session", None)
        if ctx_session is not None:
            ctx_session.runner = runner
            ctx_session.compute = compute
        result = await runner.run()
        await emit(StageFinished(
            job_id=asg.job_id, epoch=asg.epoch, stage_idx=asg.stage_idx,
            reason=result.reason, last_step=result.last_step,
        ))
        return result
    except asyncio.CancelledError:
        await emit(StageFinished(
            job_id=asg.job_id, epoch=asg.epoch, stage_idx=asg.stage_idx,
            reason="cancelled", last_step=asg.resume_step,
        ))
        raise
    except Exception as e:
        log.exception("stage failed")
        await emit(StageFinished(
            job_id=asg.job_id, epoch=asg.epoch, stage_idx=asg.stage_idx,
            reason="error", last_step=asg.resume_step, detail=str(e),
            fatal=isinstance(e, DatasetError),
        ))
        raise
    finally:
        if compute is not None:
            compute.release_ring()
        for link in (prev, nxt):
            if link is not None:
                try:
                    await link.close()
                except Exception:
                    pass
        if server is not None:
            await server.close()


# --------------------------------------------------------------------------- subprocess CLI


async def _drain_on_stdin(session: StageSession, reader: asyncio.StreamReader) -> None:
    """The daemon asks a sandboxed worker to drain with a ``{"type":"drain"}`` line on stdin."""
    while line := await reader.readline():
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if isinstance(msg, dict) and msg.get("type") == "drain":
            session.request_drain()


def _stdin_is_pipe() -> bool:
    try:
        mode = os.fstat(sys.stdin.fileno()).st_mode
    except (OSError, ValueError):
        return False
    return stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode)


def _stdio_app() -> None:
    """Read assignment JSON from argv and emit protocol messages as JSON lines."""
    import argparse

    from slashcompute.agent.http import CoordHTTP
    from slashcompute.common.logging import setup_logging
    from slashcompute.common.protocol import dump, parse_coordinator_message

    p = argparse.ArgumentParser(description="/compute sandboxed worker")
    p.add_argument("--assignment", type=Path, required=True)
    args = p.parse_args()
    setup_logging("worker")
    blob = json.loads(args.assignment.read_text())
    asg = parse_coordinator_message(blob["assignment"])
    assert isinstance(asg, StageAssignment)

    ctx = WorkerContext(
        assignment=asg, http=CoordHTTP(blob["coordinator_url"], session_token=blob.get("session_token")),
        job_dir=Path(blob["job_dir"]), data_bind=blob["data_bind"],
        data_port=int(blob["data_port"]), gpu_percent=int(blob["gpu_percent"]),
        node_id=blob["node_id"], session=StageSession(),
    )

    async def emit(msg) -> None:
        print(dump(msg), flush=True)

    async def main() -> None:
        drains = None
        if _stdin_is_pipe():
            reader = asyncio.StreamReader()
            await asyncio.get_running_loop().connect_read_pipe(
                lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
            drains = asyncio.create_task(_drain_on_stdin(ctx.session, reader))
        else:
            log.warning("stdin is not a pipe; this worker cannot be drained")
        try:
            await run_stage(ctx, emit)
        finally:
            if drains is not None:
                drains.cancel()

    asyncio.run(main())


if __name__ == "__main__":
    _stdio_app()
