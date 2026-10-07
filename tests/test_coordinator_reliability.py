"""Coordinator reliability: stalled epochs, transactional starts, checkpoint races,
agent reconnects and loop isolation."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time

import mlx.core as mx
import pytest
from fastapi.testclient import TestClient
from sqlmodel import select
from starlette.websockets import WebSocketDisconnect

from slashcompute.common import protocol as P
from slashcompute.common.canary import run_canary_mlx
from slashcompute.common.config import EngineConfig
from slashcompute.coordinator.app import create_app
from slashcompute.coordinator.core import Coordinator
from slashcompute.coordinator.db import Checkpoint, Job, StageRun, UsageRecord
from slashcompute.coordinator.partitioner import StagePlan
from slashcompute.coordinator.scheduler import EpochState
from slashcompute.jobs import LoraFinetuneSpec

GB = 1024**3


@pytest.fixture
def core(tmp_path, tiny_model, tiny_dataset):
    c = Coordinator(EngineConfig(home=tmp_path / "home", verify_rate=0.0))
    c._tiny = (tiny_model, tiny_dataset)
    return c


def _spec(core, **kw):
    tiny_model, tiny_dataset = core._tiny
    base = dict(model=str(tiny_model), dataset_path=str(tiny_dataset), steps=10,
                batch_size=2, microbatches=1, lora_rank=4, min_stages=1)
    return LoraFinetuneSpec(**(base | kw))


def test_stalled_running_epoch_is_aborted(core):
    job = core.submit(_spec(core))
    job.current = EpochState(epoch=1, plans=[])
    job.row.status = "running"
    asyncio.run(core.recovery.tick())
    assert job.row.status == "running"  # still within stall_timeout_s

    job.current.last_progress -= core.cfg.stall_timeout_s + 1
    asyncio.run(core.recovery.tick())
    assert job.current.closed
    assert job.row.status == "recovering"
    assert "no progress" in job.row.error


# ------------------------------------------------------------ reliable agent sessions


class Agent:
    """A fake agent on the real /ws/agent socket. With ``session`` it speaks the reliable
    protocol: it numbers what it sends and acknowledges only when told to."""

    def __init__(self, client, node_id, session=None, last_seq=0):
        self.node_id, self.session = node_id, session
        self.ws = client.websocket_connect("/ws/agent").__enter__()
        mem = 8 * GB
        self.ws.send_text(P.dump(P.Register(
            node_id=node_id, name=node_id, data_host="127.0.0.1", data_port=9000, gpu_percent=100,
            session_id=session, last_seq=last_seq,
            device=P.DeviceProfile(chip="test", memory_total_bytes=mem, memory_available_bytes=mem,
                                   working_set_bytes=mem, memory_contrib_bytes=mem,
                                   matmul_tflops=1.0, mem_bandwidth_gbps=100.0))))
        self.welcome = self.recv()
        assert isinstance(self.welcome, P.Welcome)

    def recv(self):
        msg = P.parse_coordinator_message(self.ws.receive_text())
        while isinstance(msg, P.Ack):
            msg = P.parse_coordinator_message(self.ws.receive_text())
        return msg

    def send(self, msg, seq=None):
        self.ws.send_text(P.dump(msg.model_copy(update={"seq": seq})))

    def ack(self, msg):
        self.send(P.Ack(upto=msg.seq))

    def pass_canary(self, seq=None):
        req = self.recv()
        assert isinstance(req, P.VerifyRequest) and req.kind == "canary"
        self.send(P.VerifyResult(verify_id=req.verify_id, kind="canary",
                                 stats=run_canary_mlx(req.seed, req.size)), seq)
        return req

    def close(self):
        self.ws.__exit__(None, None, None)


def _wait(cond, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met")


@pytest.fixture
def live(tmp_path, tiny_model, tiny_dataset):
    cfg = EngineConfig(home=tmp_path / "home", scheduler_tick_s=0.02, verify_rate=0.0,
                       canary_size=64, stage_overhead_bytes=0, heartbeat_timeout_s=60,
                       reconnect_grace_s=30)
    app = create_app(cfg)
    with TestClient(app) as client:
        yield client, app.state.core, tiny_model, tiny_dataset


def _submit(client, tiny_model, tiny_dataset):
    spec = LoraFinetuneSpec(model=str(tiny_model), dataset_path=str(tiny_dataset), steps=4,
                            batch_size=2, microbatches=1, lora_rank=4, min_stages=1, max_stages=1)
    return client.post("/jobs", json=json.loads(spec.model_dump_json())).json()["id"]


def _usage():
    return P.UsageSample(flops=1e9, tokens=10, peak_mem_bytes=1, resident_mem_bytes=1,
                         mem_byte_seconds=1.0, wall_s=1.0, busy_s=0.5)


def test_reconnect_within_grace_keeps_the_stage_and_replays(live):
    client, core, tiny_model, tiny_dataset = live
    a = Agent(client, "n1", session="s1")
    assert a.welcome.session and not a.welcome.resumed and a.welcome.reconnect_grace_s == 30
    canary = a.pass_canary(seq=1)
    a.ack(canary)
    job_id = _submit(client, tiny_model, tiny_dataset)
    asg = a.recv()
    assert isinstance(asg, P.StageAssignment) and asg.seq == canary.seq + 1
    a.close()  # the Wi-Fi drops before the agent acknowledges its assignment

    _wait(lambda: not core.registry.get("n1").connected)
    job = core.jobs[job_id]
    assert job.row.recoveries == 0 and not job.current.closed and job.row.status == "starting"

    b = Agent(client, "n1", session="s1", last_seq=canary.seq)
    assert b.welcome.resumed and b.welcome.last_seq == 1  # it had our canary result
    replayed = b.recv()
    assert replayed == asg  # same assignment, same seq: nothing lost, nothing renumbered
    b.ack(replayed)
    b.send(P.StageReady(job_id=job_id, epoch=1, stage_idx=0), seq=2)
    _wait(lambda: job.row.status == "running")
    assert job.row.recoveries == 0 and core.registry.get("n1").connected
    b.close()


def test_replayed_step_metrics_are_billed_once(live):
    client, core, tiny_model, tiny_dataset = live
    a = Agent(client, "n1", session="s1")
    a.ack(a.pass_canary(seq=1))
    job_id = _submit(client, tiny_model, tiny_dataset)
    a.ack(a.recv())
    a.send(P.StageReady(job_id=job_id, epoch=1, stage_idx=0), seq=2)
    step = P.StepMetrics(job_id=job_id, epoch=1, stage_idx=0, step=1, loss=1.0, in_digest="a",
                         out_digest="b", usage=_usage())
    a.send(step, seq=3)
    a.send(step, seq=3)  # replayed after a reconnect the coordinator didn't notice
    a.send(step, seq=4)  # or resent under a new number
    _wait(lambda: core.registry.get("n1").inbox.last == 4)
    with core.db.session() as s:
        rows = s.exec(select(UsageRecord).where(UsageRecord.job_id == job_id)).all()
    assert len(rows) == 1
    a.close()


def test_agent_that_stays_away_loses_its_place(live):
    client, core, tiny_model, tiny_dataset = live
    core.cfg.reconnect_grace_s = 0.3
    a = Agent(client, "n1", session="s1")
    a.ack(a.pass_canary(seq=1))
    job_id = _submit(client, tiny_model, tiny_dataset)
    a.recv()
    a.close()
    job = core.jobs[job_id]
    _wait(lambda: job.current.closed)
    assert core.registry.get("n1") is None
    assert job.row.recoveries == 1 and job.row.status == "recovering"

    b = Agent(client, "n1", session="s1", last_seq=2)
    assert not b.welcome.resumed  # too late: a fresh session
    b.close()


def test_agents_without_sessions_are_torn_down_at_once(live):
    client, core, tiny_model, tiny_dataset = live
    a = Agent(client, "n1")
    assert not a.welcome.session
    req = a.pass_canary()
    assert req.seq is None  # nothing is numbered for an older agent
    _submit(client, tiny_model, tiny_dataset)
    assert a.recv().seq is None
    a.close()
    _wait(lambda: core.registry.get("n1") is None)


def test_node_evicted_for_missed_heartbeats_is_disconnected(live):
    client, core, *_ = live
    core.cfg.heartbeat_timeout_s = 0.3
    a = Agent(client, "n1")
    a.pass_canary()
    _wait(lambda: core.registry.get("n1") is None)
    try:
        with pytest.raises(WebSocketDisconnect):  # before, the socket stayed open and ignored
            for _ in range(10):
                a.recv()
    finally:
        a.close()


# ------------------------------------------------------------ transactional epoch start


def _online(core, node_id, sent, on_send=None):
    mem = 8 * GB

    async def send(msg):
        sent.setdefault(node_id, []).append(msg)
        if on_send is not None:
            await on_send(node_id, msg)

    core.registry.register(P.Register(
        node_id=node_id, name=node_id, data_host="127.0.0.1", data_port=9700, gpu_percent=50,
        device=P.DeviceProfile(chip="test", memory_total_bytes=mem, memory_available_bytes=mem,
                               working_set_bytes=mem, memory_contrib_bytes=mem, matmul_tflops=1.0,
                               mem_bandwidth_gbps=100.0)), send)


def _two_stage_job(core):
    return core.submit(_spec(core, min_stages=2, max_stages=2))


def test_epoch_start_that_cannot_be_recorded_changes_nothing(core, monkeypatch):
    sent = {}
    _online(core, "n1", sent)
    _online(core, "n2", sent)
    job = _two_stage_job(core)

    def locked(*rows):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(core.db, "save_all", locked)
    assert not asyncio.run(core.scheduler.try_start(job))
    assert (job.row.epoch, job.row.status, job.current) == (0, "queued", None)
    assert all(n.assignment is None for n in core.registry.nodes.values()) and not sent
    assert "could not record epoch 1" in job.wait_reason

    monkeypatch.undo()  # the database recovers: the next tick starts the job cleanly
    assert asyncio.run(core.scheduler.try_start(job))
    assert (job.row.epoch, job.row.status) == (1, "starting")
    assert core.db.get(Job, job.id).epoch == 1
    assert sorted(type(m).__name__ for ms in sent.values() for m in ms) == ["StageAssignment"] * 2


def test_node_lost_before_its_assignment_rolls_the_epoch_back(core):
    sent, told = {}, []

    async def lose_the_other(node_id, msg):  # while one node is told, the other drops out
        if isinstance(msg, P.StageAssignment) and not told:
            told.append(node_id)
            core.registry.remove("n2" if node_id == "n1" else "n1")

    _online(core, "n1", sent, lose_the_other)
    _online(core, "n2", sent, lose_the_other)
    job = _two_stage_job(core)
    assert not asyncio.run(core.scheduler.try_start(job))

    [first] = told
    assert [type(m).__name__ for m in sent[first]] == ["StageAssignment", "CancelStage"]
    assert sent[first][1].epoch == 1 and len(sent) == 1  # the lost node was never told
    assert job.current.closed and job.row.status == "queued" and job.row.recoveries == 0
    assert core.registry.get(first).assignment is None
    with core.db.session() as s:
        runs = s.exec(select(StageRun).where(StageRun.job_id == job.id)).all()
    assert len(runs) == 2 and {r.end_reason for r in runs} == {"start rolled back"}

    # The node that was told may have begun the stage: it is free once its agent says it stopped.
    assert not core.registry.get(first).schedulable
    asyncio.run(core.handle(first, P.StageFinished(job_id=job.id, epoch=1, stage_idx=0,
                                                   reason="cancelled", last_step=0)))
    assert core.registry.get(first).schedulable
    _online(core, "n2" if first == "n1" else "n1", sent)
    assert asyncio.run(core.scheduler.try_start(job))  # tried again as if nothing happened
    assert job.row.epoch == 2 and job.row.recoveries == 0


def test_epoch_cancelled_while_assignments_go_out_sends_no_more(core):
    sent, told = {}, []

    async def cancel_job(node_id, msg):
        if isinstance(msg, P.StageAssignment) and not told:
            told.append(node_id)
            await core.cancel_job(job)

    _online(core, "n1", sent, cancel_job)
    _online(core, "n2", sent, cancel_job)
    job = _two_stage_job(core)
    assert not asyncio.run(core.scheduler.try_start(job))
    other = "n2" if told[0] == "n1" else "n1"
    assert [type(m).__name__ for m in sent[other]] == ["CancelStage"]  # never a zombie stage
    assert job.row.status == "cancelled"


# ------------------------------------------------------------ checkpoint uploads


def _stage_checkpoint(spec, start, end, step, path):
    from slashcompute.pipeline.local import build_compute

    build_compute(spec, start, end, 6).save_checkpoint(path, step)
    return path.read_bytes()


@pytest.fixture
def two_stage_epoch(core, tmp_path, monkeypatch):
    from slashcompute.coordinator import checkpoints

    job = core.submit(_spec(core))
    job.current = EpochState(epoch=1, plans=[StagePlan(0, "n0", 0, 3, 0), StagePlan(1, "n1", 3, 6, 0)])
    parts = [_stage_checkpoint(job.spec, 0, 3, 5, tmp_path / "a.safetensors"),
             _stage_checkpoint(job.spec, 3, 6, 5, tmp_path / "b.safetensors")]
    merges = []
    real = checkpoints.merge_checkpoints

    def counting(paths, out):
        merges.append(out)
        real(paths, out)

    monkeypatch.setattr(checkpoints, "merge_checkpoints", counting)
    return job, parts, merges


def test_concurrent_uploads_merge_a_step_once(core, two_stage_epoch):
    job, parts, merges = two_stage_epoch

    async def upload():
        await core.on_checkpoint_upload(job.id, 1, 0, 5, parts[0])
        # The last stage's upload arrives three times at once (its retries overlapped).
        await asyncio.gather(*(core.on_checkpoint_upload(job.id, 1, 1, 5, parts[1]) for _ in range(3)))

    asyncio.run(upload())
    assert len(merges) == 1 and job.row.last_checkpoint_step == 5
    merged = core.checkpoints.merged_path(job.id, 5)
    layers = {int(m.group(1)) for k in mx.load(str(merged))
              if (m := re.match(r"adapter/layers\.L(\d+)\.", k))}
    assert layers == set(range(6))  # a whole, valid file with both stages in it
    with core.db.session() as s:
        assert len(s.exec(select(Checkpoint).where(Checkpoint.job_id == job.id)).all()) == 1
    assert not list(merged.parent.rglob(".*"))  # no temp files left behind


def test_recorded_checkpoint_is_never_rewritten(core, two_stage_epoch):
    job, parts, merges = two_stage_epoch
    asyncio.run(core.on_checkpoint_upload(job.id, 1, 0, 5, parts[0]))
    asyncio.run(core.on_checkpoint_upload(job.id, 1, 1, 5, parts[1]))
    merged = core.checkpoints.merged_path(job.id, 5)
    before = os.stat(merged)
    # A drain right after resuming from step 5: every stage uploads step 5 again while
    # other stages may be downloading it as their resume checkpoint.
    asyncio.run(core.on_checkpoint_upload(job.id, 1, 0, 5, parts[0]))
    asyncio.run(core.on_checkpoint_upload(job.id, 1, 1, 5, parts[1]))
    after = os.stat(merged)
    assert len(merges) == 1
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)


def test_uploads_to_a_closed_epoch_or_unknown_stage_are_refused(core, two_stage_epoch):
    job, parts, merges = two_stage_epoch
    with pytest.raises(ValueError, match="no stage 2"):
        asyncio.run(core.on_checkpoint_upload(job.id, 1, 2, 5, parts[0]))
    job.current.closed = True  # aborted: a stage still running in it must not move the job on
    with pytest.raises(ValueError, match="stale"):
        asyncio.run(core.on_checkpoint_upload(job.id, 1, 0, 5, parts[0]))
    assert job.row.last_checkpoint_step == 0 and not merges


# ------------------------------------------------------------ loops and liveness


def test_database_uses_wal(core):
    with core.db.engine.connect() as conn:
        assert conn.exec_driver_sql("PRAGMA journal_mode").scalar() == "wal"


def test_blocked_loop_does_not_evict_live_nodes(core):
    core.cfg.heartbeat_interval_s, core.cfg.heartbeat_timeout_s = 0.1, 0.3
    core.cfg.scheduler_tick_s = 0.02
    _online(core, "n1", {})

    async def scenario():
        await core.recovery.tick()
        time.sleep(0.5)  # a slow write blocks the loop past the heartbeat timeout
        await core.recovery.tick()  # the node's heartbeats are still unread in its socket
        assert core.registry.get("n1") is not None
        core.registry.heartbeat("n1", "idle")  # ...and get read now
        await core.recovery.tick()
        assert core.registry.get("n1") is not None

        # A node that really went quiet is still evicted once the loop runs normally.
        for _ in range(40):
            await asyncio.sleep(0.02)
            await core.recovery.tick()
        assert core.registry.get("n1") is None

    asyncio.run(scenario())


def test_liveness_checks_keep_running_while_scheduling_is_stuck(core):
    core.cfg.scheduler_tick_s = 0.01
    recovery_ticks = []
    real = core.recovery.tick

    async def counted():
        recovery_ticks.append(1)
        await real()

    async def stuck():
        await asyncio.sleep(3600)  # e.g. profiling a model over a slow link

    async def broken():
        raise RuntimeError("verification bug")

    core.recovery.tick, core.scheduler.tick, core.verification.tick = counted, stuck, broken

    async def scenario():
        core.start()
        await asyncio.sleep(0.3)
        await core.stop()

    asyncio.run(scenario())
    assert len(recovery_ticks) >= 10


def test_silent_agent_with_a_session_is_held_then_resumes(live):
    client, core, tiny_model, tiny_dataset = live
    core.cfg.heartbeat_timeout_s = 0.3
    a = Agent(client, "n1", session="s1")
    a.ack(a.pass_canary(seq=1))
    job_id = _submit(client, tiny_model, tiny_dataset)
    asg = a.recv()
    a.ack(asg)
    # The network goes quiet without closing anything: no heartbeats arrive.
    _wait(lambda: not core.registry.get("n1").connected)
    with pytest.raises(WebSocketDisconnect):  # its socket is closed so it reconnects...
        for _ in range(10):
            a.recv()
    a.close()
    job = core.jobs[job_id]
    assert not job.current.closed and job.row.recoveries == 0  # ...but its place is held

    b = Agent(client, "n1", session="s1", last_seq=asg.seq)
    assert b.welcome.resumed
    b.send(P.Heartbeat(node_id="n1", status="loading", job_id=job_id, epoch=1))
    b.send(P.StageReady(job_id=job_id, epoch=1, stage_idx=0), seq=2)
    _wait(lambda: job.row.status == "running")
    assert job.row.recoveries == 0
    b.close()


# ------------------------------------------------------------ a stopped job frees its Mac only once it stopped


def _hb(node_id, status, job_id=None, epoch=None, **kw):
    return P.Heartbeat(node_id=node_id, status=status, job_id=job_id, epoch=epoch, **kw)


def _runs(core, job_id):
    with core.db.session() as s:
        return s.exec(select(StageRun).where(StageRun.job_id == job_id)).all()


def _stopped(core, node_id, job, epoch=1):
    asyncio.run(core.handle(node_id, P.StageFinished(job_id=job.id, epoch=epoch, stage_idx=0,
                                                     reason="cancelled", last_step=0)))


def test_cancelled_jobs_node_is_not_reassigned_until_it_lets_go(core):
    """Behind "the job says starting but never runs": a cancel freed the Mac at once, so the
    next job went to an agent still downloading the cancelled job's model and queued there."""
    sent = {}
    _online(core, "n1", sent)
    a = core.submit(_spec(core))
    assert asyncio.run(core.scheduler.try_start(a))
    b = core.submit(_spec(core))
    asyncio.run(core.cancel_job(a))
    assert isinstance(sent["n1"][-1], P.CancelStage)
    assert not core.registry.get("n1").schedulable
    assert [r.end_reason for r in _runs(core, a.id)] == ["cancelled"]
    asyncio.run(core.scheduler.tick())
    assert b.row.status == "queued" and "waiting for n1 to stop a cancelled job" in b.wait_reason

    _stopped(core, "n1", a)
    asyncio.run(core.scheduler.tick())
    assert b.row.status == "starting" and core.registry.get("n1").assignment.job_id == b.id


def test_release_confirmed_by_idle_heartbeat_or_bounded_fallback(core):
    sent = {}
    _online(core, "n1", sent)
    node = core.registry.get("n1")
    job = core.submit(_spec(core))
    assert asyncio.run(core.scheduler.try_start(job))
    asyncio.run(core.cancel_job(job))
    asyncio.run(core.handle("n1", _hb("n1", "idle")))   # may have left before it got the cancel
    assert node.releasing is not None
    node.releasing.since -= core.cfg.heartbeat_interval_s
    asyncio.run(core.handle("n1", _hb("n1", "idle")))   # sent after it: it let go
    assert node.releasing is None and node.schedulable

    job2 = core.submit(_spec(core))
    assert asyncio.run(core.scheduler.try_start(job2))
    asyncio.run(core.cancel_job(job2))
    asyncio.run(core.handle("n1", _hb("n1", "loading", job2.id, 1)))
    node.releasing.since -= core.cfg.release_timeout_s + 1
    asyncio.run(core.recovery.tick())
    assert node.releasing is not None and not node.schedulable   # still busy with it: held
    asyncio.run(core.handle("n1", _hb("n1", "loading")))        # an agent that never says
    asyncio.run(core.recovery.tick())
    assert node.releasing is None and node.schedulable           # ...is freed after the timeout


def test_late_stage_finished_confirms_release_but_is_not_applied(core):
    sent = {}
    _online(core, "n1", sent)
    _online(core, "n2", sent)
    job = core.submit(_spec(core))
    assert asyncio.run(core.scheduler.try_start(job))
    [first] = [p.node_id for p in job.current.plans]
    other = "n2" if first == "n1" else "n1"
    asyncio.run(core.recovery.abort_epoch(job, "test"))
    assert not core.registry.get(first).schedulable
    assert asyncio.run(core.scheduler.try_start(job))
    assert job.current.epoch == 2 and [p.node_id for p in job.current.plans] == [other]
    recoveries = job.row.recoveries
    _stopped(core, first, job, epoch=1)
    assert core.registry.get(first).schedulable
    assert not job.current.closed and job.row.recoveries == recoveries


def test_abort_closes_stage_runs_of_lost_nodes(core):
    sent = {}
    _online(core, "n1", sent)
    _online(core, "n2", sent)
    job = _two_stage_job(core)
    assert asyncio.run(core.scheduler.try_start(job))
    core.registry.remove("n2")
    asyncio.run(core.recovery.abort_epoch(job, "node n2 lost"))
    assert sorted(r.end_reason for r in _runs(core, job.id)) == ["aborted", "aborted"]


def test_start_timeout_waits_while_fetch_progress_advances(core):
    """A big model took longer than the start timeout to download, so the job was aborted and
    re-sent to the same Mac, behind the same download, up to 20 times."""
    core.cfg.stage_start_timeout_s = 0.3
    sent = {}
    _online(core, "n1", sent)
    job = core.submit(_spec(core))
    assert asyncio.run(core.scheduler.try_start(job))
    done = 0
    for _ in range(6):                                   # downloading for twice the timeout
        done += 1000
        asyncio.run(core.handle("n1", _hb("n1", "loading", job.id, 1, phase="fetching",
                                          fetch_done_bytes=done, fetch_total_bytes=10**6)))
        time.sleep(0.1)
        asyncio.run(core.scheduler.tick())
        assert job.row.status == "starting"
    [stage] = core.job_view(job)["stages"]
    assert stage["node_name"] == "n1" and stage["phase"] == "fetching"
    assert (stage["fetch_done_bytes"], stage["fetch_total_bytes"]) == (done, 10**6)

    time.sleep(0.4)                                      # the download stalls
    asyncio.run(core.scheduler.tick())
    assert job.row.status == "recovering" and "no progress" in job.row.error
    assert core.registry.get("n1").avoid_until > time.monotonic()


def test_start_timeout_abort_prefers_another_node(core):
    sent = {}
    _online(core, "n1", sent)
    core.registry.get("n1").avoid_until = time.monotonic() + 60   # it just stalled a start
    _online(core, "n2", sent)
    job = core.submit(_spec(core))
    assert asyncio.run(core.scheduler.try_start(job))
    assert [p.node_id for p in job.current.plans] == ["n2"]
    job2 = core.submit(_spec(core))                      # only the stalled Mac is free: use it
    assert asyncio.run(core.scheduler.try_start(job2))
    assert [p.node_id for p in job2.current.plans] == ["n1"]


def test_job_needing_more_macs_than_the_pool_does_not_block_later_jobs(core):
    sent = {}
    _online(core, "n1", sent)
    a = core.submit(_spec(core, min_stages=2))
    b = core.submit(_spec(core))
    asyncio.run(core.scheduler.tick())
    assert a.row.status == "queued" and a.wait_reason == "needs 2 Macs; the pool has 1"
    assert core.job_view(a)["wait_reason"] == "needs 2 Macs; the pool has 1"
    assert b.row.status == "starting"

    _online(core, "n2", sent)        # now the pool could run it: first in line again
    c = core.submit(_spec(core))
    asyncio.run(core.scheduler.tick())
    assert a.row.status == "queued" and c.row.status == "queued"
