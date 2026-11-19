"""Training pays a node only once the steps sampled from its epoch have been checked.

A step is charged to its job at once but held for the node. A failed replay returns the whole
hold to the job, a sampled step the node keeps back is not paid, and what is left is paid when
the epoch has ended with nothing of it still being checked.
"""

from __future__ import annotations

import secrets

import mlx.core as mx
import pytest
from sqlmodel import select

from slashcompute.common import protocol as P
from slashcompute.common.config import EngineConfig
from slashcompute.community.credits import POT_ID, InsufficientCredits
from slashcompute.coordinator.core import Coordinator
from slashcompute.coordinator.db import EarningHold, StageRun, User, Verification, now
from slashcompute.coordinator.partitioner import StagePlan
from slashcompute.coordinator.scheduler import EpochState
from slashcompute.jobs import LoraFinetuneSpec
from slashcompute.metering import stage_step_flops
from slashcompute.pipeline.model_profile import profile_model
from slashcompute.transport.serialization import digest

SEQ = 6          # padded length of the rows in a sampled step
BUDGET = 1e12


def _config(tmp_path, **kw) -> EngineConfig:
    return EngineConfig(home=tmp_path / "home", scheduler_tick_s=0.05, verify_rate=0.0, **kw)


@pytest.fixture
def core(tmp_path):
    return Coordinator(_config(tmp_path))


def _member(core, name):
    """An account that has accepted the terms, and a session for it."""
    user = User(id=secrets.token_hex(8), email=f"{name}@lan.test", password_hash="-", name=name,
                accepted_terms_at=now())
    core.db.add(user)
    return user, core.auth._issue(user)[1]


def _device(mem=8 << 30):
    return P.DeviceProfile(
        chip="test", memory_total_bytes=mem, memory_available_bytes=mem,
        working_set_bytes=mem, memory_contrib_bytes=mem,
        matmul_tflops=1.0, mem_bandwidth_gbps=100.0,
    )


class Pool:
    """A job running as one stage on ``worker``, paid for by ``requester``; ``checker`` is an
    idle Mac of another account that can replay its steps."""

    def __init__(self, core, tiny_model, tiny_dataset):
        self.core = core
        self.sent: dict[str, list] = {}
        self.requester, _ = _member(core, "requester")
        self.host, self.host_token = _member(core, "host")
        self.other, self.other_token = _member(core, "other")
        core.credits.contribute(self.requester.id, BUDGET, 0)
        self.job = core.submit(LoraFinetuneSpec(
            model=str(tiny_model), dataset_path=str(tiny_dataset), steps=20, batch_size=2,
            microbatches=1, lora_rank=4, min_stages=1))
        core.credits.reserve_job(self.requester.id, self.job.id, BUDGET / 2)
        self.job.profile = profile_model(self.job.spec.model)
        layers = self.job.profile.num_layers
        self.job.current = EpochState(epoch=1, plans=[StagePlan(0, "worker", 0, layers, 0)])
        self.job.row.status, self.job.row.epoch = "running", 1
        core.db.save(self.job.row)
        core.db.add(StageRun(job_id=self.job.id, epoch=1, stage_idx=0, node_id="worker",
                             layer_start=0, layer_end=layers))
        # What an honest stage reports for a step whose rows are SEQ long.
        self.cost = stage_step_flops(self.job.profile, 0, layers, 2 * SEQ, SEQ, True)

    async def join(self, node_id, token):
        async def send(msg):
            self.sent.setdefault(node_id, []).append(msg)

        await self.core.on_register(P.Register(
            node_id=node_id, name=node_id, device=_device(), data_host="127.0.0.1",
            data_port=9700, gpu_percent=50, session_token=token), send)
        self.core.registry.get(node_id).canary_passed = True
        return self.core.registry.get(node_id)

    def tensors(self, step):
        mx.random.seed(step)
        x = mx.random.randint(0, 50, (2, SEQ))
        return {"x_in": x, "out": mx.random.normal((2, SEQ, 8))}

    async def step(self, step, flops=None):
        t = self.tensors(step)
        await self.core._on_step("worker", P.StepMetrics(
            job_id=self.job.id, epoch=1, stage_idx=0, step=step, loss=1.0,
            in_digest=digest(t["x_in"]), out_digest=digest(t["out"]),
            usage=P.UsageSample(flops=self.cost if flops is None else flops, tokens=2 * SEQ,
                                peak_mem_bytes=1, resident_mem_bytes=1, mem_byte_seconds=1.0,
                                wall_s=1.0, busy_s=0.5)))

    async def sample(self, step) -> str:
        return await self.core.verification.request_replay(self.job, "worker", 1, 0, step)

    async def hand_over(self, vid, step, error=None):
        verification = self.core.verification
        if error is None:
            mx.save_safetensors(str(verification.bundle_path(vid)), self.tensors(step))
        await verification.on_bundle_ready("worker", P.VerifyBundleReady(
            verify_id=vid, job_id=self.job.id, stage_idx=0, step=step,
            path=None if error else "bundle", error=error))

    async def replay(self, vid, step, *, agrees=True):
        """The checker replays the step and reports an output that agrees with the worker's or not."""
        verification = self.core.verification
        await verification.tick()                    # hands the queued replay to the checker
        assert self.status(vid) == "running"
        out = self.tensors(step)["out"]
        mx.save_safetensors(str(verification.result_path(vid)),
                            {"out": out if agrees else out + 1.0})
        await verification.on_result("checker", P.VerifyResult(verify_id=vid, kind="replay"))

    def status(self, vid):
        return self.core.db.get(Verification, vid).status

    def hold(self) -> EarningHold:
        with self.core.db.session() as s:
            return s.exec(select(EarningHold).where(EarningHold.job_id == self.job.id)).one()

    def spent(self) -> float:
        return self.core.credits.job_account(self.job.id).spent_flops

    def book(self) -> float:
        """Every FLOP there is: balances, open reservations and what is held for nodes."""
        credits = self.core.credits
        ids = [self.requester.id, self.host.id, self.other.id, POT_ID]
        return (sum(credits.balance(i) + credits.reserved_in_flight(i) for i in ids)
                + sum(h.flops for h in credits.open_holds()))

    def end_epoch(self):
        self.job.current.closed = True
        self.core.recovery._close_stage_run(self.job.id, 1, 0, "done")


@pytest.fixture
async def pool(core, tiny_model, tiny_dataset):
    p = Pool(core, tiny_model, tiny_dataset)
    await p.join("worker", p.host_token)
    await p.join("checker", p.other_token)
    return p


async def test_a_held_step_cannot_be_spent(pool):
    credits = pool.core.credits
    await pool.step(1)
    assert pool.spent() == pytest.approx(pool.cost)             # the job is charged at once
    assert credits.pending_earnings(pool.host.id) == pytest.approx(pool.cost)
    assert credits.balance(pool.host.id) == 0.0
    with pytest.raises(InsufficientCredits):
        credits.reserve_job(pool.host.id, "another-job", pool.cost / 2)
    assert pool.core.verification.settle_holds() == 0           # its epoch is still running


async def test_a_replay_that_passes_is_paid_once_the_epoch_has_ended(pool):
    credits, verification = pool.core.credits, pool.core.verification
    await pool.step(1)
    vid = await pool.sample(1)
    await pool.hand_over(vid, 1)
    pool.end_epoch()
    assert verification.settle_holds() == 0                     # a sampled step is still unchecked
    assert credits.balance(pool.host.id) == 0.0
    await pool.replay(vid, 1, agrees=True)
    assert pool.status(vid) == "passed"
    assert verification.settle_holds() == 1
    assert credits.balance(pool.host.id) == pytest.approx(pool.cost)
    assert credits.pending_earnings(pool.host.id) == 0.0
    assert verification.settle_holds() == 0                     # paid once


async def test_a_failed_replay_returns_the_whole_epoch_to_the_job(pool):
    """It used to mark the step ``disputed`` and nothing else: the node kept every FLOP."""
    core, credits = pool.core, pool.core.credits
    before = pool.book()
    for step in (1, 2, 3):
        await pool.step(step)
    vid = await pool.sample(3)
    await pool.hand_over(vid, 3)
    await pool.replay(vid, 3, agrees=False)

    assert pool.status(vid) == "failed"
    assert (pool.hold().status, pool.hold().flops) == ("forfeited", 0.0)
    assert pool.hold().forfeited_flops == pytest.approx(3 * pool.cost)
    assert pool.spent() == pytest.approx(0.0)                    # all three steps, not just the sampled one
    assert credits.lifetime_spent(pool.requester.id) == pytest.approx(0.0)
    assert core.registry.get("worker").canary_passed is False   # gets no more work
    assert pool.job.current.closed and pool.job.row.status == "recovering"
    assert any(isinstance(m, P.CancelStage) for m in pool.sent["worker"])
    assert core.verification.settle_holds() == 0
    assert credits.lifetime_earned(pool.host.id) == 0.0
    assert pool.book() == pytest.approx(before)


async def test_a_node_caught_out_bills_nothing_more_in_that_epoch(pool):
    await pool.step(1)
    assert pool.core.credits.forfeit_hold(pool.job.id, 1, "worker") == pytest.approx(pool.cost)
    assert pool.core.credits.hold_step(pool.job.id, 1, "worker", pool.host.id, pool.cost) == 0.0
    assert pool.spent() == pytest.approx(0.0)


async def test_a_replay_failing_after_the_job_ended_still_refunds_the_requester(pool):
    core, credits = pool.core, pool.core.credits
    start = credits.balance(pool.requester.id) + credits.reserved_in_flight(pool.requester.id)
    await pool.step(1)
    vid = await pool.sample(1)
    await pool.hand_over(vid, 1)
    await core.cancel_job(pool.job)                             # settles: the unspent rest goes back
    assert credits.balance(pool.requester.id) == pytest.approx(start - pool.cost)
    assert core.verification.settle_holds() == 0                # the replay is still waiting

    core.registry.get("worker").assignment = None
    await pool.replay(vid, 1, agrees=False)
    assert credits.balance(pool.requester.id) == pytest.approx(start)
    assert credits.reserved_in_flight(pool.requester.id) == 0.0
    assert credits.lifetime_earned(pool.host.id) == 0.0


async def test_a_step_billed_above_what_its_rows_cost_fails_its_replay(pool):
    """The ceiling allows every row to be as long as the job permits. A node that claims it
    for short rows computes the right output and still overcharges."""
    core = pool.core
    ceiling = core._max_step_flops(pool.job, pool.job.current, 0)
    assert ceiling > pool.cost * 2
    await pool.step(1, flops=ceiling)
    assert pool.spent() == pytest.approx(ceiling)
    vid = await pool.sample(1)
    await pool.hand_over(vid, 1)
    await pool.replay(vid, 1, agrees=True)
    row = core.db.get(Verification, vid)
    assert row.status == "failed" and row.detail.startswith("billed")
    assert pool.spent() == pytest.approx(0.0)
    assert pool.hold().status == "forfeited"


async def test_a_sampled_step_kept_back_is_not_paid(pool):
    """Answering every sampled step with an error used to be a free way never to be checked."""
    core, credits = pool.core, pool.core.credits
    for step in (1, 2, 3):
        await pool.step(step)
    first = await pool.sample(1)
    await pool.hand_over(first, 1, error="bundle no longer held")
    assert pool.status(first) == "refused"
    assert pool.spent() == pytest.approx(2 * pool.cost)
    assert credits.pending_earnings(pool.host.id) == pytest.approx(2 * pool.cost)
    assert core.registry.get("worker").canary_passed is True    # once can be bad luck
    assert not pool.job.current.closed

    second = await pool.sample(2)
    await pool.hand_over(second, 2, error="bundle no longer held")
    assert pool.status(second) == "refused"
    assert pool.hold().status == "forfeited" and pool.spent() == pytest.approx(0.0)
    assert core.registry.get("worker").canary_passed is False
    assert pool.job.current.closed


async def test_a_bundle_that_is_not_what_the_node_committed_to_is_a_failed_check(pool):
    await pool.step(1)
    vid = await pool.sample(1)
    mx.save_safetensors(str(pool.core.verification.bundle_path(vid)), pool.tensors(2))
    await pool.core.verification.on_bundle_ready("worker", P.VerifyBundleReady(
        verify_id=vid, job_id=pool.job.id, stage_idx=0, step=1, path="bundle"))
    assert pool.status(vid) == "failed"
    assert pool.hold().status == "forfeited" and pool.spent() == pytest.approx(0.0)


async def test_a_stage_that_had_stopped_owes_no_bundle(pool):
    """The last steps of a stage are sampled as it winds down: by then it holds nothing."""
    core, credits = pool.core, pool.core.credits
    await pool.step(1)
    vid = await pool.sample(1)
    pool.job.current.finished[0] = "done"
    await pool.hand_over(vid, 1, error="bundle no longer held")
    assert pool.status(vid) == "error"
    assert credits.pending_earnings(pool.host.id) == pytest.approx(pool.cost)
    pool.end_epoch()
    assert core.verification.settle_holds() == 1
    assert credits.balance(pool.host.id) == pytest.approx(pool.cost)


async def test_a_bundle_that_never_comes_counts_as_kept_back(pool):
    core = pool.core
    await pool.step(1)
    await pool.step(2)
    vid = await pool.sample(1)
    await core.verification.tick()
    assert pool.status(vid) == "fetching"                       # still within its time
    core.cfg.verify_fetch_timeout_s = 0.0
    await core.verification.tick()
    assert pool.status(vid) == "refused"
    assert pool.spent() == pytest.approx(pool.cost)


async def test_a_node_that_leaves_with_a_sampled_step_is_not_paid_for_it(pool):
    core, credits = pool.core, pool.core.credits
    await pool.step(1)
    await pool.step(2)
    core.registry.get("worker").assignment = None
    one, two = await pool.sample(1), await pool.sample(2)
    await core.recovery.on_node_lost("worker", "disconnected")
    assert (pool.status(one), pool.status(two)) == ("refused", "refused")
    assert pool.spent() == pytest.approx(0.0)
    assert pool.hold().status == "held"                         # leaving is not cheating
    assert credits.lifetime_earned(pool.host.id) == 0.0


async def test_a_replay_nobody_can_run_is_dropped_and_the_node_is_paid(pool):
    """With no other Mac to replay it, the check would wait for ever and so would the pay."""
    core, credits = pool.core, pool.core.credits
    await core.recovery.on_node_lost("checker", "disconnected")
    await pool.step(1)
    vid = await pool.sample(1)
    await pool.hand_over(vid, 1)
    core.cfg.verify_queue_timeout_s = 0.0
    await core.verification.tick()
    assert pool.status(vid) == "queued"                         # its stage is still running
    pool.end_epoch()
    await core.verification.tick()
    assert pool.status(vid) == "error"
    assert credits.balance(pool.host.id) == pytest.approx(pool.cost)


async def test_an_account_does_not_check_its_own_work_on_a_public_pool(pool):
    core = pool.core
    await pool.join("second-mac", pool.host_token)             # same account as the worker
    await core.recovery.on_node_lost("checker", "disconnected")
    assert [n.node_id for n in core.verification._verifiers("worker")] == ["second-mac"]
    core.cfg.public_pool = True
    assert core.verification._verifiers("worker") == []
    await pool.join("checker", pool.other_token)
    assert [n.node_id for n in core.verification._verifiers("worker")] == ["checker"]
    core.cfg.public_pool = False                                # a LAN prefers a stranger's Mac
    assert [n.node_id for n in core.verification._verifiers("worker")] == ["checker", "second-mac"]


async def test_holds_and_open_checks_survive_a_restart(pool, tmp_path):
    await pool.step(1)
    await pool.step(2)
    fetching = await pool.sample(1)
    queued = await pool.sample(2)
    await pool.hand_over(queued, 2)

    again = Coordinator(_config(tmp_path))
    rows = {vid: again.db.get(Verification, vid) for vid in (fetching, queued)}
    assert (rows[fetching].status, rows[fetching].detail) == ("error", "coordinator restarted")
    assert rows[queued].status == "queued"
    assert again.credits.pending_earnings(pool.host.id) == pytest.approx(2 * pool.cost)
    assert again.verification.settle_holds() == 0               # one check is still to run
    again.cfg.verify_queue_timeout_s = 0.0
    again.recovery._close_stage_run(pool.job.id, 1, 0, "aborted")
    await again.verification.tick()
    assert again.credits.balance(pool.host.id) == pytest.approx(2 * pool.cost)
