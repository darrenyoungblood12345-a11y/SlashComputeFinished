"""Checking that contributed compute is real and correct.

Three mechanisms:

* Chain check (every step, free): stage k's committed output digest must
  equal stage k+1's committed input digest.
* Sampled replay (~verify_rate of steps): the stage uploads the microbatch-0
  input, output and adapters it used. The upload must match the digests it
  committed at step time, then a *different* idle node recomputes the output
  and the two are compared within a tolerance. If no other node is idle the
  replay waits (for example until the job finishes and frees its nodes).
* Canary: on join (and periodically when idle) a node runs a seeded matmul
  whose statistics the coordinator computes independently. Nodes that fail
  are excluded from scheduling.

Failures mark the affected usage records ``disputed``, and they cost money: a training step
is charged to its job at once but only *held* for the node that ran it (Credits.hold_step).

* The hold is paid once its epoch has ended and no replay of it is still open.
* A failed replay forfeits the node's whole hold for that epoch back to the job, ends the
  epoch and takes the node out of scheduling.
* A node that keeps back a sampled step while its stage is still running forfeits that step;
  ``verify_max_target_errors`` of those in one epoch count as a failed replay.
* An ``error`` is nobody's fault (no verifier, a verifier that broke, a stage that had
  already stopped) and costs nothing.
"""

from __future__ import annotations

import logging
import random
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import mlx.core as mx
import numpy as np
from sqlmodel import select

from slashcompute.common.canary import compare_stats, expected_stats
from slashcompute.common.protocol import (
    StepMetrics, UsageSample, VerifyBundleReady, VerifyFetch, VerifyRequest, VerifyResult,
)
from slashcompute.coordinator.db import Node, StageRun, Verification, now
from slashcompute.coordinator.scheduler import TERMINAL
from slashcompute.metering.flops import replay_flops, stage_step_flops
from slashcompute.transport.serialization import digest

if TYPE_CHECKING:
    from slashcompute.coordinator.core import Coordinator
    from slashcompute.coordinator.scheduler import JobRuntime

log = logging.getLogger(__name__)

OPEN = ("fetching", "queued", "running")
# Rounding room when a billed step is compared with what its replayed shape costs.
COST_SLACK = 1e-6


def relative_error(claimed: np.ndarray, replay: np.ndarray) -> float:
    a, b = claimed.astype(np.float64), replay.astype(np.float64)
    if a.shape != b.shape:
        return float("inf")
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))


class VerificationManager:
    def __init__(self, core: "Coordinator") -> None:
        self.core = core
        self.rng = random.Random()
        self._canary_expected: dict[str, tuple[int, int]] = {}
        self._drop_interrupted()

    def _drop_interrupted(self) -> None:
        """What a previous run of the coordinator left half done. No node is connected yet and
        none remembers being asked, so a fetch or a canary cannot finish and nobody is to blame;
        a replay whose bundle is already here only needs a new verifier."""
        with self.core.db.session() as s:
            for v in s.exec(select(Verification).where(
                    Verification.status.in_(["fetching", "running"]))).all():
                if v.status == "running" and v.kind == "replay":
                    v.status, v.verifier_node_id = "queued", None
                else:
                    v.status, v.detail, v.finished_at = "error", "coordinator restarted", now()
                s.add(v)
            s.commit()

    def dir(self, vid: str) -> Path:
        d = self.core.cfg.coordinator_dir / "verify" / vid
        d.mkdir(parents=True, exist_ok=True)
        return d

    def bundle_path(self, vid: str) -> Path:
        return self.dir(vid) / "bundle.safetensors"

    def result_path(self, vid: str) -> Path:
        return self.dir(vid) / "replay.safetensors"

    def _get(self, vid: str) -> Optional[Verification]:
        return self.core.db.get(Verification, vid)

    def _finish(self, v: Verification, status: str, detail: str | None = None,
                rel_error: float | None = None) -> None:
        v.status, v.detail, v.rel_error, v.finished_at = status, detail, rel_error, now()
        self.core.db.save(v)
        if status in ("failed", "refused") and v.job_id is not None and v.step is not None:
            self.core.ledger.dispute(v.job_id, v.epoch, v.stage_idx, v.step)
        log.log(logging.WARNING if status in ("failed", "refused") else logging.INFO,
                "verification %s (%s job=%s step=%s stage=%s node=%s): %s %s", v.id[:8], v.kind,
                v.job_id, v.step, v.stage_idx, v.target_node_id[:8], status, detail or "")

    # ------------------------------------------------------------ consequences

    def _stage_running(self, v: Verification) -> bool:
        """Is the stage this replay samples still running? One that has stopped has let go of
        its bundles, so failing to hand one over is then no fault of the node."""
        job = self.core.jobs.get(v.job_id)
        cur = job.current if job else None
        return (cur is not None and cur.epoch == v.epoch and not cur.closed
                and v.stage_idx not in cur.finished)

    def _take_back(self, v: Verification, *, whole: bool) -> float:
        """Return what the target was held for (this step, or its whole epoch) to the job."""
        core = self.core
        if whole:
            amount = core.credits.forfeit_hold(v.job_id, v.epoch, v.target_node_id)
        else:
            record = core.ledger.find_step(v.job_id, v.epoch, v.stage_idx, v.step)
            amount = core.credits.forfeit_hold(v.job_id, v.epoch, v.target_node_id,
                                               record.billed_flops if record else 0.0)
        job = core.jobs.get(v.job_id)
        if amount and (job is None or job.row.status in TERMINAL):
            core.credits.settle_job(v.job_id)  # the job is over: the requester gets it now
        return amount

    def _refusals(self, v: Verification) -> int:
        with self.core.db.session() as s:
            return len(s.exec(select(Verification.id).where(
                Verification.kind == "replay", Verification.status == "refused",
                Verification.job_id == v.job_id, Verification.epoch == v.epoch,
                Verification.target_node_id == v.target_node_id)).all())

    async def _not_delivered(self, v: Verification, detail: str, *, strike: bool = True) -> None:
        """The target did not hand over a step it was sampled for. Unchecked work is not paid;
        doing it repeatedly is how a node would hide what a replay would show."""
        if not self._stage_running(v):
            self._finish(v, "error", detail)
            return
        self._finish(v, "refused", detail)
        self._take_back(v, whole=False)
        limit = self.core.cfg.verify_max_target_errors
        if strike and self._refusals(v) >= limit:
            await self._caught(v, f"kept back {limit} sampled steps")

    async def _caught(self, v: Verification, reason: str) -> None:
        """A check showed this node's reported work cannot be trusted: nothing it was held for
        in this epoch is paid, it gets no more work, and the epoch restarts without it."""
        core = self.core
        amount = self._take_back(v, whole=True)
        node = core.registry.get(v.target_node_id)
        if node is not None:
            node.canary_passed = False
        row = core.db.get(Node, v.target_node_id)
        if row is not None:
            row.canary_passed = False
            core.db.save(row)
        log.warning("node %s %s in job %s epoch %s: %.3e held FLOPs returned to the job",
                    v.target_node_id[:8], reason, v.job_id, v.epoch, amount)
        job = core.jobs.get(v.job_id)
        cur = job.current if job else None
        if cur is not None and cur.epoch == v.epoch and not cur.closed:
            await core.recovery.abort_epoch(
                job, f"node {v.target_node_id[:8]} failed verification: {reason}")

    def settle_holds(self) -> int:
        """Pay every hold whose epoch has ended with no replay of it still open. Returns how
        many were paid. Holds live in the database, so this also settles what a restart found."""
        core, paid = self.core, 0
        for hold in core.credits.open_holds():
            job = core.jobs.get(hold.job_id)
            cur = job.current if job else None
            if cur is not None and cur.epoch == hold.epoch and not cur.closed:
                continue
            with core.db.session() as s:
                waiting = s.exec(select(Verification.id).where(
                    Verification.kind == "replay", Verification.status.in_(OPEN),
                    Verification.job_id == hold.job_id, Verification.epoch == hold.epoch,
                    Verification.target_node_id == hold.node_id)).first()
            if waiting is None:
                core.credits.release_hold(hold.id)
                paid += 1
        return paid

    async def _expire(self) -> None:
        """Replays nobody will finish: a bundle that never came, or no Mac free to replay it."""
        cfg = self.core.cfg
        with self.core.db.session() as s:
            waiting = list(s.exec(select(Verification).where(
                Verification.kind == "replay",
                Verification.status.in_(["fetching", "queued"]))).all())
            if not waiting:
                return
            ended = {(r.job_id, r.epoch, r.stage_idx): r.ended_at for r in s.exec(
                select(StageRun).where(
                    StageRun.job_id.in_(sorted({v.job_id for v in waiting})))).all()}
        for v in waiting:
            if v.status == "fetching":
                if now() - v.created_at > cfg.verify_fetch_timeout_s:
                    await self._not_delivered(v, "bundle not uploaded in time")
                continue
            # Queued: the job's own Macs are busy while it runs, so the wait only counts from
            # the end of its stage.
            stopped = ended.get((v.job_id, v.epoch, v.stage_idx))
            if stopped is not None and now() - max(stopped, v.created_at) > cfg.verify_queue_timeout_s:
                self._finish(v, "error", "no other Mac became free to replay it")

    def _owner(self, node_id: str) -> Optional[str]:
        node = self.core.registry.get(node_id)
        return (node.user_id if node and node.user_id else None) or self.core.credits.owner_of(node_id)

    def _verifiers(self, target_node_id: str) -> list:
        """Idle nodes that may replay ``target_node_id``'s step, best first. A node of the same
        account is a last resort on a LAN (it still catches a faulty Mac) and never used on a
        public pool, where it would let an account vouch for its own work."""
        core = self.core
        owner = self._owner(target_node_id)

        def same_account(n) -> bool:
            return owner is not None and self._owner(n.node_id) == owner

        idle = [n for n in core.registry.schedulable()
                if n.node_id != target_node_id and n.canary_passed]
        if core.cfg.public_pool:
            return [n for n in idle if not same_account(n)]
        return sorted(idle, key=same_account)

    # ------------------------------------------------------------ per step

    async def on_step(self, job: "JobRuntime", node_id: str, msg: StepMetrics) -> None:
        self._chain_check(job, node_id, msg)
        if self.rng.random() < self.core.cfg.verify_rate:
            await self.request_replay(job, node_id, msg.epoch, msg.stage_idx, msg.step)

    def _chain_check(self, job: "JobRuntime", node_id: str, msg: StepMetrics) -> None:
        key = (msg.epoch, msg.step)
        per_stage = job.digests.setdefault(key, {})
        per_stage[msg.stage_idx] = (msg.in_digest, msg.out_digest, node_id)
        for a, b in ((msg.stage_idx - 1, msg.stage_idx), (msg.stage_idx, msg.stage_idx + 1)):
            if a in per_stage and b in per_stage and per_stage[a][1] != per_stage[b][0]:
                for idx in (a, b):
                    v = Verification(id=uuid.uuid4().hex, kind="chain", job_id=job.id,
                                     epoch=msg.epoch, step=msg.step, stage_idx=idx,
                                     target_node_id=per_stage[idx][2])
                    self._finish(v, "failed", f"stage {a} output != stage {b} input")
        # keep memory bounded
        for old in [k for k in job.digests if k[0] == msg.epoch and k[1] < msg.step - 50]:
            job.digests.pop(old, None)

    async def request_replay(self, job: "JobRuntime", node_id: str, epoch: int, stage_idx: int,
                             step: int) -> str:
        v = Verification(id=uuid.uuid4().hex, kind="replay", job_id=job.id, epoch=epoch, step=step,
                         stage_idx=stage_idx, target_node_id=node_id, status="fetching")
        self.core.db.add(v)
        await self.core.send(node_id, VerifyFetch(verify_id=v.id, job_id=job.id, epoch=epoch,
                                                  stage_idx=stage_idx, step=step))
        return v.id

    # ------------------------------------------------------------ bundle

    def store_bundle(self, vid: str, data: bytes) -> None:
        v = self._get(vid)
        if v is None or v.kind != "replay" or v.status != "fetching":
            raise KeyError(vid)
        self.bundle_path(vid).write_bytes(data)

    async def on_bundle_ready(self, node_id: str, msg: VerifyBundleReady) -> None:
        v = self._get(msg.verify_id)
        if v is None or v.status != "fetching" or v.target_node_id != node_id:
            return
        if msg.error or not self.bundle_path(v.id).exists():
            await self._not_delivered(v, msg.error or "bundle missing")
            return
        record = self.core.ledger.find_step(v.job_id, v.epoch, v.stage_idx, v.step)
        matches = record is not None and await self.core.off_loop(
            _bundle_matches, self.bundle_path(v.id), record.in_digest, record.out_digest)
        v = self._get(msg.verify_id)  # re-read: it may have moved on while we hashed
        if v is None or v.status != "fetching":
            return
        if not matches:
            self._finish(v, "failed", "uploaded bundle does not match digests committed at step time")
            await self._caught(v, "uploaded a bundle that is not what it committed to")
            return
        v.status = "queued"
        self.core.db.save(v)

    # ------------------------------------------------------------ dispatch

    async def tick(self) -> None:
        await self._expire()
        await self._dispatch_replays()
        await self._dispatch_canaries()
        self.settle_holds()

    async def _dispatch_replays(self) -> None:
        core = self.core
        with core.db.session() as s:
            queued = list(s.exec(select(Verification).where(
                Verification.kind == "replay", Verification.status == "queued",
            ).order_by(Verification.created_at)).all())
        for v in queued:
            idle = self._verifiers(v.target_node_id)
            if not idle:
                continue
            verifier = idle[0]
            job = core.jobs.get(v.job_id)
            with core.db.session() as s:
                run = s.exec(select(StageRun).where(
                    StageRun.job_id == v.job_id, StageRun.epoch == v.epoch,
                    StageRun.stage_idx == v.stage_idx)).first()
            if job is None or run is None or job.profile is None:
                self._finish(v, "error", "job or stage metadata missing")
                continue
            verifier.verifying = v.id
            v.status, v.verifier_node_id = "running", verifier.node_id
            core.db.save(v)
            spec = job.spec
            await core.send(verifier.node_id, VerifyRequest(
                verify_id=v.id, kind="replay", model=spec.model, layer_start=run.layer_start,
                layer_end=run.layer_end, num_layers=job.profile.num_layers,
                lora_rank=spec.lora_rank, lora_scale=spec.lora_scale, lora_targets=spec.lora_targets,
                bundle_url=f"/verify/{v.id}/bundle",
            ))

    async def _dispatch_canaries(self) -> None:
        core = self.core
        for node in list(core.registry.nodes.values()):
            due = node.canary_passed is None or (
                node.canary_passed and time.monotonic() - node.last_canary > core.cfg.canary_interval_s)
            if not due or node.draining or node.assignment or node.releasing or node.verifying:
                continue
            seed, size = self.rng.randrange(2**31), core.cfg.canary_size
            v = Verification(id=uuid.uuid4().hex, kind="canary", target_node_id=node.node_id,
                             verifier_node_id=node.node_id, status="running")
            core.db.add(v)
            self._canary_expected[v.id] = (seed, size)
            node.verifying = v.id
            node.last_canary = time.monotonic()
            await core.send(node.node_id, VerifyRequest(verify_id=v.id, kind="canary",
                                                        seed=seed, size=size))

    # ------------------------------------------------------------ results

    def store_result(self, vid: str, data: bytes) -> None:
        v = self._get(vid)
        if v is None or v.kind != "replay" or v.status != "running":
            raise KeyError(vid)
        self.result_path(vid).write_bytes(data)

    async def on_result(self, node_id: str, msg: VerifyResult) -> None:
        core = self.core
        node = core.registry.get(node_id)
        if node is not None and node.verifying == msg.verify_id:
            node.verifying = None
        v = self._get(msg.verify_id)
        if v is None or v.status != "running" or v.verifier_node_id != node_id:
            return
        if msg.kind == "canary":
            seed, size = self._canary_expected.pop(v.id)
            if msg.error:
                passed, worst, detail = False, None, msg.error
            else:
                passed, worst = compare_stats(msg.stats, expected_stats(seed, size),
                                              core.cfg.canary_rel_tolerance)
                detail = None
            self._finish(v, "passed" if passed else "failed", detail, worst)
            if node is not None:
                node.canary_passed = passed
            row = core.db.get(Node, v.target_node_id)
            if row is not None:
                row.canary_passed = passed
                core.db.save(row)
            return

        if msg.error or not self.result_path(v.id).exists():
            self._finish(v, "error", msg.error or "replay output missing")
            return
        err, (batch, seq) = await core.off_loop(_replay_error, self.bundle_path(v.id),
                                                self.result_path(v.id))
        v = self._get(msg.verify_id)  # re-read: it may have moved on while we compared
        if v is None or v.status != "running" or v.verifier_node_id != node_id:
            return
        job = core.jobs.get(v.job_id)
        run = None
        if job is not None and job.profile is not None:
            with core.db.session() as s:
                run = s.exec(select(StageRun).where(
                    StageRun.job_id == v.job_id, StageRun.epoch == v.epoch,
                    StageRun.stage_idx == v.stage_idx)).first()
        detail = None if err <= core.cfg.verify_rel_tolerance else "replay mismatch"
        if detail is None and run is not None:
            # The output is right; so must the bill be. The bundle shows how long the step's
            # rows really were, and the job fixes how many there are.
            record = core.ledger.find_step(v.job_id, v.epoch, v.stage_idx, v.step)
            cost = stage_step_flops(job.profile, run.layer_start, run.layer_end,
                                    job.spec.batch_size * seq, seq,
                                    run.layer_end == job.profile.num_layers)
            if record is not None and record.billed_flops > cost * (1 + COST_SLACK):
                detail = (f"billed {record.billed_flops:.3e} FLOPs for a step that cost "
                          f"{cost:.3e}")
        self._finish(v, "failed" if detail else "passed", detail, err)
        if detail:
            await self._caught(v, detail)
        if run is not None:
            flops = replay_flops(job.profile, run.layer_start, run.layer_end, batch * seq,
                                 seq, run.layer_end == job.profile.num_layers)
            core.ledger.record_verify(node_id, v.job_id, UsageSample(
                flops=flops, tokens=batch * seq, peak_mem_bytes=0,
                resident_mem_bytes=0, mem_byte_seconds=0.0,
                wall_s=msg.stats.get("wall_s", 0.0), busy_s=msg.stats.get("busy_s", 0.0)))

    async def on_node_lost(self, node_id: str) -> None:
        core = self.core
        with core.db.session() as s:
            open_vs = list(s.exec(select(Verification).where(
                Verification.status.in_(["fetching", "running"]))).all())
        for v in open_vs:
            if v.status == "fetching" and v.target_node_id == node_id:
                # Its sampled step goes unchecked, so unpaid; leaving is not counted against it.
                await self._not_delivered(v, "target node left before uploading bundle",
                                          strike=False)
            elif v.status == "running" and v.verifier_node_id == node_id:
                if v.kind == "canary":
                    self._finish(v, "error", "node left during canary")
                else:  # hand it to another verifier
                    v.status, v.verifier_node_id = "queued", None
                    core.db.save(v)


# ------------------------------------------------------------ off-loop checks


def _bundle_matches(path: Path, in_digest: str, out_digest: str) -> bool:
    tensors = mx.load(str(path))
    return digest(tensors["x_in"]) == in_digest and digest(tensors["out"]) == out_digest


def _replay_error(bundle_path: Path, result_path: Path) -> tuple[float, tuple[int, int]]:
    """Relative error of the replayed output against the claimed one, and the (batch,
    seq) shape of the replayed input."""
    claimed = mx.load(str(bundle_path))
    replay = mx.load(str(result_path))["out"]
    err = relative_error(np.asarray(claimed["out"].astype(mx.float32)),
                         np.asarray(replay.astype(mx.float32)))
    x = claimed["x_in"]
    return err, (int(x.shape[0]), int(x.shape[1]))
