"""Drains, dropouts and epoch teardown.

* Drain (contributor pressed stop): ask stage 0 to stop after its current
  step. Every stage checkpoints at that step and exits. The job re-queues and
  resumes from that checkpoint without the drained node.
* Dropout (heartbeat timeout, socket loss, or a drain that overran its grace
  period): cancel the epoch's other stages and resume from the last complete
  checkpoint.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Optional

from sqlmodel import select

from slashcompute.common.protocol import CancelStage, Drain, StageFinished
from slashcompute.coordinator.db import Node, StageRun, now
from slashcompute.coordinator.scheduler import JobRuntime

if TYPE_CHECKING:
    from slashcompute.coordinator.core import Coordinator

log = logging.getLogger(__name__)


class Recovery:
    def __init__(self, core: "Coordinator") -> None:
        self.core = core
        self._last_tick: Optional[float] = None
        self._hold_expiry_until = 0.0

    def _loop_stalled(self) -> bool:
        """True while recovering from a stall of the event loop (a slow write, a blocking
        call, the machine sleeping). Heartbeats that arrived meanwhile still sit unread in
        the sockets, so judging nodes by them would evict everyone at once."""
        cfg = self.core.cfg
        now = time.monotonic()
        gap = now - self._last_tick if self._last_tick is not None else 0.0
        self._last_tick = now
        if gap > max(cfg.heartbeat_interval_s, 5 * cfg.scheduler_tick_s):
            log.warning("coordinator loop stalled for %.1fs; holding node expiry", gap)
            self._hold_expiry_until = now + 2 * cfg.heartbeat_interval_s
        return now < self._hold_expiry_until

    async def tick(self) -> None:
        core = self.core
        if not self._loop_stalled():
            for node in core.registry.expired(core.cfg.heartbeat_timeout_s):
                if node.reliable and not node.draining:
                    # Most network drops are silent: treat it like a disconnect, so an agent that
                    # comes back within the grace window resumes instead of losing its stage.
                    log.warning("node %s missed heartbeats; closing its socket and holding its "
                                "place for %.0fs", node.node_id[:8], core.cfg.reconnect_grace_s)
                    node.connected, node.disconnected_at = False, time.monotonic()
                    if node.close is not None:
                        await node.close()
                    continue
                log.warning("node %s missed heartbeats", node.node_id[:8])
                await self.on_node_lost(node.node_id, "heartbeat timeout")
            for node in core.registry.away(core.cfg.reconnect_grace_s):
                await self.on_node_lost(node.node_id,
                                        f"did not reconnect within {core.cfg.reconnect_grace_s:.0f}s")
        for node in list(core.registry.nodes.values()):
            if (node.draining and node.assignment is not None and node.drain_deadline
                    and time.monotonic() > node.drain_deadline):
                log.warning("node %s drain overran grace period", node.node_id[:8])
                job = core.jobs.get(node.assignment.job_id)
                if job is not None:
                    await self.abort_epoch(job, "drain exceeded grace period", count=False)
        self._expire_releases()
        await self._abort_stalled()

    def _expire_releases(self) -> None:
        """Bound how long a node told to stop stays out of new work when its agent never
        confirms (an older agent, a lost message). One whose heartbeats still name the job
        stays held: sending it work now is how jobs used to queue behind a download."""
        limit = self.core.cfg.release_timeout_s
        for node in self.core.registry.nodes.values():
            rel = node.releasing
            if rel is None or time.monotonic() - rel.since < limit:
                continue
            if node.still_on(rel.job_id, rel.epoch):
                if not rel.warned:
                    rel.warned = True
                    log.warning("node %s is still busy with cancelled job %s epoch %d after "
                                "%.0fs; holding it", node.node_id[:8], rel.job_id, rel.epoch, limit)
                continue
            log.warning("node %s never confirmed it stopped job %s epoch %d; freeing it",
                        node.node_id[:8], rel.job_id, rel.epoch)
            node.releasing = None

    async def _abort_stalled(self) -> None:
        """Backstop for a hang nothing else notices (a wedged worker, a link that never
        errors): a running epoch that reports no step for ``stall_timeout_s`` restarts
        from its last checkpoint."""
        limit = self.core.cfg.stall_timeout_s
        for job in list(self.core.jobs.values()):
            cur = job.current
            if (job.row.status == "running" and cur is not None and not cur.closed
                    and time.monotonic() - cur.last_progress > limit):
                await self.abort_epoch(job, f"no progress for {limit:.0f}s")

    async def on_drain(self, node_id: str) -> None:
        core = self.core
        node = core.registry.get(node_id)
        if node is None or node.draining:
            return
        node.draining = True
        node.drain_deadline = time.monotonic() + core.cfg.grace_period_s
        log.info("node %s draining", node_id[:8])
        if node.assignment is None:
            return
        job = core.jobs.get(node.assignment.job_id)
        cur = job.current if job else None
        if cur is None or cur.closed or cur.epoch != node.assignment.epoch:
            return
        if not cur.drain_requested:
            cur.drain_requested = True
            await core.send(cur.node_for(0), Drain(job_id=job.id, epoch=cur.epoch))

    async def on_node_lost(self, node_id: str, reason: str) -> None:
        core = self.core
        node = core.registry.remove(node_id)
        if node is None:
            return
        if node.connected and node.close is not None:
            # Close its socket too: an agent evicted for missed heartbeats otherwise stays
            # connected, ignored, and never re-registers.
            await node.close()
        row = core.db.get(Node, node_id)
        if row is not None:
            row.online = False
            core.db.save(row)
        log.info("node %s left (%s)", node_id[:8], reason)
        await core.verification.on_node_lost(node_id)
        if node.assignment is not None:
            job = core.jobs.get(node.assignment.job_id)
            if job and job.current and not job.current.closed and job.current.epoch == node.assignment.epoch:
                # A clean drain that already finished doesn't count as a failure.
                await self.abort_epoch(job, f"node {node_id[:8]} lost: {reason}",
                                       count=not node.draining)

    async def on_stage_finished(self, node_id: str, msg: StageFinished) -> None:
        core = self.core
        node = core.registry.get(node_id)
        if node and node.assignment and node.assignment.job_id == msg.job_id \
                and node.assignment.epoch == msg.epoch:
            node.assignment = None
        self._close_stage_run(msg.job_id, msg.epoch, msg.stage_idx, msg.reason)

        job = core.jobs.get(msg.job_id)
        if job is None or job.current is None or job.current.epoch != msg.epoch or job.current.closed:
            return
        cur = job.current
        cur.finished[msg.stage_idx] = msg.reason
        log.info("job %s epoch %d stage %d finished: %s (step %d)%s", job.id, msg.epoch,
                 msg.stage_idx, msg.reason, msg.last_step, f" {msg.detail}" if msg.detail else "")

        if msg.reason in ("error", "cancelled"):
            # A contributor stopping their Mac is not the job's fault (as with on_node_lost).
            count = not (msg.reason == "cancelled" and node is not None and node.draining)
            await self.abort_epoch(job, f"stage {msg.stage_idx} {msg.reason}: {msg.detail or ''}",
                                   count=count, fatal=msg.fatal)
            return
        if len(cur.finished) < len(cur.plans):
            return
        reasons = set(cur.finished.values())
        cur.closed = True
        if reasons == {"done"}:
            await core.complete_job(job)
        else:  # drained
            job.row.status = "recovering"
            core.db.save(job.row)
            log.info("job %s drained at step %d; will resume on remaining nodes",
                     job.id, job.row.last_checkpoint_step)

    async def abort_epoch(self, job: JobRuntime, reason: str, count: bool = True,
                          fatal: bool = False) -> None:
        core = self.core
        cur = job.current
        if cur is None or cur.closed:
            return
        cur.closed = True
        log.warning("job %s epoch %d aborted: %s", job.id, cur.epoch, reason)
        for p in cur.plans:
            self._close_stage_run(job.id, cur.epoch, p.stage_idx, "aborted")  # lost nodes' too
            node = core.registry.get(p.node_id)
            if node is None:
                continue
            mine = (node.assignment is not None and node.assignment.job_id == job.id
                    and node.assignment.epoch == cur.epoch)
            if p.stage_idx in cur.finished:
                if mine:
                    node.assignment = None
                continue
            if mine:
                core.registry.release(node, job.id, cur.epoch)   # free once its agent stops
            await core.send(p.node_id, CancelStage(job_id=job.id, epoch=cur.epoch))
        row = job.row
        if fatal:
            await core.fail_job(job, reason)
            return
        if count:
            row.recoveries += 1
        if row.recoveries > core.cfg.max_recoveries:
            await core.fail_job(job, f"too many recoveries; last: {reason}")
            return
        row.status = "recovering"
        row.error = reason
        core.db.save(row)

    def _close_stage_run(self, job_id: str, epoch: int, stage_idx: int, reason: str) -> None:
        with self.core.db.session() as s:
            run = s.exec(select(StageRun).where(
                StageRun.job_id == job_id, StageRun.epoch == epoch, StageRun.stage_idx == stage_idx,
                StageRun.ended_at == None,  # noqa: E711
            )).first()
            if run is not None:
                run.ended_at, run.end_reason = now(), reason
                s.add(run)
                s.commit()
