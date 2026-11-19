"""Raw usage ledger. Stores per-step measurements; the credit book posts from them."""

from __future__ import annotations

from sqlalchemy import func
from sqlmodel import select

from slashcompute.common.protocol import StepMetrics, UsageSample
from slashcompute.coordinator.db import Database, UsageRecord


class Ledger:
    def __init__(self, db: Database) -> None:
        self.db = db

    def record_step(self, node_id: str, msg: StepMetrics, billed_flops: float = 0.0) -> UsageRecord:
        u = msg.usage
        row = UsageRecord(
            kind="train", job_id=msg.job_id, epoch=msg.epoch, stage_idx=msg.stage_idx,
            node_id=node_id, step=msg.step, flops=u.flops, billed_flops=billed_flops,
            tokens=u.tokens,
            peak_mem_bytes=u.peak_mem_bytes, resident_mem_bytes=u.resident_mem_bytes,
            mem_byte_seconds=u.mem_byte_seconds, wall_s=u.wall_s, busy_s=u.busy_s,
            loss=msg.loss, in_digest=msg.in_digest, out_digest=msg.out_digest,
        )
        self.db.add(row)
        return row

    def record_infer(self, node_id: str, job_id: str, flops: float, tokens: int, wall_s: float) -> None:
        """One LLM request's share on one inference node (time-weighted FLOPs)."""
        self.db.add(UsageRecord(kind="infer", job_id=job_id, node_id=node_id, flops=flops, tokens=tokens,
                                wall_s=wall_s, busy_s=wall_s))

    def record_verify(self, node_id: str, job_id: str | None, usage: UsageSample) -> None:
        self.db.add(UsageRecord(
            kind="verify", job_id=job_id, node_id=node_id, flops=usage.flops,
            tokens=usage.tokens, peak_mem_bytes=usage.peak_mem_bytes,
            resident_mem_bytes=usage.resident_mem_bytes,
            mem_byte_seconds=usage.mem_byte_seconds, wall_s=usage.wall_s, busy_s=usage.busy_s,
        ))

    def find_step(self, job_id: str, epoch: int, stage_idx: int, step: int) -> UsageRecord | None:
        with self.db.session() as s:
            return s.exec(select(UsageRecord).where(
                UsageRecord.kind == "train", UsageRecord.job_id == job_id,
                UsageRecord.epoch == epoch, UsageRecord.stage_idx == stage_idx,
                UsageRecord.step == step,
            )).first()

    def dispute(self, job_id: str, epoch: int, stage_idx: int, step: int) -> None:
        with self.db.session() as s:
            rows = s.exec(select(UsageRecord).where(
                UsageRecord.job_id == job_id, UsageRecord.epoch == epoch,
                UsageRecord.stage_idx == stage_idx, UsageRecord.step == step,
            )).all()
            for r in rows:
                r.disputed = True
                s.add(r)
            s.commit()

    def summary(self) -> list[dict]:
        with self.db.session() as s:
            q = select(
                UsageRecord.node_id, UsageRecord.kind,
                func.count(UsageRecord.id), func.sum(UsageRecord.flops),
                func.sum(UsageRecord.mem_byte_seconds), func.sum(UsageRecord.busy_s),
                func.sum(UsageRecord.wall_s), func.max(UsageRecord.peak_mem_bytes),
                func.sum(func.iif(UsageRecord.disputed, UsageRecord.flops, 0.0)),
            ).group_by(UsageRecord.node_id, UsageRecord.kind)
            out = []
            for node_id, kind, n, flops, mbs, busy, wall, peak, disputed in s.exec(q).all():
                out.append({
                    "node_id": node_id, "kind": kind, "records": n, "flops": flops or 0.0,
                    "mem_byte_seconds": mbs or 0.0, "busy_s": busy or 0.0, "wall_s": wall or 0.0,
                    "peak_mem_bytes": peak or 0, "disputed_flops": disputed or 0.0,
                })
            return out

    def job_records(self, job_id: str) -> list[UsageRecord]:
        with self.db.session() as s:
            return list(s.exec(select(UsageRecord).where(UsageRecord.job_id == job_id)
                               .order_by(UsageRecord.id)).all())
