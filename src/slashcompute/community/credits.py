"""1:1 FLOP credits. Earn what you compute; spend what you take."""

from __future__ import annotations

import math
import threading


from typing import Optional

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from slashcompute.common.config import WELCOME_FLOPS
from slashcompute.coordinator.db import (
    CreditTxn, Database, EarningHold, JobAccount, NodeOwner, User, now,
)

# Holds what nobody may earn (see _host_rows), so charged FLOPs never vanish.
POT_ID = "__pot__"
# "donate", "receive" and "allocate" are no longer posted. They stay because ledgers from before
# hold such rows, and dropping the kinds would change those accounts' balances.
BALANCE_KINDS = ("earn", "reserve", "release", "donate", "receive", "allocate",
                 "welcome")


class CreditError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class InsufficientCredits(CreditError):
    """The balance does not cover a reservation (training keeps its 400; chat answers 402)."""


class Credits:
    def __init__(self, db: Database) -> None:
        self.db = db
        # Serialises check-balance-then-debit so concurrent spends cannot overdraw.
        self._spend_lock = threading.Lock()
        self._welcomed: set[str] = set()   # users known to hold their welcome credit: skip the insert

    def post(self, user_id: str, kind: str, amount: float, *, job_id: Optional[str] = None,
             node_id: Optional[str] = None, note: Optional[str] = None) -> CreditTxn:
        row = CreditTxn(user_id=user_id, kind=kind, amount=float(amount), job_id=job_id,
                        node_id=node_id, note=note)
        self.db.add(row)
        return row

    def balance(self, user_id: str) -> float:
        with self.db.session() as s:
            total = s.exec(
                select(func.coalesce(func.sum(CreditTxn.amount), 0.0)).where(
                    CreditTxn.user_id == user_id, CreditTxn.kind.in_(BALANCE_KINDS),
                )
            ).one()
        return float(total or 0.0)

    def lifetime_earned(self, user_id: str) -> float:
        with self.db.session() as s:
            total = s.exec(
                select(func.coalesce(func.sum(CreditTxn.amount), 0.0)).where(
                    CreditTxn.user_id == user_id, CreditTxn.kind == "generated",
                )
            ).one()
        return float(total or 0.0)

    def lifetime_spent(self, user_id: str) -> float:
        with self.db.session() as s:
            total = s.exec(
                select(func.coalesce(func.sum(CreditTxn.amount), 0.0)).where(
                    CreditTxn.user_id == user_id, CreditTxn.kind == "consumed",
                )
            ).one()
        return float(total or 0.0)

    def reserved_in_flight(self, user_id: str) -> float:
        with self.db.session() as s:
            rows = s.exec(select(JobAccount).where(JobAccount.user_id == user_id)).all()
        return float(sum(max(0.0, a.reserved_flops - a.spent_flops) for a in rows))

    def summary(self, user_id: str) -> dict:
        return {
            "user_id": user_id,
            "balance": self.balance(user_id),
            "lifetime_earned": self.lifetime_earned(user_id),
            "lifetime_spent": self.lifetime_spent(user_id),
            "pot": self.balance(POT_ID),
            "reserved_in_flight": self.reserved_in_flight(user_id),
            # charged to jobs for this user's nodes, not yet paid: see hold_step
            "pending_earnings": self.pending_earnings(user_id),
        }

    def list_txns(self, user_id: str, *, limit: int = 50,
                  before_id: Optional[int] = None) -> dict:
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 50
        limit = max(1, min(200, limit))
        with self.db.session() as s:
            q = select(CreditTxn).where(CreditTxn.user_id == user_id)
            if before_id is not None:
                q = q.where(CreditTxn.id < int(before_id))
            q = q.order_by(CreditTxn.id.desc())
            rows = list(s.exec(q).all())[: limit + 1]
        items = rows[:limit]
        return {
            "items": [
                {
                    "id": r.id, "kind": r.kind, "amount": r.amount,
                    "job_id": r.job_id, "node_id": r.node_id,
                    "note": r.note, "created_at": r.created_at,
                }
                for r in items
            ],
            "next_cursor": items[-1].id if len(rows) > limit else None,
        }

    def live(self, user_id: str, window_s: float = 60.0) -> dict:
        try:
            window_s = float(window_s)
        except (TypeError, ValueError):
            window_s = 60.0
        window_s = max(1.0, min(3600.0, window_s))
        cutoff = now() - window_s
        with self.db.session() as s:
            recent = list(s.exec(
                select(CreditTxn).where(
                    CreditTxn.kind == "generated", CreditTxn.created_at >= cutoff,
                )
            ).all())
        you_flops = sum(r.amount for r in recent if r.user_id == user_id)
        community_flops = sum(r.amount for r in recent if r.user_id != POT_ID)
        snap = self.summary(user_id)
        return {
            **snap,
            "window_s": window_s,
            "community_flops": community_flops,
            "you": {
                "flops": you_flops,
                "balance": snap["balance"],
                "lifetime_earned": snap["lifetime_earned"],
            },
        }

    def nodes_for(self, user_id: str) -> list[NodeOwner]:
        with self.db.session() as s:
            return list(s.exec(select(NodeOwner).where(NodeOwner.user_id == user_id)).all())

    def bind_node(self, node_id: str, user_id: str, *, take_over: bool = False) -> None:
        # Node ids are public: never hand one user's node (and its earnings) to another unless
        # the caller has established the new account may take it over (see Coordinator._check_claim).
        owner = self.owner_of(node_id)
        if owner is not None and owner != user_id and not take_over:
            raise PermissionError("node id belongs to another account")
        self.db.save(NodeOwner(node_id=node_id, user_id=user_id))

    def owner_of(self, node_id: str) -> Optional[str]:
        row = self.db.get(NodeOwner, node_id)
        return row.user_id if row else None

    def _earning_rows(self, user_id: str, flops: float, *,
                      node_id: Optional[str], job_id: Optional[str]) -> list[CreditTxn]:
        """What ``user_id`` earns for ``flops`` of work: the record of the work ("generated",
        which counts for nothing in a balance) and the credit itself, all of it theirs."""
        return [
            CreditTxn(user_id=user_id, kind=kind, amount=flops, node_id=node_id, job_id=job_id)
            for kind in ("generated", "earn")
        ]

    def _host_rows(self, owner: Optional[User], flops: float, *, node_id: Optional[str],
                   job_id: Optional[str]) -> list[CreditTxn]:
        """The rows that pay ``flops`` for a node's work. A share nobody may earn (unbound node,
        missing or banned owner) goes to the community pot so charged FLOPs never vanish."""
        if owner is None or owner.banned:
            return [CreditTxn(user_id=POT_ID, kind="earn", amount=flops, node_id=node_id,
                              job_id=job_id, note="unclaimed host share")]
        return self._earning_rows(owner.id, flops, node_id=node_id, job_id=job_id)

    def contribute(self, user_id: str, flops: float, *,
                   node_id: Optional[str] = None, job_id: Optional[str] = None) -> None:
        if flops <= 0:
            return
        self.db.add(*self._earning_rows(user_id, flops, node_id=node_id, job_id=job_id))

    def credit_host(self, owner_id: Optional[str], flops: float, *,
                    node_id: Optional[str] = None, job_id: Optional[str] = None) -> None:
        """Pay out FLOPs a job was charged for a node's work, at once. Training does not use
        this: its steps are held until they could have been checked (see hold_step)."""
        if flops <= 0:
            return
        owner = self.db.get(User, owner_id) if owner_id else None
        self.db.add(*self._host_rows(owner, flops, node_id=node_id, job_id=job_id))

    # ------------------------------------------------------------ held training earnings

    def hold_step(self, job_id: str, epoch: int, node_id: str, owner_id: Optional[str],
                  flops: float) -> float:
        """Charge a job's reserve for one training step and keep the amount back for the node
        that ran it, instead of paying it. Returns the FLOPs charged.

        What a node reports proves nothing until a sampled step of it has been replayed, so
        nothing is spendable before release_hold. A node whose hold for this epoch is already
        settled (forfeited, or the epoch was closed and paid) charges nothing more."""
        if flops <= 0:
            return 0.0
        with self._spend_lock, self.db.session() as s:
            acct = s.get(JobAccount, job_id)
            if acct is None:
                return 0.0
            hold = s.exec(select(EarningHold).where(
                EarningHold.job_id == job_id, EarningHold.epoch == epoch,
                EarningHold.node_id == node_id)).first()
            if hold is not None and hold.status != "held":
                return 0.0
            take = min(flops, acct.reserved_flops - acct.spent_flops)
            if take <= 0:
                return 0.0
            if hold is None:
                hold = EarningHold(job_id=job_id, epoch=epoch, node_id=node_id, owner_id=owner_id)
            hold.flops += take
            acct.spent_flops += take
            s.add(acct)
            s.add(hold)
            s.add(CreditTxn(user_id=acct.user_id, kind="consumed", amount=take, job_id=job_id))
            s.commit()
        return take

    def open_holds(self) -> list[EarningHold]:
        with self.db.session() as s:
            return list(s.exec(select(EarningHold).where(EarningHold.status == "held")).all())

    def pending_earnings(self, user_id: str) -> float:
        with self.db.session() as s:
            total = s.exec(
                select(func.coalesce(func.sum(EarningHold.flops), 0.0)).where(
                    EarningHold.owner_id == user_id, EarningHold.status == "held",
                )
            ).one()
        return float(total or 0.0)

    def release_hold(self, hold_id: int) -> float:
        """Pay a hold to its owner: the banned-owner rule is that of now.
        Returns the FLOPs paid; a hold that is no longer held pays nothing."""
        with self._spend_lock, self.db.session() as s:
            hold = s.get(EarningHold, hold_id)
            if hold is None or hold.status != "held":
                return 0.0
            hold.status, hold.resolved_at = "released", now()
            s.add(hold)
            if hold.flops > 0:
                owner = s.get(User, hold.owner_id) if hold.owner_id else None
                for row in self._host_rows(owner, hold.flops, node_id=hold.node_id,
                                           job_id=hold.job_id):
                    s.add(row)
            s.commit()
            return hold.flops

    def forfeit_hold(self, job_id: str, epoch: int, node_id: str,
                     flops: Optional[float] = None) -> float:
        """Take back what a node was held for this epoch (all of it, or ``flops`` of it) and
        return it to the job's reserve. Returns the FLOPs taken back.

        Taking all of it closes the hold, so later steps of that node in the epoch charge nothing.
        The job's account is only un-spent here: a caller whose job has already ended settles it
        again to hand the amount to the requester."""
        with self._spend_lock, self.db.session() as s:
            hold = s.exec(select(EarningHold).where(
                EarningHold.job_id == job_id, EarningHold.epoch == epoch,
                EarningHold.node_id == node_id)).first()
            if hold is None:
                if flops is not None:
                    return 0.0
                # Nothing charged yet: still close the epoch to this node.
                hold = EarningHold(job_id=job_id, epoch=epoch, node_id=node_id)
            if hold.status != "held":
                return 0.0
            amount = hold.flops if flops is None else max(0.0, min(float(flops), hold.flops))
            hold.flops -= amount
            hold.forfeited_flops += amount
            if flops is None:
                hold.status, hold.resolved_at = "forfeited", now()
            s.add(hold)
            acct = s.get(JobAccount, job_id)
            if amount and acct is not None:
                acct.spent_flops = max(0.0, acct.spent_flops - amount)
                s.add(acct)
                s.add(CreditTxn(user_id=acct.user_id, kind="consumed", amount=-amount,
                                job_id=job_id, node_id=node_id, note="unverified work refunded"))
            s.commit()
            return amount

    def grant_welcome(self, user_id: str, flops: float = WELCOME_FLOPS) -> float:
        """One-time sign-in credit. Returns FLOPs granted (0 if already given)."""
        if not math.isfinite(flops) or flops <= 0 or user_id in self._welcomed:
            return 0.0
        user = self.db.get(User, user_id)
        if user is None or user.banned:
            return 0.0
        try:
            self.post(user_id, "welcome", flops, note="Welcome credit")
        except IntegrityError:  # uq_credit_txns_welcome: someone else got there first
            self._welcomed.add(user_id)
            return 0.0
        self._welcomed.add(user_id)
        return float(flops)

    def reserve_job(self, user_id: str, job_id: str, flops: float) -> JobAccount:
        if not math.isfinite(flops) or flops <= 0:
            raise CreditError("Set a FLOP budget greater than zero.")
        with self._spend_lock:
            have = self.balance(user_id)
            if have < flops:
                raise InsufficientCredits(
                    f"Need {flops:.3e} FLOPs; you have {have:.3e}. Contribute first.",
                )
            acct = JobAccount(job_id=job_id, user_id=user_id, reserved_flops=flops, spent_flops=0.0)
            self.db.add(CreditTxn(user_id=user_id, kind="reserve", amount=-flops, job_id=job_id), acct)
        return acct

    def job_account(self, job_id: str) -> Optional[JobAccount]:
        return self.db.get(JobAccount, job_id)

    def consume_job(self, job_id: str, flops: float) -> float:
        """Charge a job's reserve. Returns FLOPs actually charged."""
        if flops <= 0:
            return 0.0
        with self._spend_lock, self.db.session() as s:
            acct = s.get(JobAccount, job_id)
            if acct is None:
                return 0.0
            remaining = acct.reserved_flops - acct.spent_flops
            take = min(flops, remaining)
            if take:
                acct.spent_flops += take
                s.add(acct)
                s.add(CreditTxn(user_id=acct.user_id, kind="consumed", amount=take, job_id=job_id))
                s.commit()
        return take

    def job_exhausted(self, job_id: str) -> bool:
        acct = self.job_account(job_id)
        if acct is None:
            return False
        return acct.spent_flops >= acct.reserved_flops - 1e-9

    def settle_job(self, job_id: str) -> None:
        # Settling shrinks the reserve to what was spent, so a settled account
        # has nothing left to release and repeat settles are no-ops.
        with self._spend_lock, self.db.session() as s:
            acct = s.get(JobAccount, job_id)
            if acct is None:
                return
            leftover = max(0.0, acct.reserved_flops - acct.spent_flops)
            if leftover:
                acct.reserved_flops = acct.spent_flops
                s.add(acct)
                s.add(CreditTxn(user_id=acct.user_id, kind="release", amount=leftover, job_id=job_id))
                s.commit()

    def settle_open_jobs(self, prefix: str) -> int:
        """Settle every account whose id starts with ``prefix`` and still holds unspent credits.

        For startup, when nothing that reserved under that prefix can still be running.
        Returns how many accounts it settled; a second pass finds none.
        """
        with self.db.session() as s:
            open_ids = s.exec(select(JobAccount.job_id).where(
                JobAccount.job_id.startswith(prefix, autoescape=True),
                JobAccount.reserved_flops > JobAccount.spent_flops,
            )).all()
        for job_id in open_ids:
            self.settle_job(job_id)
        return len(open_ids)

    def leaderboard(self, limit: int = 20) -> list[dict]:
        with self.db.session() as s:
            rows = s.exec(
                select(CreditTxn.user_id, func.sum(CreditTxn.amount)).where(
                    CreditTxn.kind == "generated",
                ).group_by(CreditTxn.user_id)
            ).all()
            users = {u.id: u for u in s.exec(select(User)).all()}
        ranked = sorted(
            ((uid, float(total or 0.0)) for uid, total in rows
             if uid != POT_ID and not (uid in users and users[uid].banned)),
            key=lambda x: x[1], reverse=True,
        )[:limit]
        out = []
        for uid, earned in ranked:
            u = users.get(uid)
            out.append({
                "user_id": uid,
                "name": u.name if u else uid[:8],
                "lifetime_earned": earned,
                "balance": self.balance(uid),
            })
        return out
