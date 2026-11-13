"""Community grants: propose, moderate, fund, discuss."""

from __future__ import annotations

import math
import secrets
from typing import Optional

from sqlalchemy import func
from sqlmodel import select, update

from slashcompute.community.credits import Credits
from slashcompute.community.fields import text
from slashcompute.coordinator.db import CreditTxn, Database, Grant, GrantComment, User, UserFlag, now

# Far beyond any real ask; keeps goals finite so progress math and JSON stay sane.
MAX_GOAL_FLOPS = 1e30


class GrantError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class Grants:
    def __init__(self, db: Database, credits: Credits) -> None:
        self.db = db
        self.credits = credits

    def create(self, author: User, title: str, body: str, goal_flops: float) -> Grant:
        if author.banned:
            raise GrantError("Banned accounts cannot open grants.", 403)
        title = text(title, "title", GrantError).strip()[:120]
        body = text(body, "body", GrantError).strip()[:4000]
        if len(title) < 4 or len(body) < 20:
            raise GrantError("Describe the need: a title and at least a short paragraph.")
        try:
            goal = float(goal_flops)
        except (TypeError, ValueError) as e:
            raise GrantError("Goal must be a FLOP number.") from e
        if not math.isfinite(goal) or goal <= 0:
            raise GrantError("Goal must be greater than zero.")
        if goal > MAX_GOAL_FLOPS:
            raise GrantError("Goal is too large.")
        grant = Grant(
            id=secrets.token_hex(6), author_id=author.id, title=title, body=body,
            goal_flops=goal, status="pending",
        )
        self.db.add(grant)
        return grant

    def get(self, grant_id: str) -> Grant:
        g = self.db.get(Grant, grant_id)
        if g is None:
            raise GrantError("Grant not found.", 404)
        return g

    def visible(self, g: Grant, viewer: Optional[User]) -> bool:
        return g.status == "approved" or bool(viewer and (viewer.admin or viewer.id == g.author_id))

    def review(self, admin: User, grant_id: str, approve: bool,
               note: Optional[str] = None) -> Grant:
        if not admin.admin:
            raise GrantError("Admin only.", 403)
        note = text(note, "note", GrantError)
        g = self.get(grant_id)
        if g.status != "pending":
            raise GrantError("This grant was already reviewed.")
        g.status = "approved" if approve else "declined"
        g.reviewed_at = now()
        g.reviewed_by = admin.id
        g.review_note = note.strip()[:500] or None
        self.db.save(g)
        return g

    def donate(self, donor: User, grant_id: str, flops: float, *, from_pot: bool = False) -> Grant:
        if donor.banned:
            raise GrantError("Banned accounts cannot donate.", 403)
        try:
            flops = float(flops)
        except (TypeError, ValueError) as e:
            raise GrantError("Donation must be a FLOP number.") from e
        if not math.isfinite(flops) or flops <= 0:
            raise GrantError("Donation must be greater than zero.")
        g = self.get(grant_id)
        if g.status != "approved":
            raise GrantError("Only approved grants can receive FLOPs.")
        if not from_pot and donor.id == g.author_id:
            raise GrantError("Donate to someone else's grant.")
        if from_pot:
            if not donor.admin:
                raise GrantError("Only an admin can allocate the community pot.", 403)
            self.credits.allocate_pot(g.author_id, g.id, flops)
        else:
            self.credits.donate(donor.id, g.author_id, g.id, flops)
        with self.db.session() as s:
            s.exec(update(Grant).where(Grant.id == g.id)
                   .values(received_flops=Grant.received_flops + float(flops)))
            s.commit()
        return self.get(g.id)

    def comment(self, user: User, grant_id: str, body: str) -> GrantComment:
        g = self.get(grant_id)
        if not self.visible(g, user):
            raise GrantError("Grant not found.", 404)
        if g.status == "declined":
            raise GrantError("This grant is not public.")
        body = text(body, "body", GrantError).strip()[:2000]
        if len(body) < 2:
            raise GrantError("Write a short comment.")
        row = GrantComment(grant_id=g.id, user_id=user.id, body=body)
        self.db.add(row)
        return row

    def comments(self, grant_id: str) -> list[GrantComment]:
        with self.db.session() as s:
            return list(s.exec(
                select(GrantComment).where(GrantComment.grant_id == grant_id)
                .order_by(GrantComment.created_at)
            ).all())

    def list(self, *, include_pending: bool = False, sort: str = "top",
             viewer: Optional[User] = None) -> list[Grant]:
        with self.db.session() as s:
            rows = list(s.exec(select(Grant)).all())
        visible = []
        for g in rows:
            if g.status == "approved":
                visible.append(g)
            elif include_pending or (viewer and (viewer.admin or viewer.id == g.author_id)):
                visible.append(g)
        rows = visible
        if sort == "least":
            rows.sort(key=lambda g: (g.received_flops / g.goal_flops) if g.goal_flops else 0)
        elif sort == "trending":
            rows.sort(key=lambda g: g.created_at, reverse=True)
        else:
            rows.sort(key=lambda g: g.received_flops, reverse=True)
        return rows

    def flag_user(self, admin: User, user: User, reason: str) -> UserFlag:
        if not admin.admin:
            raise GrantError("Admin only.", 403)
        reason = text(reason, "reason", GrantError).strip()[:500] or "flagged"
        user.flagged = True
        self.db.save(user)
        row = UserFlag(user_id=user.id, admin_id=admin.id, reason=reason)
        self.db.add(row)
        return row

    def list_flags(self) -> list[dict]:
        with self.db.session() as s:
            rows = list(s.exec(select(UserFlag).order_by(UserFlag.created_at.desc())).all())
        return [
            {
                "id": r.id, "user_id": r.user_id, "admin_id": r.admin_id,
                "reason": r.reason, "created_at": r.created_at,
            }
            for r in rows
        ]

    def backers(self, grant_id: str) -> int:
        """How many different people gave to this grant (an allocation from the pot is nobody's gift)."""
        with self.db.session() as s:
            return int(s.exec(
                select(func.count(func.distinct(CreditTxn.user_id))).where(
                    CreditTxn.grant_id == grant_id, CreditTxn.kind == "donate",
                )
            ).one() or 0)

    def view(self, g: Grant, names: Optional[dict[str, str]] = None) -> dict:
        names = names or {}
        return {
            "id": g.id, "author_id": g.author_id,
            "author": names.get(g.author_id, g.author_id[:8]),
            "title": g.title, "body": g.body, "goal_flops": g.goal_flops,
            "received_flops": g.received_flops, "backers": self.backers(g.id), "status": g.status,
            "created_at": g.created_at, "progress": (
                g.received_flops / g.goal_flops if g.goal_flops else 0.0
            ),
            "reviewed_at": g.reviewed_at, "reviewed_by": g.reviewed_by,
            "review_note": g.review_note,
        }
