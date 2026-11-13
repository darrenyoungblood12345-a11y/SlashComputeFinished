"""Inference credits in the same FLOP book as training: hosts earn, signed-in chatters spend."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Request

from slashcompute.community.credits import CreditError, InsufficientCredits
from slashcompute.community.http import _token
from slashcompute.coordinator.core import Coordinator
from slashcompute.inference.accounting import ACCOUNT_PREFIX, AccountingError

log = logging.getLogger(__name__)


class CoreAccounting:
    """Mirrors training (``Coordinator._on_step``): every request is logged in the usage ledger;
    hosts earn only what a signed-in user's reservation actually paid, split by their FLOPs, and
    a share no host may earn (unbound node, banned owner) goes to the community pot."""

    def __init__(self, core: Coordinator) -> None:
        self.core = core

    def requester(self, request: Request) -> Optional[str]:
        user = self.core.auth.session_user(_token(request, request.headers.get("authorization")))
        if user is None:
            if self.core.cfg.public_pool:
                raise AccountingError("Sign in first.", 401)
            return None
        if user.banned:
            raise AccountingError("This account is banned.", 403)
        return user.id

    def admit_node(self, session_token: Optional[str]) -> None:
        # same gate as a training node joining a public pool (Coordinator.on_register)
        if not self.core.cfg.public_pool:
            return
        user = self.core.auth.session_user(session_token)
        if user is None:
            raise AccountingError("Sign in first.", 401)
        if user.banned:
            raise AccountingError("This account is banned.", 403)
        if user.accepted_terms_at is None:
            raise AccountingError("Accept the terms first.", 403)

    def require_admin(self, request: Request) -> None:
        if not self.core.cfg.public_pool:
            return
        user = self.core.auth.session_user(_token(request, request.headers.get("authorization")))
        if user is None:
            raise AccountingError("Sign in first.", 401)
        if user.banned:
            raise AccountingError("This account is banned.", 403)
        if not user.admin:
            raise AccountingError("Only an admin can manage models and pipelines.", 403)

    def bind_node(self, node_id: str, session_token: Optional[str]) -> None:
        user = self.core.auth.session_user(session_token)
        if user is None:
            log.warning("inference node %s presented a bad session token", node_id)
        elif not user.banned and user.accepted_terms_at is not None:
            # The node's own token proved this is the same machine, so whoever is signed in on it
            # now takes over its future earnings (an account switch). A live training node with
            # this id still answers only to its owner.
            live = self.core.registry.get(node_id)
            if live is not None and live.user_id not in (None, user.id):
                log.warning("inference node %s id belongs to another account's live node", node_id)
                return
            self.core.credits.bind_node(node_id, user.id, take_over=True)

    def reserve(self, user_id: str, account_id: str, flops: float) -> None:
        user = self.core.auth.get(user_id)
        if user is None or user.accepted_terms_at is None:
            raise AccountingError("Accept the terms before taking from the pool.", 403)
        try:
            self.core.credits.reserve_job(user_id, account_id, flops)
        except InsufficientCredits as e:
            raise AccountingError(str(e), 402) from e   # Payment Required, as the Accounting protocol promises
        except CreditError as e:
            raise AccountingError(str(e), e.status) from e

    def record(self, account_id: str, per_node: dict[str, float], tokens: int, wall_s: float) -> None:
        credits = self.core.credits
        for node_id, flops in per_node.items():
            self.core.ledger.record_infer(node_id, account_id, flops, tokens, wall_s)
        total = sum(per_node.values())
        if credits.job_account(account_id) is None or total <= 0:
            return
        take = credits.consume_job(account_id, total)   # never more than the reservation holds
        for node_id, flops in per_node.items():
            credits.credit_host(credits.owner_of(node_id), take * flops / total, node_id=node_id,
                                job_id=account_id)

    def settle(self, account_id: str) -> None:
        self.core.credits.settle_job(account_id)

    def recover(self) -> None:
        # No request of this run has reserved yet, so every chat account still open is an orphan.
        n = self.core.credits.settle_open_jobs(ACCOUNT_PREFIX)
        if n:
            log.info("refunded %d chat reservation(s) a previous run left unsettled", n)
