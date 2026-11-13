"""How inference reports credits. The main coordinator plugs in its FLOP credits; tests use :class:`NullAccounting`."""

from __future__ import annotations

from typing import Optional, Protocol

from fastapi import Request


# Every chat reserves under its own account id with this prefix, so the holds a crash left
# behind can be told apart from training jobs at the next start.
ACCOUNT_PREFIX = "inf-"


class AccountingError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class Accounting(Protocol):
    def requester(self, request: Request) -> Optional[str]:
        """The signed-in user making a chat request (None = anonymous LAN use, free)."""
        ...

    def admit_node(self, session_token: Optional[str]) -> None:
        """May a node join? Public pools take only signed-in, consenting, unbanned accounts (raises AccountingError)."""
        ...

    def require_admin(self, request: Request) -> None:
        """Uploading/deleting models and stopping pipelines: admins only on public pools (raises AccountingError)."""
        ...

    def bind_node(self, node_id: str, session_token: Optional[str]) -> None:
        """An inference node registered with its owner's session: its earnings go to that user."""
        ...

    def reserve(self, user_id: str, account_id: str, flops: float) -> None:
        """Hold ``flops`` of the user's credits for one chat request (raises AccountingError 402)."""
        ...

    def record(self, account_id: str, per_node: dict[str, float], tokens: int, wall_s: float) -> None:
        """A request finished: credit each hosting node, log usage, consume from the reservation."""
        ...

    def settle(self, account_id: str) -> None:
        """Refund whatever the request did not use."""
        ...

    def recover(self) -> None:
        """At startup: refund the unused part of every reservation a previous run never settled
        (it was killed mid-reply, and only the request's own handler would have called settle)."""
        ...


class NullAccounting:
    """No credits: records totals in memory (standalone app and tests)."""

    def __init__(self) -> None:
        self.earned: dict[str, float] = {}
        self.records: list[dict] = []

    def requester(self, request: Request) -> Optional[str]:
        return None

    def admit_node(self, session_token: Optional[str]) -> None:
        return None

    def require_admin(self, request: Request) -> None:
        return None

    def bind_node(self, node_id: str, session_token: Optional[str]) -> None:
        return None

    def reserve(self, user_id: str, account_id: str, flops: float) -> None:
        return None

    def record(self, account_id: str, per_node: dict[str, float], tokens: int, wall_s: float) -> None:
        for node_id, flops in per_node.items():
            self.earned[node_id] = self.earned.get(node_id, 0.0) + flops
        self.records.append({"account_id": account_id, "per_node": dict(per_node), "tokens": tokens,
                             "wall_s": wall_s})

    def settle(self, account_id: str) -> None:
        return None

    def recover(self) -> None:
        return None
