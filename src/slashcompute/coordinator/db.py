"""Persistent coordinator state (SQLite via SQLModel)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from sqlalchemy import event
from sqlmodel import Field, Session, SQLModel, create_engine


def now() -> float:
    return time.time()


class Node(SQLModel, table=True):
    __tablename__ = "nodes"
    id: str = Field(primary_key=True)
    name: str
    chip: str
    memory_contrib_bytes: int
    matmul_tflops: float
    mem_bandwidth_gbps: float
    first_seen: float = Field(default_factory=now)
    last_seen: float = Field(default_factory=now)
    online: bool = True
    canary_passed: Optional[bool] = None


class Job(SQLModel, table=True):
    __tablename__ = "jobs"
    id: str = Field(primary_key=True)
    kind: str
    spec_json: str
    status: str = "queued"  # queued|starting|running|recovering|completed|failed|cancelled
    submitted_at: float = Field(default_factory=now)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    epoch: int = 0
    recoveries: int = 0
    last_checkpoint_step: int = 0
    progress_step: int = 0
    last_loss: Optional[float] = None
    error: Optional[str] = None


class StageRun(SQLModel, table=True):
    __tablename__ = "stage_runs"
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: str = Field(index=True)
    epoch: int
    stage_idx: int
    node_id: str = Field(index=True)
    layer_start: int
    layer_end: int
    started_at: float = Field(default_factory=now)
    ended_at: Optional[float] = None
    end_reason: Optional[str] = None


class UsageRecord(SQLModel, table=True):
    __tablename__ = "usage_records"
    id: Optional[int] = Field(default=None, primary_key=True)
    kind: str = "train"  # train | verify
    job_id: Optional[str] = Field(default=None, index=True)
    epoch: Optional[int] = None
    stage_idx: Optional[int] = None
    node_id: str = Field(index=True)
    step: Optional[int] = None
    flops: float
    tokens: int = 0
    peak_mem_bytes: int = 0
    resident_mem_bytes: int = 0
    mem_byte_seconds: float = 0.0
    wall_s: float = 0.0
    busy_s: float = 0.0
    loss: Optional[float] = None
    in_digest: Optional[str] = None
    out_digest: Optional[str] = None
    disputed: bool = False
    created_at: float = Field(default_factory=now)


class Verification(SQLModel, table=True):
    __tablename__ = "verifications"
    id: str = Field(primary_key=True)
    kind: str  # replay | canary | chain
    job_id: Optional[str] = Field(default=None, index=True)
    epoch: Optional[int] = None
    step: Optional[int] = None
    stage_idx: Optional[int] = None
    target_node_id: str
    verifier_node_id: Optional[str] = None
    # fetching -> queued -> running -> passed|failed ; or error
    status: str = "fetching"
    rel_error: Optional[float] = None
    detail: Optional[str] = None
    created_at: float = Field(default_factory=now)
    finished_at: Optional[float] = None


class Checkpoint(SQLModel, table=True):
    __tablename__ = "checkpoints"
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: str = Field(index=True)
    step: int
    path: str
    created_at: float = Field(default_factory=now)


class User(SQLModel, table=True):
    __tablename__ = "users"
    id: str = Field(primary_key=True)
    email: str = Field(index=True, unique=True)
    password_hash: str
    name: str
    admin: bool = False
    banned: bool = False
    flagged: bool = False
    grant_split: int = 0
    google_sub: Optional[str] = Field(default=None, index=True)
    bio: Optional[str] = None
    accepted_terms_at: Optional[float] = None
    created_at: float = Field(default_factory=now)


class SessionRow(SQLModel, table=True):
    __tablename__ = "sessions"
    token_hash: str = Field(primary_key=True)
    user_id: str = Field(index=True)
    created_at: float = Field(default_factory=now)
    expires_at: float


class CreditTxn(SQLModel, table=True):
    __tablename__ = "credit_txns"
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(index=True)
    kind: str
    amount: float
    job_id: Optional[str] = Field(default=None, index=True)
    grant_id: Optional[str] = None
    node_id: Optional[str] = None
    note: Optional[str] = None
    created_at: float = Field(default_factory=now)


class NodeOwner(SQLModel, table=True):
    __tablename__ = "node_owners"
    node_id: str = Field(primary_key=True)
    user_id: str = Field(index=True)


class JobAccount(SQLModel, table=True):
    __tablename__ = "job_accounts"
    job_id: str = Field(primary_key=True)
    user_id: str = Field(index=True)
    reserved_flops: float = 0.0
    spent_flops: float = 0.0


class Grant(SQLModel, table=True):
    __tablename__ = "grants"
    id: str = Field(primary_key=True)
    author_id: str = Field(index=True)
    title: str
    body: str
    goal_flops: float
    received_flops: float = 0.0
    status: str = "pending"
    created_at: float = Field(default_factory=now)
    reviewed_at: Optional[float] = None
    reviewed_by: Optional[str] = None
    review_note: Optional[str] = None


class GrantComment(SQLModel, table=True):
    __tablename__ = "grant_comments"
    id: Optional[int] = Field(default=None, primary_key=True)
    grant_id: str = Field(index=True)
    user_id: str
    body: str
    created_at: float = Field(default_factory=now)


class UserFlag(SQLModel, table=True):
    __tablename__ = "user_flags"
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(index=True)
    admin_id: str
    reason: str
    created_at: float = Field(default_factory=now)


def _sqlite_pragmas(dbapi_conn, _record) -> None:
    # WAL: readers (HTTP handlers on the threadpool) no longer block the event loop's writes
    # for up to the busy timeout, which stalled every heartbeat and socket behind them.
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA busy_timeout=5000")
    cur.close()


class Database:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
        event.listen(self.engine, "connect", _sqlite_pragmas)
        SQLModel.metadata.create_all(self.engine)
        self._migrate()

    def _migrate(self) -> None:
        with self.engine.connect() as conn:
            user_cols = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(users)").fetchall()}
            if "google_sub" not in user_cols:
                conn.exec_driver_sql("ALTER TABLE users ADD COLUMN google_sub VARCHAR")
            if "bio" not in user_cols:
                conn.exec_driver_sql("ALTER TABLE users ADD COLUMN bio VARCHAR")
            grant_cols = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(grants)").fetchall()}
            if grant_cols:
                if "reviewed_at" not in grant_cols:
                    conn.exec_driver_sql("ALTER TABLE grants ADD COLUMN reviewed_at FLOAT")
                if "reviewed_by" not in grant_cols:
                    conn.exec_driver_sql("ALTER TABLE grants ADD COLUMN reviewed_by VARCHAR")
                if "review_note" not in grant_cols:
                    conn.exec_driver_sql("ALTER TABLE grants ADD COLUMN review_note VARCHAR")
            # At most one welcome credit per user, even under concurrent sign-ins.
            conn.exec_driver_sql(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_credit_txns_welcome "
                "ON credit_txns (user_id) WHERE kind = 'welcome'"
            )
            # Each grant card counts its donors: without this, one scan of every credit row per card.
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_credit_txns_grant "
                "ON credit_txns (grant_id) WHERE grant_id IS NOT NULL"
            )
            conn.commit()

    def session(self) -> Session:
        return Session(self.engine, expire_on_commit=False)

    def add(self, *rows) -> None:
        with self.session() as s:
            for r in rows:
                s.add(r)
            s.commit()

    def get(self, model, key):
        with self.session() as s:
            return s.get(model, key)

    def save(self, row) -> None:
        with self.session() as s:
            s.merge(row)
            s.commit()

    def save_all(self, *rows) -> None:
        """Insert or update every row in one transaction: all are written or none is."""
        with self.session() as s:
            for row in rows:
                s.merge(row)
            s.commit()
