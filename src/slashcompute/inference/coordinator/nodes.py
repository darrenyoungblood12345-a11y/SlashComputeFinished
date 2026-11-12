"""Node rows -> planner candidates, plus the latency matrix."""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Optional

from slashcompute.inference.config import InferenceSettings
from slashcompute.inference.coordinator.planner import NodeCandidate

LIVE_STATES = ("planned", "starting", "loading", "active", "draining")


def is_online(row, s: InferenceSettings, now: Optional[float] = None) -> bool:
    now = now or time.time()
    return bool(row["last_heartbeat"]) and now - row["last_heartbeat"] <= s.OFFLINE_AFTER_SECONDS


def reserved_bytes(conn: sqlite3.Connection, exclude_pipeline: Optional[str] = None) -> dict[str, int]:
    """Memory each node already gives to live pipelines (weights + KV)."""
    rows = conn.execute(
        f"SELECT m.node_id, SUM(m.full_bytes + m.kv_bytes) AS used FROM pipeline_members m "
        f"JOIN pipelines p ON p.id = m.pipeline_id WHERE p.state IN ({','.join('?' * len(LIVE_STATES))}) "
        f"AND p.id != ? GROUP BY m.node_id", (*LIVE_STATES, exclude_pipeline or "")).fetchall()
    return {r["node_id"]: int(r["used"] or 0) for r in rows}


def head_files(conn: sqlite3.Connection, row) -> set[str]:
    """The models this node may head: the files it reports, less every uploaded model whose current copy it
    has not confirmed it holds. A report carries names only, and a file of that name on its disk may be
    another model: an upload replaced since under the same name, or its own."""
    stale = conn.execute(
        "SELECT f.name FROM model_files f LEFT JOIN node_uploads u ON u.name = f.name AND u.node_id = ? "
        "WHERE u.sha256 IS NOT f.sha256", (row["id"],)).fetchall()
    return set(json.loads(row["gguf_files_json"] or "[]")) - {r["name"] for r in stale}


def candidate(conn: sqlite3.Connection, row, s: InferenceSettings, model_id: str, reserved: int,
              now: float) -> NodeCandidate:
    files = head_files(conn, row)
    allowed = json.loads(row["allowed_models_json"]) if row["allowed_models_json"] else None
    return NodeCandidate(
        node_id=row["id"], name=row["name"], committed_bytes=int(row["committed_bytes"] or 0),
        reserved_bytes=reserved, gen_score=row["gen_score"] or 0.0, prompt_score=row["prompt_score"] or 0.0,
        can_head=bool(row["can_head"]) and (bool(row["approved_head"]) or not s.REQUIRE_HEAD_APPROVAL),
        has_file=model_id in files, build=row["llama_build"] or "",
        online=is_online(row, s, now) and bool(row["available"]) and not row["draining"] and bool(row["gen_score"]),
        hours=tuple(tuple(h) for h in json.loads(row["hours_json"] or "[[0, 24]]")),
        allowed_models=tuple(allowed) if allowed else None,
        reliability=row["reliability"],
    )


def candidates(conn: sqlite3.Connection, s: InferenceSettings, model_id: str,
               now: Optional[float] = None) -> list[NodeCandidate]:
    now = now or time.time()
    reserved = reserved_bytes(conn)
    return [candidate(conn, r, s, model_id, reserved.get(r["id"], 0), now)
            for r in conn.execute("SELECT * FROM nodes ORDER BY id").fetchall()]


def latency_matrix(conn: sqlite3.Connection) -> dict[tuple[str, str], float]:
    return {(r["a"], r["b"]): r["one_way_ms"] for r in conn.execute("SELECT a, b, one_way_ms FROM latency")}


def node_names(conn: sqlite3.Connection) -> dict[str, str]:
    return {r["id"]: r["name"] for r in conn.execute("SELECT id, name FROM nodes")}
