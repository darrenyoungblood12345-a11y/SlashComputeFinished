"""Model registry: layer tables (and FLOP profiles) from GGUF headers that head nodes report.

A model becomes ``ready`` as soon as any node reports a GGUF whose header parses. There is no
price table: speed estimates come from bytes read per token, credits from parameter counts.
The LLMs tab can stop serving a model (``disabled``) or remove it (``removed``, a tombstone the row
stays as, so replies still draining can be credited): node reports never change either back.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from typing import Optional

from slashcompute.inference.config import InferenceSettings
from slashcompute.inference.coordinator import nodes
from slashcompute.inference.coordinator.db import tx
from slashcompute.inference.coordinator.layers import ModelLayout, build_layout
from slashcompute.inference.flops import ModelFlops
from slashcompute.inference.gguf import GGUFHeader, plain_gguf  # noqa: F401 - re-exported

log = logging.getLogger(__name__)
_SHARD = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$")
SETTLED = ("ready", "disabled", "removed")   # statuses a node's report never changes


def first_shard_name(filename: str) -> str:
    m = _SHARD.match(filename)
    return f"{m.group(1)}-00001-of-{m.group(3)}.gguf" if m else filename


def store_layout(conn: sqlite3.Connection, model_id: str, layout: ModelLayout, source: str,
                 flops: Optional[ModelFlops] = None, **fields) -> None:
    now = time.time()
    with tx(conn):
        conn.execute("INSERT OR IGNORE INTO models (id, created_at) VALUES (?, ?)", (model_id, now))
        conn.execute("DELETE FROM model_layers WHERE model_id=?", (model_id,))
        conn.executemany(
            "INSERT INTO model_layers (model_id, idx, full_bytes, active_bytes, kv_bytes_per_token, kv_window) "
            "VALUES (?,?,?,?,?,?)",
            [(model_id, l.index, l.full_bytes, l.active_bytes, l.kv_bytes_per_token, l.kv_window)
             for l in layout.layers])
        sets = {"layout_json": json.dumps(layout.to_json()), "layer_source": source, "status": "ready",
                "status_reason": None, "arch": layout.arch, "moe": int(layout.moe), "updated_at": now,
                "flops_json": json.dumps(flops.to_json()) if flops else None,
                "size_bytes": layout.file_bytes, **fields}
        conn.execute(f"UPDATE models SET {', '.join(f'{k}=?' for k in sets)} WHERE id=?",
                     (*sets.values(), model_id))


def reject(conn: sqlite3.Connection, model_id: str, reason: str) -> None:
    with tx(conn):
        conn.execute("INSERT OR IGNORE INTO models (id, created_at) VALUES (?, ?)", (model_id, time.time()))
        conn.execute("UPDATE models SET status='rejected', status_reason=?, updated_at=? WHERE id=? "
                     f"AND status NOT IN ({','.join('?' * len(SETTLED))})", (reason, time.time(), model_id, *SETTLED))


def model_row(conn, model_id: str):
    return conn.execute("SELECT * FROM models WHERE id=?", (model_id,)).fetchone()


def unavailable(row, model_id: str) -> Optional[str]:
    """Why a chat for this model is refused (None: it is ready)."""
    status = row["status"] if row is not None else None
    if status == "ready":
        return None
    if status == "disabled":
        return (f"{model_id} is not being served: it was stopped in the LLMs tab. "
                "Press Serve there to use it again.")
    if status == "removed":
        return f"{model_id} was removed from this pool."
    return f"model {model_id!r} is not available"


def model_layout(row) -> ModelLayout:
    return ModelLayout.from_json(json.loads(row["layout_json"]))


def model_flops(row) -> ModelFlops:
    if row["flops_json"]:
        return ModelFlops.from_json(json.loads(row["flops_json"]))
    return ModelFlops.from_layout(model_layout(row))


def ready_models(conn) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM models WHERE status='ready' ORDER BY id").fetchall()


def est_out_s(layout: ModelLayout, s: InferenceSettings) -> float:
    """Reference seconds per output token: bytes read per token / reference memory bandwidth."""
    return max(layout.total_active_bytes / s.REF_BANDWIDTH_BYTES_PER_S, 1e-3)


def holders(conn, s: InferenceSettings, model_id: str, now: Optional[float] = None) -> list[sqlite3.Row]:
    """Online nodes that could head this model right now (have the file, may be head, and take
    work: a Mac paused for training would refuse the chat that listing it invites)."""
    now = now or time.time()
    out = []
    for r in conn.execute("SELECT * FROM nodes WHERE can_head=1").fetchall():
        files = set(json.loads(r["gguf_files_json"] or "[]"))
        approved = bool(r["approved_head"]) or not s.REQUIRE_HEAD_APPROVAL
        if model_id in files and approved and r["available"] and nodes.is_online(r, s, now):
            out.append(r)
    return out


def servable_models(conn, s: InferenceSettings) -> list[sqlite3.Row]:
    """Ready models that some online head has on disk right now."""
    return [r for r in ready_models(conn) if holders(conn, s, r["id"])]


# ------------------------------------------------------------ from a head node

def accept_node_header(conn, s: InferenceSettings, node_id: str, filename: str, size_bytes: int,
                       headers: list[dict]) -> bool:
    """A node reported the parsed header(s) of a GGUF it has on disk. Use it as the layer table
    (and FLOP profile) if the model doesn't have one yet."""
    name = first_shard_name(filename)
    row = model_row(conn, name)
    settled = row is not None and row["status"] in SETTLED
    if settled and (row["status"] == "ready" or row["layout_json"]):
        return False
    try:
        parsed = [GGUFHeader.from_json(h) for h in headers]
        layout = build_layout(name, parsed, size_bytes, s.KV_BYTES_PER_ELEMENT)
        if layout.n_layers == 0:
            raise ValueError("no transformer layers found")
        flops = ModelFlops.from_headers(parsed)
    except Exception as e:  # noqa: BLE001 - a bad header rejects the model, never the node
        log.warning("model %s from %s rejected: %s", name, node_id, e)
        if not settled:
            reject(conn, name, f"unreadable GGUF header: {e}")
        return False
    # a stopped or removed model without a layer table keeps its status (Serve / Add back can use the table)
    keep = {"status": row["status"], "status_reason": row["status_reason"]} if settled else {}
    store_layout(conn, name, layout, f"node:{node_id}", flops, size_bytes=size_bytes, **keep)
    return True
