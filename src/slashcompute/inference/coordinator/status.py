"""JSON snapshot of the inference network (polled by the /compute shell's INF tab)."""

from __future__ import annotations

import json
import math
import time
from typing import Optional

from slashcompute.inference.config import GIB, InferenceSettings
from slashcompute.inference.coordinator import nodes, planner, registry


def _named(text: Optional[str], names: dict[str, str]) -> Optional[str]:
    for node_id, name in names.items():
        text = text.replace(node_id, name) if text else text
    return text


def _min_memory_gb(row, s: InferenceSettings) -> Optional[int]:
    """Whole GiB a Mac must lend to run the model alone at the default context (the LLMs tab
    says so before a chat fails for lack of memory)."""
    if row is None or row["status"] != "ready" or not row["layout_json"]:
        return None
    try:
        need = planner.single_node_bytes(registry.model_layout(row), s)
    except (ValueError, KeyError, TypeError):
        return None
    return math.ceil(planner.commit_for(need, s) / GIB)


def snapshot(conn, s: InferenceSettings) -> dict:
    now = time.time()
    node_rows = conn.execute("SELECT * FROM nodes ORDER BY created_at").fetchall()
    names = {r["id"]: r["name"] for r in node_rows}

    pipes = conn.execute(
        "SELECT p.*, (SELECT COUNT(*) FROM json_each(m.layout_json, '$.layers')) AS n_layers FROM pipelines p "
        "JOIN models m ON m.id = p.model_id "
        "WHERE p.state NOT IN ('stopped', 'broken') OR p.stopped_at > ? ORDER BY p.created_at DESC LIMIT 12",
        (now - 120,)).fetchall()
    pipelines = []
    for p in pipes:
        members = conn.execute("SELECT * FROM pipeline_members WHERE pipeline_id=? ORDER BY position",
                               (p["id"],)).fetchall()
        pipelines.append({
            "id": p["id"], "model": p["model_id"], "state": p["state"], "ctx": p["ctx"],
            "n_layers": p["n_layers"], "est_tok_s": p["est_tok_s"], "live_tok_s": p["live_tok_s"],
            "explanation": _named(p["explanation"], names), "broken_reason": _named(p["broken_reason"], names),
            "tensor_split": json.loads(p["tensor_split"] or "[]"), "load_seconds": p["load_seconds"],
            "members": [{"node": names.get(m["node_id"], m["node_id"]), "node_id": m["node_id"], "role": m["role"],
                         "layer_start": m["layer_start"], "layer_end": m["layer_end"], "share": float(m["share"]),
                         "endpoint": m["endpoint"],
                         "memory_gb": round((m["full_bytes"] + m["kv_bytes"]) / 1e9, 1)} for m in members],
        })

    reserved = nodes.reserved_bytes(conn)
    node_list = [{
        "id": r["id"], "name": r["name"], "chip": r["chip"],
        "online": nodes.is_online(r, s, now), "available": bool(r["available"]), "draining": bool(r["draining"]),
        "committed_gb": round((r["committed_bytes"] or 0) / GIB, 1),
        "in_use_gb": round(reserved.get(r["id"], 0) / 1e9, 1),
        "gen_score": r["gen_score"], "prompt_score": r["prompt_score"], "can_head": bool(r["can_head"]),
        "models_on_disk": json.loads(r["gguf_files_json"] or "[]"),
        "downloads": json.loads(r["downloads_json"] or "{}"),
        "reliability": r["reliability"], "build": r["llama_build"],
    } for r in node_rows]

    lat = nodes.latency_matrix(conn)
    ids = [r["id"] for r in node_rows]
    matrix = [[None if a == b else lat.get((a, b), lat.get((b, a))) for b in ids] for a in ids]

    jobs = [{
        "id": j["id"], "model": j["model_id"], "pipeline": j["pipeline_id"], "state": j["state"],
        "attempt": j["attempt"], "prompt_n": j["prompt_n"], "predicted_n": j["predicted_n"], "flops": j["flops"],
        "gen_weight": j["gen_weight"], "tok_s": j["tok_s"], "error": _named(j["error"], names),
    } for j in conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 30").fetchall()]

    uploads = {r["name"]: r for r in conn.execute("SELECT * FROM model_files").fetchall()}
    model_ids = sorted({r["id"] for r in conn.execute("SELECT id FROM models")} | set(uploads))
    models = []
    for mid in model_ids:
        row = registry.model_row(conn, mid)
        heads = [r["name"] for r in registry.holders(conn, s, mid, now)]
        downloading = {n["name"]: n["downloads"][mid] for n in node_list if mid in n["downloads"]}
        up = uploads.get(mid)
        models.append({
            "id": mid, "status": row["status"] if row else "pending",
            "status_reason": row["status_reason"] if row else None,
            "size_gb": round(((row["size_bytes"] if row else None) or (up["size"] if up else 0)) / 1e9, 2),
            "arch": row["arch"] if row else None, "uploaded": up is not None,
            "heads": heads, "downloading": downloading, "servable": bool(row and row["status"] == "ready" and heads),
            "min_memory_gb": _min_memory_gb(row, s),
        })

    return {
        "now": now, "transport": s.TRANSPORT, "pinned_build": s.PINNED_LLAMA_BUILD,
        "uploads_enabled": bool(s.MODELS_DIR),
        "pipelines": pipelines, "nodes": node_list,
        "latency": {"nodes": [names[i] for i in ids], "matrix": matrix},
        "jobs": jobs, "models": models,
    }
