"""Dashboard view model: pure functions over the coordinator's raw rows.

Credits are 1:1 with FLOPs (see ``community.credits``), so everything here is
counted in FLOPs. Until accounts are back, "this Mac" is the local agent's node.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

JOB_WAITING = ("queued", "recovering")
JOB_ACTIVE = ("starting", "running")
JOB_TERMINAL = ("completed", "failed", "cancelled")
# The coordinator's /jobs has no limit parameter, so the cap is applied here: the newest rows are
# listed, every row is counted.
JOBS_SHOWN = 50
POOL_LISTS = ("/nodes", "/jobs", "/ledger")


@dataclass(frozen=True)
class PoolData:
    """Latest coordinator view: raw ``/nodes``, ``/jobs`` and ``/ledger`` rows."""

    online: bool = False
    nodes: list = field(default_factory=list)
    jobs: list = field(default_factory=list)
    ledger: list = field(default_factory=list)


def fresh_pool(previous: PoolData, nodes: Optional[list], jobs: Optional[list],
               ledger: Optional[list]) -> PoolData:
    """An online pool's latest rows. A list whose fetch failed (``None``) keeps its last-good rows,
    so a slow coordinator shows a stale leaderboard rather than one that flickers to empty."""
    return PoolData(
        online=True,
        nodes=previous.nodes if nodes is None else nodes,
        jobs=previous.jobs if jobs is None else jobs,
        ledger=previous.ledger if ledger is None else ledger,
    )


def recent_jobs(jobs: list, limit: int = JOBS_SHOWN) -> list:
    """The newest ``limit`` jobs, newest first."""
    return sorted(jobs, key=lambda j: j.get("submitted_at") or 0.0, reverse=True)[:limit]


def format_flops(flops: float) -> str:
    """``1.24e12`` -> ``"1.24 T"``."""
    v, unit = float(flops or 0.0), ""
    for u in ("K", "M", "G", "T", "P", "E"):
        if abs(v) < 1000:
            break
        v, unit = v / 1000, u
    if v == 0:
        text = "0"
    elif abs(v) >= 100:
        text = f"{v:.0f}"
    elif abs(v) >= 10:
        text = f"{v:.1f}"
    else:
        text = f"{v:.2f}"
    return f"{text} {unit}".rstrip()


def with_unit(flops: float, unit: str = "FLOPs") -> str:
    """``1.24e12`` -> ``"1.24 TFLOPs"``, ``512`` -> ``"512 FLOPs"``."""
    s = format_flops(flops)
    return f"{s}{unit}" if s[-1].isalpha() else f"{s} {unit}"


def node_flops(ledger: list, node_id: Optional[str]) -> float:
    """FLOPs a node has been credited with: all metered work minus disputed work."""
    if not node_id:
        return 0.0
    return sum(
        float(r.get("flops") or 0.0) - float(r.get("disputed_flops") or 0.0)
        for r in ledger if r.get("node_id") == node_id
    )


def split_credits(flops: float, grant_percent: int) -> dict:
    earned = max(0.0, float(flops))
    to_grants = earned * max(0, min(100, int(grant_percent))) / 100.0
    return {"earned": earned, "kept": earned - to_grants, "to_grants": to_grants}


def _node_name(nodes: list, node_id: str) -> str:
    n = next((n for n in nodes if n.get("node_id") == node_id), None)
    return str((n or {}).get("name") or node_id[:8])


def leaderboard(ledger: list, nodes: list, my_id: Optional[str]) -> list[dict]:
    """Every Mac that has contributed or is online, ranked by credited FLOPs."""
    ids = {r.get("node_id") for r in ledger} | {n.get("node_id") for n in nodes}
    ids.discard(None)
    scored = sorted(
        ((node_flops(ledger, nid), _node_name(nodes, nid), nid) for nid in ids),
        key=lambda t: (-t[0], t[1].lower()),
    )
    return [
        {"rank": i + 1, "node_id": nid, "name": name, "flops": flops, "is_me": nid == my_id}
        for i, (flops, name, nid) in enumerate(scored)
    ]


def capacity(nodes: list, jobs: list) -> dict:
    return {
        "macs": len(nodes),
        "tflops": sum(float(n.get("matmul_tflops") or 0.0) for n in nodes),
        "memory_bytes": sum(int(n.get("memory_contrib_bytes") or 0) for n in nodes),
        "running": sum(1 for j in jobs if j.get("status") in JOB_ACTIVE),
        "waiting": sum(1 for j in jobs if j.get("status") in JOB_WAITING),
    }


def job_progress(job: dict) -> float:
    if job.get("status") == "completed":
        return 1.0
    steps = int(job.get("steps") or 0)
    if steps <= 0:
        return 0.0
    return max(0.0, min(1.0, int(job.get("progress_step") or 0) / steps))


def job_card(job: dict) -> dict:
    return {**job, "progress": job_progress(job),
            "can_cancel": job.get("status") not in JOB_TERMINAL}


def overview(status: dict, pool: PoolData, my_id: Optional[str], grant_percent: int) -> dict:
    """Everything the dashboard renders, in one payload."""
    board = leaderboard(pool.ledger, pool.nodes, my_id)
    me_row = next((r for r in board if r["is_me"]), None)
    my_flops = node_flops(pool.ledger, my_id)
    flops_by_node = {r["node_id"]: r["flops"] for r in board}
    return {
        "status": status,
        "pool": {
            "online": pool.online,
            "nodes": [{**n, "flops": flops_by_node.get(n.get("node_id"), 0.0),
                       "is_me": bool(my_id) and n.get("node_id") == my_id}
                      for n in sorted(pool.nodes, key=lambda n: str(n.get("name", "")).lower())],
            "jobs": [job_card(j) for j in recent_jobs(pool.jobs)],
            "capacity": capacity(pool.nodes, pool.jobs),
        },
        "me": {
            "node_id": my_id,
            "node": next((n for n in pool.nodes if my_id and n.get("node_id") == my_id), None),
            "flops": my_flops,
            "credits": split_credits(my_flops, grant_percent),
            # A rank only means something once this Mac has done some work.
            "rank": me_row["rank"] if me_row and my_flops > 0 else None,
            "of": len(board),
        },
        "leaderboard": board,
    }
