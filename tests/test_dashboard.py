import pytest

from slashcompute.launcher.dashboard import (
    PoolData, capacity, job_card, leaderboard, node_flops, overview,
)

LEDGER = [
    {"node_id": "a", "kind": "train", "flops": 5e12, "disputed_flops": 1e12},
    {"node_id": "a", "kind": "verify", "flops": 1e12, "disputed_flops": 0.0},
    {"node_id": "b", "kind": "train", "flops": 9e12, "disputed_flops": 0.0},
]
NODES = [
    {"node_id": "a", "name": "Air", "matmul_tflops": 2.5, "memory_contrib_bytes": 8 << 30},
    {"node_id": "c", "name": "Studio", "matmul_tflops": 20.0, "memory_contrib_bytes": 64 << 30},
]


def test_node_flops_subtracts_disputed_work():
    assert node_flops(LEDGER, "a") == pytest.approx(5e12)
    assert node_flops(LEDGER, None) == 0.0


def test_leaderboard_ranks_by_net_flops_and_marks_me():
    rows = leaderboard(LEDGER, NODES, my_id="a")
    assert [(r["rank"], r["node_id"]) for r in rows] == [(1, "b"), (2, "a"), (3, "c")]
    assert [r["name"] for r in rows] == ["b", "Air", "Studio"]  # unknown names fall back to id
    assert [r["is_me"] for r in rows] == [False, True, False]


def test_capacity_and_job_card():
    jobs = [{"status": "running"}, {"status": "queued"}, {"status": "recovering"},
            {"status": "completed"}]
    assert capacity(NODES, jobs) == {"macs": 2, "tflops": 22.5, "memory_bytes": 72 << 30,
                                     "running": 1, "waiting": 2}
    assert job_card({"status": "running", "steps": 50, "progress_step": 10})["progress"] == 0.2
    done = job_card({"status": "completed", "steps": 50, "progress_step": 49})
    assert done["progress"] == 1.0 and done["can_cancel"] is False


def test_overview_without_a_registered_mac():
    ov = overview({"lan_ip": "x"}, PoolData(online=True, nodes=NODES, ledger=LEDGER), None)
    assert ov["me"]["rank"] is None and ov["me"]["node"] is None
    assert ov["me"]["of"] == 3
    assert not any(n["is_me"] for n in ov["pool"]["nodes"])


def test_no_rank_until_this_mac_has_contributed():
    nodes = [{"node_id": "me", "name": "Air"}, {"node_id": "b", "name": "Bee"}]
    ov = overview({}, PoolData(online=True, nodes=nodes), "me")
    assert ov["me"]["rank"] is None and ov["me"]["of"] == 2
