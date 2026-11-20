"""FIFO waitlist position and clock ETAs on Takes."""

from __future__ import annotations

import json

import pytest

from accounts import sign_in
from fastapi.testclient import TestClient

from slashcompute.common import protocol as P
from slashcompute.common.config import EngineConfig
from slashcompute.coordinator.app import create_app
from slashcompute.coordinator.core import Coordinator
from slashcompute.jobs import LoraFinetuneSpec


@pytest.fixture
def core(tmp_path, tiny_model, tiny_dataset):
    cfg = EngineConfig(home=tmp_path / "home", scheduler_tick_s=0.05, verify_rate=0.0)
    c = Coordinator(cfg)
    c._tiny = (tiny_model, tiny_dataset)
    return c


@pytest.fixture
def env(tmp_path, tiny_model, tiny_dataset):
    cfg = EngineConfig(home=tmp_path / "home", scheduler_tick_s=0.05, verify_rate=0.0)
    app = create_app(cfg)
    with TestClient(app) as client:
        yield client, app.state.core, tiny_model, tiny_dataset


def _spec(tiny_model, tiny_dataset, **kw):
    base = dict(model=str(tiny_model), dataset_path=str(tiny_dataset), steps=10,
                batch_size=2, microbatches=1, lora_rank=4, min_stages=1)
    return LoraFinetuneSpec(**(base | kw))


def _device(tflops=2.0):
    mem = 8 << 30
    return P.DeviceProfile(
        chip="test", memory_total_bytes=mem, memory_available_bytes=mem,
        working_set_bytes=mem, memory_contrib_bytes=mem,
        matmul_tflops=tflops, mem_bandwidth_gbps=100.0,
    )


def _usage(flops):
    return P.UsageSample(flops=flops, tokens=10, peak_mem_bytes=1, resident_mem_bytes=1,
                         mem_byte_seconds=1.0, wall_s=1.0, busy_s=0.5)


async def _send(_):
    return None


def _online(core, tflops=2.0, node_id="n1"):
    core.registry.register(P.Register(
        node_id=node_id, name="mac", device=_device(tflops),
        data_host="127.0.0.1", data_port=9700, gpu_percent=50,
    ), _send)


def _pair(core):
    tiny_model, tiny_dataset = core._tiny
    a = core.submit(_spec(tiny_model, tiny_dataset))
    b = core.submit(_spec(tiny_model, tiny_dataset))
    a.row.submitted_at = 100.0
    b.row.submitted_at = 200.0
    return a, b


def test_two_queued_no_pool_positions_without_eta(core):
    a, b = _pair(core)
    va, vb = core.job_view(a), core.job_view(b)
    assert va["queue_position"] == 1
    assert vb["queue_position"] == 2
    assert va["wait_s"] is None
    assert vb["wait_s"] is None
    assert core.job_view(a)["status"] == "queued"


def test_reserved_ahead_divides_by_pool_tflops(core):
    a, b = _pair(core)
    user, _ = sign_in(core.auth)
    core.credits.contribute(user.id, 1e15)
    core.credits.reserve_job(user.id, a.id, 2e12)
    _online(core, tflops=2.0)
    assert core.remaining_flops(a) == pytest.approx(2e12)
    assert core.job_view(b)["queue_position"] == 2
    assert core.job_view(b)["wait_s"] == pytest.approx(2e12 / (2.0 * 1e12))
    assert core.job_view(a)["wait_s"] == pytest.approx(0.0)


def test_anonymous_ahead_uses_leftover_steps_times_last_step(core):
    a, b = _pair(core)
    a.row.progress_step = 4
    a.last_step_flops = 1e11
    _online(core, tflops=2.0)
    leftover = (a.spec.steps - 4) * 1e11
    assert core.remaining_flops(a) == pytest.approx(leftover)
    assert core.job_view(b)["wait_s"] == pytest.approx(leftover / (2.0 * 1e12))
    assert core.job_view(b)["queue_position"] == 2


def test_anonymous_ahead_without_step_omits_eta(core):
    a, b = _pair(core)
    _online(core, tflops=2.0)
    assert core.remaining_flops(a) is None
    vb = core.job_view(b)
    assert vb["queue_position"] == 2
    assert vb["wait_s"] is None


def test_running_job_is_ahead_of_queued(core):
    a, b = _pair(core)
    user, _ = sign_in(core.auth)
    core.credits.contribute(user.id, 1e15)
    core.credits.reserve_job(user.id, a.id, 4e12)
    a.row.status = "running"
    _online(core, tflops=2.0)
    va, vb = core.job_view(a), core.job_view(b)
    assert va["queue_position"] is None
    assert va["wait_s"] is None
    assert vb["queue_position"] == 1
    assert vb["wait_s"] == pytest.approx(4e12 / (2.0 * 1e12))


def test_http_waitlist_follows_submitted_at(env):
    client, core, tiny_model, tiny_dataset = env
    spec = json.loads(_spec(tiny_model, tiny_dataset).model_dump_json())
    first = client.post("/jobs", json=spec)
    second = client.post("/jobs", json=spec)
    assert first.status_code == 200 and second.status_code == 200
    a, b = core.jobs[first.json()["id"]], core.jobs[second.json()["id"]]
    a.row.submitted_at = 10.0
    b.row.submitted_at = 20.0
    rows = client.get("/jobs/waitlist").json()
    assert [r["id"] for r in rows] == [a.id, b.id]
    assert [r["queue_position"] for r in rows] == [1, 2]
    listed = {j["id"]: j for j in client.get("/jobs").json()}
    assert listed[a.id]["queue_position"] == 1
    assert listed[b.id]["wait_s"] is None
