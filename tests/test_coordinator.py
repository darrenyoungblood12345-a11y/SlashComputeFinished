"""Coordinator behaviour against fake agents over the real WebSocket API."""

import json
import time

import pytest
from fastapi.testclient import TestClient

from slashcompute.common import protocol as P
from slashcompute.common.canary import compare_stats, expected_stats, run_canary_mlx
from slashcompute.common.config import EngineConfig
from slashcompute.coordinator.app import create_app
from slashcompute.coordinator.core import Coordinator
from slashcompute.jobs import LoraFinetuneSpec
from slashcompute.pipeline.local import build_compute

GB = 1024**3


class FakeAgent:
    def __init__(self, client: TestClient, node_id: str, mem: int = 8 * GB, port: int = 9000):
        self.node_id = node_id
        self.ws = client.websocket_connect("/ws/agent").__enter__()
        self.ws.send_text(P.dump(P.Register(
            node_id=node_id, name=node_id, data_host="127.0.0.1", data_port=port, gpu_percent=100,
            device=P.DeviceProfile(chip="test", memory_total_bytes=mem, memory_available_bytes=mem,
                                   working_set_bytes=mem, memory_contrib_bytes=mem,
                                   matmul_tflops=1.0, mem_bandwidth_gbps=100.0))))
        assert isinstance(self.recv(), P.Welcome)

    def recv(self):
        return P.parse_coordinator_message(self.ws.receive_text())

    def send(self, msg):
        self.ws.send_text(P.dump(msg))

    def pass_canary(self):
        req = self.recv()
        assert isinstance(req, P.VerifyRequest) and req.kind == "canary"
        self.send(P.VerifyResult(verify_id=req.verify_id, kind="canary",
                                 stats=run_canary_mlx(req.seed, req.size)))

    def close(self):
        self.ws.__exit__(None, None, None)


def _usage():
    return P.UsageSample(flops=1e9, tokens=10, peak_mem_bytes=1, resident_mem_bytes=1,
                         mem_byte_seconds=1.0, wall_s=1.0, busy_s=0.5)


def _wait(cond, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met")


@pytest.fixture
def env(tmp_path, tiny_model, tiny_dataset):
    cfg = EngineConfig(home=tmp_path / "home", scheduler_tick_s=0.02, verify_rate=0.0,
                       canary_size=64, stage_overhead_bytes=0, heartbeat_timeout_s=60)
    app = create_app(cfg)
    with TestClient(app) as client:
        yield client, app.state.core, tiny_model, tiny_dataset, tmp_path


def _spec(tiny_model, tiny_dataset, **kw):
    base = dict(model=str(tiny_model), dataset_path=str(tiny_dataset), steps=2,
                batch_size=2, microbatches=1, lora_rank=4, min_stages=2)
    return LoraFinetuneSpec(**(base | kw))


def _upload_ckpt(client, spec, a: P.StageAssignment, step, tmp_path):
    comp = build_compute(spec, a.layer_start, a.layer_end, a.num_layers)
    path = tmp_path / f"s{a.stage_idx}_e{a.epoch}_{step}.safetensors"
    comp.save_checkpoint(path, step)
    r = client.post(f"/jobs/{a.job_id}/checkpoints/{step}?epoch={a.epoch}&stage={a.stage_idx}",
                    content=path.read_bytes())
    assert r.status_code == 200, r.text


def test_full_job_lifecycle(env):
    client, core, tiny_model, tiny_dataset, tmp_path = env
    a, b = FakeAgent(client, "node-a"), FakeAgent(client, "node-b", port=9001)
    a.pass_canary(); b.pass_canary()
    _wait(lambda: all(n.canary_passed for n in core.registry.nodes.values()))

    spec = _spec(tiny_model, tiny_dataset)
    r = client.post("/jobs", json=json.loads(spec.model_dump_json()))
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]

    asg = {x.node_id: x.recv() for x in (a, b)}
    for msg in asg.values():
        assert isinstance(msg, P.StageAssignment) and msg.num_stages == 2 and msg.epoch == 1
    s0 = next(m for m in asg.values() if m.stage_idx == 0)
    s1 = next(m for m in asg.values() if m.stage_idx == 1)
    assert (s0.layer_start, s0.layer_end, s1.layer_end) == (0, s1.layer_start, 6)
    assert s0.next_peer.port == (9001 if s1 is asg["node-b"] else 9000)
    assert s0.dataset_url and s1.dataset_url is None and s0.resume_step == 0
    assert client.get(s0.dataset_url).status_code == 200

    by_stage = {0: (s0, a if asg["node-a"] is s0 else b), 1: (s1, a if asg["node-a"] is s1 else b)}
    for idx, (m, ag) in by_stage.items():
        ag.send(P.StageReady(job_id=job_id, epoch=1, stage_idx=idx))
    _wait(lambda: client.get(f"/jobs/{job_id}").json()["status"] == "running")

    for step in (1, 2):
        for idx, (m, ag) in by_stage.items():
            ag.send(P.StepMetrics(job_id=job_id, epoch=1, stage_idx=idx, step=step,
                                  loss=1.0 / step if idx == 1 else None,
                                  in_digest=f"in{idx}-{step}", out_digest=f"in{idx + 1}-{step}",
                                  usage=_usage()))
    for idx, (m, ag) in by_stage.items():
        _upload_ckpt(client, spec, m, 2, tmp_path)
        ag.send(P.StageFinished(job_id=job_id, epoch=1, stage_idx=idx, reason="done", last_step=2))
    _wait(lambda: client.get(f"/jobs/{job_id}").json()["status"] == "completed")

    job = client.get(f"/jobs/{job_id}").json()
    assert job["last_checkpoint_step"] == 2 and job["last_loss"] == 0.5
    cfg = client.get(f"/jobs/{job_id}/adapter/adapter_config.json").json()
    assert cfg["num_layers"] == 6 and cfg["lora_parameters"]["rank"] == 4
    assert client.get(f"/jobs/{job_id}/adapter/adapters.safetensors").status_code == 200

    ledger = {(r["node_id"], r["kind"]): r for r in client.get("/ledger").json()}
    assert ledger[("node-a", "train")]["flops"] == 2e9 and ledger[("node-a", "train")]["disputed_flops"] == 0
    assert not [v for v in client.get("/verifications").json() if v["kind"] == "chain"]
    a.close(); b.close()


def test_chain_mismatch_disputes(env):
    client, core, tiny_model, tiny_dataset, _ = env
    a, b = FakeAgent(client, "node-a"), FakeAgent(client, "node-b", port=9001)
    a.pass_canary(); b.pass_canary()
    _wait(lambda: all(n.canary_passed for n in core.registry.nodes.values()))
    job_id = client.post("/jobs", json=json.loads(_spec(tiny_model, tiny_dataset).model_dump_json())).json()["id"]
    asg = {x.node_id: x.recv() for x in (a, b)}
    ag = {m.stage_idx: (a if k == "node-a" else b) for k, m in asg.items()}
    ag[0].send(P.StepMetrics(job_id=job_id, epoch=1, stage_idx=0, step=1, in_digest="t",
                             out_digest="X", usage=_usage()))
    ag[1].send(P.StepMetrics(job_id=job_id, epoch=1, stage_idx=1, step=1, loss=1.0, in_digest="Y",
                             out_digest="z", usage=_usage()))
    _wait(lambda: len([v for v in client.get("/verifications").json() if v["kind"] == "chain"]) == 2)
    ledger = client.get("/ledger").json()
    assert all(r["disputed_flops"] == 1e9 for r in ledger if r["kind"] == "train")
    a.close(); b.close()


def test_failed_canary_excludes_node(env):
    client, core, *_ = env
    a = FakeAgent(client, "liar")
    req = a.recv()
    a.send(P.VerifyResult(verify_id=req.verify_id, kind="canary",
                          stats={k: 0.0 for k in ("sum", "abs_sum", "fro", "trace", "c00", "c_last", "row0_dot")}))
    _wait(lambda: core.registry.get("liar").canary_passed is False)
    assert core.registry.schedulable() == []
    a.close()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_canary_rejects_non_finite_stats(bad):
    exp = expected_stats(1, 64)
    assert compare_stats(dict(exp), exp, 0.5) == (True, 0.0)
    assert compare_stats({k: bad for k in exp}, exp, 0.5) == (False, float("inf"))
    assert compare_stats({**exp, "trace": bad}, exp, 0.5) == (False, float("inf"))


def test_node_loss_triggers_recovery_and_reschedule(env):
    client, core, tiny_model, tiny_dataset, tmp_path = env
    agents = [FakeAgent(client, f"node-{x}", port=9000 + i) for i, x in enumerate("abc")]
    try:
        for x in agents:
            x.pass_canary()
        _wait(lambda: all(n.canary_passed for n in core.registry.nodes.values()))
        spec = _spec(tiny_model, tiny_dataset, max_stages=2, steps=10)
        job_id = client.post("/jobs", json=json.loads(spec.model_dump_json())).json()["id"]
        _wait(lambda: sum(1 for n in core.registry.nodes.values() if n.assignment) == 2)
        assigned = [x for x in agents if core.registry.get(x.node_id) and core.registry.get(x.node_id).assignment]
        spare = next(x for x in agents if x not in assigned)
        msgs = {x.node_id: x.recv() for x in assigned}

        for x in assigned:
            _upload_ckpt(client, spec, msgs[x.node_id], 5, tmp_path)
        _wait(lambda: client.get(f"/jobs/{job_id}").json()["last_checkpoint_step"] == 5)
        lost, survivor = assigned
        lost.close()

        cancel = survivor.recv()
        assert isinstance(cancel, P.CancelStage) and cancel.epoch == 1
        survivor.send(P.StageFinished(job_id=job_id, epoch=1, stage_idx=msgs[survivor.node_id].stage_idx,
                                      reason="cancelled", last_step=5))
        new = {x.node_id: x.recv() for x in (survivor, spare)}
        for m in new.values():
            assert isinstance(m, P.StageAssignment) and m.epoch == 2 and m.resume_step == 5
            assert m.checkpoint_url == f"/jobs/{job_id}/checkpoints/5"
        assert client.get(f"/jobs/{job_id}/checkpoints/5").status_code == 200
        job = client.get(f"/jobs/{job_id}").json()
        assert job["recoveries"] == 1 and job["status"] == "starting"
    finally:
        for x in agents:
            try:
                x.close()
            except Exception:
                pass


def test_drain_sends_drain_to_stage0_and_resumes(env):
    client, core, tiny_model, tiny_dataset, tmp_path = env
    agents = [FakeAgent(client, f"node-{x}", port=9000 + i) for i, x in enumerate("abc")]
    try:
        for x in agents:
            x.pass_canary()
        _wait(lambda: all(n.canary_passed for n in core.registry.nodes.values()))
        spec = _spec(tiny_model, tiny_dataset, max_stages=2, steps=10)
        job_id = client.post("/jobs", json=json.loads(spec.model_dump_json())).json()["id"]
        _wait(lambda: sum(1 for n in core.registry.nodes.values() if n.assignment) == 2)
        assigned = [x for x in agents if core.registry.get(x.node_id) and core.registry.get(x.node_id).assignment]
        spare = next(x for x in agents if x not in assigned)
        msgs = {x.node_id: x.recv() for x in assigned}
        stage0 = next(x for x in assigned if msgs[x.node_id].stage_idx == 0)
        leaver = next(x for x in assigned if x is not stage0)

        leaver.send(P.DrainNotice(node_id=leaver.node_id))
        drain = stage0.recv()
        assert isinstance(drain, P.Drain) and drain.epoch == 1
        for x in assigned:
            m = msgs[x.node_id]
            _upload_ckpt(client, spec, m, 3, tmp_path)
            x.send(P.StageFinished(job_id=job_id, epoch=1, stage_idx=m.stage_idx, reason="drained", last_step=3))
        new = {x.node_id: x.recv() for x in (stage0, spare)}
        assert all(m.epoch == 2 and m.resume_step == 3 for m in new.values())
        assert client.get(f"/jobs/{job_id}").json()["recoveries"] == 0
    finally:
        for x in agents:
            try:
                x.close()
            except Exception:
                pass


def test_job_upload_accepts_dataset_file(env):
    client, core, tiny_model, tiny_dataset, _ = env
    r = client.post(
        "/jobs/upload",
        files={"dataset": ("train.jsonl", tiny_dataset.read_bytes(), "application/jsonl")},
        data={"model": str(tiny_model), "steps": 2, "min_stages": 1,
              "batch_size": 2, "microbatches": 1},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "lora_finetune"
    assert body["id"] in core.jobs


def test_fatal_stage_error_fails_the_job_without_retrying(env):
    client, core, tiny_model, tiny_dataset, _ = env
    agents = [FakeAgent(client, "node-a"), FakeAgent(client, "node-b", port=9001)]
    try:
        for x in agents:
            x.pass_canary()
        _wait(lambda: all(n.canary_passed for n in core.registry.nodes.values()))
        spec = _spec(tiny_model, tiny_dataset)
        job_id = client.post("/jobs", json=json.loads(spec.model_dump_json())).json()["id"]
        ag = {m.stage_idx: x for x in agents for m in [x.recv()]}
        ag[0].send(P.StageFinished(job_id=job_id, epoch=1, stage_idx=0, reason="error", last_step=0,
                                   detail="bad dataset: unrecognised dataset row keys", fatal=True))
        assert isinstance(ag[1].recv(), P.CancelStage)
        _wait(lambda: client.get(f"/jobs/{job_id}").json()["status"] == "failed", timeout=3)
        job = client.get(f"/jobs/{job_id}").json()
        assert job["recoveries"] == 0 and "unrecognised dataset row keys" in job["error"]
    finally:
        for x in agents:
            try:
                x.close()
            except Exception:
                pass


def test_recovered_job_drops_the_stale_abort_error(env):
    client, core, tiny_model, tiny_dataset, tmp_path = env
    agents = [FakeAgent(client, "node-a"), FakeAgent(client, "node-b", port=9001)]
    try:
        for x in agents:
            x.pass_canary()
        _wait(lambda: all(n.canary_passed for n in core.registry.nodes.values()))
        spec = _spec(tiny_model, tiny_dataset)
        job_id = client.post("/jobs", json=json.loads(spec.model_dump_json())).json()["id"]
        ag = {m.stage_idx: x for x in agents for m in [x.recv()]}
        ag[0].send(P.StageFinished(job_id=job_id, epoch=1, stage_idx=0, reason="error", last_step=0,
                                   detail="transient"))
        cancel = ag[1].recv()
        assert isinstance(cancel, P.CancelStage)
        # As a real agent does, say the stage stopped: until then its Mac takes no new work.
        ag[1].send(P.StageFinished(job_id=job_id, epoch=1, stage_idx=1, reason="cancelled",
                                   last_step=0))
        new = {x.node_id: x.recv() for x in agents}
        assert all(isinstance(m, P.StageAssignment) and m.epoch == 2 for m in new.values())
        assert "transient" in client.get(f"/jobs/{job_id}").json()["error"]

        by_stage = {m.stage_idx: (m, x) for x in agents for m in [new[x.node_id]]}
        for idx, (m, x) in by_stage.items():
            x.send(P.StageReady(job_id=job_id, epoch=2, stage_idx=idx))
        _wait(lambda: client.get(f"/jobs/{job_id}").json()["status"] == "running")
        assert client.get(f"/jobs/{job_id}").json()["error"] is None

        core.jobs[job_id].row.error = "node lost: disconnected"  # as if aborted mid-run
        for idx, (m, x) in by_stage.items():
            _upload_ckpt(client, spec, m, 2, tmp_path)
            x.send(P.StageFinished(job_id=job_id, epoch=2, stage_idx=idx, reason="done", last_step=2))
        _wait(lambda: client.get(f"/jobs/{job_id}").json()["status"] == "completed")
        assert client.get(f"/jobs/{job_id}").json()["error"] is None
        assert Coordinator(core.cfg).jobs[job_id].row.error is None
    finally:
        for x in agents:
            try:
                x.close()
            except Exception:
                pass


def test_rejected_job_does_not_come_back_after_restart(env, monkeypatch):
    monkeypatch.setattr("slashcompute.community.auth.ITERATIONS", 1)
    client, core, tiny_model, tiny_dataset, _ = env
    token = client.post("/auth/register", json={
        "email": "ada@lan.test", "password": "password1", "name": "Ada",
    }).json()["token"]
    hdr = {"Authorization": f"Bearer {token}"}
    client.post("/auth/accept-terms", headers=hdr)
    r = client.post("/jobs/upload", headers=hdr,
                    files={"dataset": ("train.jsonl", tiny_dataset.read_bytes())},
                    data={"model": str(tiny_model), "steps": 2, "batch_size": 2,
                          "microbatches": 1, "min_stages": 1, "max_flops": 1e30})
    assert r.status_code == 400, r.text
    assert client.get("/jobs").json() == []
    assert Coordinator(core.cfg).jobs == {}
    assert not any(core.checkpoints.root.iterdir())


def test_unsatisfiable_job_fails_instead_of_blocking_queue(env):
    client, core, tiny_model, tiny_dataset, _ = env
    agents = [FakeAgent(client, "node-a"), FakeAgent(client, "node-b", port=9001)]
    try:
        for x in agents:
            x.pass_canary()
        _wait(lambda: all(n.canary_passed for n in core.registry.nodes.values()))

        stuck = json.loads(_spec(tiny_model, tiny_dataset, min_stages=500).model_dump_json())
        stuck_id = client.post("/jobs", json=stuck).json()["id"]
        ok = json.loads(_spec(tiny_model, tiny_dataset).model_dump_json())
        ok_id = client.post("/jobs", json=ok).json()["id"]

        _wait(lambda: client.get(f"/jobs/{stuck_id}").json()["status"] == "failed")
        assert "6 layers" in client.get(f"/jobs/{stuck_id}").json()["error"]
        _wait(lambda: client.get(f"/jobs/{ok_id}").json()["status"] == "starting")
        for x in agents:
            msg = x.recv()
            assert isinstance(msg, P.StageAssignment) and msg.job_id == ok_id
    finally:
        for x in agents:
            try:
                x.close()
            except Exception:
                pass


@pytest.mark.parametrize("kind", [["x"], {}, 3])
def test_submit_rejects_non_string_kind(env, kind):
    client, _, tiny_model, tiny_dataset, _ = env
    body = json.loads(_spec(tiny_model, tiny_dataset).model_dump_json()) | {"kind": kind}
    r = client.post("/jobs", json=body)
    assert r.status_code == 400, r.text


def _cli(monkeypatch, client, seen=None):
    """Route the coordinator CLI's HTTP calls into the in-process app."""
    import httpx

    from slashcompute.coordinator import main

    def request(method, url, timeout=None, **kw):
        if seen is not None:
            seen.append(kw.get("headers") or {})
        return client.request(method, url, **kw)

    monkeypatch.setattr(httpx, "request", request)
    return main.app


@pytest.mark.parametrize("args", [["jobs", "nope"], ["cancel", "nope"]])
def test_cli_reports_http_errors_without_traceback(env, monkeypatch, args):
    from typer.testing import CliRunner

    res = CliRunner().invoke(_cli(monkeypatch, env[0]), args + ["--url", "http://127.0.0.1:9970"])
    assert res.exit_code == 1
    assert isinstance(res.exception, SystemExit)
    assert "not found" in res.output.lower()


def test_cli_reports_unreachable_coordinator():
    from typer.testing import CliRunner

    from slashcompute.coordinator import main

    res = CliRunner().invoke(main.app, ["jobs", "--url", "http://127.0.0.1:9971"])
    assert res.exit_code == 1
    assert isinstance(res.exception, SystemExit)
    assert "cannot reach coordinator at http://127.0.0.1:9971" in res.output


def test_cli_sends_session_token_as_bearer(env, monkeypatch):
    from typer.testing import CliRunner

    seen = []
    app = _cli(monkeypatch, env[0], seen)
    url = ["--url", "http://127.0.0.1:9970"]
    CliRunner().invoke(app, ["cancel", "nope", "--session-token", "tok1"] + url)
    CliRunner().invoke(app, ["cancel", "nope"] + url, env={"SLASHCOMPUTE_SESSION": "tok2"})
    assert [h.get("Authorization") for h in seen] == ["Bearer tok1", "Bearer tok2"]


@pytest.mark.parametrize("data,why", [
    (b'{"text": "ok"}\n{"text": "ok"\n', "dataset line 2 is not JSON"),
    (b'{"text": "ok"}\n[1, 2]\n', "dataset line 2 is not a JSON object"),
    (b'{"text": "ok"}\n\n{"nope": 1}\n', "dataset line 3 has keys ['nope']"),
    (b"\n\n", "the dataset has no rows"),
])
def test_upload_rejects_a_malformed_dataset(env, data, why):
    """It used to be queued, then crash the worker on a Mac after the model had loaded."""
    client, core, tiny_model, _, _ = env
    r = client.post("/jobs/upload", files={"dataset": ("train.jsonl", data)},
                    data={"model": str(tiny_model), "steps": "2"})
    assert r.status_code == 400 and why in r.json()["detail"], r.text
    assert core.jobs == {}


def test_upload_accepts_every_dataset_row_format(env):
    client, core, tiny_model, _, _ = env
    rows = [{"text": "a"}, {"prompt": "q", "completion": "a"},
            {"messages": [{"role": "user", "content": "q"}]}, {"tokens": [1, 2, 3]}]
    data = "\n".join(json.dumps(r) for r in rows).encode() + b"\n"
    r = client.post("/jobs/upload", files={"dataset": ("train.jsonl", data)},
                    data={"model": str(tiny_model), "steps": "2"})
    assert r.status_code == 200, r.text
    assert core.jobs[r.json()["id"]].spec.min_stages == 1   # the form's default: one Mac is enough
