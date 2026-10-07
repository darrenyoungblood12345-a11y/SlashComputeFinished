import asyncio
import json
import stat
import sys
from types import SimpleNamespace

import httpx
import pytest

from slashcompute.agent.daemon import AgentOptions, Daemon
from slashcompute.agent.http import CoordHTTP
from slashcompute.agent.worker import _stdio_app
from slashcompute.common.protocol import LoraFinetuneSpec, StageAssignment


@pytest.mark.parametrize("session_token", [None, "agent-session"])
@pytest.mark.parametrize("path", ["/jobs/j/dataset", "https://pool.example/jobs/j/dataset"])
def test_agent_downloads_and_uploads_with_optional_session(monkeypatch, tmp_path, session_token, path):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, content=b"training data")

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr(httpx, "stream", client.stream)
        monkeypatch.setattr(httpx, "post", client.post)
        coord = CoordHTTP("https://pool.example", session_token=session_token)
        dest = coord.get_file(path, tmp_path / "dataset")
        coord.put_bytes("/jobs/j/checkpoints/1", b"checkpoint", params={"stage": 0})

    assert dest.read_bytes() == b"training data"
    assert [request.method for request in requests] == ["GET", "POST"]
    assert requests[1].content == b"checkpoint"
    assert requests[1].url.params["stage"] == "0"
    expected = f"Bearer {session_token}" if session_token else None
    assert [request.headers.get("Authorization") for request in requests] == [expected, expected]


@pytest.mark.parametrize("url", [
    "https://external.example/dataset",
    "http://pool.example/dataset",
    "https://pool.example:444/dataset",
])
def test_agent_never_sends_session_to_another_origin(monkeypatch, tmp_path, url):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, content=b"dataset")

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr(httpx, "stream", client.stream)
        monkeypatch.setattr(httpx, "post", client.post)
        coord = CoordHTTP("https://pool.example", session_token="private-session")
        coord.get_file(url, tmp_path / "dataset")
        coord.put_bytes(url, b"data")

    assert all("Authorization" not in request.headers for request in requests)


@pytest.mark.parametrize("explicit_token,expected", [
    (None, "environment-session"),
    ("explicit-session", "explicit-session"),
])
def test_agent_session_reaches_http_and_sandbox_worker(monkeypatch, tmp_path, explicit_token, expected):
    monkeypatch.setenv("SLASHCOMPUTE_SESSION", "environment-session")
    opt = AgentOptions(
        url="https://pool.example", home=tmp_path, localhost=True, session_token=explicit_token,
    )
    assert opt.http.session_token == expected
    daemon = Daemon(opt)
    assignment = StageAssignment(
        job_id="j", epoch=1, stage_idx=0, num_stages=1, layer_start=0, layer_end=8,
        num_layers=8, spec=LoraFinetuneSpec(dataset_path="dataset.jsonl", steps=2),
        checkpoint_every=25, verify_ring_size=8,
    )
    job_dir = opt.paths.job_dir("j", 1)

    async def spawn(*args, **kwargs):
        return SimpleNamespace(stdout=None)

    monkeypatch.setattr("slashcompute.agent.daemon.wrap_command", lambda cmd, *_: cmd)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    from slashcompute.agent.daemon import _Stage

    stage = daemon._stage = _Stage(assignment)
    asyncio.run(daemon._start_sandboxed(stage, job_dir))
    spec_path = job_dir / "assignment.json"
    assert json.loads(spec_path.read_text())["session_token"] == expected
    assert stat.S_IMODE(spec_path.stat().st_mode) == 0o600

    contexts = []

    async def run_stage(ctx, emit):
        contexts.append(ctx)

    monkeypatch.setattr("slashcompute.agent.worker.run_stage", run_stage)
    monkeypatch.setattr("slashcompute.common.logging.setup_logging", lambda *_: None)
    monkeypatch.setattr(sys, "argv", ["worker", "--assignment", str(spec_path)])
    _stdio_app()
    assert contexts[0].http.session_token == expected
    assert contexts[0].assignment == assignment
