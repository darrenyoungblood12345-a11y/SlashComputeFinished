"""Shell + launcher side of LLM inference: the LLMs view, streaming chat/upload passthrough, the inference node."""

import json
import os
import re
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from slashcompute.launcher.controller import Launcher, LauncherSettings
from slashcompute.web.server import create_shell

SHELL = "http://127.0.0.1:8766"   # the shell refuses any Host but loopback


class FakeProc:
    def __init__(self, pid: int, argv: list[str]) -> None:
        self.pid = pid
        self.argv = argv

    def poll(self):
        return None


class FakeHTTP:
    def __init__(self, health=None) -> None:
        self.health = health

    def get(self, url: str, timeout: float = 1.0, params=None, headers=None):
        if self.health is None:
            raise ConnectionError("down")
        health = self.health

        class R:
            status_code = 200
            content = json.dumps(health).encode()
            headers = {"content-type": "application/json"}

            def json(self_inner):
                return health
        return R()


def _launcher(tmp_path, health=None):
    spawned = []
    n = {"p": 6000}

    def popen(argv, **_):
        n["p"] += 1
        spawned.append(FakeProc(n["p"], argv))
        return spawned[-1]

    launcher = Launcher(home=tmp_path, python="/opt/venv/bin/python", popen=popen, http=FakeHTTP(health),
                        discover_fn=lambda timeout=5.0: None, lan_ip_fn=lambda: "192.168.1.20",
                        port_free_fn=lambda host, port: True)
    return launcher, spawned


CURRENT = {"ok": True, "inference_nodes": 0, "inference_transport": "direct"}
OUTDATED = {"ok": True, "nodes": 4, "jobs": 1}   # /health of a coordinator from before LLM inference


def _shell(tmp_path, handler, health=CURRENT):
    launcher, _ = _launcher(tmp_path, health)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return create_shell(launcher, stream_client=client)


# ------------------------------------------------------------ UI

def test_llm_view_is_served(tmp_path):
    with TestClient(_shell(tmp_path, lambda r: httpx.Response(404)), base_url=SHELL) as c:
        html = c.get("/").content
        assert b'data-tab="llm"' in html and b'id="view-llm"' in html and b'id="chat-form"' in html
        js = c.get("/static/app.js").content
        assert b"/api/chat" in js and b"/api/models/upload" in js and b'"inference_memory_gb"' in js
        assert b"GOOGLE" not in js


# ------------------------------------------------------------ streaming passthrough

def test_chat_streams_sse_through_with_session(tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["cookie"] = request.headers.get("cookie")
        sse = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'
        return httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})

    app = _shell(tmp_path, handler)
    launcher = app.state.launcher
    launcher.save_settings(launcher.with_session(launcher.load_settings(), "tok"))   # signed in to this pool
    with TestClient(app, base_url=SHELL) as c:
        c.cookies.set("slashcompute_session", "tok")
        r = c.post("/api/chat", json={"model": "m.gguf", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert r.text.endswith("data: [DONE]\n\n")
    assert seen["url"] == "http://127.0.0.1:8765/v1/chat/completions"
    assert seen["body"]["stream"] is True
    assert seen["cookie"] == "slashcompute_session=tok"


class BreaksMidStream(httpx.AsyncByteStream):
    """A coordinator answer that dies after the first token (the coordinator restarted, a node died)."""

    async def __aiter__(self):
        yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        raise httpx.ReadError("connection lost")


def test_chat_turns_a_mid_stream_failure_into_an_error_event(tmp_path):
    def handler(request):
        return httpx.Response(200, stream=BreaksMidStream(), headers={"content-type": "text/event-stream"})

    with TestClient(_shell(tmp_path, handler), base_url=SHELL) as c:
        r = c.post("/api/chat", json={"model": "m", "messages": []})
    assert r.status_code == 200
    events = [e for e in r.text.split("\n\n") if e]
    assert events[0] == 'data: {"choices":[{"delta":{"content":"hi"}}]}'
    assert "connection lost" in json.loads(events[1][len("data: "):])["error"]["message"]
    assert events[2] == "data: [DONE]"


async def test_chat_stops_waiting_for_the_pipeline_when_the_window_leaves(tmp_path):
    """Forming a pipeline can take minutes before the first byte. A window that gave up meanwhile
    (Stop, a closed tab) must not keep the coordinator request open until then."""
    import asyncio

    asked, cancelled = asyncio.Event(), []

    async def handler(request):
        asked.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        return httpx.Response(200)

    app = _shell(tmp_path, handler)
    body = json.dumps({"model": "m", "messages": []}).encode()
    inbox = [{"type": "http.request", "body": body, "more_body": False}]

    async def receive():
        if inbox:
            return inbox.pop(0)
        await asked.wait()                 # the window leaves once the coordinator has the request
        return {"type": "http.disconnect"}

    sent = []

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "http_version": "1.1", "method": "POST", "scheme": "http",
             "path": "/api/chat", "raw_path": b"/api/chat", "root_path": "", "query_string": b"",
             "headers": [(b"host", b"127.0.0.1:8766"), (b"content-type", b"application/json"),
                         (b"content-length", str(len(body)).encode())],
             "client": ("127.0.0.1", 50000), "server": ("127.0.0.1", 8766)}
    await asyncio.wait_for(app(scope, receive, send), 10)
    assert cancelled and sent[0]["status"] == 499


def test_multipart_forms_stream_through_the_proxy(tmp_path):
    """A job's dataset goes to the coordinator as it arrives, boundary and all, off the event loop."""
    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["content_type"] = request.headers["content-type"]
        seen["body"] = await request.aread()
        return httpx.Response(200, json={"id": "job-1"})

    with TestClient(_shell(tmp_path, handler), base_url=SHELL) as c:
        r = c.post("/api/coord/jobs/upload", files={"dataset": ("d.jsonl", b'{"text":"a"}\n')},
                   data={"model": "m", "steps": "3"})
    assert r.status_code == 200 and r.json() == {"id": "job-1"}
    assert seen["url"] == "http://127.0.0.1:8765/jobs/upload"
    assert seen["content_type"].startswith("multipart/form-data; boundary=")
    assert b'name="dataset"; filename="d.jsonl"' in seen["body"] and b'{"text":"a"}' in seen["body"]
    assert b'name="steps"' in seen["body"]


def test_chat_errors_keep_their_status(tmp_path):
    def handler(request):
        return httpx.Response(503, json={"error": {"message": "no eligible head node has m.gguf on disk"}})

    with TestClient(_shell(tmp_path, handler), base_url=SHELL) as c:
        r = c.post("/api/chat", json={"model": "m.gguf", "messages": []})
    assert r.status_code == 503 and "no eligible head" in r.json()["error"]["message"]


def test_chat_without_a_coordinator_is_502(tmp_path):
    def handler(request):
        raise httpx.ConnectError("refused")

    with TestClient(_shell(tmp_path, handler), base_url=SHELL) as c:
        assert c.post("/api/chat", json={"model": "m", "messages": []}).status_code == 502


def test_chat_rejects_bodies_that_are_not_a_json_object(tmp_path):
    sent = []

    def handler(request):
        sent.append(request)
        return httpx.Response(200)

    with TestClient(_shell(tmp_path, handler), base_url=SHELL) as c:
        for raw in (b"notjson", b"[1,2]", b'"hi"', b"", b"\xff"):
            r = c.post("/api/chat", content=raw, headers={"content-type": "application/json"})
            assert r.status_code == 400, (raw, r.text)
            assert "JSON" in r.json()["detail"]
    assert sent == []


def test_upload_streams_the_file_to_the_coordinator(tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = request.read()
        return httpx.Response(200, json={"name": "m.gguf", "size": len(seen["body"])})

    data = os.urandom(300_000)
    with TestClient(_shell(tmp_path, handler), base_url=SHELL) as c:
        r = c.post("/api/models/upload", content=data, headers={"x-filename": "m.gguf"})
        assert r.status_code == 200 and r.json()["size"] == len(data)
        assert seen["url"] == "http://127.0.0.1:8765/inference/models/upload?name=m.gguf"
        assert seen["body"] == data
        # The window sends the name as a query parameter (headers mangle non-ASCII); it wins.
        r = c.post("/api/models/upload?name=Qwen2.5-7B.gguf", content=data, headers={"x-filename": "old.gguf"})
        assert r.status_code == 200
        assert httpx.URL(seen["url"]).params["name"] == "Qwen2.5-7B.gguf"


def test_upload_too_big_for_any_pool_is_refused_with_a_reason(tmp_path):
    sent = []

    def handler(request):
        sent.append(request)
        return httpx.Response(200, json={})

    with TestClient(_shell(tmp_path, handler), base_url=SHELL) as c:
        r = c.post("/api/models/upload?name=huge.gguf", content=b"GGUF",
                   headers={"content-length": str((64 << 30) + 1)})
    assert r.status_code == 413 and "huge.gguf" in r.json()["detail"] and "64 GiB" in r.json()["detail"]
    assert sent == []


def test_outdated_coordinator_is_explained_before_anything_is_sent(tmp_path):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(404, json={"detail": "Not Found"})

    with TestClient(_shell(tmp_path, handler, OUTDATED), base_url=SHELL) as c:
        up = c.post("/api/models/upload", content=b"GGUF" + bytes(1000), headers={"x-filename": "m.gguf"})
        chat = c.post("/api/chat", json={"model": "m", "messages": []})
        status = c.get("/api/status").json()
    assert up.status_code == chat.status_code == 409
    assert "older /compute" in up.json()["detail"] and "older /compute" in chat.json()["detail"]
    assert seen == []                                   # the GGUF never left this Mac
    assert status["inference_supported"] is False


def test_inference_support_follows_coordinator_health(tmp_path):
    for health, expected in ((CURRENT, True), (OUTDATED, False), (None, None)):
        launcher, _ = _launcher(tmp_path, health)
        launcher.save_settings(LauncherSettings(mode="join", url="10.0.0.5"))
        assert launcher.snapshot().inference_supported is expected


# ------------------------------------------------------------ launcher

def test_new_settings_round_trip_and_clamp(tmp_path):
    launcher, _ = _launcher(tmp_path)
    launcher.save_settings(LauncherSettings(inference=True, training=False, inference_memory_gb=24,
                                            inference_head=False, models_dir="/Volumes/m", transport="relay"))
    s = launcher.load_settings()
    assert (s.inference, s.training, s.inference_memory_gb, s.inference_head, s.models_dir, s.transport) == \
        (True, False, 24, False, "/Volumes/m", "relay")
    bad = LauncherSettings(transport="carrier-pigeon", inference_memory_gb="lots").clamp()
    assert bad.transport == "direct" and bad.inference_memory_gb == 0


def test_inference_argv(tmp_path):
    launcher, _ = _launcher(tmp_path)
    argv = launcher.inference_argv("http://127.0.0.1:8765", LauncherSettings(
        inference_memory_gb=12, inference_head=False, session_token="tok", session_url="http://127.0.0.1:8765"))
    assert argv[:4] == ["/opt/venv/bin/python", "-m", "slashcompute.inference.node", "start"]
    assert argv[argv.index("--url") + 1] == "http://127.0.0.1:8765"
    assert argv[argv.index("--memory-gb") + 1] == "12"
    assert "--no-head" in argv and "--session-token" not in argv   # the session goes through the environment
    assert launcher.coordinator_argv("relay")[-2:] == ["--inference-transport", "relay"]
    assert "--inference-transport" not in launcher.coordinator_argv()


def test_host_with_inference_spawns_the_node_and_relay_coordinator(tmp_path):
    launcher, spawned = _launcher(tmp_path)
    launcher.poll_health = lambda url: {"ok": True, "inference_transport": "relay"} if spawned else None
    launcher.start(LauncherSettings(mode="host", contribute=True, inference=True, transport="relay"))
    mods = [p.argv[2] for p in spawned]
    assert mods == ["slashcompute.coordinator.main", "slashcompute.agent.main", "slashcompute.inference.node"]
    assert spawned[0].argv[-2:] == ["--inference-transport", "relay"]
    assert "127.0.0.1" in spawned[2].argv[spawned[2].argv.index("--url") + 1]


def test_training_off_lends_only_to_inference(tmp_path):
    launcher, spawned = _launcher(tmp_path)
    launcher.poll_health = lambda url: {"ok": True} if spawned else None
    launcher.start(LauncherSettings(mode="host", contribute=True, training=False, inference=True))
    assert [p.argv[2] for p in spawned] == ["slashcompute.coordinator.main", "slashcompute.inference.node"]


def test_changed_inference_settings_restart_the_node(tmp_path, monkeypatch):
    launcher, spawned = _launcher(tmp_path, {"ok": True})
    running = {"pid": None}
    monkeypatch.setattr(launcher, "read_inference_pid", lambda: running["pid"])
    stopped = []
    monkeypatch.setattr(launcher, "_stop_inference", lambda wait=0.0: (stopped.append(1), running.update(pid=None)))
    launcher.start(LauncherSettings(mode="join", url="10.0.0.9", training=False, inference=True))
    running["pid"] = spawned[-1].pid
    launcher.start(LauncherSettings(mode="join", url="10.0.0.9", training=False, inference=True))
    assert len(spawned) == 1                                    # same settings: keep running
    launcher.start(LauncherSettings(mode="join", url="10.0.0.9", training=False, inference=True,
                                    inference_memory_gb=8))
    assert len(spawned) == 2 and stopped                        # new memory: drained and rejoined
    assert spawned[-1].argv[spawned[-1].argv.index("--memory-gb") + 1] == "8"


def test_transport_change_restarts_our_coordinator(tmp_path, monkeypatch):
    launcher, spawned = _launcher(tmp_path, {"ok": True, "inference_transport": "direct"})
    monkeypatch.setattr(launcher, "read_coordinator_pid", lambda: 4242)   # we started it
    killed = []
    monkeypatch.setattr("slashcompute.launcher.controller.os.kill", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr("slashcompute.launcher.controller.process_alive", lambda pid: False)
    monkeypatch.setattr(launcher, "wait_health", lambda url, timeout=8.0: True)
    launcher.start(LauncherSettings(mode="host", contribute=False, transport="relay"))
    assert killed and killed[0][0] == 4242
    assert spawned[0].argv[-2:] == ["--inference-transport", "relay"]


# ------------------------------------------------------------ Start serving touches only the LLM node


def _serving_launcher(tmp_path, monkeypatch, health=CURRENT):
    launcher, spawned = _launcher(tmp_path, health)
    monkeypatch.setattr(launcher, "_ensure_coordinator", lambda s, url: None)
    return launcher, spawned


def test_start_serving_never_restarts_the_training_agent(tmp_path, monkeypatch):
    """Start serving used to resend every setting through start(): a training setting saved since
    the agent started (here its memory) restarted the agent, which can't stop while it downloads
    a model, so the click failed with "Training agent is still stopping"."""
    launcher, spawned = _serving_launcher(tmp_path, monkeypatch)
    monkeypatch.setattr(launcher, "_await_node", lambda proc, timeout=8.0: None)
    monkeypatch.setattr(launcher, "_agent_running", lambda: True)
    monkeypatch.setattr(launcher, "_ensure_agent", lambda *a: pytest.fail("touched the training agent"))
    monkeypatch.setattr("slashcompute.launcher.controller.request_stop",
                        lambda paths: pytest.fail("stopped the training agent"))
    launcher.save_settings(LauncherSettings(mode="host", contribute=False, training=True, memory_gb=8))
    launcher.set_inference(True, inference_memory_gb=9, memory_gb=2, training=False)  # the last two ignored
    [node] = spawned
    assert node.argv[2] == "slashcompute.inference.node"
    assert node.argv[node.argv.index("--memory-gb") + 1] == "9"
    assert "127.0.0.1" in node.argv[node.argv.index("--url") + 1]
    s = launcher.load_settings()
    assert (s.inference, s.contribute, s.training, s.memory_gb, s.inference_memory_gb) == (True, True, True, 8, 9)
    assert launcher.inference_pid_path.read_text().strip() == str(node.pid)   # recorded at spawn


def test_second_start_serving_click_does_not_start_a_second_node(tmp_path, monkeypatch):
    launcher, spawned = _serving_launcher(tmp_path, monkeypatch)
    monkeypatch.setattr(launcher, "_await_node", lambda proc, timeout=8.0: None)
    monkeypatch.setattr("slashcompute.launcher.controller.process_alive", lambda pid: pid == 6001)
    launcher.set_inference(True)
    launcher.set_inference(True)            # before the node itself got to write its pid
    assert len(spawned) == 1


def test_node_that_exits_on_start_says_why(tmp_path, monkeypatch):
    launcher, _ = _serving_launcher(tmp_path, monkeypatch)
    monkeypatch.setattr("slashcompute.launcher.controller.process_alive", lambda pid: False)
    log = launcher.log_dir / "inference.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("an older run's noise\n")

    def popen(argv, stdout=None, **_):
        stdout.write(b"llama.cpp RPC server not found (/x/rpc-server): run scripts/build_llama.sh\n")
        stdout.flush()
        return SimpleNamespace(pid=7001, argv=argv, poll=lambda: 1)

    launcher._popen = popen
    snap = launcher.set_inference(True)
    assert not snap.inference_running
    assert "exited with code 1" in snap.last_error and "RPC server not found" in snap.last_error
    assert "older run" not in snap.last_error


def test_node_that_cannot_join_says_why(tmp_path, monkeypatch):
    launcher, spawned = _serving_launcher(tmp_path, monkeypatch)
    status = tmp_path / "inference" / "status.json"
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text(json.dumps({"pid": 6001, "reason": "unsupported",
                                  "last_error": "the coordinator has no LLM inference"}))
    snap = launcher.set_inference(True)
    assert len(spawned) == 1 and "no LLM inference" in snap.last_error


def test_stop_serving_stops_only_the_node(tmp_path, monkeypatch):
    launcher, _ = _serving_launcher(tmp_path, monkeypatch)
    stopped = []
    monkeypatch.setattr(launcher, "_stop_inference", lambda wait=0.0: stopped.append(1))
    monkeypatch.setattr("slashcompute.launcher.controller.request_stop",
                        lambda paths: pytest.fail("stopped the training agent"))
    launcher.save_settings(LauncherSettings(inference=True))
    launcher.set_inference(False)
    assert stopped and launcher.load_settings().inference is False


def test_start_serving_route_passes_only_llm_settings(tmp_path, monkeypatch):
    launcher, _ = _launcher(tmp_path, CURRENT)
    calls = []
    monkeypatch.setattr(launcher, "set_inference",
                        lambda on, **changes: (calls.append((on, changes)), launcher.snapshot())[1])
    app = create_shell(launcher, stream_client=httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(404))))
    with TestClient(app, base_url=SHELL) as c:
        r = c.post("/api/inference", json={"on": True, "inference_memory_gb": 9, "memory_gb": 2,
                                           "training": False})
    assert r.status_code == 200 and calls == [(True, {"inference_memory_gb": 9})]


def test_static_assets_are_versioned_and_revalidated(tmp_path):
    """The app window kept running an old app.js after an update, so UI fixes never showed."""
    with TestClient(_shell(tmp_path, lambda r: httpx.Response(404)), base_url=SHELL) as c:
        page = c.get("/")
        assert page.headers["cache-control"] == "no-cache"
        v = re.search(r'/static/app\.js\?v=([0-9a-f]{12})"', page.text).group(1)
        assert f'/static/app.css?v={v}"' in page.text
        js = c.get(f"/static/app.js?v={v}")
        assert js.status_code == 200 and js.headers["cache-control"] == "no-cache"
        assert b'"/api/inference"' in js.content and b'id="l-why"' in page.content
