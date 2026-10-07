"""Unload, the Serve switch and Remove for models in the LLMs tab (POST /inference/models/*)."""

import asyncio
import errno
import os
import threading
import time
from pathlib import Path

import httpx
import pytest

from inf_gguf_fixtures import tiny_gguf
from inf_harness import FakeNode, chat, fast_settings, start_harness, upload, wait_for
from slashcompute.inference.coordinator import nodes
from slashcompute.inference.coordinator.layers import synthetic_layout
from slashcompute.inference.coordinator.pipelines import serve
from slashcompute.inference.node import agent as agent_mod
from slashcompute.inference.node.agent import scan_models
from slashcompute.inference.node.fake_engine import FakeEngine

GB = 10 ** 9
QWEN = "Qwen3.8-27B-UD-Q4_K_XL.gguf"
TINY = "tiny.gguf"


def qwen_layout():
    return synthetic_layout(QWEN, n_layers=64, total_bytes=int(17.56 * GB), head_bytes=1 * GB)


class QuickLoad(FakeEngine):
    """Real-time tokens (time_scale=1), but a model loads in 0.2 s instead of 2 s."""

    def __init__(self, *a, **kw):
        super().__init__(*a, load_seconds=0.2, **kw)


class Loading(FakeEngine):
    """Loads take 2 s; counts the loads that started and the ones that returned."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.started = self.finished = 0

    async def start_head(self, pipeline_id, spec):
        self.started += 1
        try:
            return await super().start_head(pipeline_id, spec)
        finally:
            self.finished += 1


async def post(h, path: str, body) -> httpx.Response:
    async with httpx.AsyncClient(base_url=h.node_url, timeout=30) as c:
        return await c.post(path, json=body)


async def status(h) -> dict:
    async with httpx.AsyncClient(base_url=h.node_url, timeout=30) as c:
        return (await c.get("/status")).json()


async def model(h, name: str):
    return next((m for m in (await status(h))["models"] if m["id"] == name), None)


async def listed(h) -> list[str]:
    async with httpx.AsyncClient(base_url=h.url) as c:
        return [m["id"] for m in (await c.get("/v1/models")).json()["data"]]


def model_status(h, name: str) -> str:
    return h.conn.execute("SELECT status FROM models WHERE id=?", (name,)).fetchone()[0]


def reported_files(h, name: str) -> str:
    return h.conn.execute("SELECT gguf_files_json FROM nodes WHERE id=?", (h.ids[name],)).fetchone()[0]


def pipeline_states(h) -> list[str]:
    return [r["state"] for r in h.conn.execute("SELECT state FROM pipelines ORDER BY created_at")]


def idle(h) -> bool:
    """No Mac holds a pipeline or runs a llama.cpp process for one any more."""
    return all(not a.in_use and not a.engine.heads and not a.engine.workers for a in h.agents.values())


async def long_chat(h, name: str = QWEN, max_tokens: int = 150) -> asyncio.Task:
    """A reply that takes seconds (time_scale=1): returned once a head is generating it
    (the set-up of test_inf_pipelines' disconnect tests)."""
    task = asyncio.create_task(chat(h, name, content="tell me a long story", max_tokens=max_tokens))
    await wait_for(lambda: task.done() or any(a.jobs for a in h.agents.values()))
    assert not task.done(), task.result().text
    return task


def node_dir(h, name: str, kind: str = "models") -> Path:
    """FakeNode `name`'s own models folder ("models") or the app's downloads ("downloads")."""
    return Path(h.tmp) / name / kind


async def holding(h, name: str, index: int, where: str = "models", **cfg):
    """A head with tiny.gguf on disk, in its own models folder or the app's downloads."""
    d = node_dir(h, name, where)
    d.mkdir(parents=True, exist_ok=True)
    (d / TINY).write_bytes(tiny_gguf())
    return await h.add_node(FakeNode(name, 16, may_be_head=True), index, gguf_files=scan_models([str(d)]), **cfg)


@pytest.fixture
def trash(tmp_path, monkeypatch):
    """A Trash under tmp (conftest makes the real one refuse every move)."""
    bin_ = tmp_path / "Trash"
    bin_.mkdir()
    monkeypatch.setattr(agent_mod, "move_to_trash", lambda path: os.replace(path, bin_ / path.name))
    return bin_


@pytest.fixture
async def two_node(tmp_path):
    h = await start_harness(fast_settings(PIPELINE_IDLE_SECONDS=30), tmp=str(tmp_path / "nodes"))
    h.add_model(qwen_layout())
    await h.add_nodes([FakeNode("head", 16, may_be_head=True, files=(QWEN,)), FakeNode("worker", 16, gen_score=1.2)])
    yield h
    await h.stop()


@pytest.fixture
async def slow(tmp_path):
    """Two nodes generating in real time, so a test can act while a reply is in progress."""
    h = await start_harness(fast_settings(PIPELINE_IDLE_SECONDS=30), time_scale=1.0, tmp=str(tmp_path / "nodes"))
    h.add_model(qwen_layout())
    await h.add_nodes([FakeNode("head", 16, may_be_head=True, files=(QWEN,)), FakeNode("worker", 16, gen_score=1.2)],
                      engine_cls=QuickLoad)
    assert (await chat(h, QWEN, max_tokens=2)).status_code == 200   # warm: the pipeline is active
    yield h
    await h.stop()


@pytest.fixture
async def cluster(tmp_path):
    h = await start_harness(fast_settings(MODELS_DIR=str(tmp_path / "coord-models"), PIPELINE_IDLE_SECONDS=30),
                            tmp=str(tmp_path / "nodes"))
    yield h
    await h.stop()


# ------------------------------------------------------------ Unload


async def test_unload_frees_every_pipeline_and_the_next_chat_loads_it_again(two_node):
    h = two_node
    first = (await chat(h, QWEN)).json()["network"]["pipeline_id"]
    assert (await model(h, QWEN))["load"] == "loaded"
    r = await post(h, "/models/unload", {"model": QWEN})
    assert r.status_code == 200 and r.json() == {"ok": True, "pipelines": 1}
    await wait_for(lambda: idle(h))
    assert pipeline_states(h) == ["stopped"]
    assert (await model(h, QWEN))["load"] is None
    assert await listed(h) == [QWEN]                                 # not sticky
    r = await chat(h, QWEN)
    assert r.status_code == 200 and r.json()["network"]["pipeline_id"] != first
    assert (await model(h, QWEN))["load"] == "loaded"


async def test_unload_lets_the_reply_in_progress_finish(slow):
    h = slow
    reply = await long_chat(h)
    r = await post(h, "/models/unload", {"model": QWEN})
    assert r.json() == {"ok": True, "pipelines": 1}
    assert (await model(h, QWEN))["load"] == "unloading"             # at once, while the reply runs
    assert (await post(h, "/models/unload", {"model": QWEN})).json()["pipelines"] == 0   # already on its way
    res = await reply
    assert res.status_code == 200 and res.json()["usage"]["completion_tokens"] == 150
    await wait_for(lambda: idle(h))
    assert pipeline_states(h) == ["stopped"]
    assert (await model(h, QWEN))["load"] is None


# ------------------------------------------------------------ Serve switch


async def test_stop_serving_unloads_hides_and_refuses_until_serve(two_node):
    h = two_node
    assert (await chat(h, QWEN)).status_code == 200
    r = await post(h, "/models/serving", {"model": QWEN, "on": False})
    assert r.status_code == 200 and r.json() == {"ok": True, "serving": False, "pipelines": 1}
    m = await model(h, QWEN)
    assert m["status"] == "disabled" and not m["servable"] and m["restorable"]
    assert await listed(h) == []
    for stream in (False, True):
        r = await chat(h, QWEN, stream=stream)
        assert r.status_code == 404 and "not being served" in r.json()["detail"]
    await wait_for(lambda: idle(h))
    for body in ({"model": QWEN, "on": "no"}, {"model": QWEN}, {"model": QWEN, "on": None}):
        assert (await post(h, "/models/serving", body)).status_code == 400
    assert model_status(h, QWEN) == "disabled"
    r = await post(h, "/models/serving", {"model": QWEN, "on": "true"})
    assert r.json() == {"ok": True, "serving": True, "pipelines": 0}
    assert await listed(h) == [QWEN] and (await chat(h, QWEN)).status_code == 200
    assert (await post(h, "/models/serving", {"model": QWEN, "on": True})).json()["serving"] is True   # no-op


async def test_stop_serving_while_loading_refuses_the_waiting_chat_and_frees_the_head(tmp_path):
    h = await start_harness(fast_settings(PIPELINE_IDLE_SECONDS=30), time_scale=1.0, tmp=str(tmp_path / "nodes"))
    try:
        h.add_model(qwen_layout())
        await h.add_nodes([FakeNode("head", 16, may_be_head=True, files=(QWEN,)), FakeNode("worker", 16)],
                          engine_cls=Loading)
        head = h.agents["head"]
        waiting = asyncio.create_task(chat(h, QWEN, max_tokens=4))
        await wait_for(lambda: head.engine.started == 1)              # llama-server is loading the model
        t0 = time.monotonic()
        r = await post(h, "/models/serving", {"model": QWEN, "on": False})
        assert r.json() == {"ok": True, "serving": False, "pipelines": 1}
        res = await waiting
        assert res.status_code == 404 and "not being served" in res.text
        assert time.monotonic() - t0 < 1.5 and head.engine.finished == 0   # refused before the load ended
        assert pipeline_states(h) == ["broken"]
        await wait_for(lambda: head.engine.finished == 1)
        # the load that finished after its pipeline broke is stopped, not left holding memory
        await asyncio.sleep(2 * h.settings.HEARTBEAT_SECONDS)
        await wait_for(lambda: idle(h))
    finally:
        await h.stop()


async def test_a_chat_waiting_for_memory_is_refused_within_a_second_of_stop_serving(tmp_path):
    h = await start_harness(fast_settings(PIPELINE_IDLE_SECONDS=30), time_scale=1.0, tmp=str(tmp_path / "nodes"))
    a, b = "a-10b.gguf", "b-10b.gguf"
    try:
        for name in (a, b):   # one Mac that fits one of them at a time
            h.add_model(synthetic_layout(name, n_layers=32, total_bytes=10 * GB, head_bytes=GB // 2))
        await h.add_nodes([FakeNode("solo", 16, may_be_head=True, files=(a, b))], engine_cls=QuickLoad)
        assert (await chat(h, a, max_tokens=2)).status_code == 200
        reply = await long_chat(h, a, max_tokens=500)
        assert (await post(h, "/models/unload", {"model": a})).json()["pipelines"] == 1   # a drains; b must wait
        waiting = asyncio.create_task(chat(h, b, max_tokens=4))
        await wait_for(lambda: h.conn.execute("SELECT COUNT(*) FROM jobs WHERE model_id=?", (b,)).fetchone()[0])
        await asyncio.sleep(0.3)                                       # inside _create's wait for a's drain
        assert not waiting.done()
        t0 = time.monotonic()
        assert (await post(h, "/models/serving", {"model": b, "on": False})).json()["pipelines"] == 0
        res = await waiting
        assert res.status_code == 404 and "not being served" in res.text
        assert time.monotonic() - t0 < 2.5 and not reply.done()      # long before a's reply (and drain) ended
        assert (await reply).status_code == 200
    finally:
        await h.stop()


async def test_a_live_pipeline_is_not_reused_once_the_model_is_stopped(two_node):
    h = two_node
    assert (await chat(h, QWEN)).status_code == 200
    h.conn.execute("UPDATE models SET status='disabled' WHERE id=?", (QWEN,))   # the switch, before any unload
    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}
    events = [ev async for ev in serve(h.svc.mgr, QWEN, body, 4096, None, False)]
    assert [ev["type"] for ev in events] == ["error"]
    assert events[0]["status"] == 404 and "not being served" in events[0]["error"]
    assert [rt.state for rt in h.svc.mgr.runtimes.values()] == ["active"]   # refused, not reused


async def test_stop_serving_survives_node_reports_uploads_and_a_restart(tmp_path):
    settings = fast_settings(DB_PATH=str(tmp_path / "inference.sqlite3"), MODELS_DIR=str(tmp_path / "coord-models"))
    h = await start_harness(settings, tmp=str(tmp_path / "nodes"))
    try:
        head = await h.add_node(FakeNode("head", 16, may_be_head=True), 0, gguf_files=[])
        assert (await upload(h, TINY, tiny_gguf())).status_code == 200
        await wait_for(lambda: TINY in reported_files(h, "head"))
        assert (await chat(h, TINY)).status_code == 200
        assert (await post(h, "/models/serving", {"model": TINY, "on": False})).json()["serving"] is False
        await head.register()                                          # the Mac re-registers with its files
        await head._post("/nodes/models", {"files": scan_models(head.cfg.model_dirs)})
        assert (await upload(h, TINY, tiny_gguf())).status_code == 200   # and the model is uploaded again
        assert model_status(h, TINY) == "disabled"
        r = await chat(h, TINY)
        assert r.status_code == 404 and "not being served" in r.json()["detail"]
    finally:
        await h.stop()
    h = await start_harness(settings, tmp=str(tmp_path / "nodes-again"))   # restarted on the same database
    try:
        await h.add_node(FakeNode("head", 16, may_be_head=True), 0,
                         gguf_files=scan_models([str(tmp_path / "nodes" / "head" / "downloads")]))
        assert model_status(h, TINY) == "disabled" and await listed(h) == []
        assert (await chat(h, TINY)).status_code == 404
        assert (await post(h, "/models/serving", {"model": TINY, "on": True})).json()["serving"] is True
        assert (await chat(h, TINY)).status_code == 200
    finally:
        await h.stop()


# ------------------------------------------------------------ Remove


async def test_remove_clears_every_copy(cluster, trash):
    h = cluster
    await h.add_node(FakeNode("A", 16, may_be_head=True), 0, gguf_files=[])
    assert (await upload(h, TINY, tiny_gguf())).status_code == 200            # pushed to A's downloads
    await wait_for(lambda: TINY in reported_files(h, "A"))
    await holding(h, "B", 1, trash_removed=True)                              # in B's own models folder
    assert (await chat(h, TINY)).status_code == 200
    r = await post(h, "/models/remove", {"model": TINY})
    assert r.status_code == 200
    assert r.json() == {"ok": True, "model": TINY, "macs": ["A", "B"], "offline": [], "uploaded": True}
    assert model_status(h, TINY) == "removed"
    await wait_for(lambda: not h.svc.removals)
    assert not (Path(h.settings.MODELS_DIR) / TINY).exists()
    assert not (node_dir(h, "A", "downloads") / TINY).exists()
    assert not (node_dir(h, "B") / TINY).exists() and [p.name for p in trash.iterdir()] == [TINY]
    assert reported_files(h, "A") == reported_files(h, "B") == "[]"
    assert await model(h, TINY) is None and await listed(h) == []
    r = await chat(h, TINY)
    assert r.status_code == 404 and "removed" in r.json()["detail"]
    await wait_for(lambda: idle(h))


async def test_a_host_mac_sharing_the_upload_folder_removes_without_errors(cluster, trash):
    h = cluster
    # the host's node downloads into the coordinator's own models folder (~/.slashcompute/models)
    await h.add_node(FakeNode("host", 16, may_be_head=True), 0, gguf_files=[], download_dir=h.settings.MODELS_DIR,
                     trash_removed=True)
    assert (await upload(h, TINY, tiny_gguf())).status_code == 200
    await wait_for(lambda: TINY in reported_files(h, "host"))
    assert (await post(h, "/models/remove", {"model": TINY})).json()["macs"] == ["host"]
    await wait_for(lambda: not h.svc.removing(TINY))
    assert h.svc.removals == {}                                               # no error, nothing kept
    assert reported_files(h, "host") == "[]" and not any(trash.iterdir())
    assert not any(Path(h.settings.MODELS_DIR).iterdir())


async def test_remove_mid_reply_lets_the_reply_finish(slow):
    h = slow
    records = len(h.svc.accounting.records)
    reply = await long_chat(h)
    r = await post(h, "/models/remove", {"model": QWEN})
    assert r.json()["macs"] == ["head"]
    m = await model(h, QWEN)
    assert m["status"] == "removed" and m["load"] == "unloading"
    res = await reply
    assert res.status_code == 200 and res.json()["usage"]["completion_tokens"] == 150
    job = h.conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 1").fetchone()
    assert job["state"] == "done" and job["predicted_n"] == 150
    assert len(h.svc.accounting.records) == records + 1                       # credited from the tombstone
    await wait_for(lambda: idle(h))
    assert await model(h, QWEN) is None                                       # gone once it drained


async def test_a_removed_model_stays_gone_until_added_back_or_uploaded_again(cluster, monkeypatch):
    h = cluster
    keeper = await holding(h, "keeper", 0)                    # keeps its models-folder copy (--keep-removed)
    assert (await post(h, "/models/remove", {"model": TINY})).json()["macs"] == ["keeper"]
    await wait_for(lambda: not h.svc.removing(TINY))
    # stale reports from the Mac that still has the file don't bring it back
    await keeper.register()
    await keeper._post("/nodes/models", {"files": scan_models(keeper.cfg.model_dirs)})
    m = await model(h, TINY)
    assert m["status"] == "removed" and m["restorable"] and not m["servable"]
    assert m["held_by"] == [{"name": "keeper", "online": True, "removing": False, "error": None, "kept": True}]
    assert await listed(h) == []
    r = await chat(h, TINY)
    assert r.status_code == 404 and "removed" in r.json()["detail"]
    assert (await post(h, "/models/serving", {"model": TINY, "on": False})).status_code == 409   # nothing to stop

    # Add back
    assert (await post(h, "/models/serving", {"model": TINY, "on": True})).json() == \
        {"ok": True, "serving": True, "pipelines": 0}
    assert await listed(h) == [TINY] and (await chat(h, TINY)).status_code == 200

    # removed again, then uploaded again
    assert (await post(h, "/models/remove", {"model": TINY})).status_code == 200
    await wait_for(lambda: not h.svc.removing(TINY))
    assert (await upload(h, TINY, tiny_gguf())).status_code == 200
    m = await model(h, TINY)
    assert m["status"] == "ready" and m["uploaded"] and m["held_by"][0]["kept"] is False

    # while a Mac is still removing it, an upload (or Add back) is refused: the Mac would delete it again
    gate = threading.Event()
    real = agent_mod.remove_model_files

    def slow_remove(*a):
        gate.wait(10)
        return real(*a)

    monkeypatch.setattr(agent_mod, "remove_model_files", slow_remove)
    try:
        assert (await post(h, "/models/remove", {"model": TINY})).status_code == 200
        assert (await model(h, TINY))["held_by"][0]["removing"] is True
        r = await upload(h, TINY, tiny_gguf())
        assert r.status_code == 409 and "still being removed" in r.json()["detail"]
        assert (await post(h, "/models/serving", {"model": TINY, "on": True})).status_code == 409
    finally:
        gate.set()
    await wait_for(lambda: not h.svc.removing(TINY))
    assert (await upload(h, TINY, tiny_gguf())).status_code == 200
    assert model_status(h, TINY) == "ready"


async def test_a_layer_table_reported_for_a_removed_model_is_kept_but_the_model_stays_removed(cluster):
    h = cluster
    # an older Mac that never answers remove_model reports the model by name only (no header yet)
    old = await h.add_node(FakeNode("old", 16, may_be_head=True, files=(TINY,)), 0)
    real = old.dispatch

    async def older(kind, p):
        if kind == "remove_model":
            raise ValueError(f"unknown command {kind}")
        return await real(kind, p)

    old.dispatch = older
    assert (await post(h, "/models/remove", {"model": TINY})).json()["macs"] == ["old"]
    await wait_for(lambda: not h.svc.removing(TINY))
    m = await model(h, TINY)
    assert m["status"] == "removed" and not m["restorable"]
    assert (await post(h, "/models/serving", {"model": TINY, "on": True})).status_code == 409   # upload it again
    d = node_dir(h, "old")
    d.mkdir(parents=True)
    (d / TINY).write_bytes(tiny_gguf())
    await old._post("/nodes/models", {"files": scan_models([str(d)])})
    m = await model(h, TINY)
    assert m["status"] == "removed" and m["restorable"]
    assert (await post(h, "/models/serving", {"model": TINY, "on": True})).json()["serving"] is True
    assert model_status(h, TINY) == "ready" and (await chat(h, TINY)).status_code == 200


async def test_failed_removals_are_shown_per_mac_and_remove_again_clears_them(cluster, monkeypatch, tmp_path):
    h = cluster
    old = await holding(h, "old", 0, where="downloads")
    real = old.dispatch
    updated = False

    async def older(kind, p):
        if kind == "remove_model" and not updated:
            raise ValueError(f"unknown command {kind}")       # what a node from before remove_model answers
        return await real(kind, p)

    old.dispatch = older
    await holding(h, "B", 1, trash_removed=True)

    def locked(path):
        raise PermissionError(errno.EACCES, f"Permission denied: {path}")

    monkeypatch.setattr(agent_mod, "move_to_trash", locked)
    r = await post(h, "/models/remove", {"model": TINY})
    assert r.json()["macs"] == ["old", "B"]
    await wait_for(lambda: not h.svc.removing(TINY))
    m = await model(h, TINY)
    held = {x["name"]: x for x in m["held_by"]}
    assert held["old"]["error"].startswith("runs an older /compute") and TINY in held["old"]["error"]
    assert held["B"]["error"] == f"{TINY}: Permission denied: {TINY}"            # names, never folders
    assert str(tmp_path) not in str(await status(h))
    assert (node_dir(h, "old", "downloads") / TINY).exists() and (node_dir(h, "B") / TINY).exists()

    updated = True                                                              # the Mac is updated...
    bin_ = tmp_path / "Trash"
    bin_.mkdir()
    monkeypatch.setattr(agent_mod, "move_to_trash", lambda path: os.replace(path, bin_ / path.name))  # ...Trash ok
    r = await post(h, "/models/remove", {"model": TINY})                         # Remove again
    assert r.status_code == 200 and r.json()["macs"] == ["old", "B"] and r.json()["uploaded"] is False
    await wait_for(lambda: not h.svc.removing(TINY))
    assert h.svc.removals == {} and await model(h, TINY) is None
    assert not (node_dir(h, "old", "downloads") / TINY).exists() and (bin_ / TINY).exists()
    assert (await post(h, "/models/remove", {"model": TINY})).status_code == 404   # nothing left anywhere


async def test_a_mac_that_keeps_removed_models_and_an_offline_one_are_listed(cluster, trash):
    h = cluster
    await holding(h, "keeper", 0)                                    # started with --keep-removed (the default)
    await holding(h, "away", 1, trash_removed=True)
    await h.drop("away")
    await wait_for(lambda: not nodes.is_online(
        h.conn.execute("SELECT * FROM nodes WHERE id=?", (h.ids["away"],)).fetchone(), h.settings))
    r = await post(h, "/models/remove", {"model": TINY})
    assert r.json() == {"ok": True, "model": TINY, "macs": ["keeper"], "offline": ["away"], "uploaded": False}
    await wait_for(lambda: not h.svc.removing(TINY))
    assert (node_dir(h, "keeper") / TINY).exists() and not any(trash.iterdir())
    m = await model(h, TINY)
    assert m["status"] == "removed"
    assert m["held_by"] == [{"name": "keeper", "online": True, "removing": False, "error": None, "kept": True},
                            {"name": "away", "online": False, "removing": False, "error": None, "kept": False}]


async def test_the_older_uis_delete_never_moves_files_to_the_trash(cluster, trash):
    h = cluster
    await holding(h, "B", 0, trash_removed=True)
    async with httpx.AsyncClient(base_url=h.node_url) as c:
        assert (await c.delete(f"/models/{TINY}")).status_code == 200
    await wait_for(lambda: not h.svc.removing(TINY))
    assert (node_dir(h, "B") / TINY).exists() and not any(trash.iterdir())
    assert (await model(h, TINY))["held_by"][0]["kept"] is True


async def test_bad_requests(two_node):
    h = two_node
    async with httpx.AsyncClient(base_url=h.node_url) as c:
        assert (await c.post("/models/unload", content=b"not json")).status_code == 400
        assert (await c.post("/models/unload", json=[QWEN])).status_code == 400
    for path in ("/models/unload", "/models/serving", "/models/remove"):
        assert (await post(h, path, {"model": 5, "on": True})).status_code == 400
        assert (await post(h, path, {"on": True})).status_code == 400
        r = await post(h, path, {"model": "missing.gguf", "on": True})
        assert r.status_code == 404 and r.json()["detail"]
    for name in ("../x.gguf", ".x.gguf", "x.txt"):
        assert (await post(h, "/models/remove", {"model": name})).status_code == 400
    assert model_status(h, QWEN) == "ready"
