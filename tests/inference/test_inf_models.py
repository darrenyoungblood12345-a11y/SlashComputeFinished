"""Model registry without a price table, GGUF upload, and distribution to heads."""

import asyncio
import hashlib
from pathlib import Path

import httpx
import pytest

from inf_gguf_fixtures import tiny_gguf
from inf_harness import FakeNode, chat, fast_settings, heads_of, start_harness, upload, wait_for
from slashcompute.inference.coordinator import registry
from slashcompute.inference.node.agent import scan_models


@pytest.fixture
async def cluster(tmp_path):
    h = await start_harness(fast_settings(MODELS_DIR=str(tmp_path / "coord-models")), tmp=str(tmp_path / "nodes"))
    yield h
    await h.stop()


def reported_files(h, name: str) -> str:
    return h.conn.execute("SELECT gguf_files_json FROM nodes WHERE id=?", (h.ids[name],)).fetchone()[0]


async def models(h) -> list[str]:
    async with httpx.AsyncClient(base_url=h.url) as c:
        return [m["id"] for m in (await c.get("/v1/models")).json()["data"]]


# ------------------------------------------------------------ registry

async def test_any_parseable_gguf_a_head_reports_becomes_servable(tmp_path):
    h = await start_harness(fast_settings(), tmp=str(tmp_path))
    try:
        data = tiny_gguf()
        d = tmp_path / "head" / "models"
        d.mkdir(parents=True)
        (d / "tiny.gguf").write_bytes(data)
        (d / "broken.gguf").write_bytes(b"GGUF" + b"\0" * 8)
        files = scan_models([str(d)])
        assert [f["name"] for f in files] == ["tiny.gguf"]       # unreadable files are skipped on the node
        await h.add_node(FakeNode("head", 16, may_be_head=True), 0, gguf_files=files)
        row = h.conn.execute("SELECT * FROM models WHERE id='tiny.gguf'").fetchone()
        assert row["status"] == "ready" and row["layer_source"].startswith("node:") and row["flops_json"]
        assert await models(h) == ["tiny.gguf"]
        assert (await chat(h, "tiny.gguf")).status_code == 200
        # listed only while a head that has the file is online
        await h.drop("head")
        await asyncio.sleep(0.6)
        assert await models(h) == []
    finally:
        await h.stop()


async def test_a_bad_header_rejects_the_model_not_the_node(cluster):
    h = cluster
    bad = [{"name": "odd.gguf", "size": 10, "headers": [{"nope": True}]}]
    await h.add_node(FakeNode("head", 16, may_be_head=True), 0, gguf_files=bad)
    row = h.conn.execute("SELECT * FROM models WHERE id='odd.gguf'").fetchone()
    assert row["status"] == "rejected" and "header" in row["status_reason"]
    assert h.svc.online == {h.ids["head"]}


# ------------------------------------------------------------ upload + distribution

async def test_upload_is_validated(cluster):
    h = cluster
    assert (await upload(h, "../evil.gguf", tiny_gguf())).status_code == 400
    assert (await upload(h, "notes.txt", b"hello")).status_code == 400
    r = await upload(h, "junk.gguf", b"definitely not a gguf file")
    assert r.status_code == 422
    assert not any(Path(h.settings.MODELS_DIR).iterdir())       # nothing left behind


async def test_upload_is_pushed_to_heads_and_becomes_servable(cluster):
    h = cluster
    await h.add_node(FakeNode("head", 16, may_be_head=True), 0, gguf_files=[])
    await h.add_node(FakeNode("worker", 16), 1, gguf_files=[])
    data = tiny_gguf()
    r = await upload(h, "tiny.gguf", data)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sha256"] == hashlib.sha256(data).hexdigest() and body["size"] == len(data)
    assert body["pushed"] == 1                                   # only head-capable nodes get it
    got = Path(h.agents["head"].cfg.download_dir) / "tiny.gguf"
    await wait_for(lambda: got.exists())
    assert got.read_bytes() == data
    await wait_for(lambda: heads_of(h, "tiny.gguf") == ["head"])
    assert "tiny.gguf" in reported_files(h, "head")
    assert not (Path(h.agents["worker"].cfg.download_dir) / "tiny.gguf").exists()
    assert await models(h) == ["tiny.gguf"]
    r = await chat(h, "tiny.gguf")
    assert r.status_code == 200, r.text


async def test_a_head_with_the_same_file_skips_the_download_and_late_heads_get_it(cluster, tmp_path):
    h = cluster
    data = tiny_gguf()
    assert (await upload(h, "tiny.gguf", data)).status_code == 200
    # already on disk (same bytes) but not yet reported: verified by sha256, not downloaded again
    have = tmp_path / "nodes" / "copy" / "models"
    have.mkdir(parents=True)
    (have / "tiny.gguf").write_bytes(data)
    await h.add_node(FakeNode("copy", 16, may_be_head=True), 0, gguf_files=[])
    # a head that joins after the upload is sent the file when it registers
    await h.add_node(FakeNode("late", 16, may_be_head=True), 1, gguf_files=[])

    await wait_for(lambda: heads_of(h, "tiny.gguf") == ["copy", "late"])
    assert "tiny.gguf" in reported_files(h, "copy") and "tiny.gguf" in reported_files(h, "late")
    assert not (Path(h.agents["copy"].cfg.download_dir) / "tiny.gguf").exists()
    assert (Path(h.agents["late"].cfg.download_dir) / "tiny.gguf").read_bytes() == data
    async with httpx.AsyncClient(base_url=h.node_url) as c:
        st = (await c.get("/status")).json()
    (m,) = [m for m in st["models"] if m["id"] == "tiny.gguf"]
    assert m["uploaded"] and m["servable"] and sorted(m["heads"]) == ["copy", "late"]


def pipeline_states(h) -> list[str]:
    return [r["state"] for r in h.conn.execute("SELECT state FROM pipelines ORDER BY created_at")]


def hold_downloads(agent) -> asyncio.Event:
    """This Mac's downloads wait until the returned event is set."""
    gate, real = asyncio.Event(), agent.download_model

    async def held(p):
        await gate.wait()
        return await real(p)

    agent.download_model = held
    return gate


def other_model(data: bytes) -> bytes:
    """Another model with the same header and size: only its bytes (its sha256) tell the two apart."""
    return data[:-1] + b"\x01"


async def test_a_different_model_uploaded_under_the_same_name_replaces_the_heads_copy(cluster):
    """The head was skipped because it already reported a file of that name: it went on loading the old
    model while the pool listed and charged for the new one."""
    h = cluster
    head = await h.add_node(FakeNode("head", 16, may_be_head=True), 0, gguf_files=[])
    old = tiny_gguf()
    new = other_model(old)
    assert (await upload(h, "tiny.gguf", old)).status_code == 200
    got = Path(head.cfg.download_dir) / "tiny.gguf"
    await wait_for(lambda: heads_of(h, "tiny.gguf") == ["head"])
    assert (await chat(h, "tiny.gguf")).status_code == 200            # a pipeline holds the old model
    assert pipeline_states(h) == ["active"]

    r = await upload(h, "tiny.gguf", old)                             # the same file again: nothing to do
    assert r.status_code == 200 and r.json()["pushed"] == 0
    assert heads_of(h, "tiny.gguf") == ["head"] and pipeline_states(h) == ["active"]

    gate = hold_downloads(head)
    r = await upload(h, "tiny.gguf", new)
    assert r.status_code == 200 and r.json()["pushed"] == 1, r.text   # sent although the name is there
    # the head still has only the old bytes: it must not serve them as the new model
    assert heads_of(h, "tiny.gguf") == [] and await models(h) == []
    assert "tiny.gguf" in reported_files(h, "head") and got.read_bytes() == old
    await wait_for(lambda: pipeline_states(h) == ["stopped"])         # and the pipeline on the old model is gone
    gate.set()
    await wait_for(lambda: heads_of(h, "tiny.gguf") == ["head"])
    assert got.read_bytes() == new
    assert await models(h) == ["tiny.gguf"]
    assert (await chat(h, "tiny.gguf")).status_code == 200
    assert pipeline_states(h) == ["stopped", "active"]                # loaded again, from the new file


@pytest.mark.parametrize("folder", ["models", "downloads"])
async def test_a_mac_whose_own_file_has_that_name_does_not_head_another_upload(cluster, tmp_path, folder):
    """Its engine loads the copy in its own models folder: a download next to it would never be used. And a
    file in the app's folder that this pool did not send (another pool's, the Mac's own upload when it
    hosts) is not this pool's to write over."""
    h = cluster
    own = tmp_path / "nodes" / "keeper" / folder
    own.mkdir(parents=True)
    mine = tiny_gguf()
    (own / "tiny.gguf").write_bytes(mine)
    keeper = await h.add_node(FakeNode("keeper", 16, may_be_head=True), 0, gguf_files=scan_models([str(own)]))
    assert heads_of(h, "tiny.gguf") == ["keeper"]                     # its own model, known from its report
    await h.add_node(FakeNode("fresh", 16, may_be_head=True), 1, gguf_files=[])

    r = await upload(h, "tiny.gguf", other_model(mine))
    assert r.status_code == 200 and r.json()["pushed"] == 2, r.text
    await wait_for(lambda: heads_of(h, "tiny.gguf") == ["fresh"] and not h.svc.pushing)
    assert heads_of(h, "tiny.gguf") == ["fresh"]                      # keeper refused: it never heads the upload
    assert (own / "tiny.gguf").read_bytes() == mine                   # its own file is left alone
    assert sorted(p.name for p in Path(keeper.cfg.models_dir).parent.rglob("*.gguf*")) == ["tiny.gguf"]
    assert "tiny.gguf" in reported_files(h, "keeper")                 # and is still shown as on that Mac


async def test_a_mac_back_from_offline_with_a_replaced_upload_fetches_the_new_one(cluster):
    h = cluster
    mac = await h.add_node(FakeNode("mac", 16, may_be_head=True), 0, gguf_files=[])
    old = tiny_gguf()
    assert (await upload(h, "tiny.gguf", old)).status_code == 200
    await wait_for(lambda: heads_of(h, "tiny.gguf") == ["mac"])
    await mac.cancel_tasks()                                          # the Mac sleeps
    await wait_for(lambda: not registry.holders(h.conn, h.settings, "tiny.gguf"))
    r = await upload(h, "tiny.gguf", other_model(old))
    assert r.status_code == 200 and r.json()["pushed"] == 0           # nobody online to send it to
    for loop in (mac.command_loop, mac.heartbeat_loop, mac.models_loop):   # it wakes, with the old file
        mac._spawn(loop())
    await wait_for(lambda: heads_of(h, "tiny.gguf") == ["mac"])
    assert (Path(mac.cfg.download_dir) / "tiny.gguf").read_bytes() == other_model(old)


async def test_heads_confirm_the_uploads_they_hold_after_a_restart_without_a_record(tmp_path):
    """A pool upgraded from before the per-head record: the heads are asked once and serve again."""
    settings = fast_settings(DB_PATH=str(tmp_path / "inference.sqlite3"), MODELS_DIR=str(tmp_path / "coord-models"),
                             OFFLINE_AFTER_SECONDS=5)
    h = await start_harness(settings, tmp=str(tmp_path / "nodes"))
    try:
        await h.add_node(FakeNode("head", 16, may_be_head=True), 0, gguf_files=[])
        assert (await upload(h, "tiny.gguf", tiny_gguf())).status_code == 200
        await wait_for(lambda: heads_of(h, "tiny.gguf") == ["head"])
        port = int(h.url.rsplit(":", 1)[1])
        h.server.should_exit = True                                   # the pool restarts; the Mac keeps running
        await h._task
        h.conn.execute("DELETE FROM node_uploads")
        await h.start(port)
        await wait_for(lambda: heads_of(h, "tiny.gguf") == ["head"])
        assert (await chat(h, "tiny.gguf")).status_code == 200
    finally:
        await h.stop()


async def test_files_are_only_served_to_nodes(cluster):
    h = cluster
    assert (await upload(h, "tiny.gguf", tiny_gguf())).status_code == 200
    async with httpx.AsyncClient(base_url=h.node_url) as c:
        assert (await c.get("/models/files/tiny.gguf")).status_code == 401
        assert (await c.delete("/models/tiny.gguf")).status_code == 200
        assert (await c.get("/status")).json()["models"] == []          # removed, and no Mac holds a copy
        assert not (Path(h.settings.MODELS_DIR) / "tiny.gguf").exists()
        assert (await c.delete("/models/tiny.gguf")).status_code == 404   # nothing left to remove


async def test_shared_token_guards_uploads_and_chat(tmp_path):
    h = await start_harness(fast_settings(TOKEN="s3cret", MODELS_DIR=str(tmp_path / "m")))
    try:
        assert (await upload(h, "tiny.gguf", tiny_gguf())).status_code == 401
        async with httpx.AsyncClient(base_url=h.url) as c:
            assert (await c.get("/v1/models")).status_code == 401
            ok = await c.get("/v1/models", headers={"X-Inference-Token": "s3cret"})
            assert ok.status_code == 200
    finally:
        await h.stop()


async def test_a_paused_head_does_not_make_its_models_servable(tmp_path):
    """A head busy training refuses chats: listing its models enabled Send for a chat that failed."""
    from slashcompute.inference.coordinator import registry
    from slashcompute.inference.coordinator.status import snapshot

    h = await start_harness(fast_settings(), tmp=str(tmp_path))
    try:
        d = tmp_path / "head" / "models"
        d.mkdir(parents=True)
        (d / "tiny.gguf").write_bytes(tiny_gguf())
        await h.add_node(FakeNode("head", 16, may_be_head=True), 0, gguf_files=scan_models([str(d)]))
        assert [r["id"] for r in registry.servable_models(h.conn, h.settings)] == ["tiny.gguf"]
        [m] = snapshot(h.conn, h.settings)["models"]
        assert m["servable"] and isinstance(m["min_memory_gb"], int) and m["min_memory_gb"] >= 2
        h.conn.execute("UPDATE nodes SET available=0")
        assert registry.servable_models(h.conn, h.settings) == []
        assert not snapshot(h.conn, h.settings)["models"][0]["servable"]
    finally:
        await h.stop()
