"""Model registry without a price table, GGUF upload, and distribution to heads."""

import asyncio
import hashlib
from pathlib import Path

import httpx
import pytest

from inf_gguf_fixtures import tiny_gguf
from inf_harness import FakeNode, chat, fast_settings, start_harness, upload, wait_for
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
    await wait_for(lambda: "tiny.gguf" in reported_files(h, "head"))
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

    await wait_for(lambda: "tiny.gguf" in reported_files(h, "copy") and "tiny.gguf" in reported_files(h, "late"))
    assert not (Path(h.agents["copy"].cfg.download_dir) / "tiny.gguf").exists()
    assert (Path(h.agents["late"].cfg.download_dir) / "tiny.gguf").read_bytes() == data
    async with httpx.AsyncClient(base_url=h.node_url) as c:
        st = (await c.get("/status")).json()
    (m,) = [m for m in st["models"] if m["id"] == "tiny.gguf"]
    assert m["uploaded"] and m["servable"] and sorted(m["heads"]) == ["copy", "late"]


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
