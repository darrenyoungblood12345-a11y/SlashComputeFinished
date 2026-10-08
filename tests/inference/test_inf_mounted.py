"""Inference mounted inside the real /compute coordinator: middleware, lifespan, FLOP credits, training priority."""

import asyncio
import json
from dataclasses import replace

import httpx
import pytest

from inf_harness import FakeNode, chat, fast_settings, start_harness
from slashcompute.common.config import EngineConfig
from slashcompute.coordinator.app import create_app
from fastapi import HTTPException

from slashcompute.inference.coordinator.layers import synthetic_layout
from slashcompute.inference.coordinator.service import ctx_for
from slashcompute.inference.node.agent import training_busy
from slashcompute.inference.node.engine import EngineError
from slashcompute.inference.node.fake_engine import FakeEngine

GB = 10 ** 9
QWEN = "Qwen3.8-27B-UD-Q4_K_XL.gguf"


@pytest.fixture(autouse=True)
def fast_hash(monkeypatch):
    monkeypatch.setattr("slashcompute.community.auth.ITERATIONS", 1)


@pytest.fixture
async def pool(tmp_path, request):
    cfg = EngineConfig(home=tmp_path / "home", scheduler_tick_s=0.05, verify_rate=0.0,
                       public_pool=getattr(request, "param", False))
    h = await start_harness(fast_settings(PIPELINE_IDLE_SECONDS=30),
                            app_factory=lambda: create_app(cfg, inference=fast_settings(PIPELINE_IDLE_SECONDS=30)),
                            tmp=str(tmp_path / "nodes"))
    h.add_model(synthetic_layout(QWEN, n_layers=64, total_bytes=int(17.56 * GB), head_bytes=1 * GB))
    yield h
    await h.stop()


def core(h):
    return h.app.state.core


def account(h, email, terms=True, balance=0.0):
    auth = core(h).auth
    auth.register(email, "password1", email.split("@")[0])
    user, token = auth.login(email, "password1")
    if terms:
        user = auth.accept_terms(user)
    if balance:
        core(h).credits.post(user.id, "earn", balance)
    return user, token


async def two_nodes(h, **kw):
    await h.add_node(FakeNode("head", 16, may_be_head=True, files=(QWEN,)), 0, **kw)
    await h.add_node(FakeNode("worker", 16, gen_score=1.2), 1, **kw)


async def test_streaming_chat_through_the_main_coordinator(pool):
    h = pool
    await two_nodes(h)
    r = await chat(h, QWEN, max_tokens=12, stream=True)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/event-stream")
    lines = [l for l in r.text.splitlines() if l.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(l[6:]) for l in lines[:-2]]
    assert len(chunks) == 12 and all(c["choices"][0]["delta"]["content"] for c in chunks)
    net = json.loads(lines[-2][6:])["network"]
    assert {m["node"] for m in net["members"]} == {"head", "worker"}
    async with httpx.AsyncClient(base_url=h.url) as c:
        health = (await c.get("/health")).json()
        assert health["inference_nodes"] == 2 and health["inference_transport"] == "direct"
        assert (await c.get("/nodes")).status_code == 200       # the training API is untouched
        ledger = (await c.get("/ledger")).json()
        infer = {row["node_id"]: row for row in ledger if row["kind"] == "infer"}
        assert set(infer) == {h.ids["head"], h.ids["worker"]}
        assert sum(row["flops"] for row in infer.values()) == pytest.approx(net["flops"])


@pytest.mark.parametrize("pool", [False, True], indirect=True, ids=["lan", "public"])
@pytest.mark.parametrize("authentication", ["bearer", "cookie"])
async def test_signed_in_chatter_pays_hosts_in_flops(pool, authentication):
    h = pool
    host, host_token = account(h, "host@lan.test")
    chatter, chat_token = account(h, "chatter@lan.test", balance=1e16)
    await two_nodes(h, session_token=host_token)
    credits = core(h).credits
    headers = ({"Authorization": f"Bearer {chat_token}"} if authentication == "bearer"
               else {"Cookie": f"slashcompute_session={chat_token}"})
    r = await chat(h, QWEN, max_tokens=8, headers=headers)
    assert r.status_code == 200, r.text
    flops = r.json()["network"]["flops"]
    assert flops > 0
    assert credits.lifetime_earned(host.id) == pytest.approx(flops)       # both nodes belong to the host
    assert credits.balance(chatter.id) == pytest.approx(1e16 - flops)       # reservation consumed, rest refunded
    assert credits.balance(host.id) == pytest.approx(flops * (1 - host.grant_split / 100))


async def test_anonymous_chat_is_free_and_earns_nobody(pool):
    h = pool
    host, host_token = account(h, "host@lan.test")
    await two_nodes(h, session_token=host_token)
    r = await chat(h, QWEN, max_tokens=4)
    assert r.status_code == 200
    assert core(h).credits.lifetime_earned(host.id) == 0      # same rule as anonymous training jobs
    usage = [row for row in core(h).ledger.summary() if row["kind"] == "infer"]
    assert usage and all(row["flops"] > 0 for row in usage)  # but the work is still on the books


@pytest.mark.parametrize("pool", [False, True], indirect=True, ids=["lan", "public"])
async def test_broke_or_unconsented_chatters_are_refused(pool):
    h = pool
    _, host_token = account(h, "host@lan.test")
    await two_nodes(h, session_token=host_token)
    _, broke = account(h, "broke@lan.test")
    r = await chat(h, QWEN, headers={"Authorization": f"Bearer {broke}"})
    assert r.status_code == 402 and "Contribute first" in r.text   # Payment Required (training says 400)
    _, no_terms = account(h, "new@lan.test", terms=False, balance=1e16)
    r = await chat(h, QWEN, headers={"Authorization": f"Bearer {no_terms}"})
    assert r.status_code == 403


async def test_hanging_up_on_a_non_streaming_chat_charges_only_the_partial_work(tmp_path):
    cfg = EngineConfig(home=tmp_path / "home", scheduler_tick_s=0.05, verify_rate=0.0)
    h = await start_harness(fast_settings(), time_scale=1.0, tmp=str(tmp_path / "nodes"),
                            app_factory=lambda: create_app(cfg, inference=fast_settings()))
    try:
        h.add_model(synthetic_layout(QWEN, n_layers=64, total_bytes=int(17.56 * GB), head_bytes=1 * GB))
        host, host_token = account(h, "host@lan.test")
        chatter, chat_token = account(h, "chatter@lan.test", balance=1e18)
        await two_nodes(h, session_token=host_token)
        credits = core(h).credits
        start = credits.balance(chatter.id)
        async with httpx.AsyncClient(base_url=h.url, timeout=30) as c:
            req = asyncio.create_task(c.post("/v1/chat/completions", headers={"Authorization": f"Bearer {chat_token}"},
                                             json={"model": QWEN, "max_tokens": 1500,
                                                   "messages": [{"role": "user", "content": "go on"}]}))
            for _ in range(100):
                await asyncio.sleep(0.1)
                if h.agents["head"].jobs:
                    break
            await asyncio.sleep(0.5)
            req.cancel()  # curl -m
            with pytest.raises(asyncio.CancelledError):
                await req
        svc = h.app.state.inference
        for _ in range(50):
            await asyncio.sleep(0.1)
            job = svc.conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 1").fetchone()
            if job["state"] != "running" and credits.reserved_in_flight(chatter.id) == 0:
                break
        assert job["state"] == "cancelled" and 1 <= job["predicted_n"] < 1500
        assert credits.reserved_in_flight(chatter.id) == 0                     # settled once
        spent = start - credits.balance(chatter.id)
        assert spent == pytest.approx(job["flops"]) and spent > 0              # only the partial work
        assert credits.lifetime_earned(host.id) == pytest.approx(job["flops"])
    finally:
        await h.stop()


@pytest.mark.parametrize("pool", [True], indirect=True, ids=["public"])
@pytest.mark.parametrize("headers", [
    {},
    {"Authorization": "Bearer invalid-session"},
    {"Cookie": "slashcompute_session=invalid-session"},
], ids=["missing-session", "invalid-bearer", "invalid-cookie"])
@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
async def test_public_chat_requires_an_account_before_running_inference(pool, headers, stream):
    host, host_token = account(pool, "host@lan.test")
    await two_nodes(pool, session_token=host_token)
    r = await chat(pool, QWEN, headers=headers, stream=stream)
    assert r.status_code == 401 and "Sign in first" in r.text
    assert not [row for row in core(pool).ledger.summary() if row["kind"] == "infer"]
    assert core(pool).credits.lifetime_earned(host.id) == 0


@pytest.mark.parametrize("pool", [True], indirect=True, ids=["public"])
async def test_shared_inference_secret_does_not_replace_a_public_account(pool):
    service = pool.app.state.inference
    service.s = replace(service.s, TOKEN="shared-pool-secret")
    r = await chat(pool, QWEN, headers={"x-inference-token": "shared-pool-secret"})
    assert r.status_code == 401 and "Sign in first" in r.text


@pytest.mark.parametrize("pool", [False, True], indirect=True, ids=["lan", "public"])
async def test_banned_chatter_is_refused(pool):
    user, token = account(pool, "banned@lan.test", balance=1e16)
    core(pool).auth.set_banned(user, True)
    r = await chat(pool, QWEN, headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 403 and "banned" in r.text


async def register(h, session_token=None):
    async with httpx.AsyncClient(base_url=h.node_url) as c:
        return await c.post("/nodes/register", json={"name": "stranger", "llama_build": "b11160-fake",
                                                     "session_token": session_token})


@pytest.mark.parametrize("pool", [True], indirect=True, ids=["public"])
async def test_public_inference_node_must_belong_to_a_consenting_account(pool):
    h = pool
    assert (await register(h)).status_code == 401                      # no session: it would serve anonymously
    assert (await register(h, "invalid-session")).status_code == 401
    _, no_terms = account(h, "new@lan.test", terms=False)
    assert (await register(h, no_terms)).status_code == 403
    banned, banned_token = account(h, "banned@lan.test")
    core(h).auth.set_banned(banned, True)
    assert (await register(h, banned_token)).status_code == 403
    assert not h.conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    host, host_token = account(h, "host@lan.test")
    r = await register(h, host_token)
    assert r.status_code == 200
    assert core(h).credits.owner_of(r.json()["node_id"]) == host.id     # its earnings have an owner


async def test_lan_inference_node_may_register_without_an_account(pool):
    assert (await register(pool)).status_code == 200


@pytest.mark.parametrize("pool", [False, True], indirect=True, ids=["lan", "public"])
async def test_public_model_and_pipeline_management_is_admin_only(pool):
    h = pool
    public = core(h).cfg.public_pool
    admin, admin_token = account(h, "admin@lan.test")                   # the first account is the admin
    _, user_token = account(h, "user@lan.test", balance=1e16)
    assert admin.admin
    await two_nodes(h, session_token=admin_token)
    assert (await chat(h, QWEN, max_tokens=4, headers={"Authorization": f"Bearer {user_token}"})).status_code == 200
    [pipeline_id] = h.svc.mgr.runtimes

    async def manage(headers, pipeline="p-missing"):
        missing = {"model": "missing.gguf"}
        async with httpx.AsyncClient(base_url=h.node_url, headers=headers) as c:
            return [(await c.post("/models/upload", params={"name": "m.txt"}, content=b"")).status_code,
                    (await c.delete("/models/missing.gguf")).status_code,
                    (await c.post(f"/pipelines/{pipeline}/stop")).status_code,
                    (await c.post("/models/unload", json=missing)).status_code,
                    (await c.post("/models/serving", json={**missing, "on": False})).status_code,
                    (await c.post("/models/remove", json=missing)).status_code]

    allowed = [400, 404, 404, 404, 404, 404]                            # past auth: bad file name, unknown names
    assert await manage({}) == ([401] * 6 if public else allowed)
    assert await manage({"Authorization": "Bearer invalid"}) == ([401] * 6 if public else allowed)
    user = {"Authorization": f"Bearer {user_token}"}
    assert await manage(user) == ([403] * 6 if public else allowed)
    if public:
        assert (await manage(user, pipeline_id))[2] == 403 and pipeline_id in h.svc.mgr.runtimes
        async with httpx.AsyncClient(base_url=h.node_url, headers=user) as c:
            for path, body in (("/models/serving", {"model": QWEN, "on": False}), ("/models/unload", {"model": QWEN}),
                               ("/models/remove", {"model": QWEN})):
                assert (await c.post(path, json=body)).status_code == 403
        assert h.svc.mgr.runtimes[pipeline_id].state == "active"
        assert h.conn.execute("SELECT status FROM models WHERE id=?", (QWEN,)).fetchone()[0] == "ready"
    assert await manage({"Cookie": f"slashcompute_session={admin_token}"}) == allowed   # the web shell's cookie
    stopper = {"Authorization": f"Bearer {admin_token}"} if public else {}
    assert (await manage(stopper, pipeline_id))[2] == 200


async def test_non_numeric_max_tokens_is_a_400(pool):
    h = pool
    await two_nodes(h)
    _, token = account(h, "chatter@lan.test", balance=1e16)
    for headers in ({"Authorization": f"Bearer {token}"}, {}):
        r = await chat(h, QWEN, max_tokens="abc", headers=headers)
        assert r.status_code == 400 and "max_tokens" in r.text


@pytest.mark.parametrize("bad", [0, -5, True, "abc"])
async def test_max_tokens_must_be_a_positive_integer(pool, bad):
    h = pool
    await two_nodes(h)
    _, token = account(h, "chatter@lan.test", balance=1e16)
    for headers in ({"Authorization": f"Bearer {token}"}, {}):
        for field in ("max_tokens", "max_completion_tokens", "n_predict"):
            async with httpx.AsyncClient(base_url=h.url) as c:
                r = await c.post("/v1/chat/completions", headers=headers, json={
                    "model": QWEN, "messages": [{"role": "user", "content": "hi"}], field: bad})
            assert r.status_code == 400 and "max_tokens" in r.text, (field, r.text)


class RecordingEngine(FakeEngine):
    bodies: list = []

    def complete(self, pipeline_id, body):
        RecordingEngine.bodies.append(body)
        return super().complete(pipeline_id, body)


@pytest.mark.parametrize("limit, expect", [
    ({}, 256),                                                 # no limit: capped at the default we reserve for
    ({"max_completion_tokens": 300}, 300),                     # the newer OpenAI name is honored, not ignored
    ({"n_predict": 7}, 7),
    ({"max_tokens": 5, "n_predict": 4000}, 5),                 # one budget; the engine never sees a bigger one
])
async def test_engine_is_capped_at_the_budget_the_chatter_paid_for(pool, limit, expect):
    h = pool
    RecordingEngine.bodies = []
    host, host_token = account(h, "host@lan.test")
    chatter, chat_token = account(h, "chatter@lan.test", balance=1e16)
    await two_nodes(h, session_token=host_token, engine_cls=RecordingEngine)
    async with httpx.AsyncClient(base_url=h.url, timeout=30) as c:
        r = await c.post("/v1/chat/completions", headers={"Authorization": f"Bearer {chat_token}"}, json={
            "model": QWEN, "messages": [{"role": "user", "content": "Write a long story about a dragon."}], **limit})
    assert r.status_code == 200, r.text
    sent = RecordingEngine.bodies[-1]
    assert sent["max_tokens"] == expect and "n_predict" not in sent and "max_completion_tokens" not in sent
    assert r.json()["usage"]["completion_tokens"] == expect
    flops = r.json()["network"]["flops"]
    credits = core(h).credits
    assert credits.lifetime_earned(host.id) == pytest.approx(flops)
    assert credits.balance(chatter.id) == pytest.approx(1e16 - flops)   # charged for every token generated


class DenseEngine(FakeEngine):
    """Counts prompt tokens like llama.cpp does for dense text (one per digit, plus the chat template),
    rejects prompts that don't fit the pipeline's context, and generates slowly (weight above the default)."""

    async def complete(self, pipeline_id, body):
        ctx = self.heads[pipeline_id]["ctx"]
        prompt_n = sum(len(m["content"].encode()) + 5 for m in body["messages"]) + 5
        if prompt_n + body["max_tokens"] > ctx:
            raise EngineError(f"request ({prompt_n} tokens) exceeds the available context size ({ctx} tokens)",
                              pipeline_broken=False, status=400)
        async for ev in super().complete(pipeline_id, body):
            if ev["type"] == "final":
                n = ev["timings"]["predicted_n"]
                ev["timings"] = {"cache_n": 0, "prompt_n": prompt_n, "prompt_per_second": 1000.0,
                                 "predicted_n": n, "predicted_per_second": 1e6 if n == 1 else 10.0}
            yield ev


async def test_reservation_covers_the_real_charge_of_dense_prompts_and_slow_replies(pool):
    h = pool
    svc = h.app.state.inference
    host, host_token = account(h, "host@lan.test")
    chatter, chat_token = account(h, "chatter@lan.test", balance=1e16)
    await two_nodes(h, session_token=host_token, engine_cls=DenseEngine)
    credits = core(h).credits
    headers = {"Authorization": f"Bearer {chat_token}"}
    paid = 0.0
    # 2010 real prompt tokens (~510 by len(json)/4); then a slow reply after a 1-token one "ran" at 1e6 tok/s
    for content, max_tokens, live_tok_s in (("1" * 2000, 1, None), ("Write an essay about rivers.", 200, 10.0)):
        r = await chat(h, QWEN, content=content, max_tokens=max_tokens, headers=headers)
        assert r.status_code == 200, r.text
        paid += r.json()["network"]["flops"]
        assert credits.balance(chatter.id) == pytest.approx(1e16 - paid)    # charged in full, never capped
        assert credits.lifetime_earned(host.id) == pytest.approx(paid)      # so the hosts are paid in full
        assert credits.reserved_in_flight(chatter.id) == 0                  # and the rest of the hold refunded
        assert svc.conn.execute("SELECT live_tok_s FROM pipelines").fetchone()["live_tok_s"] == live_tok_s
    assert r.json()["network"]["gen_weight"] == svc.s.GEN_WEIGHT_MAX


async def test_pipeline_context_is_sized_from_the_prompt_byte_bound(pool):
    await two_nodes(pool, engine_cls=DenseEngine)
    r = await chat(pool, QWEN, content="1" * 6000, max_tokens=5)        # 6010 tokens: needs 8192, not 4096
    assert r.status_code == 200, r.text
    assert [rt.ctx for rt in pool.app.state.inference.mgr.runtimes.values()] == [8192]


def test_ctx_is_capped_at_the_model_context_length():
    s = fast_settings()
    body = {"messages": [{"role": "user", "content": "1" * 6000}], "max_tokens": 3000}
    assert ctx_for(body, s, 32768) == 16384
    assert ctx_for(body, s, 8192) == 8192             # the engine rejects the prompt if it really doesn't fit
    with pytest.raises(HTTPException) as e:
        ctx_for({**body, "max_tokens": 8192}, s, 8192)
    assert e.value.status_code == 400


@pytest.mark.parametrize("model, max_tokens", [(QWEN, 1e30), (QWEN, 131072), ("Small.gguf", 8192)])
async def test_max_tokens_beyond_the_model_context_is_a_400(pool, model, max_tokens):
    pool.add_model(replace(synthetic_layout("Small.gguf", n_layers=8, total_bytes=GB, head_bytes=GB // 8),
                           max_ctx=8192))   # context_length from its GGUF; QWEN has none (MAX_CTX)
    await two_nodes(pool)
    r = await chat(pool, model, max_tokens=max_tokens)
    assert r.status_code == 400 and "context length" in r.text, r.text
    assert not pool.app.state.inference.mgr.runtimes                       # nothing planned for it


async def test_training_on_a_mac_drains_its_inference_pipelines(pool):
    h = pool
    busy = {"worker": False}
    await h.add_node(FakeNode("head", 16, may_be_head=True, files=(QWEN,)), 0)
    await h.add_node(FakeNode("worker", 16), 1, busy_fn=lambda: busy["worker"])
    await h.add_node(FakeNode("spare", 16, gen_score=0.5), 2)
    r = await chat(h, QWEN, max_tokens=4)
    first = r.json()["network"]
    assert "worker" in {m["node"] for m in first["members"]}
    busy["worker"] = True        # the MLX agent picked up a training stage on this Mac
    for _ in range(40):
        await asyncio.sleep(0.05)
        if h.conn.execute("SELECT state FROM pipelines WHERE id=?", (first["pipeline_id"],)).fetchone()[0] == "stopped":
            break
    assert h.conn.execute("SELECT state FROM pipelines WHERE id=?", (first["pipeline_id"],)).fetchone()[0] == "stopped"
    assert h.agents["worker"].last_reason == "training on this Mac"
    r2 = await chat(h, QWEN, max_tokens=4)
    assert r2.status_code == 200
    assert "worker" not in {m["node"] for m in r2.json()["network"]["members"]}


def test_training_busy_reads_the_mlx_agent_status(tmp_path):
    import os

    agent = tmp_path / "agent"
    agent.mkdir()
    status = agent / "status.json"
    assert not training_busy(str(status))                       # no MLX agent at all
    status.write_text(json.dumps({"status": "running"}))
    assert not training_busy(str(status))                       # stale: no live agent pid
    (agent / "agent.pid").write_text(f"{os.getpid()}\n")
    assert training_busy(str(status))
    status.write_text(json.dumps({"status": "idle"}))
    assert not training_busy(str(status))
