"""Accounts, 1:1 credits, and the engine hooks that keep them honest."""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

from accounts import fake_google, http_sign_in, id_token, sign_in
from fastapi.testclient import TestClient
from sqlmodel import select
from starlette.websockets import WebSocketDisconnect

from slashcompute.common import protocol as P
from slashcompute.common.config import WELCOME_FLOPS, EngineConfig
from slashcompute.community.auth import AuthError
from slashcompute.community.credits import POT_ID, CreditError
from slashcompute.coordinator.app import create_app
from slashcompute.coordinator.core import Coordinator
from slashcompute.coordinator.db import CreditTxn, User
from slashcompute.coordinator.inference_accounting import CoreAccounting
from slashcompute.coordinator.partitioner import StagePlan
from slashcompute.coordinator.scheduler import EpochState
from slashcompute.jobs import LoraFinetuneSpec
from slashcompute.pipeline.model_profile import profile_model


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


def _device(mem=8 << 30):
    return P.DeviceProfile(
        chip="test", memory_total_bytes=mem, memory_available_bytes=mem,
        working_set_bytes=mem, memory_contrib_bytes=mem,
        matmul_tflops=1.0, mem_bandwidth_gbps=100.0,
    )


def _usage(flops=1e9):
    return P.UsageSample(flops=flops, tokens=10, peak_mem_bytes=1, resident_mem_bytes=1,
                         mem_byte_seconds=1.0, wall_s=1.0, busy_s=0.5)


def _spec(tiny_model, tiny_dataset, **kw):
    base = dict(model=str(tiny_model), dataset_path=str(tiny_dataset), steps=2,
                batch_size=2, microbatches=1, lora_rank=4, min_stages=1)
    return LoraFinetuneSpec(**(base | kw))


def _one_stage_epoch(job):
    """Epoch 1 as the scheduler starts it on one Mac: the whole model as stage 0. Steps are
    billed against the stage the coordinator planned, so a step test needs one."""
    job.profile = profile_model(job.spec.model)
    return EpochState(epoch=1, plans=[StagePlan(0, "n1", 0, job.profile.num_layers, 0)])


def _pay_out(core, job):
    """End the job's epoch and pay what its nodes were held for, as the coordinator's tick does
    once nothing sampled from that epoch is left to check."""
    job.current.closed = True
    return core.verification.settle_holds()


def _account(auth, email="ada@lan.test", name="Ada"):
    return sign_in(auth, email, name)


# --------------------------------------------------------------------------- auth


def test_first_user_is_admin_second_is_not(core):
    a, _ = _account(core.auth, "a@lan.test", name="A")
    b, _ = _account(core.auth, "b@lan.test", name="B")
    assert a.admin is True
    assert b.admin is False


def test_terms_and_profile(core):
    user, _ = _account(core.auth)
    assert user.accepted_terms_at is None
    user = core.auth.accept_terms(user)
    assert user.accepted_terms_at is not None
    user = core.auth.update_profile(user, bio="  hello  ")
    assert user.bio == "hello"
    assert core.auth.public_view(user)["bio"] == "hello"
    assert "grant_split" not in core.auth.public_view(user)
    with pytest.raises(AuthError):
        core.auth.update_profile(user, name="  ")


def test_summary_reports_everything_earned_as_the_owners(core):
    user, _ = _account(core.auth)
    other, _ = _account(core.auth, "o@lan.test")
    core.credits.contribute(user.id, 100.0)
    core.credits.contribute(other.id, 40.0)                   # someone else's earnings are theirs alone
    s = core.credits.summary(user.id)
    assert (s["lifetime_earned"], s["balance"], s["pot"]) == (100.0, 100.0, 0.0)
    assert "lifetime_to_grants" not in s and "lifetime_kept" not in s


def test_logout_drops_session(core):
    _, token = _account(core.auth)
    assert core.auth.user_from_token(token) is not None
    core.auth.logout(token)
    assert core.auth.user_from_token(token) is None


# --------------------------------------------------------------------------- credits


def test_contribute_is_one_to_one_and_all_the_owners(core):
    user, _ = _account(core.auth)
    core.credits.contribute(user.id, 100.0, node_id="n1")
    assert core.credits.lifetime_earned(user.id) == 100.0
    assert core.credits.balance(user.id) == 100.0
    assert core.credits.balance(POT_ID) == 0.0
    core.credits.contribute(user.id, 100.0)
    assert core.credits.balance(user.id) == 200.0
    assert core.credits.lifetime_earned(user.id) == 200.0


def test_reserve_consume_settle(core):
    user, _ = _account(core.auth)
    core.credits.contribute(user.id, 1000.0)
    with pytest.raises(CreditError):
        core.credits.reserve_job(user.id, "job-x", 5000.0)
    core.credits.reserve_job(user.id, "job-1", 400.0)
    assert core.credits.balance(user.id) == 600.0
    assert core.credits.summary(user.id)["reserved_in_flight"] == 400.0
    assert core.credits.consume_job("job-1", 150.0) == 150.0
    assert core.credits.lifetime_spent(user.id) == 150.0
    assert core.credits.job_exhausted("job-1") is False
    assert core.credits.summary(user.id)["reserved_in_flight"] == 250.0
    assert core.credits.consume_job("job-1", 300.0) == 250.0
    assert core.credits.lifetime_spent(user.id) == 400.0
    assert core.credits.job_exhausted("job-1") is True
    core.credits.settle_job("job-1")
    assert core.credits.balance(user.id) == 600.0
    assert core.credits.reserved_in_flight(user.id) == 0.0


def test_settle_releases_leftover(core):
    user, _ = _account(core.auth)
    core.credits.contribute(user.id, 100.0)
    core.credits.reserve_job(user.id, "job-2", 80.0)
    core.credits.consume_job("job-2", 30.0)
    core.credits.settle_job("job-2")
    assert core.credits.balance(user.id) == 70.0
    assert core.credits.lifetime_spent(user.id) == 30.0


def test_rows_from_when_credits_could_be_given_away_still_count(core):
    """Donations and pot allocations are no longer made, but a ledger from before holds them: the
    accounts they moved credits between keep the balances they had."""
    donor, _ = _account(core.auth, "d@lan.test", name="Donor")
    recip, _ = _account(core.auth, "r@lan.test", name="Recip")
    core.credits.contribute(donor.id, 100.0)
    core.credits.post(POT_ID, "earn", 50.0, note=f"tithe from {donor.id}")
    core.credits.post(donor.id, "donate", -20.0)
    core.credits.post(recip.id, "receive", 20.0)
    core.credits.post(POT_ID, "allocate", -50.0)
    core.credits.post(recip.id, "receive", 50.0)
    assert core.credits.balance(donor.id) == 80.0
    assert core.credits.balance(recip.id) == 70.0
    assert core.credits.balance(POT_ID) == 0.0


def test_leaderboard_ranks_generated(core):
    a, _ = _account(core.auth, "a@lan.test", name="Ann")
    b, _ = _account(core.auth, "b@lan.test", name="Bea")
    core.credits.contribute(a.id, 10.0)
    core.credits.contribute(b.id, 50.0)
    board = core.credits.leaderboard()
    assert [row["name"] for row in board] == ["Bea", "Ann"]
    assert board[0]["lifetime_earned"] == 50.0


def test_settle_open_jobs_refunds_only_the_accounts_under_its_prefix(core):
    user, _ = _account(core.auth)
    core.credits.post(user.id, "earn", 1000.0)
    core.credits.reserve_job(user.id, "inf-crashed", 400.0)      # a chat cut off by a crash
    core.credits.consume_job("inf-crashed", 150.0)
    core.credits.reserve_job(user.id, "inf-settled", 100.0)      # one that finished normally
    core.credits.consume_job("inf-settled", 30.0)
    core.credits.settle_job("inf-settled")
    core.credits.reserve_job(user.id, "a1b2c3d4e5f6", 200.0)     # a training job, still running
    core.credits.reserve_job(user.id, "inf_x", 50.0)             # "_" is not a wildcard
    assert core.credits.balance(user.id) == pytest.approx(1000 - 400 - 30 - 200 - 50)

    assert core.credits.settle_open_jobs("inf-") == 1
    assert core.credits.balance(user.id) == pytest.approx(1000 - 150 - 30 - 200 - 50)
    assert core.credits.reserved_in_flight(user.id) == pytest.approx(250.0)   # the other two holds
    acct = core.credits.job_account("inf-crashed")
    assert acct.reserved_flops == acct.spent_flops == 150.0
    # nothing left open: running it again releases nothing twice
    assert core.credits.settle_open_jobs("inf-") == 0
    assert core.credits.balance(user.id) == pytest.approx(1000 - 150 - 30 - 200 - 50)
    with core.db.session() as s:
        releases = s.exec(select(CreditTxn).where(CreditTxn.kind == "release")).all()
    assert sorted((r.job_id, r.amount) for r in releases) == [("inf-crashed", 250.0), ("inf-settled", 70.0)]


# --------------------------------------------------------------------------- moderation


def test_flag_marks_the_account_and_is_admin_only(core):
    admin, _ = _account(core.auth, "admin@lan.test")
    member, _ = _account(core.auth, "m@lan.test", name="Member")
    with pytest.raises(AuthError, match="Admin") as refused:
        core.auth.flag_user(member, admin, "no")
    assert refused.value.status == 403
    assert core.auth.get(admin.id).flagged is False and core.auth.list_flags() == []
    core.auth.flag_user(admin, member, "  spam  ")
    core.auth.flag_user(admin, member, "")
    assert core.auth.get(member.id).flagged is True
    assert [(f["user_id"], f["admin_id"], f["reason"]) for f in core.auth.list_flags()] in (
        [(member.id, admin.id, "flagged"), (member.id, admin.id, "spam")],
        [(member.id, admin.id, "spam"), (member.id, admin.id, "flagged")],   # same timestamp
    )


# --------------------------------------------------------------------------- http


def _hdr(token):
    return {"Authorization": f"Bearer {token}"}


def test_http_terms_required_to_take_and_failed_reserve_drops_job(env):
    client, core, tiny_model, tiny_dataset = env
    token = http_sign_in(client, "ada@lan.test", "Ada")
    spec = json.loads(_spec(tiny_model, tiny_dataset).model_dump_json())
    spec["max_flops"] = 2 * WELCOME_FLOPS
    r = client.post("/jobs", json=spec, headers=_hdr(token))
    assert r.status_code == 403
    assert "terms" in r.json()["detail"].lower()
    assert core.jobs == {}
    client.post("/auth/accept-terms", headers=_hdr(token))
    r = client.post("/jobs", json=spec, headers=_hdr(token))
    assert r.status_code == 400
    assert "Contribute" in r.json()["detail"]
    assert core.jobs == {}


def test_http_authenticated_take_reserves(env):
    client, core, tiny_model, tiny_dataset = env
    token = http_sign_in(client, "ada@lan.test", "Ada")
    client.post("/auth/accept-terms", headers=_hdr(token))
    user = core.auth.user_from_token(token)
    core.credits.contribute(user.id, 1e12)
    spec = json.loads(_spec(tiny_model, tiny_dataset).model_dump_json())
    spec["max_flops"] = 2e11
    r = client.post("/jobs", json=spec, headers=_hdr(token))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["reserved_flops"] == 2e11
    assert body["user_id"] == user.id
    assert core.credits.balance(user.id) == pytest.approx(WELCOME_FLOPS + 8e11)


def test_anonymous_submit_still_works(env):
    client, core, tiny_model, tiny_dataset = env
    spec = json.loads(_spec(tiny_model, tiny_dataset).model_dump_json())
    r = client.post("/jobs", json=spec)
    assert r.status_code == 200, r.text
    assert r.json()["reserved_flops"] is None
    assert r.json()["id"] in core.jobs


def test_http_leaderboard_is_public_and_the_user_list_is_admin_only(env):
    client, core, *_ = env
    http_sign_in(client, "admin@lan.test", "Admin")
    member_tok = http_sign_in(client, "m@lan.test", "Member")
    donor_tok = http_sign_in(client, "d@lan.test", "Donor")
    donor = core.auth.user_from_token(donor_tok)
    core.credits.contribute(donor.id, 80.0)
    client.cookies.clear()
    board = client.get("/community/leaderboard").json()
    assert board[0]["name"] == "Donor"
    users = client.get("/admin/users", headers=_hdr(member_tok))
    assert users.status_code == 403


def test_http_ban_blocks_take(env):
    client, core, tiny_model, tiny_dataset = env
    admin_tok = http_sign_in(client, "admin@lan.test", "Admin")
    member_tok = http_sign_in(client, "m@lan.test", "Member")
    member = core.auth.user_from_token(member_tok)
    client.post("/auth/accept-terms", headers=_hdr(member_tok))
    core.credits.contribute(member.id, 1e12)
    client.post(f"/admin/users/{member.id}/ban", json={"banned": True},
                headers=_hdr(admin_tok))
    spec = json.loads(_spec(tiny_model, tiny_dataset).model_dump_json())
    spec["max_flops"] = 1e6
    r = client.post("/jobs", json=spec, headers=_hdr(member_tok))
    assert r.status_code == 403, r.text
    assert core.jobs == {}


# --------------------------------------------------------------------------- welcome credit


def _welcome_rows(client, token):
    items = client.get("/credits/transactions", headers=_hdr(token)).json()["items"]
    return [row for row in items if row["kind"] == "welcome"]


def test_welcome_credit_on_register_is_spendable_once(env):
    client, core, tiny_model, tiny_dataset = env
    token = http_sign_in(client, "ada@lan.test", "Ada")
    me = client.get("/credits/me", headers=_hdr(token)).json()
    assert me["balance"] == WELCOME_FLOPS == 1e15
    assert me["lifetime_earned"] == 0.0
    rows = _welcome_rows(client, token)
    assert len(rows) == 1 and rows[0]["note"] == "Welcome credit"
    for _ in range(2):
        http_sign_in(client, "ada@lan.test", "Ada")
    assert core.credits.balance(core.auth.user_from_token(token).id) == WELCOME_FLOPS

    client.post("/auth/accept-terms", headers=_hdr(token))
    spec = json.loads(_spec(tiny_model, tiny_dataset).model_dump_json())
    spec["max_flops"] = WELCOME_FLOPS
    r = client.post("/jobs", json=spec, headers=_hdr(token))
    assert r.status_code == 200, r.text
    assert core.credits.balance(r.json()["user_id"]) == 0.0


def test_welcome_credit_backfilled_on_login(env):
    client, core, *_ = env
    user, _ = _account(core.auth, "old@lan.test", "Old")  # pre-feature account
    assert core.credits.balance(user.id) == 0.0
    token = http_sign_in(client, "old@lan.test", "Old")
    http_sign_in(client, "old@lan.test", "Old")
    assert core.credits.balance(user.id) == WELCOME_FLOPS
    assert len(_welcome_rows(client, token)) == 1


def test_welcome_credit_backfilled_for_a_stored_session(env):
    # Upgraded while signed in: the app keeps polling /auth/me and never signs in again.
    client, core, *_ = env
    user, token = _account(core.auth, "kept@lan.test")
    assert core.credits.balance(user.id) == 0.0
    for _ in range(3):
        me = client.get("/auth/me", headers=_hdr(token)).json()
    assert me["credits"]["balance"] == WELCOME_FLOPS
    assert len(_welcome_rows(client, token)) == 1

    banned, banned_token = _account(core.auth, "ban@lan.test")
    core.auth.set_banned(banned, True)
    client.get("/auth/me", headers=_hdr(banned_token))
    assert core.credits.balance(banned.id) == 0.0


def test_welcome_credit_granted_once_under_concurrency(core):
    import threading

    user, _ = _account(core.auth)
    start = threading.Barrier(16)
    granted = []

    def sign_in():
        start.wait()
        granted.append(core.credits.grant_welcome(user.id))

    threads = [threading.Thread(target=sign_in) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(granted) == [0.0] * 15 + [WELCOME_FLOPS]
    assert core.credits.balance(user.id) == WELCOME_FLOPS


def test_welcome_credit_skips_banned_and_disabled(env, tmp_path, monkeypatch):
    client, core, *_ = env
    user, _ = _account(core.auth, "ban@lan.test", "Ban")
    core.auth.set_banned(user, True)
    nonce = client.post("/auth/google/challenge").json()["nonce"]
    r = client.post("/auth/google", json={"id_token": id_token("ban@lan.test", nonce)})
    assert r.status_code == 403
    assert core.credits.grant_welcome(user.id) == 0.0
    assert core.credits.balance(user.id) == 0.0

    monkeypatch.setenv("SLASHCOMPUTE_WELCOME_FLOPS", "0")
    cfg = EngineConfig.from_env(home=tmp_path / "off", verify_rate=0.0)
    assert cfg.welcome_flops == 0.0
    with TestClient(create_app(cfg)) as off:
        token = http_sign_in(off, "ada@lan.test", "Ada")
        assert off.get("/credits/me", headers=_hdr(token)).json()["balance"] == 0.0


# --------------------------------------------------------------------------- engine hooks


def test_register_binds_only_after_terms(core):
    user, token = _account(core.auth)
    msg = P.Register(node_id="n1", name="mac", device=_device(),
                     data_host="127.0.0.1", data_port=9700, gpu_percent=50,
                     session_token=token)

    async def send(_):
        return None

    asyncio.run(core.on_register(msg, send))
    assert core.registry.get("n1").user_id is None
    core.auth.accept_terms(user)
    asyncio.run(core.on_register(msg, send))
    assert core.registry.get("n1").user_id == user.id
    assert core.credits.owner_of("n1") == user.id


def test_banned_session_closes_agent(env):
    client, core, *_ = env
    token = http_sign_in(client, "ada@lan.test", "Ada")
    user = core.auth.session_user(token)
    core.auth.accept_terms(user)
    core.auth.set_banned(user, True)
    with pytest.raises(WebSocketDisconnect) as e:
        with client.websocket_connect("/ws/agent") as ws:
            ws.send_text(P.dump(P.Register(
                node_id="banned", name="mac", device=_device(),
                data_host="127.0.0.1", data_port=9700, gpu_percent=50,
                session_token=token,
            )))
            ws.receive_text()
    assert e.value.code == 4003


def test_steps_earn_and_spend_one_to_one(core):
    tiny_model, tiny_dataset = core._tiny
    user, token = _account(core.auth)
    core.auth.accept_terms(user)
    core.credits.contribute(user.id, 1e12)
    job = core.submit(_spec(tiny_model, tiny_dataset))
    core.credits.reserve_job(user.id, job.id, 5e9)
    job.current = _one_stage_epoch(job)
    job.row.status = "running"

    async def send(_):
        return None

    asyncio.run(core.on_register(P.Register(
        node_id="n1", name="mac", device=_device(), data_host="127.0.0.1",
        data_port=9700, gpu_percent=50, session_token=token,
    ), send))

    asyncio.run(core._on_step("n1", P.StepMetrics(
        job_id=job.id, epoch=1, stage_idx=0, step=1, loss=1.0,
        in_digest="in", out_digest="out", usage=_usage(1e9),
    )))
    # Charged to the job at once, but only held for the node: nothing to spend yet.
    assert core.credits.lifetime_spent(user.id) == pytest.approx(1e9)
    assert core.credits.lifetime_earned(user.id) == pytest.approx(1e12)
    assert core.credits.balance(user.id) == pytest.approx(1e12 - 5e9)
    assert core.credits.summary(user.id)["pending_earnings"] == pytest.approx(1e9)
    assert core.verification.settle_holds() == 0    # the epoch is still running

    assert _pay_out(core, job) == 1
    assert core.credits.summary(user.id)["pending_earnings"] == 0.0
    assert core.credits.lifetime_earned(user.id) == pytest.approx(1e12 + 1e9)
    assert core.credits.lifetime_spent(user.id) == pytest.approx(1e9)
    assert core.credits.balance(user.id) == pytest.approx(1e12 - 5e9 + 1e9)


def test_anonymous_job_does_not_mint_credits(core):
    tiny_model, tiny_dataset = core._tiny
    user, token = _account(core.auth)
    core.auth.accept_terms(user)
    job = core.submit(_spec(tiny_model, tiny_dataset))
    job.current = _one_stage_epoch(job)

    async def send(_):
        return None

    asyncio.run(core.on_register(P.Register(
        node_id="n1", name="mac", device=_device(), data_host="127.0.0.1",
        data_port=9700, gpu_percent=50, session_token=token,
    ), send))
    asyncio.run(core._on_step("n1", P.StepMetrics(
        job_id=job.id, epoch=1, stage_idx=0, step=1, loss=1.0,
        in_digest="in", out_digest="out", usage=_usage(1e9),
    )))
    assert core.credits.lifetime_earned(user.id) == 0.0
    assert core.credits.job_account(job.id) is None


def test_budget_exhaust_cancels_job(core):
    tiny_model, tiny_dataset = core._tiny
    user, token = _account(core.auth)
    core.auth.accept_terms(user)
    core.credits.contribute(user.id, 1e12)
    job = core.submit(_spec(tiny_model, tiny_dataset))
    core.credits.reserve_job(user.id, job.id, 1e9)
    job.current = _one_stage_epoch(job)
    job.row.status = "running"

    async def send(_):
        return None

    asyncio.run(core.on_register(P.Register(
        node_id="n1", name="mac", device=_device(), data_host="127.0.0.1",
        data_port=9700, gpu_percent=50, session_token=token,
    ), send))
    asyncio.run(core._on_step("n1", P.StepMetrics(
        job_id=job.id, epoch=1, stage_idx=0, step=1, loss=1.0,
        in_digest="in", out_digest="out", usage=_usage(1e9),
    )))
    assert job.row.status == "cancelled"
    assert job.row.error == "FLOP budget spent"
    assert core.verification.settle_holds() == 1    # the cancel ended the epoch
    assert core.credits.balance(user.id) == pytest.approx(1e12 - 1e9 + 1e9)


def test_a_step_pays_its_owner_in_full(core):
    """Also for an account that once chose to send part of its earnings to the community pot:
    that setting is still in its row, and no longer does anything."""
    tiny_model, tiny_dataset = core._tiny
    user, token = _account(core.auth)
    core.auth.accept_terms(user)
    core.db.update(User, user.id, grant_split=40)
    core.credits.contribute(user.id, 1e12)
    job = core.submit(_spec(tiny_model, tiny_dataset))
    core.credits.reserve_job(user.id, job.id, 1e10)
    job.current = _one_stage_epoch(job)
    job.row.status = "running"

    async def send(_):
        return None

    asyncio.run(core.on_register(P.Register(
        node_id="n1", name="mac", device=_device(), data_host="127.0.0.1",
        data_port=9700, gpu_percent=50, session_token=token,
    ), send))
    asyncio.run(core._on_step("n1", P.StepMetrics(
        job_id=job.id, epoch=1, stage_idx=0, step=1, loss=1.0,
        in_digest="in", out_digest="out", usage=_usage(100.0),
    )))
    before = core.credits.balance(user.id)
    _pay_out(core, job)
    assert core.credits.balance(POT_ID) == 0.0
    assert core.credits.balance(user.id) == pytest.approx(before + 100.0)
    assert core.credits.lifetime_earned(user.id) == pytest.approx(1e12 + 100.0)


def test_last_step_mints_only_charged_flops(core):
    tiny_model, tiny_dataset = core._tiny
    user, token = _account(core.auth)
    core.auth.accept_terms(user)
    core.credits.contribute(user.id, 1e12)
    job = core.submit(_spec(tiny_model, tiny_dataset))
    core.credits.reserve_job(user.id, job.id, 1.5e9)
    job.current = _one_stage_epoch(job)
    job.row.status = "running"

    async def send(_):
        return None

    asyncio.run(core.on_register(P.Register(
        node_id="n1", name="mac", device=_device(), data_host="127.0.0.1",
        data_port=9700, gpu_percent=50, session_token=token,
    ), send))
    asyncio.run(core._on_step("n1", P.StepMetrics(
        job_id=job.id, epoch=1, stage_idx=0, step=1, loss=1.0,
        in_digest="in", out_digest="out", usage=_usage(2e9),
    )))
    assert core.credits.lifetime_spent(user.id) == pytest.approx(1.5e9)
    core.verification.settle_holds()
    assert core.credits.lifetime_earned(user.id) == pytest.approx(1e12 + 1.5e9)
    assert job.row.status == "cancelled"
    assert job.row.error == "FLOP budget spent"


def _ledger_total(core):
    """Every FLOP in the book: all balances (pot included) plus reservations still in flight and what
    is held for nodes until their epoch's checks are in."""
    with core.db.session() as s:
        ids = [u.id for u in s.exec(select(User)).all()] + [POT_ID]
    held = sum(h.flops for h in core.credits.open_holds())
    return sum(core.credits.balance(i) + core.credits.reserved_in_flight(i) for i in ids) + held


def _host_and_chatter(core):
    host, host_tok = _account(core.auth, "host@lan.test", name="Host")
    chatter, _ = _account(core.auth, "bea@lan.test", name="Bea")
    core.auth.accept_terms(host)
    core.auth.accept_terms(chatter)
    core.credits.contribute(chatter.id, 1e12)
    return host, host_tok, chatter


@pytest.mark.parametrize("banned", [False, True])
def test_inference_share_without_eligible_host_goes_to_pot(core, banned):
    host, host_tok, chatter = _host_and_chatter(core)
    acct = CoreAccounting(core)
    if banned:
        acct.bind_node("n1", host_tok)
        core.auth.set_banned(host, True)
    core.credits.reserve_job(chatter.id, "chat-1", 1e12)
    before = _ledger_total(core)

    acct.record("chat-1", {"n1": 4.64e11}, tokens=10, wall_s=1.0)

    assert core.credits.lifetime_spent(chatter.id) == pytest.approx(4.64e11)
    assert core.credits.lifetime_earned(host.id) == 0.0
    assert core.credits.balance(POT_ID) == pytest.approx(4.64e11)
    assert _ledger_total(core) == pytest.approx(before)


@pytest.mark.parametrize("banned", [False, True])
def test_training_share_without_eligible_host_goes_to_pot(core, banned):
    tiny_model, tiny_dataset = core._tiny
    host, host_tok, chatter = _host_and_chatter(core)
    job = core.submit(_spec(tiny_model, tiny_dataset))
    core.credits.reserve_job(chatter.id, job.id, 5e9)
    job.current = _one_stage_epoch(job)
    job.row.status = "running"

    async def send(_):
        return None

    asyncio.run(core.on_register(P.Register(
        node_id="n1", name="mac", device=_device(), data_host="127.0.0.1",
        data_port=9700, gpu_percent=50, session_token=host_tok if banned else None,
    ), send))
    if banned:
        core.auth.set_banned(host, True)
    before = _ledger_total(core)
    asyncio.run(core._on_step("n1", P.StepMetrics(
        job_id=job.id, epoch=1, stage_idx=0, step=1, loss=1.0,
        in_digest="in", out_digest="out", usage=_usage(1e9),
    )))
    assert core.credits.lifetime_spent(chatter.id) == pytest.approx(1e9)
    assert _ledger_total(core) == pytest.approx(before)     # held, not lost
    _pay_out(core, job)
    assert core.credits.lifetime_earned(host.id) == 0.0
    assert core.credits.balance(POT_ID) == pytest.approx(1e9)
    assert _ledger_total(core) == pytest.approx(before)


def test_leaderboard_excludes_banned(core):
    a, _ = _account(core.auth, "a@lan.test", name="A")
    b, _ = _account(core.auth, "b@lan.test", name="B")
    core.credits.contribute(a.id, 10.0)
    core.credits.contribute(b.id, 50.0)
    core.auth.set_banned(b, True)
    assert [r["user_id"] for r in core.credits.leaderboard()] == [a.id]


def test_session_without_max_flops_abandons_job(env):
    client, core, tiny_model, tiny_dataset = env
    token = http_sign_in(client, "ada@lan.test", "Ada")
    client.post("/auth/accept-terms", headers=_hdr(token))
    spec = json.loads(_spec(tiny_model, tiny_dataset).model_dump_json())
    r = client.post("/jobs", json=spec, headers=_hdr(token))
    assert r.status_code == 400
    assert "max_flops" in r.json()["detail"]
    assert core.jobs == {}
    client.cookies.clear()
    r = client.post("/jobs", json=spec, headers=_hdr(token))
    assert r.status_code == 400
    assert core.jobs == {}
    http_sign_in(client, "ada@lan.test", "Ada")      # the cookie now carries the session
    r = client.post("/jobs", json=spec)
    assert r.status_code == 400
    assert "max_flops" in r.json()["detail"]
    assert core.jobs == {}


def test_http_community_lists_live_and_admin(env):
    client, core, tiny_model, tiny_dataset = env
    admin_tok = http_sign_in(client, "admin@lan.test", "Admin")
    member_tok = http_sign_in(client, "m@lan.test", "Member")
    donor_tok = http_sign_in(client, "d@lan.test", "Donor")
    client.post("/auth/accept-terms", headers=_hdr(admin_tok))
    client.post("/auth/accept-terms", headers=_hdr(member_tok))
    client.post("/auth/accept-terms", headers=_hdr(donor_tok))
    donor = core.auth.user_from_token(donor_tok)
    member = core.auth.user_from_token(member_tok)
    core.credits.contribute(donor.id, 200.0)
    core.credits.bind_node("n-mac", member.id)

    me = client.get("/credits/me", headers=_hdr(donor_tok)).json()
    assert me["balance"] == WELCOME_FLOPS + 200.0
    assert me["reserved_in_flight"] == 0.0
    assert me["pot"] == 0.0

    tx = client.get("/credits/transactions?limit=2", headers=_hdr(donor_tok)).json()
    assert len(tx["items"]) == 2
    assert {row["kind"] for row in tx["items"]} <= {"earn", "generated"}
    assert all("created_at" in row and "grant_id" not in row for row in tx["items"])
    page = client.get(
        f"/credits/transactions?limit=1&before_id={tx['items'][0]['id']}",
        headers=_hdr(donor_tok),
    ).json()
    assert len(page["items"]) == 1
    assert page["items"][0]["id"] < tx["items"][0]["id"]

    live = client.get("/credits/live?window_s=60", headers=_hdr(donor_tok)).json()
    assert live["window_s"] == 60.0
    assert live["you"]["flops"] == pytest.approx(200.0)
    assert live["community_flops"] == pytest.approx(200.0)
    assert live["reserved_in_flight"] == 0.0

    nodes = client.get("/auth/me/nodes", headers=_hdr(member_tok)).json()
    assert nodes == [{
        "node_id": "n-mac", "user_id": member.id, "online": False,
        "status": None, "gpu_percent": None,
    }]

    # A window from before the split was removed may still send it: it is ignored, not refused.
    bio = client.patch("/auth/me", json={"bio": "trains adapters", "grant_split": 40},
                       headers=_hdr(member_tok))
    assert bio.status_code == 200, bio.text
    assert bio.json()["user"]["bio"] == "trains adapters"
    assert "grant_split" not in bio.json()["user"]

    core.credits.contribute(member.id, 1e12)
    spec = json.loads(_spec(tiny_model, tiny_dataset).model_dump_json())
    spec["max_flops"] = 2e11
    taken = client.post("/jobs", json=spec, headers=_hdr(member_tok))
    assert taken.status_code == 200, taken.text
    job_id = taken.json()["id"]
    mine = client.get("/jobs?mine=1", headers=_hdr(member_tok)).json()
    assert [j["id"] for j in mine] == [job_id]
    others = client.get("/jobs?mine=1", headers=_hdr(donor_tok)).json()
    assert others == []

    client.cookies.clear()
    refused = client.post(f"/admin/users/{donor.id}/flag", json={"reason": "no"},
                          headers=_hdr(member_tok))
    assert refused.status_code == 403, refused.text
    flagged = client.post(f"/admin/users/{member.id}/flag", json={"reason": "spam"},
                          headers=_hdr(admin_tok))
    assert flagged.json()["user"]["flagged"] is True
    flags = client.get("/admin/flags", headers=_hdr(admin_tok)).json()
    assert len(flags) == 1
    assert flags[0]["user_id"] == member.id
    assert flags[0]["reason"] == "spam"
    assert client.get("/admin/flags", headers=_hdr(member_tok)).status_code == 403


def test_my_nodes_shows_live_inference_nodes_online(env):
    client, core, _, _ = env
    token = http_sign_in(client, "host@lan.test", "Host")
    user = core.auth.user_from_token(token)
    inference = client.app.state.inference
    for node_id, beat in [("inf-live", time.time()), ("inf-gone", time.time() - 3600)]:
        inference.conn.execute("INSERT INTO nodes (id, name, last_heartbeat, created_at) VALUES (?,?,?,?)",
                               (node_id, node_id, beat, beat))
        core.credits.bind_node(node_id, user.id)
    nodes = {n["node_id"]: n for n in client.get("/auth/me/nodes", headers=_hdr(token)).json()}
    assert nodes["inf-live"] == {"node_id": "inf-live", "user_id": user.id, "online": True,
                                 "status": "idle", "gpu_percent": None}
    assert nodes["inf-gone"]["online"] is False and nodes["inf-gone"]["status"] is None


@pytest.mark.parametrize("method,path", [
    ("get", "/grants"), ("post", "/grants"), ("get", "/grants/g1"), ("post", "/grants/g1/donate"),
    ("post", "/grants/g1/comments"), ("post", "/admin/grants/g1/review"),
])
def test_http_grants_routes_are_gone(env, method, path):
    client, *_ = env
    admin_tok = http_sign_in(client, "admin@lan.test", "Admin")
    r = client.request(method, path, json={} if method == "post" else None, headers=_hdr(admin_tok))
    assert r.status_code == 404, r.text


def test_http_admin_cannot_ban_self_or_last_admin(env):
    client, core, *_ = env
    admin_tok = http_sign_in(client, "admin@lan.test", "Admin")
    admin = core.auth.user_from_token(admin_tok)
    r = client.post(f"/admin/users/{admin.id}/ban", json={}, headers=_hdr(admin_tok))
    assert r.status_code == 400, r.text
    assert client.get("/admin/users", headers=_hdr(admin_tok)).status_code == 200
    # A second admin may ban the first, but not then the one admin left.
    other_tok = http_sign_in(client, "other@lan.test", "Other")
    other = core.auth.user_from_token(other_tok)
    other.admin = True
    core.db.save(other)
    r = client.post(f"/admin/users/{admin.id}/ban", json={}, headers=_hdr(other_tok))
    assert r.status_code == 200, r.text
    r = client.post(f"/admin/users/{other.id}/ban", json={}, headers=_hdr(other_tok))
    assert r.status_code == 400, r.text
    assert client.get("/admin/users", headers=_hdr(other_tok)).status_code == 200


# --------------------------------------------------------------------------- engine hooks


def _race(n, fn):
    """Run fn(i) on n threads released together; return (results, errors)."""
    barrier = threading.Barrier(n)
    results, errors = [], []

    def run(i):
        barrier.wait()
        try:
            results.append(fn(i))
        except Exception as e:  # noqa: BLE001 - the test inspects what was raised
            errors.append(e)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, errors


def test_concurrent_google_signups_make_exactly_one_admin(core):
    fake_google(core.auth)
    tokens = [id_token(f"g{i}@lan.test", core.auth.google_challenge(), f"G{i}") for i in range(8)]
    results, errors = _race(8, lambda i: core.auth.login_google(tokens[i]))
    assert errors == []
    assert sum(u.admin for u, _ in results) == 1
    assert sum(core.auth.get(u.id).admin for u, _ in results) == 1


def test_concurrent_same_google_account_signs_in_once(core):
    fake_google(core.auth)
    tokens = [id_token("g@lan.test", core.auth.google_challenge(), "Gia") for _ in range(8)]
    results, errors = _race(8, lambda i: core.auth.login_google(tokens[i]))
    assert errors == []
    assert len({u.id for u, _ in results}) == 1


@pytest.mark.parametrize("method,path,payload", [
    ("post", "/auth/google", {"id_token": 5}),
    ("post", "/auth/google", {"id_token": ["x"]}),
    ("patch", "/auth/me", {"name": 5}),
    ("patch", "/auth/me", {"bio": 5}),
    ("post", "/admin/users/{member}/flag", {"reason": 5}),
])
def test_http_wrongly_typed_fields_are_400(env, method, path, payload):
    client, core, *_ = env
    admin_tok = http_sign_in(client, "admin@lan.test", "Admin")
    client.post("/auth/accept-terms", headers=_hdr(admin_tok))
    member, _ = _account(core.auth, "m@lan.test", "Member")
    client.cookies.clear()
    url = path.format(member=member.id)
    r = client.request(method, url, json=payload, headers=_hdr(admin_tok))
    assert r.status_code == 400, r.text
    assert core.auth.user_count() == 2
    assert core.auth.get(member.id).flagged is False


def test_non_finite_amounts_are_rejected(core):
    user, _ = _account(core.auth)
    core.credits.contribute(user.id, 100.0)
    for bad in (float("nan"), float("inf")):
        with pytest.raises(CreditError):
            core.credits.reserve_job(user.id, "job-nan", bad)
    assert core.credits.job_account("job-nan") is None
    assert core.credits.balance(user.id) == 100.0


@pytest.mark.parametrize("max_flops", ["nan", "inf"])
def test_http_non_finite_max_flops_abandons_job(env, max_flops):
    client, core, tiny_model, tiny_dataset = env
    token = http_sign_in(client, "ada@lan.test", "Ada")
    client.post("/auth/accept-terms", headers=_hdr(token))
    spec = json.loads(_spec(tiny_model, tiny_dataset).model_dump_json())
    spec["max_flops"] = max_flops
    r = client.post("/jobs", json=spec, headers=_hdr(token))
    assert r.status_code == 400
    assert core.jobs == {}
    r = client.post("/jobs/upload", headers=_hdr(token),
                    files={"dataset": ("train.jsonl", tiny_dataset.read_bytes())},
                    data={"model": str(tiny_model), "steps": 2, "batch_size": 2,
                          "microbatches": 1, "min_stages": 1, "max_flops": max_flops})
    assert r.status_code == 400
    assert core.jobs == {}


def test_http_unexpected_reserve_error_abandons_job(env, monkeypatch):
    client, core, tiny_model, tiny_dataset = env
    token = http_sign_in(client, "ada@lan.test", "Ada")
    client.post("/auth/accept-terms", headers=_hdr(token))

    def boom(*a, **kw):
        raise RuntimeError("db down")
    monkeypatch.setattr(core.credits, "reserve_job", boom)
    spec = json.loads(_spec(tiny_model, tiny_dataset).model_dump_json())
    spec["max_flops"] = 1e9
    with pytest.raises(RuntimeError):
        client.post("/jobs", json=spec, headers=_hdr(token))
    assert core.jobs == {}


def _race_count(n, fn):
    """Run fn from n threads released together; return how many calls succeeded."""
    gate = threading.Barrier(n)
    ok = []

    def run():
        gate.wait()
        try:
            fn()
            ok.append(1)
        except CreditError:
            pass

    threads = [threading.Thread(target=run) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return len(ok)


def test_concurrent_spends_cannot_overdraw(core):
    member, _ = _account(core.auth, "m@lan.test")
    core.credits.contribute(member.id, 40.0)
    jobs = iter(range(16))
    assert _race_count(16, lambda: core.credits.reserve_job(member.id, f"job-{next(jobs)}", 40.0)) == 1
    assert core.credits.balance(member.id) == 0.0


def test_concurrent_consume_and_settle_keep_job_books_straight(core):
    user, _ = _account(core.auth)
    core.credits.contribute(user.id, 100.0)
    core.credits.reserve_job(user.id, "job-r", 80.0)
    # Every charge lands on the account: no lost updates to spent_flops.
    assert _race_count(16, lambda: core.credits.consume_job("job-r", 1.0)) == 16
    assert core.credits.job_account("job-r").spent_flops == 16.0
    assert core.credits.lifetime_spent(user.id) == 16.0
    # The leftover is released exactly once however many settles race.
    _race_count(16, lambda: core.credits.settle_job("job-r"))
    assert core.credits.balance(user.id) == 84.0
    assert core.credits.reserved_in_flight(user.id) == 0.0
    assert core.credits.consume_job("job-r", 1.0) == 0.0


def test_http_admin_flags_reject_string_booleans(env):
    client, core, *_ = env
    admin_tok = http_sign_in(client, "admin@lan.test", "Admin")
    member_tok = http_sign_in(client, "m@lan.test", "Member")
    member = core.auth.user_from_token(member_tok)
    client.post("/auth/accept-terms", headers=_hdr(member_tok))
    # bool("false") is True: these used to ban.
    for bad in ("yes", 0, 1, [], "False"):
        r = client.post(f"/admin/users/{member.id}/ban", json={"banned": bad},
                        headers=_hdr(admin_tok))
        assert r.status_code == 400, (bad, r.text)
    assert core.auth.get(member.id).banned is False
    r = client.post(f"/admin/users/{member.id}/ban", json={"banned": "false"},
                    headers=_hdr(admin_tok))
    assert r.status_code == 200, r.text
    assert core.auth.get(member.id).banned is False
    client.post(f"/admin/users/{member.id}/ban", json={}, headers=_hdr(admin_tok))
    assert core.auth.get(member.id).banned is True


# --------------------------------------------------------------------------- engine hooks
