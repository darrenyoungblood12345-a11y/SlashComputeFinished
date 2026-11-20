"""Accounts are Google accounts: who Google verifies, keyed by Google's id for them.

There is no password to guess, register first or leave behind, and nothing a request read
earlier can be written back over what the account has become since.
"""

from __future__ import annotations

import sqlite3
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from accounts import GOOGLE_CLIENT_ID, fake_google, http_sign_in, id_token, sign_in
from slashcompute.common.config import WELCOME_FLOPS, EngineConfig
from slashcompute.community import auth as auth_module
from slashcompute.community.auth import Auth, AuthError, verify_google_id_token
from slashcompute.community.credits import POT_ID, Credits
from slashcompute.coordinator.app import create_app
from slashcompute.coordinator.core import Coordinator
from slashcompute.coordinator.db import Database, SessionRow, User


@pytest.fixture
def core(tmp_path):
    return Coordinator(EngineConfig(home=tmp_path / "home", verify_rate=0.0))


@pytest.fixture
def env(tmp_path):
    app = create_app(EngineConfig(home=tmp_path / "home", scheduler_tick_s=0.05, verify_rate=0.0))
    with TestClient(app) as client:
        yield client, app.state.core


def _hdr(token):
    return {"Authorization": f"Bearer {token}"}


# --------------------------------------------------------------------------- Google's token

GOOGLE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
SOMEONE_ELSES_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _claims(**over):
    t = int(time.time())
    return {"iss": "https://accounts.google.com", "aud": GOOGLE_CLIENT_ID, "sub": "1234567890",
            "email": "Ada@Example.com", "email_verified": True, "name": "Ada", "nonce": "n-1",
            "iat": t, "exp": t + 3600} | over


def _signed(claims, key=GOOGLE_KEY, **kw):
    return jwt.encode(claims, key, algorithm="RS256", **kw)


def _verify(token):
    return verify_google_id_token(token, GOOGLE_CLIENT_ID, signing_key=lambda _: GOOGLE_KEY.public_key())


def test_a_token_signed_by_google_for_this_app_names_its_holder():
    assert _verify(_signed(_claims())) == {
        "email": "ada@example.com", "name": "Ada", "sub": "1234567890", "nonce": "n-1"}
    assert _verify(_signed(_claims(iss="accounts.google.com")))["sub"] == "1234567890"


@pytest.mark.parametrize("claims", [
    _claims(aud="another-app.apps.googleusercontent.com"),      # issued to some other app
    _claims(iss="https://accounts.example.com"),                # not issued by Google
    _claims(exp=int(time.time()) - 3600),                       # expired
    _claims(email_verified=False),
    _claims(email=""),
    {k: v for k, v in _claims().items() if k != "sub"},
    {k: v for k, v in _claims().items() if k != "exp"},
])
def test_a_token_that_does_not_prove_a_verified_google_identity_is_refused(claims):
    with pytest.raises(AuthError) as e:
        _verify(_signed(claims))
    assert e.value.status == 401


def test_a_token_google_did_not_sign_is_refused():
    with pytest.raises(AuthError) as e:
        _verify(_signed(_claims(), key=SOMEONE_ELSES_KEY))
    assert e.value.status == 401
    unsigned = jwt.encode(_claims(), key="", algorithm="none")
    with pytest.raises(AuthError):
        _verify(unsigned)
    with pytest.raises(AuthError):
        _verify("not-a-token")


def test_google_being_unreachable_is_not_a_bad_sign_in():
    def offline(_):
        raise jwt.PyJWKClientConnectionError("no route to host")

    with pytest.raises(AuthError) as e:
        verify_google_id_token(_signed(_claims()), GOOGLE_CLIENT_ID, signing_key=offline)
    assert e.value.status == 503


# --------------------------------------------------------------------------- signing in


def test_sign_in_creates_the_account_once_and_finds_it_after(core):
    user, token = sign_in(core.auth, "g@lan.test", "Gia")
    assert (user.email, user.name, user.google_sub) == ("g@lan.test", "Gia", "sub-g@lan.test")
    assert user.admin is True                               # the first account runs the pool
    assert core.auth.user_from_token(token).id == user.id
    again, token2 = sign_in(core.auth, "g@lan.test", "Gia")
    assert again.id == user.id and core.auth.user_count() == 1
    assert core.auth.user_from_token(token) is not None     # a second sign-in leaves the first alone
    other, _ = sign_in(core.auth, "o@lan.test", "Oz")
    assert other.admin is False


def test_sign_in_is_off_without_a_client_id(env, monkeypatch):
    client, core = env
    monkeypatch.delenv("SLASHCOMPUTE_GOOGLE_CLIENT_ID")
    with pytest.raises(AuthError) as e:
        core.auth.login_google("anything")
    assert e.value.status == 501
    assert client.get("/auth/providers").json() == {"google": False, "google_client_id": None}
    assert client.post("/auth/google/challenge").status_code == 501
    assert client.post("/auth/google", json={"id_token": "x"}).status_code == 501


def test_a_token_is_only_good_for_the_sign_in_it_was_requested_for(core, tmp_path):
    """Every pool accepts tokens issued to the same app. Without the pool's own nonce in it, a
    token handed to one pool could be replayed to sign in as that person at another."""
    fake_google(core.auth)
    token = id_token("ada@lan.test", core.auth.google_challenge())
    core.auth.login_google(token)
    with pytest.raises(AuthError) as e:
        core.auth.login_google(token)                        # used once
    assert e.value.status == 401

    elsewhere = Coordinator(EngineConfig(home=tmp_path / "elsewhere", verify_rate=0.0))
    fake_google(elsewhere.auth)
    with pytest.raises(AuthError):                           # a nonce this pool never issued
        elsewhere.auth.login_google(id_token("ada@lan.test", core.auth.google_challenge()))
    with pytest.raises(AuthError):
        elsewhere.auth.login_google(id_token("ada@lan.test", ""))
    assert elsewhere.auth.user_count() == 0


def test_a_sign_in_left_too_long_has_to_start_again(core, monkeypatch):
    fake_google(core.auth)
    monkeypatch.setattr(auth_module, "NONCE_TTL_S", -1.0)
    with pytest.raises(AuthError):
        core.auth.login_google(id_token("ada@lan.test", core.auth.google_challenge()))


def test_nonces_nobody_came_back_for_do_not_pile_up(core, monkeypatch):
    monkeypatch.setattr(auth_module, "MAX_NONCES", 4)
    nonces = [core.auth.google_challenge() for _ in range(10)]
    assert len(core.auth._nonces) == 4
    fake_google(core.auth)
    core.auth.login_google(id_token("ada@lan.test", nonces[-1]))     # the newest still works
    with pytest.raises(AuthError):
        core.auth.login_google(id_token("ada@lan.test", nonces[0]))


def test_admin_email_is_granted_to_whoever_google_says_holds_it(core, monkeypatch):
    monkeypatch.setenv("SLASHCOMPUTE_ADMIN_EMAIL", "Ops@lan.test ")
    sign_in(core.auth, "first@lan.test")
    other, _ = sign_in(core.auth, "other@lan.test")
    assert other.admin is False
    ops, token = sign_in(core.auth, "ops@lan.test")
    assert ops.admin is True and core.auth.user_from_token(token).admin is True


def test_a_banned_account_cannot_sign_in(core):
    user, token = sign_in(core.auth)
    core.auth.set_banned(user, True)
    with pytest.raises(AuthError) as e:
        sign_in(core.auth)
    assert e.value.status == 403
    assert core.auth.user_from_token(token) is None
    banned = core.auth.session_user(token)
    assert banned is not None and banned.banned is True


def test_logout_drops_session(core):
    _, token = sign_in(core.auth)
    core.auth.logout(token)
    assert core.auth.user_from_token(token) is None


# --------------------------------------------------------------------------- accounts from before


def _password_account(db, email, sub=None, password_hash="210000$00ff$00ff"):
    """An account as the email-and-password sign-up left it, with a live session."""
    user = User(id=f"id-{email}", email=email, name="Mallory", password_hash=password_hash,
                google_sub=sub)
    token = f"session-of-{email}"
    db.add(user, SessionRow(token_hash=auth_module._token_hash(token), user_id=user.id,
                            expires_at=time.time() + 3600))
    return user, token


def test_upgrading_ends_every_session_a_password_opened(tmp_path):
    """Whoever registered an address with a password may not own it: nobody checked."""
    db = Database(tmp_path / "coordinator.db")
    _, squatter = _password_account(db, "victim@lan.test")
    _, linked = _password_account(db, "linked@lan.test", sub="sub-linked")
    _, google = _password_account(db, "g@lan.test", sub="sub-g", password_hash="google$abc")

    db = Database(tmp_path / "coordinator.db")              # the new version opens the same file
    auth = Auth(db)
    assert auth.session_user(squatter) is None
    assert auth.session_user(linked) is None                # it still had the password it was made with
    assert auth.session_user(google) is not None            # only ever opened through Google
    assert auth.get_by_email("victim@lan.test").password_hash == ""
    assert auth.user_count() == 3                           # the accounts and what they hold stay


def test_a_pool_from_when_there_were_grants_opens_with_its_balances(tmp_path):
    """Grants are gone, and so is the share of earnings an account could send to the community
    pot. A pool's database from before still has their tables, columns and ledger rows: it
    opens, nobody's balance moves, and what is earned from now on is all the earner's."""
    path = tmp_path / "coordinator.db"
    Database(path)
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE grants (
                id VARCHAR PRIMARY KEY, author_id VARCHAR NOT NULL, title VARCHAR NOT NULL,
                body VARCHAR NOT NULL, goal_flops FLOAT NOT NULL, received_flops FLOAT NOT NULL,
                status VARCHAR NOT NULL, created_at FLOAT NOT NULL);
            CREATE TABLE grant_comments (
                id INTEGER PRIMARY KEY, grant_id VARCHAR NOT NULL, user_id VARCHAR NOT NULL,
                body VARCHAR NOT NULL, created_at FLOAT NOT NULL);
            ALTER TABLE credit_txns ADD COLUMN grant_id VARCHAR;
            CREATE INDEX ix_credit_txns_grant ON credit_txns (grant_id) WHERE grant_id IS NOT NULL;
            INSERT INTO users (id, email, password_hash, name, admin, banned, flagged, grant_split, created_at)
            VALUES ('donor', 'd@lan.test', '', 'Donor', 1, 0, 0, 40, 1.0),
                   ('author', 'a@lan.test', '', 'Author', 0, 0, 0, 0, 1.0);
            INSERT INTO grants VALUES ('g1', 'author', 'Need compute', 'For a parser.', 100.0, 30.0, 'approved', 1.0);
            INSERT INTO grant_comments VALUES (1, 'g1', 'donor', 'good luck', 2.0);
            INSERT INTO credit_txns (user_id, kind, amount, grant_id, note, created_at)
            VALUES ('donor', 'generated', 100.0, NULL, NULL, 1.0),
                   ('donor', 'earn', 60.0, NULL, NULL, 1.0),
                   ('__pot__', 'earn', 40.0, NULL, 'tithe from donor', 1.0),
                   ('donor', 'donate', -20.0, 'g1', NULL, 2.0),
                   ('author', 'receive', 20.0, 'g1', NULL, 2.0),
                   ('__pot__', 'allocate', -10.0, 'g1', NULL, 3.0),
                   ('author', 'receive', 10.0, 'g1', NULL, 3.0);
        """)

    db = Database(path)                                     # the new version opens the same file
    auth, credits = Auth(db), Credits(db)
    assert (credits.balance("donor"), credits.balance("author"), credits.balance(POT_ID)) == (40.0, 30.0, 30.0)
    assert credits.lifetime_earned("donor") == 100.0
    assert [t["kind"] for t in credits.list_txns("donor")["items"]] == ["donate", "earn", "generated"]

    credits.credit_host("donor", 50.0)                      # the 40% it once chose no longer applies
    assert (credits.balance("donor"), credits.balance(POT_ID)) == (90.0, 30.0)
    newcomer, token = sign_in(auth, "new@lan.test", "New")  # an account made now fits the old table
    assert auth.session_user(token).id == newcomer.id and auth.user_count() == 3
    assert "grant_split" not in auth.public_view(auth.get("donor"))
    with sqlite3.connect(path) as conn:                     # nothing of the old pool was deleted
        assert conn.execute("SELECT count(*) FROM grants").fetchone() == (1,)
        assert conn.execute("SELECT count(*) FROM grant_comments").fetchone() == (1,)


def test_google_sign_in_takes_over_the_account_holding_its_email(core):
    """Hermes scan google-password-account-preclaim-0002: someone registers the victim's address
    before the victim arrives, and kept access next to them afterwards."""
    squatter, squatter_token = _password_account(core.db, "victim@lan.test")
    core.credits.contribute(squatter.id, 500.0)

    owner, token = sign_in(core.auth, "victim@lan.test", "Vic", sub="sub-victim")
    assert owner.id == squatter.id and owner.google_sub == "sub-victim"
    assert core.credits.balance(owner.id) == 500.0          # the account and its credits carry over
    assert core.auth.session_user(squatter_token) is None
    assert core.auth.user_from_token(token).id == owner.id
    # Signing in again keeps the owner's own sessions.
    _, token2 = sign_in(core.auth, "victim@lan.test", "Vic", sub="sub-victim")
    assert core.auth.user_from_token(token) is not None
    assert core.auth.user_from_token(token2) is not None


def test_concurrent_takeovers_keep_every_new_session(core):
    import threading

    _, squatter_token = _password_account(core.db, "victim@lan.test")
    fake_google(core.auth)
    tokens = [id_token("victim@lan.test", core.auth.google_challenge(), sub="sub-victim")
              for _ in range(8)]
    start, sessions, errors = threading.Barrier(8), [], []

    def go(tok):
        start.wait()
        try:
            sessions.append(core.auth.login_google(tok)[1])
        except Exception as e:      # noqa: BLE001 - the test inspects what was raised
            errors.append(e)

    threads = [threading.Thread(target=go, args=(t,)) for t in tokens]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert core.auth.session_user(squatter_token) is None
    # Only the first sign-in revokes: a later one must not sign the others out again.
    assert all(core.auth.user_from_token(s) is not None for s in sessions)


def test_there_is_no_password_way_in(env):
    client, core = env
    for path in ("/auth/register", "/auth/login"):
        r = client.post(path, json={"email": "ada@lan.test", "password": "password1", "name": "Ada"})
        assert r.status_code in (404, 405), path
    assert core.auth.user_count() == 0
    assert not hasattr(core.auth, "register") and not hasattr(core.auth, "login")


# --------------------------------------------------------------------------- stale copies


def test_a_request_holding_an_old_copy_cannot_undo_what_happened_since(core):
    """Every change used to save the whole row it had read for the request. A profile edit that
    raced a takeover wrote the old owner back; one that raced a ban lifted it."""
    admin, _ = sign_in(core.auth, "admin@lan.test")
    user, _ = sign_in(core.auth, "victim@lan.test", sub="sub-old")
    stale = core.auth.get(user.id)                           # what a slow request is still holding

    sign_in(core.auth, "victim@lan.test", sub="sub-new")     # the account changes hands
    core.auth.set_banned(user, True)                         # and is banned

    core.auth.update_profile(stale, name="Renamed", bio="hi")
    core.auth.accept_terms(stale)
    core.auth.flag_user(admin, stale, "spam")
    now = core.auth.get(user.id)
    assert (now.google_sub, now.banned) == ("sub-new", True)
    assert (now.name, now.bio, now.flagged) == ("Renamed", "hi", True)
    assert now.accepted_terms_at is not None


def test_a_ban_that_lands_during_a_sign_in_still_stops_it(core):
    user, _ = sign_in(core.auth)
    stale = core.auth.get(user.id)
    core.auth.set_banned(user, True)
    with pytest.raises(AuthError) as e:
        core.auth._issue(stale)
    assert e.value.status == 403


def test_changing_an_account_that_is_gone_is_an_error_not_a_resurrection(core):
    ghost = User(id="nobody", email="ghost@lan.test", name="Ghost")
    with pytest.raises(AuthError) as e:
        core.auth.update_profile(ghost, name="Back")
    assert e.value.status == 404
    assert core.auth.get("nobody") is None


# --------------------------------------------------------------------------- over HTTP


def test_http_sign_in_cookie_me_and_sign_out(env):
    client, core = env
    assert client.get("/auth/providers").json() == {"google": True, "google_client_id": GOOGLE_CLIENT_ID}
    fake_google(core.auth)
    challenge = client.post("/auth/google/challenge").json()
    assert challenge["client_id"] == GOOGLE_CLIENT_ID
    r = client.post("/auth/google", json={"id_token": id_token("ada@lan.test", challenge["nonce"], "Ada")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["user"]["admin"] is True and body["user"]["accepted_terms"] is False
    assert r.cookies.get("slashcompute_session")
    me = client.get("/auth/me", headers=_hdr(body["token"])).json()
    assert me["user"]["email"] == "ada@lan.test"
    assert me["credits"]["balance"] == WELCOME_FLOPS
    patched = client.patch("/auth/me", json={"bio": "hello"}, headers=_hdr(body["token"]))
    assert patched.json()["user"]["bio"] == "hello"
    client.post("/auth/logout")
    assert client.get("/auth/me").json()["user"] is None
    # The same token again: its nonce is spent.
    r = client.post("/auth/google", json={"id_token": id_token("ada@lan.test", challenge["nonce"], "Ada")})
    assert r.status_code == 401


def test_http_welcome_credit_is_granted_once_however_often_one_signs_in(env):
    client, core = env
    token = http_sign_in(client, "g@lan.test", "Gia")
    http_sign_in(client, "g@lan.test", "Gia")
    assert core.credits.balance(core.auth.user_from_token(token).id) == WELCOME_FLOPS
