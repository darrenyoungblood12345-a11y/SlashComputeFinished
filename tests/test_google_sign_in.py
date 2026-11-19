"""Signing in from the app: the shell sends the user's browser to Google and takes it from there.

The pool here is the real coordinator; only Google is stood in for (its consent page is never
opened, and its token endpoint is a function that checks what a real one checks).
"""

from __future__ import annotations

import json
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient

from accounts import GOOGLE_CLIENT_ID, fake_google, id_token
from slashcompute.common.config import EngineConfig
from slashcompute.coordinator.app import create_app
from slashcompute.launcher.controller import Launcher, LauncherSettings, stateless_http
from slashcompute.web.google_sign_in import (
    CONSENT_URL, TOKEN_URL, SignIns, pkce_challenge, redirect_uri, result_page,
)
from slashcompute.web.server import SHELL_PORT, create_shell

SHELL = f"http://127.0.0.1:{SHELL_PORT}"   # the shell refuses any Host but loopback
POOL = "http://testserver"
CALLBACK = "/api/auth/google/callback"


@pytest.fixture(autouse=True)
def _unknown_cmdlines(monkeypatch):
    monkeypatch.setattr("slashcompute.launcher.controller.process_cmdline", lambda pid: None)


class FakeGoogle:
    """Google's token endpoint: a code is redeemed once, by whoever holds the PKCE verifier for
    the challenge it was issued against, and yields an ID token carrying the request's nonce."""

    def __init__(self, email="ada@lan.test"):
        self.email = email
        self.codes: dict[str, dict] = {}
        self.forms: list[dict] = []

    def consent(self, url: str, code: str = "code-1") -> dict:
        """The user approves on the consent page ``url``. Returns what the browser is sent back with."""
        assert url.startswith(CONSENT_URL + "?")
        q = {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}
        self.codes[code] = q
        return {"state": q["state"], "code": code}

    def token(self, form: dict):
        self.forms.append(form)
        asked = self.codes.pop(form.get("code"), None)
        ok = (asked is not None and form.get("grant_type") == "authorization_code"
              and form.get("client_id") == asked["client_id"]
              and form.get("redirect_uri") == asked["redirect_uri"]
              and pkce_challenge(form.get("code_verifier", "")) == asked["code_challenge"])
        if not ok:
            return _response({"error": "invalid_grant", "error_description": "Bad Request"}, 400)
        return _response({"id_token": id_token(self.email, asked["nonce"], "Ada"), "access_token": "x"})


def _response(payload, status=200):
    class R:
        status_code = status
        content = json.dumps(payload).encode()
        headers = {"content-type": "application/json"}

        def json(self):
            return payload

    return R()


class Http:
    """What the shell dials: the pool for everything, except Google's token endpoint."""

    def __init__(self, pool, google: FakeGoogle):
        self.pool, self.google = pool, google

    def post(self, url, **kw):
        if url == TOKEN_URL:
            return self.google.token(kw["data"])
        return self.pool.post(url, **kw)

    def __getattr__(self, name):
        return getattr(self.pool, name)


@pytest.fixture
def app(tmp_path):
    """(shell client, launcher, the pool's core, Google, the URLs sent to the browser)."""
    coord_app = create_app(EngineConfig(home=tmp_path / "coord", scheduler_tick_s=0.05))
    with TestClient(coord_app) as coord:
        fake_google(coord_app.state.core.auth)
        google, opened = FakeGoogle(), []
        launcher = Launcher(
            home=tmp_path / "shell", python="/opt/venv/bin/python", popen=lambda argv, **_: None,
            http=Http(stateless_http(transport=coord._transport), google),
            discover_fn=lambda timeout=5.0: None, lan_ip_fn=lambda: "192.168.1.20",
            port_free_fn=lambda host, port: True,
        )
        launcher.save_settings(LauncherSettings(mode="join", url=POOL))
        shell = create_shell(launcher, open_browser=lambda url: opened.append(url) or True)
        with TestClient(shell, base_url=SHELL) as c:
            yield c, launcher, coord_app.state.core, google, opened


def test_sign_in_goes_through_the_browser_and_ends_with_a_stored_session(app):
    c, launcher, core, google, opened = app
    assert c.get("/api/auth/google/status").json()["state"] == "idle"
    assert c.post("/api/auth/google/start").json() == {"ok": True}
    assert c.get("/api/auth/google/status").json()["state"] == "waiting"

    [url] = opened
    q = {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}
    assert q["client_id"] == GOOGLE_CLIENT_ID and q["response_type"] == "code"
    assert q["redirect_uri"] == f"http://127.0.0.1:{SHELL_PORT}{CALLBACK}"
    assert set(q["scope"].split()) == {"openid", "email", "profile"}
    assert q["code_challenge_method"] == "S256" and len(q["code_challenge"]) == 43
    assert q["nonce"] in core.auth._nonces                   # the pool's own, so the token is for it

    r = c.get(CALLBACK, params=google.consent(url))
    assert r.status_code == 200 and "You are signed in" in r.text
    assert r.headers["cache-control"] == "no-store"
    s = launcher.load_settings()
    assert s.session_url == POOL and core.auth.user_from_token(s.session_token).email == "ada@lan.test"
    assert s.session_token not in r.text                     # the token never reaches a page
    assert c.get("/api/auth/google/status").json()["state"] == "done"
    assert c.get("/api/settings").json()["has_session"] is True
    assert c.get("/api/coord/auth/me").json()["user"]["email"] == "ada@lan.test"
    assert "client_secret" not in google.forms[0]            # none configured: none sent


def test_the_client_secret_is_sent_when_the_build_has_one(app, monkeypatch):
    c, _, _, google, opened = app
    monkeypatch.setenv("SLASHCOMPUTE_GOOGLE_CLIENT_SECRET", "not-really-secret")
    c.post("/api/auth/google/start")
    assert c.get(CALLBACK, params=google.consent(opened[0])).status_code == 200
    assert google.forms[0]["client_secret"] == "not-really-secret"


def test_the_browser_is_only_let_back_in_once_and_only_for_a_sign_in_started_here(app):
    c, launcher, core, google, opened = app
    r = c.get(CALLBACK, params={"state": "made-up", "code": "code-1"})
    assert r.status_code == 400 and "expired or was already used" in r.text
    assert c.get(CALLBACK).status_code == 400
    assert google.forms == []                                # nothing was sent to Google for it

    c.post("/api/auth/google/start")
    back = google.consent(opened[0])
    assert c.get(CALLBACK, params=back).status_code == 200
    first = launcher.load_settings().session_token
    google.codes["code-1"] = {}                              # even if Google took the code again
    assert c.get(CALLBACK, params=back).status_code == 400
    assert launcher.load_settings().session_token == first
    assert core.auth.user_count() == 1


def test_a_code_from_another_sign_in_does_not_pass(app):
    """PKCE: the code is only worth something to the shell that holds the verifier it was asked with."""
    c, launcher, _, google, opened = app
    c.post("/api/auth/google/start")
    c.post("/api/auth/google/start")
    mine, other = (google.consent(url, code) for url, code in zip(opened, ("code-a", "code-b")))
    r = c.get(CALLBACK, params={"state": mine["state"], "code": other["code"]})
    assert r.status_code == 502 and "Google refused" in r.text
    assert launcher.load_settings().session_token == ""
    assert c.get("/api/auth/google/status").json()["state"] == "failed"


def test_declining_at_google_is_reported_and_signs_nobody_in(app):
    c, launcher, _, google, opened = app
    c.post("/api/auth/google/start")
    state = google.consent(opened[0])["state"]
    r = c.get(CALLBACK, params={"state": state, "error": "access_denied"})
    assert r.status_code == 400 and "access_denied" in r.text
    assert c.get("/api/auth/google/status").json() == {
        "state": "failed", "message": "Google did not sign you in (access_denied)."}
    assert launcher.load_settings().session_token == ""


def test_a_pool_that_refuses_the_account_is_reported(app):
    c, launcher, core, google, opened = app
    c.post("/api/auth/google/start")
    c.get(CALLBACK, params=google.consent(opened[0]))
    core.auth.set_banned(core.auth.get_by_email("ada@lan.test"), True)
    c.post("/api/settings", json={"session_token": ""})
    c.post("/api/auth/google/start")
    r = c.get(CALLBACK, params=google.consent(opened[1]))
    assert r.status_code == 403 and "banned" in r.text
    assert launcher.load_settings().session_token == ""


def test_a_session_for_a_pool_the_app_has_left_is_not_kept(app):
    c, launcher, _, google, opened = app
    c.post("/api/auth/google/start")
    c.post("/api/settings", json={"mode": "join", "url": "10.0.0.2"})    # Connect moved on meanwhile
    assert c.get(CALLBACK, params=google.consent(opened[0])).status_code == 200
    assert launcher.load_settings().session_token == ""


def test_sign_in_cannot_start_without_a_browser_or_a_client_id(app, monkeypatch, tmp_path):
    c, launcher, _, _, opened = app
    shell = create_shell(launcher, open_browser=lambda url: False)
    with TestClient(shell, base_url=SHELL) as no_browser:
        r = no_browser.post("/api/auth/google/start")
        assert r.status_code == 500 and "browser" in r.json()["detail"]
        assert no_browser.get("/api/auth/google/status").json()["state"] == "failed"
    monkeypatch.delenv("SLASHCOMPUTE_GOOGLE_CLIENT_ID")
    r = c.post("/api/auth/google/start")
    assert r.status_code == 501 and opened == []


def test_a_pool_built_for_another_app_is_refused_before_the_browser_opens(app, monkeypatch):
    c, _, _, _, opened = app
    monkeypatch.setattr("slashcompute.community.http.google_client_id", lambda: "other.apps.googleusercontent.com")
    r = c.post("/api/auth/google/start")
    assert r.status_code == 409 and "different build" in r.json()["detail"]
    assert opened == []


def test_the_callback_is_refused_from_another_site(app):
    """The shell answers only to its own loopback name: a page elsewhere cannot aim a browser's
    request at it under another Host."""
    c, _, _, google, opened = app
    c.post("/api/auth/google/start")
    r = c.get(CALLBACK, params=google.consent(opened[0]), headers={"host": "evil.example"})
    assert r.status_code == 403


def test_pending_sign_ins_are_bounded_and_single_use():
    sign_ins = SignIns()
    states = []
    for i in range(40):
        url = sign_ins.begin(POOL, f"nonce-{i}", GOOGLE_CLIENT_ID, 8766)
        states.append(parse_qs(urlsplit(url).query)["state"][0])
    assert sign_ins.take(states[0]) is None                  # pushed out by newer ones
    assert sign_ins.take(states[-1]).pool == POOL
    assert sign_ins.take(states[-1]) is None
    assert sign_ins.take("") is None


def test_the_result_page_escapes_what_it_is_told():
    page = result_page(False, 'Google did not sign you in (<script>alert("x")</script>).')
    assert "<script>" not in page and "&lt;script&gt;" in page
    assert redirect_uri(8776) == "http://127.0.0.1:8776/api/auth/google/callback"
