"""Accounts for tests: Google sign-ins checked by a stand-in for Google.

The stand-in takes an "ID token" that is simply its claims as JSON, so a test says who is signing
in without a key pair. Everything after that check (the pool's nonce, finding or creating the
account, issuing the session) is the real code.
"""

from __future__ import annotations

import json

GOOGLE_CLIENT_ID = "tests.apps.googleusercontent.com"


def fake_google(auth) -> None:
    auth.verify_google = lambda id_token, client_id: json.loads(id_token)


def id_token(email: str, nonce: str, name: str = "", sub: str | None = None) -> str:
    return json.dumps({"email": email, "name": name, "sub": sub or f"sub-{email}", "nonce": nonce})


def sign_in(auth, email: str = "ada@lan.test", name: str = "Ada", sub: str | None = None):
    """(user, session token) for a Google account: created the first time, found after that."""
    fake_google(auth)
    return auth.login_google(id_token(email, auth.google_challenge(), name, sub))


def http_sign_in(client, email: str = "ada@lan.test", name: str = "Ada",
                 sub: str | None = None, *, accept_terms: bool = False) -> str:
    """The same over the pool's HTTP API (``client`` is a TestClient of it). Returns the session token."""
    fake_google(client.app.state.core.auth)
    nonce = client.post("/auth/google/challenge").json()["nonce"]
    r = client.post("/auth/google", json={"id_token": id_token(email, nonce, name, sub)})
    assert r.status_code == 200, r.text
    token = r.json()["token"]
    if accept_terms:
        r = client.post("/auth/accept-terms", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200, r.text
    return token
