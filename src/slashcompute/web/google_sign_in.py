"""Sign in with Google from the desktop app, through the system browser.

Google refuses to show its sign-in page inside an embedded web view (the app's window is one),
and gives a desktop app one supported way instead: open the user's own browser, and have Google
send it back to a loopback address the app listens on. The shell already listens on one.

1. The shell asks the pool for a nonce, and opens Google's consent page in the browser.
2. Google sends the browser back to the shell with a one-time code.
3. The shell swaps the code for an ID token and hands that to the pool, which checks it and
   issues a session. The token never reaches a page.

PKCE ties the code to the shell that asked for it, ``state`` ties the browser's return to a
sign-in this shell started, and the nonce ties the ID token to the pool it was requested for.
"""

from __future__ import annotations

import base64
import hashlib
import html
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlencode

CONSENT_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
CALLBACK_PATH = "/api/auth/google/callback"
# How long the browser has to come back.
PENDING_TTL_S = 600.0
MAX_PENDING = 32


class SignInError(Exception):
    pass


@dataclass(frozen=True)
class Pending:
    """A sign-in the browser has not come back from yet."""

    pool: str           # the coordinator it was started for, as the shell dials it
    verifier: str       # PKCE: proves to Google that the code is being redeemed by who asked
    client_id: str
    expires: float


def pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def redirect_uri(port: int) -> str:
    # The loopback address itself: "localhost" can resolve elsewhere or be firewalled.
    return f"http://127.0.0.1:{port}{CALLBACK_PATH}"


class SignIns:
    """The sign-ins in flight and how the last one ended. Each is redeemed at most once."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[str, Pending] = {}
        self._outcome: dict = {"state": "idle", "message": ""}

    def begin(self, pool: str, nonce: str, client_id: str, port: int) -> str:
        """Record a new sign-in and return the consent page to open for it."""
        state, verifier = secrets.token_urlsafe(24), secrets.token_urlsafe(48)
        t = time.time()
        with self._lock:
            for old in [k for k, p in self._pending.items() if p.expires < t]:
                del self._pending[old]
            while len(self._pending) >= MAX_PENDING:
                del self._pending[next(iter(self._pending))]
            self._pending[state] = Pending(pool, verifier, client_id, t + PENDING_TTL_S)
            self._outcome = {"state": "waiting", "message": ""}
        return CONSENT_URL + "?" + urlencode({
            "client_id": client_id,
            "redirect_uri": redirect_uri(port),
            "response_type": "code",
            "scope": "openid email profile",
            "code_challenge": pkce_challenge(verifier),
            "code_challenge_method": "S256",
            "state": state,
            "nonce": nonce,
            "prompt": "select_account",
        })

    def take(self, state: str) -> Optional[Pending]:
        with self._lock:
            pending = self._pending.pop(state, None) if state else None
        return pending if pending is not None and pending.expires >= time.time() else None

    def finish(self, ok: bool, message: str = "") -> None:
        with self._lock:
            self._outcome = {"state": "done" if ok else "failed", "message": message}

    def outcome(self) -> dict:
        with self._lock:
            return dict(self._outcome)


def exchange_code(http, code: str, pending: Pending, client_secret: str, port: int) -> str:
    """Swap Google's one-time code for the ID token. ``http`` is a synchronous httpx-style client."""
    form = {
        "grant_type": "authorization_code",
        "code": code,
        "code_verifier": pending.verifier,
        "client_id": pending.client_id,
        "redirect_uri": redirect_uri(port),
    }
    if client_secret:
        form["client_secret"] = client_secret
    try:
        r = http.post(TOKEN_URL, data=form, timeout=15.0)
    except Exception as e:
        raise SignInError(f"Could not reach Google: {e}") from e
    try:
        data = r.json()
    except Exception:
        data = None
    if r.status_code != 200 or not isinstance(data, dict) or not isinstance(data.get("id_token"), str):
        why = data.get("error_description") or data.get("error") if isinstance(data, dict) else None
        raise SignInError(f"Google refused the sign-in{f' ({why})' if why else ''}.")
    return data["id_token"]


def result_page(ok: bool, message: str) -> str:
    """What the browser tab shows once it is back: the app's window is where things continue."""
    title = "You are signed in" if ok else "Sign-in did not finish"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>/compute · {html.escape(title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body {{ margin: 0; min-height: 100vh; display: grid; place-items: center; background: #0b0b0b;
         color: #f2f2f2; font: 16px/1.5 -apple-system, system-ui, sans-serif; }}
  main {{ max-width: 28rem; padding: 2rem; }}
  h1 {{ font-size: 1.5rem; margin: 0 0 .5rem; color: {"#d6ff00" if ok else "#ff6b5e"}; }}
  p {{ margin: 0; color: #b5b5b5; }}
</style></head>
<body><main><h1>{html.escape(title)}</h1><p>{html.escape(message)}</p></main></body></html>"""
