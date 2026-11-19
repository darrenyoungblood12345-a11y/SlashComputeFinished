"""Accounts are Google accounts. First user is admin. Sessions are bearer tokens.

Google is the directory: an account is whoever Google says holds a verified email, keyed by
Google's stable id for them (``sub``). There are no passwords here. Typing an address into a form
proves nothing about who owns it, and an account opened that way was open to whoever got there
first.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import threading
import time
from typing import Callable, Optional

import jwt
from sqlalchemy import func
from sqlmodel import select

from slashcompute.common.config import google_client_id
from slashcompute.community.fields import text
from slashcompute.coordinator.db import Database, SessionRow, User, now

SESSION_TTL_S = 30 * 24 * 3600
GOOGLE_ISSUERS = ("https://accounts.google.com", "accounts.google.com")
GOOGLE_KEYS_URL = "https://www.googleapis.com/oauth2/v3/certs"
# A sign-in has this long between asking this pool for a nonce and coming back with a token.
NONCE_TTL_S = 600.0
MAX_NONCES = 4096

__all__ = ["Auth", "AuthError", "google_client_id", "verify_google_id_token"]


class AuthError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# Google's signing keys, fetched once and kept: a sign-in is checked here, not by asking Google
# about each token.
_google_keys = jwt.PyJWKClient(GOOGLE_KEYS_URL, cache_keys=True, lifespan=6 * 3600, timeout=8)


def _google_signing_key(id_token: str):
    return _google_keys.get_signing_key_from_jwt(id_token).key


def verify_google_id_token(id_token: str, client_id: str,
                           signing_key: Callable[[str], object] = _google_signing_key) -> dict:
    """Check a Google ID token: signed by Google, issued to this app, not expired, and for a
    verified email. Returns who it names and the nonce it was issued for.
    Swap in tests via Auth.verify_google."""
    try:
        claims = jwt.decode(
            id_token, signing_key(id_token), algorithms=["RS256"], audience=client_id,
            issuer=GOOGLE_ISSUERS, leeway=30,
            options={"require": ["exp", "iat", "iss", "aud", "sub"]},
        )
    except jwt.PyJWKClientConnectionError as e:
        raise AuthError("Could not reach Google to check the sign-in.", 503) from e
    except jwt.PyJWTError as e:
        raise AuthError("Google sign-in failed.", 401) from e
    email = str(claims.get("email") or "").strip().lower()
    if not email or claims.get("email_verified") not in (True, "true"):
        raise AuthError("Google email is not verified.", 401)
    return {"email": email, "name": str(claims.get("name") or "").strip(),
            "sub": str(claims["sub"]), "nonce": str(claims.get("nonce") or "")}


class Auth:
    def __init__(self, db: Database,
                 verify_google: Callable[[str, str], dict] = verify_google_id_token) -> None:
        self.db = db
        self.verify_google = verify_google
        # Serializes "which account is this?" with creating or rebinding it, so concurrent
        # sign-ins on a fresh coordinator can't all see an empty table and become admin.
        self._create_lock = threading.Lock()
        self._nonces: dict[str, float] = {}     # nonce -> when it stops being accepted
        self._nonce_lock = threading.Lock()

    def user_count(self) -> int:
        with self.db.session() as s:
            return int(s.exec(select(func.count()).select_from(User)).one())

    # ------------------------------------------------------------ sign-in

    def google_challenge(self) -> str:
        """A nonce for one sign-in at this pool. Every pool accepts tokens for the same app, so
        without it a token handed to one pool could be replayed to sign in at another."""
        nonce, t = secrets.token_urlsafe(24), time.time()
        with self._nonce_lock:
            for old in [n for n, until in self._nonces.items() if until < t]:
                del self._nonces[old]
            while len(self._nonces) >= MAX_NONCES:      # dicts keep insertion order: oldest first
                del self._nonces[next(iter(self._nonces))]
            self._nonces[nonce] = t + NONCE_TTL_S
        return nonce

    def _spend_nonce(self, nonce: object) -> None:
        with self._nonce_lock:
            until = self._nonces.pop(nonce, None) if isinstance(nonce, str) and nonce else None
        if until is None or until < time.time():
            raise AuthError("That sign-in has expired or was not started here. Try again.", 401)

    def login_google(self, id_token: str) -> tuple[User, str]:
        client_id = google_client_id()
        if not client_id:
            raise AuthError("Google sign-in is not configured.", 501)
        id_token = text(id_token, "id_token", AuthError).strip()
        if not id_token:
            raise AuthError("Google sign-in failed.", 401)
        info = self.verify_google(id_token, client_id)
        self._spend_nonce(info.get("nonce"))
        return self._issue(self._account_for(info))

    def _account_for(self, info: dict) -> User:
        """The account of a verified Google identity: its own, a new one, or one that holds its
        email and is now bound to it. Read and written in one transaction under the lock, so
        nothing decided here rests on a copy of the row another sign-in has since changed."""
        sub, email = info["sub"], info["email"]
        admin_env = (os.environ.get("SLASHCOMPUTE_ADMIN_EMAIL") or "").strip().lower()
        with self._create_lock, self.db.session() as s:
            user = (s.exec(select(User).where(User.google_sub == sub)).first()
                    or s.exec(select(User).where(User.email == email)).first())
            if user is None:
                first = s.exec(select(func.count()).select_from(User)).one() == 0
                user = User(id=secrets.token_hex(8), email=email, google_sub=sub, admin=first,
                            name=(info.get("name") or email.split("@")[0])[:80] or "member")
            elif user.google_sub != sub:
                # The email was held by an account Google never vouched for (or by another Google
                # identity that no longer has it). Its real owner is here now: whoever had it
                # before keeps no way in.
                user.google_sub = sub
                for row in s.exec(select(SessionRow).where(SessionRow.user_id == user.id)).all():
                    s.delete(row)
            # The configured admin address, now that Google has verified who holds it.
            if admin_env and user.email == email == admin_env and not user.banned:
                user.admin = True
            s.add(user)
            s.commit()
            return user

    def _issue(self, user: User) -> tuple[User, str]:
        token = secrets.token_urlsafe(32)
        with self.db.session() as s:
            current = s.get(User, user.id)      # as it is now: a ban may have landed meanwhile
            if current is None:
                raise AuthError("Sign-in failed.", 401)
            if current.banned:
                raise AuthError("This account is banned.", 403)
            s.add(SessionRow(
                token_hash=_token_hash(token),
                user_id=current.id,
                expires_at=time.time() + SESSION_TTL_S,
            ))
            s.commit()
            return current, token

    def logout(self, token: str) -> None:
        with self.db.session() as s:
            row = s.get(SessionRow, _token_hash(token))
            if row is not None:
                s.delete(row)
                s.commit()

    def session_user(self, token: Optional[str]) -> Optional[User]:
        """User for this token, including banned accounts. Expired or missing → None."""
        if not token:
            return None
        with self.db.session() as s:
            row = s.get(SessionRow, _token_hash(token))
            if row is None or row.expires_at < time.time():
                return None
            return s.get(User, row.user_id)

    def user_from_token(self, token: Optional[str]) -> Optional[User]:
        user = self.session_user(token)
        if user is None or user.banned:
            return None
        return user

    def get(self, user_id: str) -> Optional[User]:
        return self.db.get(User, user_id)

    def get_by_email(self, email: str) -> Optional[User]:
        with self.db.session() as s:
            return s.exec(select(User).where(User.email == email)).first()

    def get_by_google_sub(self, sub: str) -> Optional[User]:
        with self.db.session() as s:
            return s.exec(select(User).where(User.google_sub == sub)).first()

    # ------------------------------------------------------------ changes

    def _update(self, user: User, **fields) -> User:
        """Change these columns of the account and nothing else. ``user`` was read for the request
        and may be stale: writing the whole copy back would undo whatever changed since, a ban or
        a new owner included."""
        current = self.db.update(User, user.id, **fields)
        if current is None:
            raise AuthError("No such account.", 404)
        return current

    def accept_terms(self, user: User) -> User:
        return self._update(user, accepted_terms_at=now())

    def update_profile(self, user: User, name: Optional[str] = None,
                       grant_split: Optional[int] = None,
                       bio: Optional[str] = None) -> User:
        changes: dict = {}
        if name is not None:
            name = text(name, "name", AuthError).strip()[:80]
            if not name:
                raise AuthError("Name cannot be empty.")
            changes["name"] = name
        if grant_split is not None:
            try:
                split = int(grant_split)
            except (TypeError, ValueError) as e:
                raise AuthError("grant_split must be 0–100.") from e
            if split < 0 or split > 100:
                raise AuthError("grant_split must be 0–100.")
            changes["grant_split"] = split
        if bio is not None:
            changes["bio"] = text(bio, "bio", AuthError).strip()[:280] or None
        return self._update(user, **changes)

    def set_banned(self, user: User, banned: bool) -> User:
        return self._update(user, banned=banned)

    def set_flagged(self, user: User, flagged: bool) -> User:
        return self._update(user, flagged=flagged)

    def public_view(self, user: User) -> dict:
        return {
            "id": user.id, "email": user.email, "name": user.name,
            "admin": user.admin, "banned": user.banned, "flagged": user.flagged,
            "grant_split": user.grant_split,
            "accepted_terms": user.accepted_terms_at is not None,
            "bio": user.bio or "",
        }
