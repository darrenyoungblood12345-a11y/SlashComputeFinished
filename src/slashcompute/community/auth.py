"""Email + password accounts. First user is admin. Sessions are bearer tokens."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import threading
import time
from typing import Callable, Optional

from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from slashcompute.community.fields import text
from slashcompute.coordinator.db import Database, SessionRow, User, now

ITERATIONS = 210_000
SESSION_TTL_S = 30 * 24 * 3600
# Something before the @, a dotted domain after it, no whitespace anywhere.
EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


class AuthError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def _hash_password(password: str, salt: Optional[bytes] = None) -> str:
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)
    return f"{ITERATIONS}${salt.hex()}${dk.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        iters_s, salt_hex, dk_hex = stored.split("$", 2)
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iters_s),
        )
        return hmac.compare_digest(dk.hex(), dk_hex)
    except (ValueError, TypeError):
        return False


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def normalize_email(email: str) -> str:
    return text(email, "email", AuthError).strip().lower()


def google_client_id() -> str:
    return (os.environ.get("SLASHCOMPUTE_GOOGLE_CLIENT_ID") or "").strip()


def verify_google_id_token(id_token: str, client_id: str) -> dict:
    """Check a Google ID token. Swap in tests via Auth.verify_google."""
    import httpx

    r = httpx.get(
        "https://oauth2.googleapis.com/tokeninfo",
        params={"id_token": id_token},
        timeout=8.0,
    )
    if r.status_code != 200:
        raise AuthError("Google sign-in failed.", 401)
    data = r.json()
    if data.get("aud") != client_id:
        raise AuthError("Google token is for a different app.", 401)
    email = (data.get("email") or "").strip().lower()
    if not email or str(data.get("email_verified")).lower() not in ("true", "1"):
        raise AuthError("Google email is not verified.", 401)
    sub = data.get("sub")
    if not sub:
        raise AuthError("Google sign-in failed.", 401)
    return {"email": email, "name": (data.get("name") or "").strip(), "sub": str(sub)}


class Auth:
    def __init__(self, db: Database,
                 verify_google: Callable[[str, str], dict] = verify_google_id_token) -> None:
        self.db = db
        self.verify_google = verify_google
        # Serializes "is this the first account?" with the insert, so concurrent
        # sign-ups on a fresh coordinator can't all see an empty table and become admin.
        self._create_lock = threading.Lock()

    def user_count(self) -> int:
        with self.db.session() as s:
            return len(s.exec(select(User)).all())

    def _create(self, user: User) -> Optional[User]:
        """Insert a new account, making it admin if it is the first. None if the email is taken."""
        with self._create_lock:
            if self.get_by_email(user.email) is not None:
                return None
            user.admin = user.admin or self.user_count() == 0
            try:
                self.db.add(user)
            except IntegrityError:
                return None
        return user

    def register(self, email: str, password: str, name: str) -> User:
        email = normalize_email(email)
        if not EMAIL_RE.fullmatch(email):
            raise AuthError("Enter a real email address.")
        password = text(password, "password", AuthError)
        if len(password) < 8:
            raise AuthError("Password must be at least 8 characters.")
        name = (text(name, "name", AuthError) or email.split("@")[0]).strip()[:80] or "member"
        if self.get_by_email(email) is not None:
            raise AuthError("That email is already registered.")
        # Never admin from SLASHCOMPUTE_ADMIN_EMAIL here: typing an address into this form
        # proves nothing about who owns it. Only a verified Google sign-in grants that.
        user = self._create(User(
            id=secrets.token_hex(8),
            email=email,
            password_hash=_hash_password(password),
            name=name,
        ))
        if user is None:
            raise AuthError("That email is already registered.")
        return user

    def login(self, email: str, password: str) -> tuple[User, str]:
        user = self.get_by_email(normalize_email(email))
        password = text(password, "password", AuthError)
        if user is None or user.password_hash.startswith("google$") or not _verify_password(password, user.password_hash):
            raise AuthError("Email or password is wrong.", 401)
        return self._issue(user)

    def _issue(self, user: User) -> tuple[User, str]:
        if user.banned:
            raise AuthError("This account is banned.", 403)
        token = secrets.token_urlsafe(32)
        self.db.add(SessionRow(
            token_hash=_token_hash(token),
            user_id=user.id,
            expires_at=time.time() + SESSION_TTL_S,
        ))
        return user, token

    def login_google(self, id_token: str) -> tuple[User, str]:
        client_id = google_client_id()
        if not client_id:
            raise AuthError("Google sign-in is not configured.", 501)
        id_token = text(id_token, "id_token", AuthError).strip()
        if not id_token:
            raise AuthError("Google sign-in failed.", 401)
        info = self.verify_google(id_token, client_id)
        admin_env = (os.environ.get("SLASHCOMPUTE_ADMIN_EMAIL") or "").strip().lower()
        user = self.get_by_google_sub(info["sub"]) or self.get_by_email(info["email"])
        if user is None:
            # A concurrent sign-in may create this account first; then just use theirs.
            user = self._create(User(
                id=secrets.token_hex(8),
                email=info["email"],
                password_hash="google$" + secrets.token_hex(16),
                name=(info["name"] or info["email"].split("@")[0])[:80] or "member",
                admin=admin_env == info["email"],
                google_sub=info["sub"],
            )) or self.get_by_email(info["email"])
        if user is None:
            raise AuthError("Google sign-in failed.", 401)
        if user.google_sub != info["sub"] or not user.password_hash.startswith("google$"):
            user = self._reclaim_for_google(user.id, info["sub"])
        # The configured admin address may have been registered with a password before its
        # owner arrived. Google has now verified it, and the reclaim above left no other way in.
        if admin_env and user.email == info["email"] == admin_env and not user.admin and not user.banned:
            user.admin = True
            self.db.save(user)
        return self._issue(user)

    def _reclaim_for_google(self, user_id: str, sub: str) -> User:
        """Bind a verified Google identity to an existing account and revoke what came before.

        Nobody checked who registered this email: left in place, the password and sessions of
        whoever claimed it first would keep opening the account next to its real owner.
        """
        with self._create_lock, self.db.session() as s:
            user = s.get(User, user_id)
            # A concurrent sign-in got here first: keep the session it has just issued.
            if user.google_sub == sub and user.password_hash.startswith("google$"):
                return user
            user.google_sub = sub
            user.password_hash = "google$" + secrets.token_hex(16)
            for row in s.exec(select(SessionRow).where(SessionRow.user_id == user_id)).all():
                s.delete(row)
            s.add(user)
            s.commit()
            return user

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

    def accept_terms(self, user: User) -> User:
        user.accepted_terms_at = now()
        self.db.save(user)
        return user

    def update_profile(self, user: User, name: Optional[str] = None,
                       grant_split: Optional[int] = None,
                       bio: Optional[str] = None) -> User:
        if name is not None:
            name = text(name, "name", AuthError).strip()[:80]
            if not name:
                raise AuthError("Name cannot be empty.")
            user.name = name
        if grant_split is not None:
            try:
                split = int(grant_split)
            except (TypeError, ValueError) as e:
                raise AuthError("grant_split must be 0–100.") from e
            if split < 0 or split > 100:
                raise AuthError("grant_split must be 0–100.")
            user.grant_split = split
        if bio is not None:
            user.bio = text(bio, "bio", AuthError).strip()[:280] or None
        self.db.save(user)
        return user

    def set_banned(self, user: User, banned: bool) -> User:
        user.banned = banned
        self.db.save(user)
        return user

    def set_flagged(self, user: User, flagged: bool) -> User:
        user.flagged = flagged
        self.db.save(user)
        return user

    def public_view(self, user: User) -> dict:
        return {
            "id": user.id, "email": user.email, "name": user.name,
            "admin": user.admin, "banned": user.banned, "flagged": user.flagged,
            "grant_split": user.grant_split,
            "accepted_terms": user.accepted_terms_at is not None,
            "bio": user.bio or "",
        }
