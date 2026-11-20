"""HTTP surface for accounts, credits, and admin."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Request
from sqlmodel import select

from slashcompute.community.auth import AuthError, google_client_id
from slashcompute.community.terms import TERMS
from slashcompute.common.jsonbool import body_bool
from slashcompute.coordinator.core import Coordinator
from slashcompute.coordinator.db import User


def _token(request: Request, authorization: Optional[str]) -> Optional[str]:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return request.cookies.get("slashcompute_session")


def _raise(exc: Exception):
    status = getattr(exc, "status", 400)
    raise HTTPException(status, str(exc)) from exc


def mount_community(app, core: Coordinator) -> None:
    r = APIRouter(tags=["community"])

    def require(request: Request, authorization: Optional[str] = None) -> User:
        user = core.auth.session_user(_token(request, authorization))
        if user is None:
            raise HTTPException(401, "Sign in first.")
        if user.banned:
            raise HTTPException(403, "This account is banned.")
        return user

    @r.get("/auth/terms")
    def terms():
        return {"text": TERMS}

    @r.get("/auth/providers")
    def providers():
        cid = google_client_id()
        return {"google": bool(cid), "google_client_id": cid or None}

    @r.post("/auth/google/challenge")
    def google_challenge():
        """Start a sign-in at this pool: the nonce the Google token must carry to be accepted here."""
        cid = google_client_id()
        if not cid:
            raise HTTPException(501, "Google sign-in is not configured.")
        return {"nonce": core.auth.google_challenge(), "client_id": cid}

    @r.post("/auth/google")
    def google(body: dict, request: Request):
        try:
            user, token = core.auth.login_google(body.get("id_token", ""))
        except AuthError as e:
            _raise(e)
        core.credits.grant_welcome(user.id, core.cfg.welcome_flops)
        request.state.session_token = token
        return {"user": core.auth.public_view(user), "token": token}

    @r.post("/auth/logout")
    def logout(request: Request, authorization: Optional[str] = Header(default=None)):
        tok = _token(request, authorization)
        if tok:
            core.auth.logout(tok)
        return {"ok": True}

    @r.get("/auth/me")
    def me(request: Request, authorization: Optional[str] = Header(default=None)):
        user = core.auth.session_user(_token(request, authorization))
        if user is None:
            return {"user": None}
        # A session from before the welcome credit never passes through sign-in again: grant it here.
        core.credits.grant_welcome(user.id, core.cfg.welcome_flops)
        return {"user": core.auth.public_view(user), "credits": core.credits.summary(user.id)}

    @r.post("/auth/accept-terms")
    def accept_terms(request: Request, authorization: Optional[str] = Header(default=None)):
        user = require(request, authorization)
        return {"user": core.auth.public_view(core.auth.accept_terms(user))}

    @r.patch("/auth/me")
    def patch_me(body: dict, request: Request, authorization: Optional[str] = Header(default=None)):
        user = require(request, authorization)
        try:
            user = core.auth.update_profile(user, name=body.get("name"), bio=body.get("bio"))
        except AuthError as e:
            _raise(e)
        return {"user": core.auth.public_view(user)}

    @r.get("/auth/me/nodes")
    def my_nodes(request: Request, authorization: Optional[str] = Header(default=None)):
        user = require(request, authorization)
        live = {n.node_id: n for n in core.registry.nodes.values()}
        inference = getattr(app.state, "inference", None)   # inference nodes are bound to owners too
        out = []
        for row in core.credits.nodes_for(user.id):
            n = live.get(row.node_id)
            inf = inference.node_state(row.node_id) if n is None and inference is not None else None
            out.append({
                "node_id": row.node_id,
                "user_id": n.user_id if n is not None and n.user_id else row.user_id,
                "online": n is not None or bool(inf and inf["online"]),
                "status": n.status if n is not None else inf["status"] if inf else None,
                "gpu_percent": n.gpu_percent if n is not None else None,
            })
        return out

    @r.get("/credits/me")
    def credits_me(request: Request, authorization: Optional[str] = Header(default=None)):
        user = require(request, authorization)
        return core.credits.summary(user.id)

    @r.get("/credits/transactions")
    def transactions(request: Request, limit: int = 50, before_id: Optional[int] = None,
                     authorization: Optional[str] = Header(default=None)):
        user = require(request, authorization)
        return core.credits.list_txns(user.id, limit=limit, before_id=before_id)

    @r.get("/credits/live")
    def live(request: Request, window_s: float = 60,
             authorization: Optional[str] = Header(default=None)):
        user = require(request, authorization)
        return core.credits.live(user.id, window_s=window_s)

    @r.get("/community/leaderboard")
    def leaderboard():
        return core.credits.leaderboard()

    @r.post("/admin/users/{user_id}/ban")
    def ban(user_id: str, body: dict, request: Request,
            authorization: Optional[str] = Header(default=None)):
        admin = require(request, authorization)
        if not admin.admin:
            raise HTTPException(403, "Admin only.")
        user = core.auth.get(user_id)
        if user is None:
            raise HTTPException(404, "No such user.")
        # Never lock the instance out of its own admin surface.
        if user.id == admin.id:
            raise HTTPException(400, "You cannot ban yourself.")
        if user.admin and not user.banned:
            with core.db.session() as s:
                admins = s.exec(select(User).where(User.admin == True, User.banned == False)).all()  # noqa: E712
            if len(admins) <= 1:
                raise HTTPException(400, "Cannot ban the last admin.")
        return {"user": core.auth.public_view(core.auth.set_banned(user, body_bool(body, "banned", True)))}

    @r.post("/admin/users/{user_id}/flag")
    def flag(user_id: str, body: dict, request: Request,
             authorization: Optional[str] = Header(default=None)):
        admin = require(request, authorization)
        user = core.auth.get(user_id)
        if user is None:
            raise HTTPException(404, "No such user.")
        try:
            core.auth.flag_user(admin, user, body.get("reason", ""))
        except AuthError as e:
            _raise(e)
        return {"user": core.auth.public_view(user)}

    @r.get("/admin/users")
    def users(request: Request, authorization: Optional[str] = Header(default=None)):
        admin = require(request, authorization)
        if not admin.admin:
            raise HTTPException(403, "Admin only.")
        with core.db.session() as s:
            rows = s.exec(select(User)).all()
        return [core.auth.public_view(u) for u in rows]

    @r.get("/admin/flags")
    def flags(request: Request, authorization: Optional[str] = Header(default=None)):
        admin = require(request, authorization)
        if not admin.admin:
            raise HTTPException(403, "Admin only.")
        return core.auth.list_flags()

    app.include_router(r)
