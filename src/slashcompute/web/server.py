"""Local shell on :8766 — UI, launcher control, coordinator proxy."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import os
import time
import traceback
from dataclasses import asdict, fields, replace
from http.cookiejar import CookieJar, DefaultCookiePolicy
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

import anyio
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import ClientDisconnect

from slashcompute.common.config import DEMO_MODEL_CANDIDATES, DEV_MODEL
from slashcompute.common.jsonbool import body_bool
from slashcompute.launcher.controller import (
    FINISHES, INFERENCE_SETTINGS, OUTDATED_COORDINATOR, Launcher, LauncherError, LauncherSettings,
    supports_inference,
)
from slashcompute.launcher.dashboard import POOL_LISTS, PoolData, fresh_pool, overview
from slashcompute.web import sample_grants

STATIC = Path(__file__).resolve().parent / "static"
SHELL_HOST = os.environ.get("SLASHCOMPUTE_SHELL_HOST", "127.0.0.1")
SHELL_PORT = int(os.environ.get("SLASHCOMPUTE_SHELL_PORT", "8766"))
# Bump when the shell changes: a running older shell is then replaced instead of reused.
SHELL_GENERATION = 8
MODELS = [DEV_MODEL, *DEMO_MODEL_CANDIDATES]
SESSION_COOKIE = "slashcompute_session"
NO_COORDINATOR = "No coordinator URL. Host or enter one, then Start."
_PROXY_BLOCK = {"verify"}
_SORTS = ("top", "trending", "least")
_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_SESSION_FIELDS = frozenset({"session_token", "session_url"})
_SETTING_FIELDS = tuple(f.name for f in fields(LauncherSettings) if f.name not in _SESSION_FIELDS)
# Forming an LLM pipeline (loading weights over RPC) can take minutes before the first token.
STREAM_TIMEOUT = httpx.Timeout(None, connect=5.0)
PROXY_TIMEOUT = httpx.Timeout(60.0, connect=5.0)
# While waiting on the coordinator's first byte, how often to check that the window still wants it.
DISCONNECT_POLL_S = 0.5
POOL_LIST_TIMEOUT = 1.5
MAX_UPLOAD_BYTES = 64 << 30     # no Mac pool loads a bigger GGUF; refuse before relaying it
DRAIN_LIMIT = 4 << 20           # how much of a refused upload to take so the browser gets the answer


def _proxy_blocked(path: str) -> bool:
    parts = []
    for part in path.replace("\\", "/").split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            return True
        parts.append(part)
    return not parts or parts[0].lower() in _PROXY_BLOCK


def _hostname(value: str) -> str:
    try:
        return (urlsplit(value).hostname or "").rstrip(".")
    except ValueError:
        return ""


def _same_origin(origin: str, hosts: frozenset[str], port: int) -> bool:
    """The shell's own pages: a local host on the shell's port. Any other port (a dev server, a page
    from another local app) is a different origin even though the SameSite cookie still rides along."""
    try:
        parts = urlsplit(origin)
        origin_port = parts.port or {"http": 80, "https": 443}.get(parts.scheme)
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and _hostname(origin) in hosts and origin_port == port


def local_hosts(bind: str) -> frozenset[str]:
    """Names the shell answers to: loopback, plus the bind address when it is a specific LAN one."""
    bind = bind.strip().strip("[]").lower()
    return _LOOPBACK | {bind} if bind and bind not in ("0.0.0.0", "::") else _LOOPBACK


def local_only(app, hosts: frozenset[str], port: int):
    """ASGI guard: a foreign Host is DNS rebinding, a foreign Origin on a write is CSRF. Both get 403."""
    async def guard(scope, receive, send):
        if scope["type"] != "http":
            return await app(scope, receive, send)
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        origin = headers.get("origin")
        if _hostname(f"//{headers.get('host', '')}") not in hosts:
            detail = "Host not allowed."
        elif scope["method"] not in _SAFE_METHODS and origin is not None and not _same_origin(origin, hosts, port):
            detail = "Cross-origin request refused."
        else:
            return await app(scope, receive, send)
        await JSONResponse({"detail": detail}, status_code=403)(scope, receive, send)
    return guard


def public_settings(s: LauncherSettings, session: str) -> dict:
    """Settings as the UI sees them: whether this pool's session is stored, never the token itself."""
    out = asdict(s)
    del out["session_token"], out["session_url"]
    out["has_session"] = bool(session)
    return out


def settings_from_body(body: dict, stored: LauncherSettings) -> LauncherSettings:
    """The stored settings with the body's fields changed: a partial body (the slider alone, Connect's
    mode and url) leaves everything else as it was. The session is always the stored one (the UI
    never sees it to send it back): a body's session_token is a sign-in or sign-out, which the
    caller binds to the pool that issued it."""
    changes = {name: body[name] for name in _SETTING_FIELDS if name in body}
    return replace(stored, **changes).clamp()


def _forward_headers(request: Request, session: str = "") -> dict[str, str]:
    """What the coordinator gets to identify the caller. `session` is the stored token when the pool
    we are dialling issued it, else "". The browser's Authorization goes through as is. Of its
    cookies only the session goes through, and only to that pool: the cookie belongs to the shell's
    origin, so after Connect it still holds the old pool's token. No cookie (pywebview's private
    mode drops them between launches) falls back to the stored token as a Bearer."""
    headers = {}
    if request.headers.get("authorization"):
        headers["authorization"] = request.headers["authorization"]
    cookie = request.cookies.get(SESSION_COOKIE)
    if session and cookie:
        headers["cookie"] = f"{SESSION_COOKIE}={cookie}"
    elif session and "authorization" not in headers:
        headers["authorization"] = f"Bearer {session}"
    if token := os.environ.get("SLASHCOMPUTE_INF_TOKEN"):
        headers["x-inference-token"] = token   # internet-facing pools: shared secret for LLM routes
    return headers


def _set_cookies(headers) -> list[str]:
    if headers is None:
        return []
    get_list = getattr(headers, "get_list", None)
    if callable(get_list):
        return [c for c in (get_list("set-cookie") or []) if c]
    raw = headers.get("set-cookie")
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [c for c in raw if c]
    return [raw]


def _cookie_response(r) -> Response:
    media = "application/json"
    if getattr(r, "headers", None) is not None:
        media = r.headers.get("content-type", media) or media
    resp = Response(content=r.content, status_code=r.status_code, media_type=media)
    for cookie in _set_cookies(getattr(r, "headers", None)):
        resp.headers.append("set-cookie", cookie)
    return resp


def _coord_detail(r) -> str:
    try:
        data = r.json()
    except Exception:
        text = getattr(r, "text", None) or ""
        return text or "Coordinator error."
    detail = data.get("detail", data) if isinstance(data, dict) else data
    return detail if isinstance(detail, str) else json.dumps(detail)


def _validation_detail(errors: list) -> str:
    """FastAPI's list of {loc, msg} as one line the window can show."""
    parts = []
    for err in errors:
        where = ".".join(str(p) for p in err.get("loc", ()) if p != "body") or "body"
        parts.append(f"{where}: {err.get('msg', 'invalid')}")
    return "; ".join(parts) or "Invalid request."


def _sse_error(message: str) -> bytes:
    event = json.dumps({"error": {"message": message}})
    return f"data: {event}\n\ndata: [DONE]\n\n".encode()


def _content_length(request: Request) -> Optional[int]:
    raw = request.headers.get("content-length", "")
    return int(raw) if raw.isdigit() else None


async def _drain(request: Request, limit: int = DRAIN_LIMIT) -> None:
    """Take (up to `limit` of) a request body before refusing it: an answer sent while the browser is
    still uploading resets the connection, and the window shows a network error, not the reason."""
    taken = 0
    with contextlib.suppress(ClientDisconnect):
        async for chunk in request.stream():
            taken += len(chunk)
            if taken >= limit:
                break


def _grant_card(row: dict) -> dict:
    goal = float(row.get("goal_flops") or row.get("goal") or 0)
    raised = float(row.get("received_flops") or row.get("raised") or 0)
    progress = float(row.get("progress") or 0)
    if not progress and goal:
        progress = raised / goal
    return {
        "id": row.get("id"),
        "title": row.get("title", ""),
        "author": row.get("author", ""),
        "summary": row.get("body") or row.get("summary") or "",
        "goal": goal,
        "raised": raised,
        "progress": progress,
        "remaining": max(0.0, goal - raised),
        "backers": int(row.get("backers") or 0),
        "tag": str(row.get("status") or "approved"),
        "status": row.get("status", "approved"),
    }


def _sample_board(sort: str, *, online: bool, share: float = 0.0,
                  available: Optional[float] = None, pledged: float = 0.0,
                  leaders: Optional[list] = None) -> dict:
    mode = sort if sort in _SORTS else "top"
    board = sample_grants.board(sample_grants.sample_book(), mode, share)
    board["online"] = online
    board["leaders"] = leaders or []
    board["pledged"] = pledged
    if available is not None:
        board["available"] = available
    return board


def _stateless_async_client() -> httpx.AsyncClient:
    """Like the launcher's client: it relays for many browsers, so it never keeps a cookie."""
    jar = CookieJar(policy=DefaultCookiePolicy(allowed_domains=[]))
    return httpx.AsyncClient(timeout=STREAM_TIMEOUT, cookies=jar)


def create_shell(launcher: Optional[Launcher] = None,
                 stream_client: Optional[httpx.AsyncClient] = None) -> FastAPI:
    launch = launcher or Launcher()
    streams = stream_client or _stateless_async_client()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await streams.aclose()

    app = FastAPI(title="/compute", lifespan=lifespan)
    app.add_middleware(local_only, hosts=local_hosts(SHELL_HOST), port=SHELL_PORT)
    app.state.launcher = launch
    app.state.pool = PoolData()         # last good rows, so a slow fetch never blanks the dashboard
    app.state.pool_url = ""

    def shell_log(what: str, exc: BaseException) -> None:
        """The traceback goes to <home>/logs/shell.log; the window only gets the one-line detail."""
        try:
            with (launch.log_dir / "shell.log").open("a") as fh:
                fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {what}\n")
                traceback.print_exception(exc, file=fh)
        except OSError:
            pass

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        shell_log(f"{request.method} {request.url.path}", exc)
        return JSONResponse({"detail": f"{type(exc).__name__}: {exc}"}, status_code=500)

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError):
        return JSONResponse({"detail": _validation_detail(exc.errors())}, status_code=422)

    def fwd(request: Request, s: Optional[LauncherSettings] = None) -> dict[str, str]:
        s = launch.load_settings() if s is None else s
        return _forward_headers(request, launch.session_for(s))

    def coord_base() -> str:
        base = launch.proxy_url()
        if not base:
            raise HTTPException(503, NO_COORDINATOR)
        return base.rstrip("/")

    def build(method: str, url: str, **kw) -> httpx.Request:
        try:
            return streams.build_request(method, url, **kw)
        except httpx.InvalidURL as e:
            raise HTTPException(400, f"Invalid coordinator address: {e}") from e

    async def send_upstream(request: Request, req: httpx.Request, watch: bool) -> httpx.Response:
        """Open the coordinator's response. Forming a pipeline can take minutes before the first byte;
        when the window has given up meanwhile (Stop, a closed tab) stop waiting and drop the request.
        `watch` is off while the request body is still streaming in (polling for a disconnect would
        eat a chunk of it): there, the body itself tells us the browser is gone."""
        if not watch:
            return await streams.send(req, stream=True)
        task = asyncio.ensure_future(streams.send(req, stream=True))
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=DISCONNECT_POLL_S)
                if done:
                    return task.result()
                if await request.is_disconnected():
                    raise ClientDisconnect()
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(BaseException):
                    await task

    async def relay_stream(request: Request, req: httpx.Request, *, watch_disconnect: bool = True) -> Response:
        """Pass a coordinator response through as it arrives (LLM tokens, upload results)."""
        try:
            r = await send_upstream(request, req, watch_disconnect)
        except ClientDisconnect:
            raise HTTPException(499, "The window stopped waiting.") from None
        except httpx.RequestError as e:
            raise HTTPException(502, f"Coordinator unreachable: {e}") from e
        media = r.headers.get("content-type", "application/json")
        sse = media.startswith("text/event-stream")

        async def body():
            try:
                async for chunk in r.aiter_bytes():
                    yield chunk
            except httpx.HTTPError as e:
                # Status and headers are already out. An SSE stream gets the failure as an event the
                # window renders (then [DONE]); anything else just ends early.
                shell_log(f"{req.method} {req.url} failed mid-stream", e)
                if sse:
                    yield _sse_error(f"Lost the coordinator mid-answer: {e or type(e).__name__}")
            finally:
                with anyio.CancelScope(shield=True):   # release the connection even on a disconnect
                    await r.aclose()

        return StreamingResponse(body(), status_code=r.status_code, media_type=media)

    async def inference_base() -> str:
        """Coordinator URL for LLM routes. An older coordinator 404s them with a bare "Not Found", so say
        why up front, before the request (or a multi-GB upload) is sent."""
        base = coord_base()
        if supports_inference(await asyncio.to_thread(launch.poll_health, base)) is False:
            raise HTTPException(409, OUTDATED_COORDINATOR)
        return base

    @app.post("/api/chat")
    async def chat(request: Request):
        """Streaming chat with the pool's LLMs (OpenAI-style SSE from the coordinator)."""
        try:
            body = await request.json()
        except ValueError:
            raise HTTPException(400, "Send a JSON chat request.") from None
        if not isinstance(body, dict):
            raise HTTPException(400, "Send a JSON chat request.")
        body["stream"] = True
        req = build("POST", f"{await inference_base()}/v1/chat/completions", json=body,
                    headers=fwd(request), timeout=STREAM_TIMEOUT)
        return await relay_stream(request, req)

    @app.post("/api/models/upload")
    async def upload_model(request: Request):
        """Stream a GGUF to the coordinator without holding it in memory."""
        name = request.query_params.get("name") or request.headers.get("x-filename", "")
        length = _content_length(request)
        headers = {**fwd(request), "content-type": "application/octet-stream"}
        if length is not None:
            headers["content-length"] = str(length)
        try:
            if length is not None and length > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"{name or 'That file'} is over {MAX_UPLOAD_BYTES >> 30} GiB: "
                                         "no pool of Macs can load it.")
            base = await inference_base()
        except HTTPException:
            await _drain(request)
            raise
        req = build("POST", f"{base}/inference/models/upload", params={"name": name},
                    content=request.stream(), headers=headers, timeout=STREAM_TIMEOUT)
        return await relay_stream(request, req, watch_disconnect=False)

    @app.get("/")
    def index():
        path = STATIC / "index.html"
        if not path.is_file():
            raise HTTPException(500, "UI missing")
        # Ask for the scripts by content version: the window kept an old app.js after updates.
        v = asset_version()
        page = path.read_text().replace('/static/app.js"', f'/static/app.js?v={v}"') \
            .replace('/static/app.css"', f'/static/app.css?v={v}"')
        return HTMLResponse(page, headers={"Cache-Control": "no-cache"})

    @app.get("/api/shell")
    def shell_info():
        return {"ok": True, "generation": SHELL_GENERATION, "assets": asset_version(), "proxy": "coord"}

    def shown(s: LauncherSettings) -> dict:
        return public_settings(s, launch.session_for(s))

    def body_settings(body: dict) -> LauncherSettings:
        s = settings_from_body(body, launch.load_settings())
        if "session_token" in body:   # signed in or out: the session belongs to the pool that issued it
            s = launch.with_session(s, str(body["session_token"] or ""))
        return s

    @app.get("/api/settings")
    def get_settings():
        return shown(launch.load_settings())

    @app.post("/api/settings")
    def post_settings(body: dict):
        s = body_settings(body)
        launch.save_settings(s)
        if "session_token" in body:
            launch.rebind_session()   # what runs here now earns for the signed-in account
        return shown(s)

    @app.get("/api/status")
    def status():
        snap = launch.snapshot()
        return {**asdict(snap), **shown(launch.load_settings()),
                "coordinator_url": launch.coordinator_url(launch.load_settings()),
                "finishes": list(FINISHES)}

    @app.post("/api/start")
    def start(body: dict):
        s = body_settings(body)
        try:
            snap = launch.start(s)
        except LauncherError as e:
            raise HTTPException(400, str(e)) from e
        return {**asdict(snap), **shown(launch.load_settings())}

    @app.post("/api/inference")
    def set_inference(body: dict):
        """Start or stop serving LLMs on this Mac, leaving the training agent alone."""
        changes = {k: body[k] for k in INFERENCE_SETTINGS if k in body}
        snap = launch.set_inference(bool(body.get("on")), **changes)
        return {**asdict(snap), **shown(launch.load_settings())}

    @app.post("/api/stop")
    def stop():
        return asdict(launch.stop())

    @app.post("/api/stop-agent")
    def stop_agent():
        return asdict(launch.stop_agent())

    def fetch_rows(url: str, path: str) -> Optional[list]:
        """One of the pool's lists, or None when the coordinator did not answer in time (an empty
        list is an answer: nothing there)."""
        try:
            r = launch._http.get(f"{url.rstrip('/')}{path}", timeout=POOL_LIST_TIMEOUT)
            if r.status_code != 200:
                return None
            data = r.json()
        except Exception:
            return None
        return data if isinstance(data, list) else []

    async def fetch_pool(url: str) -> PoolData:
        """Nodes, jobs and ledger side by side; whatever a slow coordinator leaves out keeps its
        last-good rows (from this pool only)."""
        previous = app.state.pool if app.state.pool_url == url else PoolData()
        rows = await asyncio.gather(*(asyncio.to_thread(fetch_rows, url, path) for path in POOL_LISTS))
        return fresh_pool(previous, *rows)

    @app.get("/api/overview")
    async def get_overview():
        """Status, pool, this Mac and the leaderboard in one poll."""
        s = launch.load_settings()
        snap = await asyncio.to_thread(launch.snapshot, s)
        url = launch.proxy_url(s)
        # snapshot() just polled /health: its answer says whether the lists are worth asking for.
        pool = await fetch_pool(url) if snap.coordinator_up else PoolData()
        app.state.pool, app.state.pool_url = pool, url
        status = {**asdict(snap), **shown(s), "coordinator_url": launch.coordinator_url(s),
                  "models": MODELS, "public_url": launch.cfg.public_url or ""}
        return overview(status, pool, launch.my_node_id(), s.grant_split)

    # ------------------------------------------------------------ live grants

    def amount(value) -> float:
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise HTTPException(400, "Enter a number.") from None
        # float() takes "NaN"/"Infinity", which would reach the coordinator as non-standard JSON.
        if not math.isfinite(v) or v <= 0:
            raise HTTPException(400, "Enter a positive, finite number.")
        return v

    def coord_call(request: Request, method: str, path: str, *,
                   params: Optional[dict] = None, payload: Optional[dict] = None,
                   required: bool = True):
        s = launch.load_settings()
        base = launch.proxy_url(s)
        if not base:
            if required:
                raise HTTPException(503, NO_COORDINATOR)
            return None
        url = f"{base.rstrip('/')}/{path.lstrip('/')}"
        headers = fwd(request, s)
        try:
            if method == "GET":
                r = launch._http.get(url, params=params or None,
                                     headers=headers or None, timeout=30.0)
            else:
                body = json.dumps(payload or {}).encode()
                fwd_headers = dict(headers)
                fwd_headers["content-type"] = "application/json"
                r = launch._http.post(url, content=body, headers=fwd_headers, timeout=30.0)
        except httpx.InvalidURL as e:
            if required:
                raise HTTPException(400, f"Invalid coordinator address {base}: {e}") from e
            return None
        except (httpx.RequestError, ConnectionError, OSError) as e:
            if required:
                raise HTTPException(502, f"Coordinator unreachable at {base}: {e}") from e
            return None
        if r.status_code >= 400:
            if required:
                raise HTTPException(r.status_code, _coord_detail(r))
            return None
        try:
            return r.json()
        except Exception:
            return None

    def grant_board(request: Request, sort: str = "top") -> dict:
        mode = sort if sort in _SORTS else "top"
        rows = coord_call(request, "GET", "/grants", params={"sort": mode}, required=False)
        if rows is None:
            return _sample_board(mode, online=False, available=0.0)
        if not isinstance(rows, list):
            rows = []
        cards = [_grant_card(g) for g in rows if isinstance(g, dict)]
        approved = [g for g in cards if g["status"] == "approved"]
        pending = [g for g in cards if g["status"] == "pending"]
        me = coord_call(request, "GET", "/auth/me", required=False) or {}
        user = me.get("user") if isinstance(me, dict) else None
        credits = me.get("credits") if isinstance(me, dict) else None
        available = float((credits or {}).get("balance") or 0)
        pledged = 0.0
        txns = coord_call(request, "GET", "/credits/transactions",
                          params={"limit": 50}, required=False)
        if isinstance(txns, dict):
            pledged = sum(
                abs(float(t.get("amount") or 0))
                for t in txns.get("items") or []
                if t.get("kind") == "donate"
            )
        raw_leaders = coord_call(request, "GET", "/community/leaderboard", required=False)
        me_id = user.get("id") if isinstance(user, dict) else None
        leaders = []
        if isinstance(raw_leaders, list):
            for i, row in enumerate(raw_leaders, 1):
                if not isinstance(row, dict):
                    continue
                leaders.append({
                    "rank": i,
                    "user_id": row.get("user_id"),
                    "name": row.get("name", ""),
                    "flops": float(row.get("lifetime_earned") or 0),
                    "is_me": bool(me_id and row.get("user_id") == me_id),
                })
        share = float((user or {}).get("grant_split") or 0)
        if not approved and not pending:
            return _sample_board(mode, online=True, share=share, available=available,
                                 pledged=pledged, leaders=leaders)
        return {
            "sample": False, "online": True,
            "grants": approved, "pending": pending,
            "pledged": pledged, "starter": 0.0,
            "share": share,
            "available": available, "leaders": leaders,
        }

    @app.get("/api/grants")
    def list_grants(request: Request, sort: str = "top"):
        return grant_board(request, sort)

    @app.post("/api/grants")
    def request_grant(body: dict, request: Request):
        goal = amount(body.get("goal"))
        coord_call(request, "POST", "/grants", payload={
            "title": str(body.get("title", "")),
            "body": str(body.get("summary", "") or body.get("body", "")),
            "goal_flops": goal,
        })
        return grant_board(request, str(body.get("sort", "top")))

    @app.post("/api/grants/{grant_id}/fund")
    def fund_grant(grant_id: str, body: dict, request: Request):
        coord_call(request, "POST", f"/grants/{grant_id}/donate", payload={
            "flops": amount(body.get("amount") if body.get("amount") is not None
                            else body.get("flops")),
        })
        return grant_board(request, str(body.get("sort", "top")))

    @app.post("/api/grants/{grant_id}/review")
    def review_grant(grant_id: str, body: dict, request: Request):
        coord_call(request, "POST", f"/admin/grants/{grant_id}/review", payload={
            "approve": body_bool(body, "approve"),
            "note": body.get("note"),
        })
        return grant_board(request, str(body.get("sort", "top")))

    @app.post("/api/discover")
    def discover():
        found = launch.find_on_lan()
        if not found:
            raise HTTPException(404, "No coordinator found on the LAN.")
        s = launch.load_settings()
        s.url = found
        launch.save_settings(s)
        return {"url": found}

    async def relay_form(request: Request, url: str, headers: dict[str, str], content_type: str) -> httpx.Response:
        """A form with a file (a job's dataset) goes through as it arrives, boundary and all: the
        browser's content-type names the boundary, so the body needs no re-encoding."""
        fwd_headers = {**headers, "content-type": content_type}
        length = _content_length(request)
        if length is not None:
            fwd_headers["content-length"] = str(length)
        req = build("POST", url, content=request.stream(), headers=fwd_headers, timeout=PROXY_TIMEOUT)
        return await streams.send(req)

    @app.api_route("/api/coord/{path:path}", methods=["GET", "POST", "PATCH", "DELETE"])
    async def proxy(path: str, request: Request):
        if _proxy_blocked(path):
            raise HTTPException(404, "not proxied")
        s = launch.load_settings()
        base = launch.proxy_url(s)
        if not base:
            raise HTTPException(503, NO_COORDINATOR)
        url = f"{base.rstrip('/')}/{path}"
        headers = fwd(request, s)
        http = launch._http   # synchronous: every call runs off the event loop
        ct = request.headers.get("content-type", "")
        try:
            if request.method == "POST" and ct.startswith("multipart/"):
                r = await relay_form(request, url, headers, ct)
            elif request.method == "DELETE":
                r = await asyncio.to_thread(http.delete, url, headers=headers or None, timeout=30.0)
            elif request.method == "GET":
                r = await asyncio.to_thread(http.get, url, params=request.query_params.multi_items() or None,
                                            headers=headers or None, timeout=30.0)
            else:
                fwd_headers = {**headers, "content-type": ct or "application/json"}
                body = await request.body()
                send = http.patch if request.method == "PATCH" else http.post
                r = await asyncio.to_thread(send, url, content=body, headers=fwd_headers, timeout=60.0)
        except ClientDisconnect:
            raise HTTPException(499, "The window stopped sending.") from None
        except httpx.InvalidURL as e:
            raise HTTPException(400, f"Invalid coordinator address {base}: {e}") from e
        except (httpx.RequestError, ConnectionError, OSError) as e:
            raise HTTPException(502, f"Coordinator unreachable at {base}: {e}") from e
        return _cookie_response(r)

    if STATIC.is_dir():
        app.mount("/static", FreshStaticFiles(directory=STATIC), name="static")
    return app


class FreshStaticFiles(StaticFiles):
    """Static files the window revalidates every time (it gets a 304 when nothing changed)."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


def asset_version() -> str:
    """Changes whenever app.js or app.css does."""
    h = hashlib.sha256()
    for name in ("app.js", "app.css"):
        with contextlib.suppress(OSError):
            h.update((STATIC / name).read_bytes())
    return h.hexdigest()[:12]
