import json
import shutil
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from slashcompute.common.config import EngineConfig
from slashcompute.coordinator.app import create_app
from slashcompute.launcher.controller import Launcher, LauncherSettings, launch_record, stateless_http
from slashcompute.web.server import create_shell

SHELL = "http://127.0.0.1:8766"   # the shell refuses any Host but loopback


@pytest.fixture(autouse=True)
def _unknown_cmdlines(monkeypatch):
    """The fake pids in these tests belong to no process (or, worse, to some unrelated real one):
    their command lines are unreadable, which leaves the pid files trusted as before."""
    monkeypatch.setattr("slashcompute.launcher.controller.process_cmdline", lambda pid: None)


class FakeProc:
    def __init__(self, pid: int, argv: list[str]) -> None:
        self.pid = pid
        self.argv = argv

    def poll(self):
        return None


class FakeHTTP:
    def __init__(self, health=None) -> None:
        self.health = health

    def get(self, url: str, timeout: float = 1.0, params=None, headers=None):
        if self.health is None:
            raise ConnectionError("down")
        class R:
            status_code = 200
            content = b'{"ok":true,"nodes":1,"jobs":0}'
            headers = {"content-type": "application/json"}
            def json(self_inner):
                return {"ok": True, "nodes": 1, "jobs": 0}
        return R()

    def post(self, url: str, content=None, headers=None, timeout: float = 1.0, **_):
        class R:
            status_code = 200
            content = b'{"user":{"id":"u1","email":"ada@lan.test"},"token":"sess"}'
            headers = {"content-type": "application/json",
                       "set-cookie": "slashcompute_session=sess; HttpOnly; SameSite=lax"}
            def json(self_inner):
                return {"user": {"id": "u1", "email": "ada@lan.test"}, "token": "sess"}
        return R()


def _shell(tmp_path, **kw):
    spawned = []
    n = {"p": 5000}

    def popen(argv, **_):
        n["p"] += 1
        proc = FakeProc(n["p"], argv)
        spawned.append(proc)
        return proc

    launcher = Launcher(
        home=tmp_path,
        python="/opt/venv/bin/python",
        popen=popen,
        http=kw.pop("http", FakeHTTP()),
        discover_fn=kw.pop("discover_fn", lambda timeout=5.0: "http://10.0.0.9:8765"),
        lan_ip_fn=lambda: "192.168.1.20",
        port_free_fn=lambda host, port: True,
    )
    app = create_shell(launcher)
    return app, launcher, spawned


def test_index_and_css(tmp_path):
    app, _, _ = _shell(tmp_path)
    with TestClient(app, base_url=SHELL) as c:
        r = c.get("/")
        assert r.status_code == 200
        assert b"COMPUTE" in r.content
        for tab in (b"contributions", b"usage", b"grants", b"pool"):
            assert b'data-tab="' + tab + b'"' in r.content
        assert b'data-ink="signal"' in r.content
        assert b"/static/logo.png" in r.content
        assert c.get("/static/logo.png").status_code == 200
        assert b'data-file="community"' not in r.content
        assert b">COM<" not in r.content
        assert b'data-mode="public"' in r.content
        assert b"Public pool" in r.content
        js = c.get("/static/app.js")
        assert js.status_code == 200
        assert b"AUTH.sc" not in js.content
        assert b"/api/overview" in js.content
        assert b"GOOGLE" not in js.content
        assert b"connect-public" in js.content
        assert b'mode: "host", contribute: false' not in js.content   # hosting must not stop contributing
        css = c.get("/static/app.css")
        assert css.status_code == 200
        assert b"IBM Plex Sans" in css.content
        assert b"archivo-black" not in css.content
        assert c.get("/static/fonts/ibm-plex-sans-regular.woff2").status_code == 200
        assert c.get("/static/fonts/ibm-plex-mono-regular.woff2").status_code == 200


def test_logout_rebuilds_grant_board(tmp_path):
    # The admin review queue is only toggled by renderGrants(); signing out has
    # to reload the board or Approve/Decline stays on screen for the next user.
    app, _, _ = _shell(tmp_path)
    with TestClient(app, base_url=SHELL) as c:
        js = c.get("/static/app.js").text
    start = js.index("  logout: (btn) =>")
    body = js[start:js.index("\n  }),", start)]
    assert "state.user = null" in body
    assert "await loadGrants()" in body
    assert body.index("state.user = null") < body.index("await loadGrants()")


def test_settings_and_status(tmp_path):
    app, launcher, _ = _shell(tmp_path)
    with TestClient(app, base_url=SHELL) as c:
        r = c.post("/api/settings", json={"mode": "join", "url": "10.1.2.3",
                                          "gpu_percent": 40, "finish": "signal"})
        assert r.status_code == 200
        body = r.json()
        assert body["mode"] == "join" and body["finish"] == "signal"
        assert body["url"] == "10.1.2.3"
        assert body["has_session"] is False
        assert body["grant_split"] == 0
        got = c.get("/api/settings").json()
        assert got["gpu_percent"] == 40
        st = c.get("/api/status").json()
        assert st["lan_ip"] == "192.168.1.20"
        assert "carbon" in st["finishes"]
    assert launcher.load_settings().finish == "signal"


def test_shell_refuses_foreign_host_and_cross_origin_writes(tmp_path):
    app, launcher, _ = _shell(tmp_path)
    launcher.save_settings(LauncherSettings(session_token="secret-sess"))
    stopped = []
    real_stop = launcher.stop_agent
    launcher.stop_agent = lambda: stopped.append(1) or real_stop()
    with TestClient(app, base_url=SHELL) as c:
        # DNS rebinding: evil.example resolves to 127.0.0.1 but the Host header gives it away.
        for path in ("/api/settings", "/api/status", "/"):
            assert c.get(path, headers={"host": "evil.example:8766"}).status_code == 403
        # CSRF: a "simple" cross-site POST (no preflight) must not act.
        for origin in ("https://evil.example", "null", "http://127.0.0.1.evil.example"):
            r = c.post("/api/stop-agent", headers={"origin": origin, "content-type": "text/plain"})
            assert r.status_code == 403, origin
        for path in ("/api/stop", "/api/discover", "/api/coord/auth/logout"):
            assert c.post(path, headers={"origin": "https://evil.example"}).status_code == 403
        # Another local port is another origin (a dev server, a page from some other local app), even
        # though it is same-site and so still gets the SameSite=Lax session cookie.
        for origin in ("http://localhost:3000", "http://127.0.0.1:9810", "http://[::1]", "https://127.0.0.1"):
            r = c.post("/api/stop-agent", headers={"origin": origin, "content-type": "text/plain"})
            assert r.status_code == 403, origin
        assert not stopped
        assert launcher.load_settings().session_token == "secret-sess"
        # The UI itself: loopback Host with any port and loopback name; Origin on the shell's own port.
        for host in ("127.0.0.1:8766", "localhost:9810", "[::1]:8766", "localhost"):
            assert c.get("/api/shell", headers={"host": host}).status_code == 200, host
        for origin in ("http://127.0.0.1:8766", "http://localhost:8766", "http://[::1]:8766"):
            r = c.post("/api/stop-agent", headers={"origin": origin})
            assert r.status_code == 200, origin
        assert c.post("/api/stop-agent").status_code == 200   # no Origin: not a browser
        assert c.get("/api/settings", headers={"origin": "https://evil.example"}).status_code == 200
    assert len(stopped) == 4


def test_shell_allows_configured_lan_bind_host(tmp_path, monkeypatch):
    monkeypatch.setattr("slashcompute.web.server.SHELL_HOST", "192.168.1.20")
    app, _, _ = _shell(tmp_path)
    with TestClient(app, base_url="http://192.168.1.20:8766") as c:
        assert c.get("/api/shell").status_code == 200
        assert c.post("/api/settings", json={}, headers={"origin": "http://192.168.1.20:8766"}).status_code == 200
        assert c.post("/api/settings", json={}, headers={"origin": "http://192.168.1.20:3000"}).status_code == 403
        assert c.get("/api/shell", headers={"host": "evil.example"}).status_code == 403


def test_settings_never_expose_session_token(tmp_path):
    app, launcher, _ = _shell(tmp_path)
    with TestClient(app, base_url=SHELL) as c:
        assert c.post("/api/settings", json={"session_token": "secret-sess"}).json()["has_session"] is True
        for path in ("/api/settings", "/api/status", "/api/overview"):
            assert b"secret-sess" not in c.get(path).content, path
        assert c.get("/api/settings").json()["has_session"] is True
        # The UI round-trips settings without the token; that must not sign it out.
        assert c.post("/api/settings", json={"gpu_percent": 30}).json()["has_session"] is True
        assert launcher.load_settings().session_token == "secret-sess"
        assert c.post("/api/settings", json={"session_token": ""}).json()["has_session"] is False
    assert launcher.load_settings().session_token == ""


def test_sign_in_rebinds_running_processes_to_the_pool_that_issued_it(tmp_path):
    app, launcher, _ = _shell(tmp_path)
    rebinds = []
    launcher.rebind_session = lambda: rebinds.append(launcher.session_for(launcher.load_settings()))
    pool_a, pool_b = {"mode": "join", "url": "10.0.0.1"}, {"mode": "join", "url": "10.0.0.2"}
    with TestClient(app, base_url=SHELL) as c:
        assert c.post("/api/settings", json={**pool_a, "session_token": "sess"}).json()["has_session"] is True
        assert rebinds == ["sess"]                    # the running agent and LLM node now earn for it
        c.post("/api/settings", json={**pool_a, "gpu_percent": 30})
        assert rebinds == ["sess"]
        # Another pool never sees this session; coming back to the first one finds it again.
        assert c.post("/api/settings", json=pool_b).json()["has_session"] is False
        assert c.post("/api/settings", json=pool_a).json()["has_session"] is True


def test_start_stop_and_discover(tmp_path, monkeypatch):
    http = FakeHTTP()
    app, launcher, spawned = _shell(tmp_path, http=http)
    monkeypatch.setattr("slashcompute.launcher.controller.process_alive",
                        lambda pid: any(p.pid == pid for p in spawned))
    killed = []   # Stop SIGTERMs the fake coordinator's pid: a real process 5001 must never get it
    monkeypatch.setattr("slashcompute.launcher.controller.os.kill", lambda pid, sig: killed.append((pid, sig)))

    def health_after(url):
        if spawned:
            http.health = {"ok": True, "nodes": 0, "jobs": 0}
            return {"ok": True, "nodes": 0, "jobs": 0}
        return None

    launcher.poll_health = health_after
    with TestClient(app, base_url=SHELL) as c:
        r = c.post("/api/start", json={"mode": "host", "gpu_percent": 50, "contribute": True})
        assert r.status_code == 200, r.text
        assert len(spawned) == 2
        found = c.post("/api/discover").json()
        assert found["url"] == "http://10.0.0.9:8765"
        c.post("/api/stop")
    assert (spawned[0].pid, 15) in killed


def test_join_start_requires_url(tmp_path):
    app, _, spawned = _shell(tmp_path)
    with TestClient(app, base_url=SHELL) as c:
        r = c.post("/api/start", json={"mode": "join", "url": ""})
        assert r.status_code == 400
        assert spawned == []


def test_proxy_allows_health_and_blocks_other(tmp_path):
    app, launcher, _ = _shell(tmp_path, http=FakeHTTP({"ok": True}))
    launcher.save_settings(LauncherSettings(mode="host"))
    with TestClient(app, base_url=SHELL) as c:
        assert c.get("/api/coord/health").status_code == 200
        assert c.get("/api/coord/auth/me").status_code == 200
        assert c.get("/api/coord/verify/secret").status_code == 404
        assert c.get("/api/coord/jobs/../verify/secret").status_code == 404
        assert c.get("/api/coord/VERIFY/secret").status_code == 404
        info = c.get("/api/shell").json()
    assert info["generation"] >= 8 and info["proxy"] == "coord"
    # The window reuses a running shell only when its generation and its app.js/app.css match.
    assert len(info["assets"]) == 12 and int(info["assets"], 16) >= 0


class RoutedHTTP:
    """Answers GETs by path. ``routes`` maps path -> JSON payload."""

    def __init__(self, routes: dict) -> None:
        self.routes = routes

    def get(self, url: str, timeout: float = 1.0, params=None, headers=None):
        path = "/" + url.split("://", 1)[-1].split("/", 1)[-1]
        if path not in self.routes:
            raise ConnectionError("down")
        payload = self.routes[path]

        class R:
            status_code = 200

            def json(self_inner):
                return payload
        return R()


POOL = {
    "/health": {"ok": True, "nodes": 2, "jobs": 2},
    "/nodes": [
        {"node_id": "me", "name": "Air", "matmul_tflops": 2.0, "memory_contrib_bytes": 8 << 30,
         "gpu_percent": 50},
        {"node_id": "b", "name": "Studio", "matmul_tflops": 20.0,
         "memory_contrib_bytes": 64 << 30, "gpu_percent": 80},
    ],
    "/jobs": [
        {"id": "old", "status": "completed", "steps": 10, "progress_step": 10, "submitted_at": 1},
        {"id": "new", "status": "running", "steps": 10, "progress_step": 4, "submitted_at": 2},
    ],
    "/ledger": [
        {"node_id": "me", "kind": "train", "flops": 4e12, "disputed_flops": 0.0},
        {"node_id": "b", "kind": "train", "flops": 9e12, "disputed_flops": 0.0},
    ],
}


def test_overview_offline(tmp_path):
    app, _, _ = _shell(tmp_path, http=RoutedHTTP({}))
    with TestClient(app, base_url=SHELL) as c:
        ov = c.get("/api/overview").json()
    assert ov["pool"]["online"] is False
    assert ov["pool"]["jobs"] == [] and ov["leaderboard"] == []
    assert ov["me"]["flops"] == 0 and ov["me"]["rank"] is None
    assert ov["status"]["lan_ip"] == "192.168.1.20"
    assert ov["status"]["models"][0].endswith("0.5B-Instruct-4bit")
    assert "public_url" in ov["status"]


def test_settings_accept_public_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("SLASHCOMPUTE_PUBLIC_URL", "https://pool.example.com")
    app, launcher, _ = _shell(tmp_path)
    with TestClient(app, base_url=SHELL) as c:
        r = c.post("/api/settings", json={"mode": "public", "url": "", "gpu_percent": 40})
        assert r.status_code == 200
        assert r.json()["mode"] == "public"
        ov = c.get("/api/overview").json()
        assert ov["status"]["mode"] == "public"
        assert ov["status"]["public_url"] == "https://pool.example.com"
        assert ov["status"]["coordinator_url"] == "https://pool.example.com"


def test_overview_online_ranks_this_mac(tmp_path):
    app, launcher, _ = _shell(tmp_path, http=RoutedHTTP(POOL))
    launcher.save_settings(LauncherSettings(mode="host", grant_split=25))
    launcher.paths.node_id_file.write_text("me\n")
    with TestClient(app, base_url=SHELL) as c:
        ov = c.get("/api/overview").json()
    assert ov["pool"]["online"] is True
    assert [j["id"] for j in ov["pool"]["jobs"]] == ["new", "old"]
    assert ov["pool"]["jobs"][0]["progress"] == 0.4 and ov["pool"]["jobs"][0]["can_cancel"]
    assert ov["pool"]["capacity"]["macs"] == 2 and ov["pool"]["capacity"]["running"] == 1
    assert ov["me"]["flops"] == 4e12
    assert ov["me"]["credits"] == {"earned": 4e12, "kept": 3e12, "to_grants": 1e12}
    assert (ov["me"]["rank"], ov["me"]["of"]) == (2, 2)
    assert [n["is_me"] for n in ov["pool"]["nodes"]] == [True, False]


def test_overview_keeps_the_last_good_lists_when_a_fetch_times_out(tmp_path):
    http = RoutedHTTP(dict(POOL))   # RoutedHTTP fails any path it has no route for
    app, launcher, _ = _shell(tmp_path, http=http)
    launcher.save_settings(LauncherSettings(mode="host"))
    launcher.paths.node_id_file.write_text("me\n")
    with TestClient(app, base_url=SHELL) as c:
        first = c.get("/api/overview").json()
        assert first["me"]["flops"] == 4e12 and len(first["pool"]["jobs"]) == 2
        del http.routes["/jobs"], http.routes["/ledger"]
        again = c.get("/api/overview").json()
        assert again["pool"]["online"] is True
        assert again["pool"]["jobs"] == first["pool"]["jobs"]
        assert again["leaderboard"] == first["leaderboard"] and again["me"]["flops"] == 4e12
        # An empty answer is news, not a timeout.
        http.routes["/jobs"] = []
        assert c.get("/api/overview").json()["pool"]["jobs"] == []
        # Another pool starts from nothing: pool A's rows are never shown for pool B.
        launcher.save_settings(LauncherSettings(mode="join", url="10.0.0.2"))
        other = c.get("/api/overview").json()
        assert other["pool"]["online"] is True and other["me"]["flops"] == 0
        # The coordinator going away clears the view.
        http.routes.clear()
        gone = c.get("/api/overview").json()
    assert gone["pool"]["online"] is False and gone["pool"]["jobs"] == [] and gone["leaderboard"] == []


def test_overview_lists_fifty_jobs_but_never_hides_one_that_can_still_be_cancelled(tmp_path):
    """The newest fifty, except that a job still queued or running stays listed however old it is:
    its Cancel button used to vanish once fifty newer jobs had finished. Every job is counted."""
    jobs = [{"id": f"j{i}", "status": "completed", "steps": 10, "progress_step": 10, "submitted_at": i}
            for i in range(60)]
    jobs[0].update(status="queued", progress_step=0)      # the oldest of all, still waiting
    jobs[7].update(status="running", progress_step=3)
    http = RoutedHTTP({**POOL, "/jobs": jobs})
    app, launcher, _ = _shell(tmp_path, http=http)
    launcher.save_settings(LauncherSettings(mode="host"))
    with TestClient(app, base_url=SHELL) as c:
        ov = c.get("/api/overview").json()
        ids = [j["id"] for j in ov["pool"]["jobs"]]
        assert len(ids) == 50 and ids[:3] == ["j59", "j58", "j57"]
        assert ids[-2:] == ["j7", "j0"]                               # kept, in order
        assert [j["can_cancel"] for j in ov["pool"]["jobs"][-2:]] == [True, True]
        assert "j12" in ids and "j11" not in ids                      # finished ones fill the rest
        assert (ov["pool"]["capacity"]["running"], ov["pool"]["capacity"]["waiting"]) == (1, 1)
        # More than fifty unfinished: all of them are listed.
        http.routes["/jobs"] = [{**j, "status": "running"} for j in jobs[:55]] + jobs[55:]
        ov = c.get("/api/overview").json()
    assert len(ov["pool"]["jobs"]) == 55 and all(j["can_cancel"] for j in ov["pool"]["jobs"])
    assert ov["pool"]["capacity"]["running"] == 55


def test_stop_agent_endpoint_leaves_coordinator(tmp_path, monkeypatch):
    kills = []
    monkeypatch.setattr("os.kill", lambda pid, sig: kills.append((pid, sig)))
    monkeypatch.setattr("slashcompute.agent.daemon._alive", lambda pid: True)
    app, launcher, _ = _shell(tmp_path)
    (tmp_path / "coordinator.pid").write_text("111\n")
    launcher.paths.pid_file.write_text("222\n")
    with TestClient(app, base_url=SHELL) as c:
        assert c.post("/api/stop-agent").status_code == 200
    assert (222, 15) in kills and (111, 15) not in kills


class GrantHTTP:
    """In-memory coordinator stand-in for the live grants adapter."""

    def __init__(self) -> None:
        self.grants = [
            {"id": "g1", "title": "Parser", "author": "Ada",
             "body": "Need FLOPs for a parser thesis project.",
             "goal_flops": 50e12, "received_flops": 10e12, "status": "approved", "progress": 0.2},
            {"id": "g2", "title": "Waiting", "author": "Bea",
             "body": "Waiting for review of this grant request.",
             "goal_flops": 20e12, "received_flops": 0.0, "status": "pending", "progress": 0.0},
        ]
        self.donated = 0.0

    def _path(self, url: str) -> str:
        return "/" + url.split("://", 1)[-1].split("/", 1)[-1].split("?")[0]

    def _resp(self, payload, status=200):
        raw = json.dumps(payload).encode()

        class R:
            status_code = status
            content = raw
            headers = {"content-type": "application/json"}

            def json(self_inner):
                return payload
        return R()

    def get(self, url: str, timeout: float = 1.0, params=None, headers=None):
        path = self._path(url)
        if path == "/grants":
            return self._resp(self.grants)
        if path == "/auth/me":
            return self._resp({
                "user": {"id": "u1", "name": "Ada", "admin": True, "grant_split": 10,
                         "accepted_terms": True},
                "credits": {"balance": 80e12, "lifetime_earned": 100e12},
            })
        if path == "/credits/transactions":
            return self._resp({"items": [
                {"kind": "donate", "amount": -self.donated} if self.donated else {"kind": "earn", "amount": 1},
            ]})
        if path == "/community/leaderboard":
            return self._resp([{"user_id": "u1", "name": "Ada", "lifetime_earned": 100e12}])
        raise ConnectionError("down")

    def post(self, url: str, content=None, headers=None, timeout: float = 1.0, **_):
        path = self._path(url)
        body = json.loads(content or b"{}")
        if path == "/grants":
            title = str(body.get("title", "")).strip()
            text = str(body.get("body", "")).strip()
            if len(title) < 4 or len(text) < 20:
                return self._resp({"detail": "Describe the need: a title and at least a short paragraph."}, 400)
            row = {"id": "g3", "title": title, "author": "You", "body": text,
                   "goal_flops": float(body.get("goal_flops") or 0), "received_flops": 0.0,
                   "status": "pending", "progress": 0.0}
            self.grants.append(row)
            return self._resp(row)
        if path.endswith("/donate"):
            gid = path.split("/")[2]
            g = next((x for x in self.grants if x["id"] == gid), None)
            if g is None or g["status"] != "approved":
                return self._resp({"detail": "Only approved grants can receive FLOPs."}, 400)
            flops = float(body.get("flops") or 0)
            if flops <= 0:
                return self._resp({"detail": "Enter a number."}, 400)
            g["received_flops"] += flops
            g["progress"] = g["received_flops"] / g["goal_flops"]
            self.donated += flops
            return self._resp(g)
        if "/admin/grants/" in path and path.endswith("/review"):
            gid = path.split("/")[3]
            g = next((x for x in self.grants if x["id"] == gid), None)
            if g is None or g["status"] != "pending":
                return self._resp({"detail": "This grant was already reviewed."}, 400)
            g["status"] = "approved" if body.get("approve") else "declined"
            return self._resp(g)
        return self._resp({"detail": "not found"}, 404)


def test_proxy_forwards_set_cookie(tmp_path):
    app, launcher, _ = _shell(tmp_path, http=FakeHTTP({"ok": True}))
    launcher.save_settings(LauncherSettings(mode="host"))
    with TestClient(app, base_url=SHELL) as c:
        r = c.post("/api/coord/auth/login", json={"email": "ada@lan.test", "password": "password1"})
        assert r.status_code == 200
        assert "slashcompute_session=sess" in r.headers.get("set-cookie", "")


def test_proxy_never_keeps_a_cookie_but_lends_the_stored_session(tmp_path, monkeypatch):
    monkeypatch.setattr("slashcompute.community.auth.ITERATIONS", 1)
    coord_app = create_app(EngineConfig(home=tmp_path / "coord", scheduler_tick_s=0.05))
    with TestClient(coord_app) as coord:
        app, launcher, _ = _shell(tmp_path / "shell", http=stateless_http(transport=coord._transport))
        launcher.save_settings(LauncherSettings(mode="join", url="http://testserver"))
        with TestClient(app, base_url=SHELL) as c:
            r = c.post("/api/coord/auth/register", json={
                "email": "ada@lan.test", "password": "password1", "name": "Ada"})
            assert r.status_code == 200, r.text
            assert "slashcompute_session=" in r.headers.get("set-cookie", "")
            # The shell stored the token as it relayed the reply: a poll before the window has saved
            # it is already signed in (the board used to flicker to Sign in in between).
            assert c.get("/api/settings").json()["has_session"] is True
            assert launcher.load_settings().session_token == r.json()["token"]
            c.post("/api/settings", json={"session_token": r.json()["token"]})   # as the window does
            assert c.get("/api/coord/auth/me").json()["user"]["email"] == "ada@lan.test"
            # The window's cookie is never what signs it in: the launcher's copy, bound to this
            # pool, does, as a Bearer token the coordinator accepts.
            c.cookies.clear()
            assert c.get("/api/coord/auth/me").json()["user"]["email"] == "ada@lan.test"
            # Signed out: nothing stored and nothing in the browser leaves nothing to lend.
            c.post("/api/settings", json={"session_token": ""})
            assert c.get("/api/coord/auth/me").json()["user"] is None
            # A stored token the pool does not know (it expired, or the coordinator's database was
            # reset) is dropped the first time the pool says so, not lent forever.
            c.post("/api/settings", json={"session_token": "from-before-the-reset"})
            assert c.get("/api/settings").json()["has_session"] is True
            assert c.get("/api/coord/auth/me").json()["user"] is None
            assert c.get("/api/settings").json()["has_session"] is False
        assert not launcher._http.cookies


class HeaderHTTP(FakeHTTP):
    """Records the headers each GET carried to the coordinator."""

    def __init__(self) -> None:
        super().__init__({"ok": True})
        self.seen: list[dict] = []

    def get(self, url: str, timeout: float = 1.0, params=None, headers=None):
        self.seen.append(dict(headers or {}))
        return super().get(url, timeout, params, headers)


def test_proxy_lends_the_stored_session_to_its_pool_and_never_the_browsers_cookie(tmp_path):
    http = HeaderHTTP()
    app, launcher, _ = _shell(tmp_path, http=http)
    pool_a = LauncherSettings(mode="join", url="10.0.0.1")
    launcher.save_settings(launcher.with_session(pool_a, "tok-a"))
    with TestClient(app, base_url=SHELL) as c:
        # The stored session signs the window in, cookie or no cookie. pywebview on macOS ignores
        # storage_path, so the WKWebView jar is shared by every home on this Mac and may hold a stale
        # token from another one: forwarded, it used to shadow the valid stored token.
        c.get("/api/coord/auth/me", headers={"cookie": "theme=dark; slashcompute_session=stale-other-home"})
        assert http.seen[-1] == {"authorization": "Bearer tok-a"}
        c.get("/api/coord/auth/me", headers={"cookie": "slashcompute_session=tok-a"})
        assert http.seen[-1] == {"authorization": "Bearer tok-a"}
        c.get("/api/coord/auth/me")
        assert http.seen[-1] == {"authorization": "Bearer tok-a"}
        # The browser's own Authorization wins over the stored session.
        c.get("/api/coord/auth/me", headers={"authorization": "Bearer mine"})
        assert http.seen[-1] == {"authorization": "Bearer mine"}
        # Connected to another pool: the browser still holds pool A's cookie, which must not reach B,
        # and the stored token (A's) is not lent to B either.
        c.post("/api/settings", json={"mode": "join", "url": "10.0.0.2"})
        c.get("/api/coord/auth/me", headers={"cookie": "slashcompute_session=tok-a"})
        assert http.seen[-1] == {}
        c.get("/api/coord/auth/me")
        assert http.seen[-1] == {}
        # Back on pool A everything is as it was.
        c.post("/api/settings", json={"mode": "join", "url": "10.0.0.1"})
        c.get("/api/coord/auth/me", headers={"cookie": "slashcompute_session=tok-a"})
        assert http.seen[-1] == {"authorization": "Bearer tok-a"}


def _json_response(payload, status: int = 200):
    raw = json.dumps(payload).encode()

    class R:
        status_code = status
        content = raw
        headers = {"content-type": "application/json"}

        def json(self_inner):
            return payload
    return R()


class SessionHTTP:
    """A coordinator that knows which session tokens are live, as the real one behaves: /auth/me
    answers 200 {user: null} for a token it does not know (never 401); a sign-in issues "fresh";
    `strict` paths 401 an unknown Bearer but answer a stranger; `private` paths 401 anyone unknown.
    Every call's headers are recorded."""

    def __init__(self, live=("sess",), strict=(), private=()) -> None:
        self.live, self.strict, self.private = set(live), set(strict), set(private)
        self.seen: list[dict] = []

    @staticmethod
    def _path(url: str) -> str:
        return "/" + url.split("://", 1)[-1].split("/", 1)[-1].split("?")[0]

    @staticmethod
    def _bearer(headers) -> str | None:
        auth = (headers or {}).get("authorization", "")
        return auth[7:] if auth.lower().startswith("bearer ") else None

    def get(self, url: str, timeout: float = 1.0, params=None, headers=None):
        self.seen.append(dict(headers or {}))
        path, token = self._path(url), self._bearer(headers)
        if path == "/auth/me":
            return _json_response({"user": {"id": "u1", "email": "ada@lan.test"} if token in self.live else None})
        if path in self.private and token not in self.live:
            return _json_response({"detail": "Sign in first."}, 401)
        if path in self.strict and token is not None and token not in self.live:
            return _json_response({"detail": "Sign in first."}, 401)
        return _json_response({"ok": True, "path": path})

    def post(self, url: str, content=None, headers=None, timeout: float = 1.0, **_):
        self.seen.append(dict(headers or {}))
        path = self._path(url)
        if path in ("/auth/login", "/auth/register", "/auth/google"):
            if json.loads(content or b"{}").get("password") == "wrong":
                return _json_response({"detail": "Wrong password."}, 401)
            self.live.add("fresh")
            return _json_response({"user": {"id": "u1", "email": "ada@lan.test"}, "token": "fresh"})
        return _json_response({"detail": "not found"}, 404)


def test_proxy_forgets_a_stored_session_its_pool_no_longer_knows(tmp_path):
    """A token that expired, or from before the coordinator's database was reset, used to be lent to
    every request forever: /auth/me kept answering {user: null} (it never 401s) and every submit
    "Sign in first.", with nothing to clear it but a manual sign-out."""
    http = SessionHTTP(live=["sess"])
    app, launcher, _ = _shell(tmp_path, http=http)
    launcher.save_settings(launcher.with_session(LauncherSettings(mode="host"), "stale"))
    with TestClient(app, base_url=SHELL) as c:
        assert c.get("/api/settings").json()["has_session"] is True
        assert c.get("/api/coord/auth/me").json()["user"] is None
        assert http.seen[-1] == {"authorization": "Bearer stale"}
        assert c.get("/api/settings").json()["has_session"] is False
        c.get("/api/coord/auth/me")
        assert http.seen[-1] == {}                                  # nothing dead is lent again
        # A live token stays stored, and so does one the browser answered for itself: when the
        # window sends its own Authorization the shell lent nothing and learns nothing.
        launcher.save_settings(launcher.with_session(launcher.load_settings(), "sess"))
        assert c.get("/api/coord/auth/me").json()["user"]["email"] == "ada@lan.test"
        assert c.get("/api/coord/auth/me", headers={"authorization": "Bearer stranger"}).json()["user"] is None
        assert c.get("/api/settings").json()["has_session"] is True
    assert launcher.load_settings().session_token == "sess"


def test_proxy_retries_a_refused_lent_token_as_a_stranger(tmp_path):
    """A 401 to a request the shell signed in itself: asked again without the token, an answer means
    the token was what was wrong, so it is dropped. A route that needs a sign-in 401s both times
    and the token is kept (the next /auth/me poll settles it)."""
    http = SessionHTTP(live=[], strict=["/grants"], private=["/auth/me/nodes"])
    app, launcher, _ = _shell(tmp_path, http=http)
    launcher.save_settings(launcher.with_session(LauncherSettings(mode="host"), "stale"))
    with TestClient(app, base_url=SHELL) as c:
        r = c.get("/api/coord/auth/me/nodes")
        assert r.status_code == 401 and "Sign in" in r.json()["detail"]
        assert [h.get("authorization") for h in http.seen] == ["Bearer stale", None]
        assert launcher.load_settings().session_token == "stale"
        http.seen.clear()
        r = c.get("/api/coord/grants")
        assert r.status_code == 200 and r.json()["path"] == "/grants"
        assert [h.get("authorization") for h in http.seen] == ["Bearer stale", None]
        assert launcher.load_settings().session_token == ""
        http.seen.clear()
        assert c.get("/api/coord/grants").status_code == 200
        assert http.seen == [{}]                                    # one try, nothing to lend


def test_proxy_stores_the_session_a_sign_in_issues_as_it_relays_the_reply(tmp_path):
    """The window saves the token itself, after the reply; a poll in between found no stored session
    and the board flickered to Sign in. Stored at once, the next poll is signed in and a sign-out
    revokes the right token."""
    http = SessionHTTP(live=[])
    app, launcher, _ = _shell(tmp_path, http=http)
    launcher.save_settings(LauncherSettings(mode="join", url="10.0.0.1"))
    with TestClient(app, base_url=SHELL) as c:
        r = c.post("/api/coord/auth/login", json={"email": "ada@lan.test", "password": "wrong"})
        assert r.status_code == 401 and launcher.load_settings().session_token == ""
        r = c.post("/api/coord/auth/login", json={"email": "ada@lan.test", "password": "password1"})
        assert r.status_code == 200 and r.json()["token"] == "fresh"
        s = launcher.load_settings()
        assert (s.session_token, s.session_url) == ("fresh", "http://10.0.0.1:8765")
        assert c.get("/api/settings").json()["has_session"] is True
        c.cookies.clear()
        assert c.get("/api/coord/auth/me").json()["user"]["email"] == "ada@lan.test"
        assert http.seen[-1] == {"authorization": "Bearer fresh"}
        # The window's own save of the same token changes nothing (and re-binds what runs here).
        assert c.post("/api/settings", json={"session_token": "fresh"}).json()["has_session"] is True
        # Another pool never sees it.
        c.post("/api/settings", json={"mode": "join", "url": "10.0.0.2"})
        c.get("/api/coord/auth/me")
        assert http.seen[-1] == {}


def test_a_sign_in_through_the_proxy_rebinds_the_running_agent_to_its_session(tmp_path, monkeypatch):
    """The window no longer posts the token back after a sign-in: storing it as the reply is relayed
    must also restart what runs here with it, or the agent keeps earning for nobody until the next
    Start."""
    monkeypatch.delenv("SLASHCOMPUTE_SESSION", raising=False)
    app, launcher, spawned = _shell(tmp_path, http=SessionHTTP(live=[]))
    sessions, spawn = [], launcher._popen

    def popen(argv, env=None, **kw):
        sessions.append((env or {}).get("SLASHCOMPUTE_SESSION", ""))
        return spawn(argv, env=env, **kw)

    launcher._popen = popen
    launcher.save_settings(LauncherSettings(mode="join", url="10.0.0.1"))
    pool = launcher.proxy_url(launcher.load_settings())
    launcher.paths.pid_file.write_text("77\n")                                   # our agent, no session
    (tmp_path / "agent.args").write_text(json.dumps(launch_record(launcher.agent_argv(pool, 50), "")))
    running = {77: True}
    monkeypatch.setattr("slashcompute.launcher.controller.process_alive", lambda pid: running.get(pid, False))

    def stop(paths):
        running[77] = False
        paths.clear_pid()

    monkeypatch.setattr("slashcompute.launcher.controller.request_stop", stop)
    with TestClient(app, base_url=SHELL) as c:
        assert c.get("/api/status").json()["agent_running"]
        r = c.post("/api/coord/auth/login", json={"email": "ada@lan.test", "password": "password1"})
        assert r.status_code == 200 and r.json()["token"] == "fresh"
        assert c.get("/api/settings").json()["has_session"] is True
    assert launcher.load_settings().session_token == "fresh"
    assert [p.argv for p in spawned] == [launcher.agent_argv(pool, 50)] and sessions == ["fresh"]
    assert json.loads((tmp_path / "agent.args").read_text()) == launch_record(spawned[0].argv, "fresh")


def test_start_merges_its_body_under_the_lock_so_a_save_meanwhile_survives(tmp_path):
    """Start merged the body into the settings it had loaded and then waited for the lock: a slider
    save that landed in between was overwritten by what Start saved once it had the lock."""
    app, launcher, _ = _shell(tmp_path, http=FakeHTTP({"ok": True}))
    launcher.save_settings(LauncherSettings(mode="host", gpu_percent=50))
    with TestClient(app, base_url=SHELL) as c:
        with launcher._lock:                   # another Start (or a poll) holds the lock
            connect = threading.Thread(target=lambda: c.post(
                "/api/start", json={"mode": "join", "url": "10.0.0.5", "training": False}))
            connect.start()
            time.sleep(0.2)                    # it has its body and waits for the lock
            launcher.save_settings(replace(launcher.load_settings(), gpu_percent=30))   # the slider lands
        connect.join(10)
    assert not connect.is_alive()
    s = launcher.load_settings()
    assert (s.mode, s.url, s.gpu_percent, s.training) == ("join", "10.0.0.5", 30, False)


def test_overview_answers_with_the_last_snapshot_while_a_start_holds_the_lock(tmp_path, monkeypatch):
    """A Start holds the launcher's lock for up to ~25 s; every poll used to wait behind it and the
    window's pills froze. Bounded, the poll answers at once with what the last snapshot said."""
    monkeypatch.setattr("slashcompute.web.server.OVERVIEW_WAIT_S", 0.1)
    app, launcher, _ = _shell(tmp_path, http=FakeHTTP({"ok": True}))
    launcher.save_settings(LauncherSettings(mode="host"))
    with TestClient(app, base_url=SHELL) as c:
        first = c.get("/api/overview").json()["status"]
        assert first["coordinator_up"] is True and first["agent_restart_pending"] is False
        launcher._http.health = None           # the pool went away...
        with launcher._lock:                   # ...while a Start holds the lock
            t0 = time.monotonic()
            held = c.get("/api/overview").json()["status"]
            assert time.monotonic() - t0 < 1.0
        assert held["coordinator_up"] is True  # the last snapshot, not a stall
        fresh = c.get("/api/overview").json()["status"]
    assert fresh["coordinator_up"] is False    # free again: a full poll


@pytest.mark.parametrize("path", ["/api/overview", "/api/coord/auth/me"])
def test_launcher_calls_stay_off_the_event_loop(tmp_path, path):
    """Start and Start serving hold the launcher's lock for seconds. A handler that read the settings
    on the event loop waited there, and with the loop the whole window froze: polls, static files,
    the chat stream. While one request is stuck in the launcher, another must still be answered."""
    app, launcher, _ = _shell(tmp_path, http=FakeHTTP({"ok": True}))
    launcher.save_settings(LauncherSettings(mode="host"))
    real, inside = launcher.load_settings, threading.Event()

    def held_up():
        inside.set()
        time.sleep(0.4)          # a Start holding the lock
        return real()

    launcher.load_settings = held_up
    with TestClient(app, base_url=SHELL) as c:
        stuck = threading.Thread(target=lambda: c.get(path))
        stuck.start()
        assert inside.wait(5)
        t0 = time.monotonic()
        assert c.get("/api/shell").status_code == 200
        quick = time.monotonic() - t0
        stuck.join()
    assert quick < 0.15, f"the shell froze for {quick:.2f}s while a handler waited on the launcher"


def test_partial_settings_keep_the_stored_fields(tmp_path):
    app, launcher, _ = _shell(tmp_path)
    launcher.save_settings(LauncherSettings(mode="join", url="10.1.2.3", inference=True, grant_split=25,
                                            memory_gb=8, models_dir="/Volumes/m", transport="relay",
                                            training=False, inference_memory_gb=12))
    with TestClient(app, base_url=SHELL) as c:
        body = c.post("/api/settings", json={"gpu_percent": 30}).json()   # the slider alone
    assert body["gpu_percent"] == 30
    s = launcher.load_settings()
    assert (s.mode, s.url, s.inference, s.grant_split, s.memory_gb, s.models_dir, s.transport,
            s.training, s.inference_memory_gb) == \
        ("join", "10.1.2.3", True, 25, 8, "/Volumes/m", "relay", False, 12)
    assert body["mode"] == "join" and body["models_dir"] == "/Volumes/m"
    # The body never sets the session fields directly (only session_token, as a sign-in or sign-out).
    assert launcher.load_settings().session_url == ""


def test_unexpected_errors_are_json_and_logged(tmp_path):
    app, launcher, _ = _shell(tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    launcher.snapshot = boom
    with TestClient(app, base_url=SHELL, raise_server_exceptions=False) as c:
        r = c.get("/api/status")
        assert r.status_code == 500 and r.json() == {"detail": "RuntimeError: boom"}
        # FastAPI's 422 list becomes one readable line.
        bad = c.post("/api/settings", json=[1, 2])
        assert bad.status_code == 422 and isinstance(bad.json()["detail"], str)
        assert "body" in bad.json()["detail"] and "dictionary" in bad.json()["detail"]
    log = (tmp_path / "logs" / "shell.log").read_text()
    assert "GET /api/status" in log and "RuntimeError: boom" in log and "Traceback" in log


def test_live_grants_empty_when_coordinator_down(tmp_path):
    app, _, _ = _shell(tmp_path, http=RoutedHTTP({}))
    with TestClient(app, base_url=SHELL) as c:
        board = c.get("/api/grants?sort=least").json()
    assert board["sample"] is True
    assert board["online"] is False
    assert any("Irish-language" in g["title"] for g in board["grants"])
    assert board["pending"]


def test_sample_grants_when_coordinator_has_none(tmp_path):
    app, launcher, _ = _shell(tmp_path, http=RoutedHTTP({**POOL, "/grants": []}))
    launcher.save_settings(LauncherSettings(mode="host"))
    with TestClient(app, base_url=SHELL) as c:
        board = c.get("/api/grants?sort=top").json()
    assert board["sample"] is True
    assert board["online"] is True
    assert any("Irish-language" in g["title"] for g in board["grants"])
    assert all(g["id"].startswith("g") for g in board["grants"])


def test_live_grants_are_not_mixed_with_samples(tmp_path):
    grant = {"id": "live-1", "title": "Parser", "author": "Ada",
             "body": "Need FLOPs for a parser.", "goal_flops": 50e12,
             "received_flops": 10e12, "status": "approved"}
    app, launcher, _ = _shell(tmp_path, http=RoutedHTTP({**POOL, "/grants": [grant]}))
    launcher.save_settings(LauncherSettings(mode="host"))
    with TestClient(app, base_url=SHELL) as c:
        board = c.get("/api/grants?sort=top").json()
    assert board["sample"] is False
    assert [g["id"] for g in board["grants"]] == ["live-1"]
    assert not any("Irish-language" in g["title"] for g in board["grants"])


# Runs app.js under Node with a stub DOM; fetch answers from the `routes` the
# test swaps in, so a poll can see the coordinator come and go.


APP_JS = Path(__file__).resolve().parents[1] / "src/slashcompute/web/static/app.js"


DOM_HARNESS = r"""
const fs = require("fs");
const vm = require("vm");
const [appJs, phasesFile] = process.argv.slice(1);
const phases = JSON.parse(fs.readFileSync(phasesFile, "utf8"));
const stub = () => new Proxy(function () {}, {
  get: (_, k) => (k === Symbol.toPrimitive ? () => "" : k === "then" ? undefined : stub()),
  apply: () => stub(),
});
const els = new Map();
const element = (sel) => {
  if (!els.has(sel)) {
    const props = { textContent: "", innerHTML: "", className: "", hidden: false, disabled: false, value: "",
      dataset: {}, style: { setProperty() {} }, classList: { toggle() {}, add() {}, remove() {} } };
    els.set(sel, new Proxy(props, { get: (o, k) => (k in o ? o[k] : stub()) }));
  }
  return els.get(sel);
};
let routes = {};
const ctx = {
  document: { querySelector: element, querySelectorAll: () => [], addEventListener() {},
    createElement: element, body: stub(), activeElement: null },
  window: { setInterval() {}, setTimeout() {}, clearTimeout() {}, confirm: () => true,
    addEventListener() {}, removeEventListener() {} },
  CSS: { escape: (s) => s }, navigator: stub(), XMLHttpRequest: function () {},
  FormData: function () {}, console, calls: [],
  fetch: async (path, opts = {}) => {
    ctx.calls.push({ path, method: opts.method || "GET", body: opts.body ? JSON.parse(opts.body) : null });
    const body = routes[path.split("?")[0]];
    const ok = body !== undefined;
    return { ok, status: ok ? 200 : 404, statusText: "x",
      text: async () => JSON.stringify(ok ? body : { detail: "nope" }) };
  },
};
vm.createContext(ctx);
const settle = () => new Promise((r) => setTimeout(r, 30));
(async () => {
  const seen = [];
  for (const [i, phase] of phases.entries()) {
    routes = phase.routes;
    if (i === 0) {
      vm.runInContext(fs.readFileSync(appJs, "utf8"), ctx);
      vm.runInContext("showTab('grants')", ctx);
    } else {
      vm.runInContext(phase.run, ctx);
    }
    await settle();
    seen.push({ pill: element("#g-pill").textContent, open: element("#g-open").textContent,
      list: element("#g-list").innerHTML, probe: phase.probe && vm.runInContext(phase.probe, ctx) });
  }
  console.log(JSON.stringify(seen));
})();
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_grants_tab_follows_coordinator_on_poll(tmp_path):
    grant = {"id": "g1", "title": "Parser", "author": "Ada", "body": "Need FLOPs for a parser.",
             "goal_flops": 50e12, "received_flops": 10e12, "status": "approved"}
    app, launcher, _ = _shell(tmp_path / "up", http=RoutedHTTP({**POOL, "/grants": [grant]}))
    launcher.save_settings(LauncherSettings(mode="host"))
    with TestClient(app, base_url=SHELL) as c:
        live = {"/api/overview": c.get("/api/overview").json(), "/api/grants": c.get("/api/grants").json()}
    assert live["/api/overview"]["status"]["coordinator_up"] and live["/api/grants"]["online"]
    app, _, _ = _shell(tmp_path / "down", http=RoutedHTTP({}))
    with TestClient(app, base_url=SHELL) as c:
        down = {"/api/overview": c.get("/api/overview").json(), "/api/grants": c.get("/api/grants").json()}
    assert not down["/api/overview"]["status"]["coordinator_up"]
    board = live["/api/grants"]
    more = {**live, "/api/grants": {**board, "grants": board["grants"] + [{**board["grants"][0], "id": "g2"}]}}
    phases = [
        {"routes": live},
        {"routes": down, "run": "poll()"},
        {"routes": live, "run": "poll()"},
        # Same coordinator, new grant: only the slow refresh picks it up.
        {"routes": more, "run": "poll()"},
        {"routes": more, "run": "state.grantsAt -= 60000; poll()"},
    ]
    (tmp_path / "phases.json").write_text(json.dumps(phases))
    out = subprocess.run(["node", "-e", DOM_HARNESS, str(APP_JS), str(tmp_path / "phases.json")],
                         capture_output=True, text=True, timeout=30, check=True)
    seen = json.loads(out.stdout.strip().splitlines()[-1])
    assert (seen[0]["pill"], seen[0]["open"]) == ("Live", "1")
    assert "Fund this grant" in seen[0]["list"]
    assert (seen[1]["pill"], seen[1]["open"]) == ("Demo", "6")
    assert "Fund this grant" not in seen[1]["list"]
    assert "Irish-language" in seen[1]["list"]
    assert "Demo grant" in seen[1]["list"]
    assert (seen[2]["pill"], seen[2]["open"]) == ("Live", "1")
    assert seen[3]["open"] == "1"
    assert seen[4]["open"] == "2"


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_chat_shows_a_thinking_models_reasoning(tmp_path):
    """reasoning_content deltas render as a dim Thinking… block before the answer, so a thinking
    model's reply is never blank, even when it runs out of tokens before answering."""
    log = 'renderChat(); $("#l-log").innerHTML'
    phases = [
        {"routes": {}},
        {"routes": {}, "probe": log, "run": """
            state.llm.messages = [{ role: "user", content: "Is 1001 a prime number?" },
                                  { role: "assistant", content: "", live: true }];
            applyChatEvent(state.llm.messages[1], { choices: [{ delta: { role: "assistant" } }] });
            applyChatEvent(state.llm.messages[1], { choices: [{ delta: { reasoning_content: "1001 = 7 * 11" } }] });
            applyChatEvent(state.llm.messages[1], { choices: [{ delta: { reasoning_content: " * 13 <so no>" } }] });"""},
        {"routes": {}, "probe": log,
         "run": 'applyChatEvent(state.llm.messages[1], { choices: [{ delta: { content: "No." } }] })'},
        # the user opened it: the next render keeps it open
        {"routes": {}, "probe": log, "run": "state.llm.messages[1].thinkOpen = true"},
        {"routes": {}, "probe": log, "run": """
            const r = { role: "assistant", content: "", live: true };
            state.llm.messages.push(r);
            applyChatEvent(r, { choices: [{ delta: { reasoning_content: "Hmm" }, finish_reason: "length" }] });
            r.live = false;"""},
    ]
    (tmp_path / "phases.json").write_text(json.dumps(phases))
    out = subprocess.run(["node", "-e", DOM_HARNESS, str(APP_JS), str(tmp_path / "phases.json")],
                         capture_output=True, text=True, timeout=30, check=True)
    thinking, answered, reopened, ran_out = [s["probe"] for s in json.loads(out.stdout.strip().splitlines()[-1])[1:]]
    assert "<summary>Thinking…</summary>" in thinking and 'open=""' in thinking
    assert "1001 = 7 * 11 * 13 &lt;so no&gt;" in thinking and "<p>…</p>" not in thinking
    assert "<summary>Thoughts</summary>" in answered and 'open=""' not in answered
    assert answered.index("1001 = 7") < answered.index("<p>No.</p>")
    assert 'open=""' in reopened
    assert "Ran out of tokens while thinking" in ran_out.split("Hmm")[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_start_hosting_stays_clickable_after_the_coordinator_fails(tmp_path):
    """A coordinator that died (its port was taken) must not leave Start hosting disabled."""
    def probe(st):
        return (f'state.settings = {{ mode: "host" }}; state.ov = {{ status: {json.dumps(st)} }}; '
                'renderPool(); [$("#p-start").disabled, $("#p-start").textContent]')
    port = "Port 8765 is already in use — quit the other /compute or coordinator, then Start hosting."
    cases = [
        {"coordinator_pid": 4001, "coordinator_up": True, "last_error": ""},
        {"coordinator_pid": None, "coordinator_up": False, "last_error": port},
        # a status from before the launcher noticed the exit, or a pid it cannot vouch for
        {"coordinator_pid": 4001, "coordinator_up": False, "last_error": port},
    ]
    phases = [{"routes": {}}] + [{"routes": {}, "run": "0", "probe": probe(st)} for st in cases]
    (tmp_path / "phases.json").write_text(json.dumps(phases))
    out = subprocess.run(["node", "-e", DOM_HARNESS, str(APP_JS), str(tmp_path / "phases.json")],
                         capture_output=True, text=True, timeout=30, check=True)
    hosting, failed, stale = [s["probe"] for s in json.loads(out.stdout.strip().splitlines()[-1])[1:]]
    assert hosting == [True, "Hosting"]
    assert failed == [False, "Start hosting"]
    assert stale == [False, "Start hosting"]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_contributing_pill_shows_the_running_share(tmp_path):
    """Moving the slider while contributing saves the next share; the pills keep the live one."""
    def probe(node):
        ov = {"status": {"agent_running": True}, "me": {"flops": 0, **({"node": node} if node else {})}}
        return (f'state.settings = {{ gpu_percent: 30 }}; state.ov = {json.dumps(ov)}; '
                'renderContributions(); renderSidebar(); [$("#c-pill").textContent, $("#side-agent").textContent]')
    phases = [{"routes": {}}] + [{"routes": {}, "run": "0", "probe": probe(n)}
                                 for n in ({"gpu_percent": 50}, None)]
    (tmp_path / "phases.json").write_text(json.dumps(phases))
    out = subprocess.run(["node", "-e", DOM_HARNESS, str(APP_JS), str(tmp_path / "phases.json")],
                         capture_output=True, text=True, timeout=30, check=True)
    live, unlisted = [s["probe"] for s in json.loads(out.stdout.strip().splitlines()[-1])[1:]]
    assert live == ["Contributing · 50%", "Contributing · 50%"]
    # not in the pool's node list yet: fall back to the saved share
    assert unlisted == ["Contributing · 30%", "Contributing · 30%"]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_sign_out_after_registering_shows_sign_in(tmp_path):
    routes = {"/api/coord/auth/logout": {}, "/api/settings": {"session_token": ""}}
    phases = [
        {"routes": {}},
        {"routes": routes, "run": """
            state.user = { id: "u1", email: "ada@example.com", accepted_terms: true };
            actions["auth-mode"]();
            actions.logout(null);""",
         "probe": """[state.authMode, $("#auth-submit").textContent, $("[data-act='auth-mode']").textContent]"""},
    ]
    (tmp_path / "phases.json").write_text(json.dumps(phases))
    out = subprocess.run(["node", "-e", DOM_HARNESS, str(APP_JS), str(tmp_path / "phases.json")],
                         capture_output=True, text=True, timeout=30, check=True)
    assert json.loads(out.stdout.strip().splitlines()[-1])[1]["probe"] == ["login", "Sign in", "Create account"]


def test_live_grants_flow(tmp_path):
    T = 1e12
    app, launcher, _ = _shell(tmp_path, http=GrantHTTP())
    launcher.save_settings(LauncherSettings(mode="host"))
    with TestClient(app, base_url=SHELL) as c:
        board = c.get("/api/grants?sort=top").json()
        assert board["sample"] is False and board["online"] is True
        assert board["grants"][0]["summary"].startswith("Need FLOPs")
        assert board["grants"][0]["goal"] == 50e12
        assert board["pending"][0]["title"] == "Waiting"
        assert board["leaders"][0]["name"] == "Ada"
        assert board["available"] == 80e12

        funded = c.post("/api/grants/g1/fund", json={"amount": 10 * T})
        assert funded.status_code == 200, funded.text
        assert funded.json()["pledged"] == 10 * T
        assert funded.json()["grants"][0]["raised"] == 20e12

        made = c.post("/api/grants", json={
            "title": "Lecture notes",
            "summary": "Fine-tune a helper on my course notes for first years.",
            "goal": 50 * T,
        })
        assert made.status_code == 200, made.text
        assert any(g["title"] == "Lecture notes" for g in made.json()["pending"])
        assert c.post("/api/grants", json={"title": "", "summary": "x", "goal": 1}).status_code == 400
        assert c.post("/api/grants", json={"title": "t", "summary": "x",
                                           "goal": "1e309"}).status_code == 400
        assert c.post("/api/grants/g1/fund", json={"amount": "nan"}).status_code == 400

        approved = c.post("/api/grants/g2/review", json={"approve": True}).json()
        assert any(g["id"] == "g2" for g in approved["grants"])
        again = c.post("/api/grants/g2/review", json={"approve": True})
        assert again.status_code == 400
        # bool("false") is True: the shell used to forward this as an approval.
        assert c.post("/api/grants/g3/review", json={"approve": "yes"}).status_code == 400
        declined = c.post("/api/grants/g3/review", json={"approve": "false"})
        assert declined.status_code == 200, declined.text
        assert all(g["id"] != "g3" for g in declined.json()["grants"])


def test_grant_amounts_must_be_positive_and_finite(tmp_path):
    http = GrantHTTP()
    app, launcher, _ = _shell(tmp_path, http=http)
    launcher.save_settings(LauncherSettings(mode="host"))
    summary = "Fine-tune a helper on my course notes for first years."
    with TestClient(app, base_url=SHELL) as c:
        # Raw bodies: the JSON spec has no NaN/Infinity, but Python's parser (and float("inf")) accepts them.
        for bad in ("NaN", "Infinity", "-Infinity", "0", "-5", '"nan"', '"inf"', '"1e999"'):
            made = c.post("/api/grants", content=f'{{"title": "Lecture notes", "summary": "{summary}", '
                                                 f'"goal": {bad}}}',
                          headers={"content-type": "application/json"})
            assert made.status_code == 400, (bad, made.text)
            assert "finite" in made.json()["detail"]
            funded = c.post("/api/grants/g1/fund", content=f'{{"amount": {bad}}}',
                            headers={"content-type": "application/json"})
            assert funded.status_code == 400, (bad, funded.text)
        assert c.post("/api/grants/g1/fund", json={"amount": "lots"}).status_code == 400
        assert c.get("/api/grants").status_code == 200
    assert len(http.grants) == 2 and http.donated == 0.0


def test_non_finite_settings_are_clamped_not_500(tmp_path):
    app, _, _ = _shell(tmp_path)
    with TestClient(app, base_url=SHELL) as c:
        r = c.post("/api/settings", content='{"gpu_percent": Infinity, "grant_split": -Infinity, '
                                            '"memory_gb": NaN, "inference_memory_gb": 1e999}',
                   headers={"content-type": "application/json"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["gpu_percent"], body["grant_split"], body["memory_gb"], body["inference_memory_gb"]) == \
        (50, 0, 0, 0)


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_connect_moves_what_this_mac_lends_and_labels_follow_the_mode(tmp_path):
    """Connect restarts the running agent and LLM node on the new pool (it used to only save the
    address), and a coordinator still running here reads as Hosting only in host mode."""
    up = {"agent_running": True, "inference_running": True, "coordinator_pid": 4001, "coordinator_up": True}
    routes = {"/api/start": {"last_error": ""},
              "/api/overview": {"status": {"coordinator_up": True, "coordinator_url": "http://10.0.0.8:8765"}}}

    def labels(mode):
        return (f'state.settings = {{ mode: "{mode}" }}; state.ov = {{ status: {json.dumps(up)} }}; '
                'renderSidebar(); renderPool(); [$("#side-title").textContent, $("#p-status").innerHTML]')

    def connect(confirmed):
        return (f'calls.length = 0; window.confirm = () => {json.dumps(confirmed)}; '
                f'state.settings = {{ mode: "join", url: "", gpu_percent: 50 }}; state.ov = {{ status: {json.dumps(up)} }}; '
                '$("#url").value = "10.0.0.8"; actions.connect({})')

    starts = 'calls.filter((c) => c.path === "/api/start").map((c) => c.body)'
    phases = [{"routes": {}},
              {"routes": {}, "run": "0", "probe": labels("host")},
              {"routes": {}, "run": "0", "probe": labels("join")},
              {"routes": routes, "run": connect(False), "probe": starts},
              {"routes": routes, "run": connect(True), "probe": starts}]
    (tmp_path / "phases.json").write_text(json.dumps(phases))
    out = subprocess.run(["node", "-e", DOM_HARNESS, str(APP_JS), str(tmp_path / "phases.json")],
                         capture_output=True, text=True, timeout=30, check=True)
    host, join, declined, started = [s["probe"] for s in json.loads(out.stdout.strip().splitlines()[-1])[1:]]
    assert host[0] == "Hosting" and "hosting here" in host[1] and "Pool hosted here" not in host[1]
    assert join[0] == "Joined" and "reachable" in join[1] and "Pool hosted here" in join[1]
    assert declined == []   # keeping the pool hosted here: nothing changes
    assert len(started) == 1
    assert {k: started[0][k] for k in ("mode", "url", "training", "inference")} == {
        "mode": "join", "url": "10.0.0.8", "training": True, "inference": True}


def _dom_probes(tmp_path, phases):
    """Runs app.js under the Node DOM harness; each phase's probe after the first (the load)."""
    (tmp_path / "phases.json").write_text(json.dumps(phases))
    out = subprocess.run(["node", "-e", DOM_HARNESS, str(APP_JS), str(tmp_path / "phases.json")],
                         capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return [s["probe"] for s in json.loads(out.stdout.strip().splitlines()[-1])[1:]]


# A LAN pool whose coordinator has model controls: net(models) is its /inference/status, model(status, more)
# one row of it. (Not pool(): app.js has its own.)
LLM_POOL = """
  state.keys = {}; state.user = null; state.settings = { mode: "host" }; state.busy.clear();
  state.ov = { status: { coordinator_up: true } };
  globalThis.model = (status, more = {}) => ({ id: "q.gguf", status, status_reason: null, size_gb: 0.5, arch: "qwen3",
    uploaded: false, heads: ["Studio"], downloading: {}, servable: status === "ready", min_memory_gb: 2, load: null,
    held_by: [], restorable: true, ...more });
  globalThis.net = (models, more = {}) => ({ model_controls: true, nodes: [], pipelines: [], models, ...more });
  globalThis.air = { name: "Air", online: false, removing: false, error: null, kept: false, kept_folder: null };
"""

# The harness has no DOM, so renderModels' patch loop would find no cards. These stand in for them: one
# per .job in the markup last written to #l-catalog, with its .meta, .notes and a stub per button. Writing
# the markup again replaces every card, as a browser does, so a patched row can be told from a rebuilt one.
JOB_CARDS = """{
  const card = (job) => {
    const meta = { textContent: "" }, notes = { innerHTML: "", dataset: {} };
    const buttons = (job.match(/<button /g) || []).map(() => ({ disabled: false }));
    return { meta, notes, buttons, querySelector: (s) => ({ ".meta": meta, ".notes": notes })[s] || null,
             querySelectorAll: (s) => (s === ".acts .btn" ? buttons : []) };
  };
  let markup = "";
  globalThis.jobCards = [];
  Object.defineProperty($("#l-catalog"), "innerHTML", { configurable: true, get: () => markup,
    set: (html) => { markup = html; jobCards = html.split('<div class="job ').slice(1).map(card); } });
  document.querySelectorAll = (sel) => (sel === "#l-catalog .job" ? jobCards : []);
}
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_llm_rows_offer_the_buttons_each_model_state_allows(tmp_path):
    """Unload only while loaded or loading, Stop serving / Serve by the switch, Remove again and Add
    back on a removed model once nothing is in flight (a pool's uploaded copy that outlived Remove
    included); a kept copy names its folder; no buttons for a non-admin on a public pool or under a
    coordinator without model controls."""
    rows = """(() => {
      const row = (m, manage = true) => {
        const v = llmRow(m, manage);
        return { tag: v.tag, tone: v.tone, acts: v.acts.map((a) => a.label + (a.on ? `:${a.on}` : "")), notes: v.notes.map((n) => n[0]) };
      };
      const kept = (folder) => row(model("removed", { restorable: false,
        held_by: [{ ...air, online: true, kept: true, kept_folder: folder }] })).notes;
      const stuck = (more, manage) => row(model("removed", { restorable: false, uploaded: true, ...more }), manage);
      return {
        loaded: row(model("ready", { load: "loaded" })),
        loading: row(model("ready", { load: "loading" })),
        idle: row(model("ready")),
        unloading: row(model("ready", { load: "unloading" })),
        disabled: row(model("disabled", { servable: false })),
        offline: row(model("removed", { restorable: false, held_by: [air] })),
        oldRecord: row(model("removed", { restorable: false, held_by: [{ ...air, superseded: true }] })),
        restorable: row(model("removed", { load: "unloading" })),
        removing: row(model("removed", { held_by: [{ ...air, online: true, removing: true }] })),
        failed: row(model("removed", { restorable: false, held_by: [{ ...air, online: true, error: "q.gguf: Operation not permitted" }] })),
        kept: [kept("models"), kept("app"), kept(null)],
        stuck: stuck({ upload_error: "Operation not permitted" }),
        unknown: stuck({ upload_error: null }),
        unmanaged: stuck({ upload_error: "Operation not permitted" }, false),
        deleted: row(model("removed", { restorable: false, uploaded: false, held_by: [air] })).notes,
        rejected: row(model("rejected", { servable: false, status_reason: "unknown arch" })),
        nobody: llmRow(model("ready", { load: "loaded" }), false).acts,
      };
    })()"""
    catalog = 'state.keys = {}; renderModels(); [canManageModels(), $("#l-catalog").innerHTML, $("#l-count").textContent]'
    lan = 'state.llm.net = net([model("ready", { load: "loaded" }), model("removed", { id: "old.gguf", held_by: [air] })]);'
    public = 'state.settings = { mode: "public" }; state.user = { id: "u1", admin: %s }; state.llm.net = net([model("ready")]);'
    older = 'state.llm.net = net([model("ready", { load: "loaded" })], { model_controls: undefined });'
    phases = [{"routes": {}},
              {"routes": {}, "run": LLM_POOL, "probe": rows},
              {"routes": {}, "run": LLM_POOL + lan, "probe": catalog},
              {"routes": {}, "run": LLM_POOL + public % "false", "probe": catalog},
              {"routes": {}, "run": LLM_POOL + public % "true", "probe": catalog},
              {"routes": {}, "run": LLM_POOL + older, "probe": catalog}]
    r, lan_html, stranger, admin, old = _dom_probes(tmp_path, phases)

    assert r["loaded"] == {"tag": "loaded", "tone": "is-active", "acts": ["Unload", "Stop serving:0", "Remove"], "notes": []}
    assert r["loading"]["tag"] == "loading" and r["loading"]["acts"] == ["Unload", "Stop serving:0", "Remove"]
    assert r["idle"] == {"tag": "", "tone": "is-active", "acts": ["Stop serving:0", "Remove"], "notes": []}
    assert r["unloading"]["tag"] == "unloading" and r["unloading"]["acts"] == ["Stop serving:0", "Remove"]
    assert any("reply in progress finishes first" in n for n in r["unloading"]["notes"])
    assert r["disabled"]["tag"] == "not served" and r["disabled"]["acts"] == ["Serve:1", "Remove"]
    assert r["disabled"]["notes"][0].startswith("Stopped for the whole pool")
    assert r["offline"] == {"tag": "removed", "tone": "", "acts": ["Remove again"],
                            "notes": ["Air is offline and keeps its copy. Press Remove again once it is back."]}
    # an old record of a Mac that is online under a new id: not "offline", and Remove again clears it
    assert r["oldRecord"]["acts"] == ["Remove again"]
    assert r["oldRecord"]["notes"] == ["An old record of Air still lists a copy. Press Remove again."]
    assert r["restorable"]["tag"] == "removed" and r["restorable"]["acts"] == ["Add back:1"]
    assert r["removing"] == {"tag": "removing", "tone": "", "acts": [], "notes": ["Removing from Air…"]}
    assert r["failed"]["acts"] == ["Remove again"] and r["failed"]["notes"] == ["Air: q.gguf: Operation not permitted"]
    assert r["kept"] == [
        ["Kept in Air's own models folder. Delete it there by hand if you want it gone."],
        ["Kept in Air's app folder (~/.slashcompute/models): this pool has no record of sending that copy. "
         "Delete it there by hand if you want it gone."],
        ["Kept on Air. Delete it there by hand if you want it gone."],   # an older node does not say where
    ]
    # the coordinator could not delete the pool's uploaded copy: the row stays, with why and Remove again
    assert r["stuck"] == {"tag": "removed", "tone": "is-waiting", "acts": ["Remove again"], "notes": [
        "The pool's uploaded copy could not be deleted: Operation not permitted. Press Remove again."]}
    assert r["unknown"]["acts"] == ["Remove again"] and r["unknown"]["tone"] == ""   # e.g. after the host restarted
    assert r["unknown"]["notes"] == ["The pool's uploaded copy is still there. Press Remove again."]
    assert r["unmanaged"]["acts"] == [] and r["unmanaged"]["notes"] == [
        "The pool's uploaded copy could not be deleted: Operation not permitted."]
    assert not any("uploaded copy" in n for n in r["deleted"])
    assert r["rejected"]["tag"] == "rejected" and r["rejected"]["acts"] == ["Remove"]
    assert r["nobody"] == []

    manage, html, count = lan_html
    assert manage is True and count == "1"   # the removed row is listed but not counted
    assert 'data-act="llm-unload" data-model="q.gguf"' in html and 'data-act="llm-remove"' in html
    assert '<div class="acts">' in html and '<span class="tag ok">loaded</span>' in html
    assert 'data-model="old.gguf"' in html and ">Remove again<" in html
    manage, html, _ = stranger
    assert manage is False and "data-act" not in html and '<div class="acts">' not in html
    assert "admin can unload" in html
    manage, html, _ = admin
    assert manage is True and 'data-act="llm-serving"' in html
    assert '<span class="tag' not in html   # an idle ready model has no tag
    manage, html, _ = old
    assert "data-act" not in html and "older /compute" in html


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_llm_catalog_keeps_its_buttons_across_download_progress_and_patches_rows_in_place(tmp_path):
    """A download's progress changes every poll: renderModels patches it into the row's card, never a
    rebuild that replaces the buttons under the pointer. A tag change does rebuild. Each card gets its
    own model's line, notes and busy buttons."""
    rebuild = """(() => {
      state.llm.net = net([model("ready", { load: "loaded" })]);
      renderModels();
      const [card] = jobCards;
      state.llm.net.models[0].downloading = { Air: 0.4 };
      renderModels();
      const kept = [jobCards[0] === card, card.meta.textContent];
      state.llm.net.models[0].load = "unloading";
      renderModels();
      return { kept, rebuilt: jobCards[0] !== card, meta: jobCards[0].meta.textContent };
    })()"""
    rows = """(() => {
      state.llm.net = net([model("ready", { downloading: { Air: 0.4 } }), model("removed", { id: "old.gguf", held_by: [air] }),
        model("removed", { id: "up.gguf", restorable: false, uploaded: true, upload_error: "Operation not permitted" })]);
      state.busy.add(llmKey("old.gguf"));
      renderModels();
      return { cards: jobCards.map((c) => [c.meta.textContent, c.notes.innerHTML, c.buttons.map((b) => b.disabled)]),
               stuck: $("#l-catalog").innerHTML.split('<div class="job ')[3] };
    })()"""
    patch = """(() => {
      const meta = { textContent: "" };
      const notes = { innerHTML: "", dataset: {} };
      const buttons = [{ disabled: false }, { disabled: false }];
      const card = { querySelector: (s) => ({ ".meta": meta, ".notes": notes })[s] || null,
                     querySelectorAll: (s) => (s === ".acts .btn" ? buttons : []) };
      const v = llmRow(model("removed", { held_by: [air] }), true);
      patchModelCard(card, v, true);
      const first = [meta.textContent, notes.innerHTML, buttons.map((b) => b.disabled)];
      notes.innerHTML = "untouched";   // the same notes again are not written again
      patchModelCard(card, v, false);
      return { first, again: [notes.innerHTML, buttons.map((b) => b.disabled)] };
    })()"""
    phases = [{"routes": {}},
              {"routes": {}, "run": LLM_POOL + JOB_CARDS, "probe": rebuild},
              {"routes": {}, "run": LLM_POOL + JOB_CARDS, "probe": rows},
              {"routes": {}, "run": LLM_POOL, "probe": patch}]
    rebuilt, rendered, patched = _dom_probes(tmp_path, phases)
    assert rebuilt["kept"] == [True, "0.5 GB · qwen3 · downloading on Air 40%"]
    assert rebuilt["rebuilt"] is True
    assert rebuilt["meta"] == "0.5 GB · qwen3 · downloading on Air 40%"   # a rebuilt card is patched too
    assert rendered["cards"] == [
        ["0.5 GB · qwen3 · downloading on Air 40%", "", [False, False]],
        ["0.5 GB · qwen3 · removed from this pool",
         '<p class="note info">Air is offline and keeps its copy. Press Remove again once it is back.</p>', [True, True]],
        # no Mac holds it, but the pool's uploaded copy outlived Remove: listed, with why, until Remove again works
        ["0.5 GB · qwen3 · removed from this pool",
         '<p class="note ">The pool&#39;s uploaded copy could not be deleted: Operation not permitted. Press Remove again.</p>',
         [False]],
    ]
    assert rendered["stuck"].startswith('is-waiting">') and 'data-act="llm-remove" data-model="up.gguf"' in rendered["stuck"]
    assert ">Remove again<" in rendered["stuck"]
    meta, notes, disabled = patched["first"]
    assert meta == "0.5 GB · qwen3 · removed from this pool"
    assert notes == '<p class="note info">Air is offline and keeps its copy. Press Remove again once it is back.</p>'
    assert disabled == [True, True]
    assert patched["again"] == ["untouched", [False, False]]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_llm_remove_asks_first_and_names_what_happens_on_each_mac(tmp_path):
    """The confirm names the Macs and says what Remove does there: copies this pool sent are deleted,
    any other copy goes to the Trash (LAN) or stays (public); a download under way is named too, and a
    Mac that came back under a new id is not also called offline. Cancel sends nothing, OK sends one
    POST with the model in the body (never in the path). The toast says what was done, and is an error
    giving the coordinator's reason when the pool's uploaded copy could not be deleted."""
    held = ('state.llm.net = net([model("ready", { uploaded: true, held_by: '
            '[{ ...air, name: "Studio", online: true }, air] })]);')
    questions = """[removeQuestion(llmModel("q.gguf"), false), removeQuestion(model("ready", { id: "z.gguf" }), false),
                    removeQuestion(llmModel("q.gguf"), true),
                    removeQuestion(model("ready", { id: "d.gguf", downloading: { Studio: 0.4 } }), false),
                    removeQuestion(model("ready", { id: "s.gguf", held_by: [{ ...air, name: "Studio", superseded: true },
                      { ...air, name: "Studio", online: true }, { ...air, name: "Studio", online: true }, air] }), false),
                    removeQuestion(model("ready", { id: "t.gguf", held_by: [{ ...air, name: "Studio" },
                      { ...air, name: "Studio", online: true }] }), false)]"""
    remove = ('calls.length = 0; globalThis.asked = []; globalThis.toasts = []; '
              'toast = (text, tone = "ok") => toasts.push([text, tone]); '
              'window.confirm = (q) => { asked.push(q); return %s; }; '
              'actions["llm-remove"]({ dataset: { model: "q.gguf" }, textContent: "Remove" });')
    sent = ('[calls.filter((c) => c.path.includes("/inference/models")).map((c) => [c.method, c.path, c.body]),'
            ' asked.length, toasts]')
    removed = {"ok": True, "model": "q.gguf", "macs": ["Studio"], "offline": ["Air"], "uploaded": True,
               "upload_error": None}
    routes = {"/api/coord/inference/models/remove": removed}
    stuck = {"/api/coord/inference/models/remove": {**removed, "upload_error": "Operation not permitted"}}
    phases = [{"routes": {}},
              {"routes": {}, "run": LLM_POOL + held, "probe": questions},
              {"routes": routes, "run": LLM_POOL + held + remove % "false", "probe": sent},
              {"routes": routes, "run": LLM_POOL + held + remove % "true", "probe": sent},
              {"routes": stuck, "run": LLM_POOL + held + remove % "true", "probe": sent}]
    (lan, nobody, public, copying, moved, namesake), cancelled, confirmed, failed = _dom_probes(tmp_path, phases)
    assert lan.startswith("Remove q.gguf from this pool?")
    assert "unloaded now" in lan and "a reply in progress finishes first" in lan
    assert "The pool's uploaded copy is deleted." in lan
    assert ("On Studio: copies this pool sent are deleted. Any other copy (in a Mac's own models folder, say) "
            "goes to that Mac's Trash.") in lan
    assert "Macs that are offline keep their copy for now: Air. Press Remove again once they are back." in lan
    assert "downloading" not in lan
    assert "No Mac has a copy" in nobody and "Trash" not in nobody and "uploaded copy" not in nobody
    assert ("On Studio: copies this pool sent are deleted. Any other copy (in a Mac's own models folder, say) "
            "stays where it is: a public pool never moves files to the Trash.") in public
    assert "goes to that Mac's Trash" not in public
    # Studio's old, offline row is the same Mac back under a new id: named once, as online
    assert "On Studio: copies this pool sent" in moved
    assert "Macs that are offline keep their copy for now: Air." in moved
    # an offline Mac that only shares a name with an online one is another Mac: it is named as offline
    assert "On Studio: copies this pool sent" in namesake
    assert "Macs that are offline keep their copy for now: Studio." in namesake
    # a head part-way through a download is not a holder yet, but it finishes and keeps that copy
    assert "No Mac has a copy" not in copying
    assert "Macs still downloading it keep the copy they finish: Studio. Press Remove again once they are done." in copying
    assert cancelled == [[], 1, []]   # asked, nothing sent
    post = ["POST", "/api/coord/inference/models/remove", {"model": "q.gguf"}]
    assert confirmed == [[post], 1, [["Removed q.gguf. Deleting its copies on Studio. Offline, still holding it: Air.", "ok"]]]
    assert failed == [[post], 1, [["Removed q.gguf. The pool's uploaded copy could not be deleted: Operation not permitted. "
                                   "Deleting its copies on Studio. Offline, still holding it: Air.", "bad"]]]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_llm_why_offers_serve_for_a_stopped_model(tmp_path):
    """A model whose Serve switch is off: the box under Send says so and its Serve posts on=true (a
    real boolean: the coordinator refuses 1); someone who may not manage models is told to ask the
    admin. The picker drops a removed model and labels a stopped one. Stop serving in the Models list
    posts on=false, a boolean too."""
    setup = LLM_POOL + """
      state.llm.net = net([model("disabled", { servable: false }), model("removed", { id: "gone.gguf", held_by: [air] })]);
      state.llm.models = []; state.llm.model = "q.gguf"; state.llm.error = "";
      globalThis.st = { coordinator_up: true, inference_running: true, inference_status: { node_id: "n1", available: true } };"""
    why = """(() => {
      const admin = llmWhy(st, {}, state.llm, false);
      state.settings = { mode: "public" }; state.user = { id: "u1", admin: false };
      const stranger = llmWhy(st, {}, state.llm, false);
      state.settings = { mode: "host" }; state.user = null;
      renderLlm();
      return { admin, stranger, choices: llmChoices(state.llm).map((m) => m.id), picker: $("#l-model").innerHTML,
               pipe: $("#l-pipe").innerHTML };
    })()"""
    click = ('{ renderWhy(llmWhy(st, {}, state.llm, false)); calls.length = 0; '
             'const b = $("#l-why-act"); actions[b.dataset.act](b); }')
    # The Models list's Stop serving, clicked with the data-* attributes it was rendered with.
    stop = """
      state.llm.net = net([model("ready")]); renderModels(); calls.length = 0;
      { const tag = $("#l-catalog").innerHTML.match(/<button [^>]*>Stop serving</)[0];
        const dataset = Object.fromEntries([...tag.matchAll(/data-(\\w+)="([^"]*)"/g)].map((m) => [m[1], m[2]]));
        actions[dataset.act]({ dataset, textContent: "Stop serving" }); }"""
    # JSON.parse'd in the harness: Python reads {"on": 1} as equal to {"on": True}, so the JS type is checked too.
    sent = ('calls.filter((c) => c.path === "/api/coord/inference/models/serving")'
            '.map((c) => [c.body, typeof c.body.on])')
    routes = {"/api/coord/inference/models/serving": {"ok": True, "serving": True, "pipelines": 0}}
    phases = [{"routes": {}},
              {"routes": {}, "run": setup, "probe": why},
              {"routes": routes, "run": setup + click, "probe": sent},
              {"routes": routes, "run": setup + stop, "probe": sent}]
    shown, posted, stopped = _dom_probes(tmp_path, phases)
    assert shown["admin"]["text"].startswith("q.gguf is not being served")
    assert shown["admin"]["action"] == {"label": "Serve", "act": "llm-serving", "model": "q.gguf", "on": "1"}
    assert "ask the pool's admin" in shown["stranger"]["text"].lower() and "action" not in shown["stranger"]
    assert shown["choices"] == ["q.gguf"]
    assert "q.gguf · 0.5 GB · stopped" in shown["picker"] and "gone.gguf" not in shown["picker"]
    assert "stopped for the whole pool" in shown["pipe"] and "Press Serve" in shown["pipe"]
    assert posted == [[{"model": "q.gguf", "on": True}, "boolean"]]
    assert stopped == [[{"model": "q.gguf", "on": False}, "boolean"]]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_llm_why_button_follows_the_box_while_another_model_is_busy(tmp_path):
    """The button under Send offers what the box explains, even while the model it last named is busy
    with an action pressed elsewhere (Remove in the Models list) or this Mac starts serving from the
    switch. Only an action pressed on the button itself keeps its label until done."""
    setup = LLM_POOL + """
      state.llm.net = net([model("disabled", { servable: false })]);
      state.llm.models = []; state.llm.model = "q.gguf"; state.llm.error = "";
      state.ov = { status: { coordinator_up: true, inference_running: true,
                             inference_status: { node_id: "n1", available: true } } };
      globalThis.button = () => { const b = $("#l-why-act");
        return [$("#l-why-text").textContent, b.textContent, b.dataset.act, b.dataset.model, b.hidden, b.disabled]; };"""
    elsewhere = """(() => {
      renderLlm();
      const serve = button();
      state.busy.add(llmKey("q.gguf"));   // Remove pressed on q.gguf in the Models list; its refresh picks y.gguf
      state.llm.net = net([model("removed", { held_by: [air] }), model("ready", { id: "y.gguf" })]);
      state.llm.model = "y.gguf";
      renderLlm();
      const waiting = button();
      state.ov.status.inference_running = false;
      renderLlm();
      const start = button();
      actions["toggle-llm"]($("#l-toggle"));   // this Mac starts serving from the switch, not from the box
      renderLlm();
      return { serve, waiting, start, toggled: button() };
    })()"""
    # (Set in the probe: the switch's start above ends while this phase settles, and its poll clears state.ov.)
    pressed = """(() => {
      state.llm.net = net([model("ready", { id: "y.gguf" })]); state.llm.model = "y.gguf";
      state.ov = { status: { coordinator_up: true, inference_running: false } };
      renderLlm();
      const b = $("#l-why-act");
      actions[b.dataset.act](b);
      renderLlm();   // a poll while it starts: the box says so, the button keeps its label
      return button();
    })()"""
    phases = [{"routes": {}},
              {"routes": {}, "run": setup, "probe": elsewhere},
              {"routes": {}, "run": setup, "probe": pressed}]
    shown, own = _dom_probes(tmp_path, phases)
    assert shown["serve"][1:] == ["Serve", "llm-serving", "q.gguf", False, False]
    assert shown["waiting"][0] == "Waiting for y.gguf to reach this Mac." and shown["waiting"][4] is True
    assert shown["start"] == ["No Mac is serving y.gguf yet.", "Start serving on this Mac", "llm-serve", "", False, False]
    assert shown["toggled"][0] == "Starting llama.cpp on this Mac…" and shown["toggled"][4] is True
    assert own == ["Starting llama.cpp on this Mac…", "Starting…", "llm-serve", "", False, True]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_llm_pipeline_card_shows_the_serving_pipeline_and_keeps_unload_across_speed_updates(tmp_path):
    """The pipeline serving now is shown before one still draining; the live speed doesn't rebuild the
    card; a draining pipeline's Unload is a disabled Unloading…, and an older coordinator keeps the
    per-pipeline Unload."""
    pipes = ('state.llm.model = "q.gguf"; state.llm.net = net([model("ready", { load: "loaded" })], { pipelines: ['
             '{ id: "p-old", model: "q.gguf", state: "draining", members: [], explanation: "" },'
             '{ id: "p-new", model: "q.gguf", state: "active", members: [], explanation: "", live_tok_s: 5 }] });')
    card = """(() => {
      renderLlm();
      const first = [$("#l-pipe-tag").textContent, $("#l-pipe").innerHTML, state.keys["llm-pipe"]];
      state.llm.net.pipelines[1].live_tok_s = 9;
      renderLlm();
      const steady = state.keys["llm-pipe"] === first[2];
      const draining = pipeUnload(state.llm.net.pipelines[0]);
      state.llm.net.model_controls = false;
      return { first: first.slice(0, 2), steady, draining, older: pipeUnload(state.llm.net.pipelines[1]) };
    })()"""
    phases = [{"routes": {}}, {"routes": {}, "run": LLM_POOL + pipes, "probe": card}]
    (shown,) = _dom_probes(tmp_path, phases)
    tag, html = shown["first"]
    assert tag == "active"
    assert 'data-act="llm-unload" data-model="q.gguf">Unload<' in html and "data-pipeline" not in html
    assert shown["steady"] is True
    assert "disabled" in shown["draining"] and "Unloading…" in shown["draining"] and "data-act" not in shown["draining"]
    assert 'data-pipeline="p-new"' in shown["older"]
