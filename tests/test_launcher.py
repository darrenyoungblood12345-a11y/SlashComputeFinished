import json
import signal
import socket
import stat
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from slashcompute.launcher.controller import (
    AGENT_STOP_MARGIN_S, Launcher, LauncherError, LauncherSettings, health_timeout, launch_record,
    normalize_url, write_private,
)


@pytest.fixture(autouse=True)
def _unknown_cmdlines(monkeypatch):
    """The fake pids in these tests belong to no process: their command lines are unreadable, which
    leaves the pid files trusted as before. Tests of the ownership check set their own."""
    monkeypatch.setattr("slashcompute.launcher.controller.process_cmdline", lambda pid: None)


def _this_mac(monkeypatch, interfaces: dict | None = None, hostname: str = "test-mac.local",
              names: dict[str, list[str]] | None = None) -> None:
    """What is_own_url sees of this Mac: its interfaces' addresses, its hostname, and the names
    that resolve (none by default: no DNS or mDNS lookup ever leaves a test)."""
    addrs = {ifname: [SimpleNamespace(family=socket.AF_INET6 if ":" in ip else socket.AF_INET, address=ip)
                      for ip in ips] for ifname, ips in (interfaces or {}).items()}
    resolves = {k.lower(): v for k, v in (names or {}).items()}

    def getaddrinfo(host, port, *a, **kw):
        if host.lower() not in resolves:
            raise socket.gaierror(8, "nodename nor servname provided, or not known")
        return [(socket.AF_INET6 if ":" in ip else socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))
                for ip in resolves[host.lower()]]

    monkeypatch.setattr("slashcompute.launcher.controller.psutil.net_if_addrs", lambda: addrs)
    monkeypatch.setattr("slashcompute.launcher.controller.socket.gethostname", lambda: hostname)
    monkeypatch.setattr("slashcompute.launcher.controller.socket.getaddrinfo", getaddrinfo)


@pytest.fixture(autouse=True)
def _one_plain_mac(monkeypatch):
    """By default this Mac has no interfaces beyond the injected lan_ip and resolves no names, so
    is_own_url never depends on the machine running the tests."""
    _this_mac(monkeypatch)


def _fake_clock(monkeypatch) -> dict:
    """time.monotonic/time.sleep on a clock that only sleeping advances: waits take no real time."""
    clock = {"t": 0.0}
    monkeypatch.setattr("slashcompute.launcher.controller.time.monotonic", lambda: clock["t"])
    monkeypatch.setattr("slashcompute.launcher.controller.time.sleep",
                        lambda seconds: clock.__setitem__("t", clock["t"] + seconds))
    return clock


class FakeProc:
    def __init__(self, pid: int, argv: list[str], env: dict | None = None) -> None:
        self.pid = pid
        self.argv = argv
        self.env = env or {}
        self.returncode = None   # set to make the process "exit"
        self.signals: list[int] = []   # what the launcher sent through the Popen

    @property
    def session(self) -> str:
        return self.env.get("SLASHCOMPUTE_SESSION", "")

    def poll(self):
        return self.returncode

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


class FakeHTTP:
    def __init__(self, health=None) -> None:
        self.health = health
        self.urls: list[str] = []

    def get(self, url: str, timeout: float = 1.0):
        self.urls.append(url)
        if self.health is None:
            raise ConnectionError("down")
        return FakeResponse(self.health)


def _launcher(tmp_path: Path, **kw) -> Launcher:
    spawned: list[FakeProc] = []
    next_pid = {"n": 4000}

    def popen(argv, **kw):
        next_pid["n"] += 1
        proc = FakeProc(next_pid["n"], argv, kw.get("env"))
        spawned.append(proc)
        return proc

    launcher = Launcher(
        home=tmp_path,
        python="/opt/venv/bin/python",
        popen=popen,
        http=kw.pop("http", FakeHTTP()),
        discover_fn=kw.pop("discover_fn", lambda timeout=5.0: None),
        lan_ip_fn=kw.pop("lan_ip_fn", lambda: "192.168.1.20"),
        port_free_fn=kw.pop("port_free_fn", lambda host, port: True),
    )
    launcher._spawned = spawned  # type: ignore[attr-defined]
    return launcher


def _signed_in(launcher: Launcher, token: str, **kw) -> LauncherSettings:
    """Settings holding `token`, issued by the pool they point at."""
    return launcher.with_session(LauncherSettings(**kw), token)


def test_normalize_url():
    assert normalize_url("") == ""
    assert normalize_url("192.168.1.10") == "http://192.168.1.10:8765"
    assert normalize_url("192.168.1.10:9000") == "http://192.168.1.10:9000"
    assert normalize_url("http://10.0.0.2:8765/") == "http://10.0.0.2:8765"
    assert normalize_url("pool.example.com", scheme="https") == "https://pool.example.com:8765"
    assert normalize_url("https://pool.example.com/") == "https://pool.example.com"
    assert health_timeout("https://pool.example.com") == 5.0
    assert health_timeout("http://10.0.0.1:8765") == 1.0


def test_settings_persist(tmp_path):
    launcher = _launcher(tmp_path)
    s = LauncherSettings(mode="join", url="http://10.0.0.1:8765", gpu_percent=75,
                         contribute=False)
    launcher.save_settings(s)
    raw = json.loads((tmp_path / "launcher.json").read_text())
    assert raw["mode"] == "join"
    assert raw["gpu_percent"] == 75
    loaded = launcher.load_settings()
    assert loaded == s.clamp()


def test_settings_corrupt_and_clamp(tmp_path):
    launcher = _launcher(tmp_path)
    (tmp_path / "launcher.json").write_text("not-json")
    assert launcher.load_settings() == LauncherSettings()
    launcher.save_settings(LauncherSettings(mode="nope", gpu_percent=999, contribute=1))
    s = launcher.load_settings()
    assert s.mode == "host"
    assert s.gpu_percent == 100
    assert s.contribute is True
    assert LauncherSettings(mode="public").clamp().mode == "public"


def test_coordinator_and_agent_argv(tmp_path):
    launcher = _launcher(tmp_path)
    assert launcher.coordinator_argv() == [
        "/opt/venv/bin/python", "-m", "slashcompute.coordinator.main", "serve",
        "--home", str(tmp_path),
    ]
    assert launcher.agent_argv("http://192.168.1.20:8765", 40) == [
        "/opt/venv/bin/python", "-m", "slashcompute.agent.main", "start",
        "--url", "http://192.168.1.20:8765", "--gpu-percent", "40",
        "--home", str(tmp_path),
    ]


def test_host_url_uses_lan_ip(tmp_path):
    launcher = _launcher(tmp_path, lan_ip_fn=lambda: "10.1.2.3")
    assert launcher.coordinator_url(LauncherSettings(mode="host")) == "http://10.1.2.3:8765"
    assert launcher.coordinator_url(LauncherSettings(mode="join", url="10.1.2.9")) == (
        "http://10.1.2.9:8765"
    )
    assert launcher.coordinator_url(LauncherSettings(mode="public", url="pool.example.com")) == (
        "https://pool.example.com:8765"
    )
    launcher.cfg.public_url = "https://pool.example.com"
    assert launcher.coordinator_url(LauncherSettings(mode="public", url="")) == (
        "https://pool.example.com"
    )


def test_start_host_spawns_coordinator_and_agent(tmp_path, monkeypatch):
    http = FakeHTTP()
    launcher = _launcher(tmp_path, http=http)

    def health_after_spawn(url: str):
        if launcher._spawned:  # type: ignore[attr-defined]
            http.health = {"ok": True, "nodes": 0, "jobs": 0}
            return {"ok": True, "nodes": 0, "jobs": 0}
        return None

    def alive(pid: int) -> bool:
        return any(p.pid == pid for p in launcher._spawned)  # type: ignore[attr-defined]

    monkeypatch.setattr("slashcompute.launcher.controller.process_alive", alive)
    launcher.poll_health = health_after_spawn  # type: ignore[method-assign]
    snap = launcher.start(LauncherSettings(mode="host", gpu_percent=50, contribute=True))
    argv_lists = [p.argv for p in launcher._spawned]  # type: ignore[attr-defined]
    assert argv_lists[0][:4] == ["/opt/venv/bin/python", "-m", "slashcompute.coordinator.main", "serve"]
    assert argv_lists[1][2:4] == ["slashcompute.agent.main", "start"]
    assert "--url" in argv_lists[1] and "127.0.0.1" in argv_lists[1][argv_lists[1].index("--url") + 1]
    assert "--no-sandbox" not in argv_lists[1]
    assert (tmp_path / "coordinator.pid").read_text().strip() == str(launcher._spawned[0].pid)
    assert json.loads((tmp_path / "agent.args").read_text()) == launch_record(argv_lists[1], "")
    assert stat.S_IMODE((tmp_path / "agent.args").stat().st_mode) == 0o600
    assert snap.last_error == ""


BIND_ERROR = (b"ERROR:    [Errno 48] error while attempting to bind on address ('0.0.0.0', 8765): "
              b"[errno 48] address already in use\n")


def test_coordinator_that_cannot_bind_is_reported_and_hosting_can_be_retried(tmp_path, monkeypatch):
    # Another coordinator held :8765 but did not answer /health: ours logged the bind error and
    # exited 3. The launcher used to keep reporting its (zombie) pid with no error, so Start
    # hosting stayed disabled and the pool offline for good.
    launcher = _launcher(tmp_path)
    spawn = launcher._popen

    def dies_on_bind(argv, **kw):
        proc = spawn(argv, **kw)
        kw["stdout"].write(BIND_ERROR)
        proc.returncode = 3
        return proc

    launcher._popen = dies_on_bind
    # An exited child nobody reaped still answers kill(0).
    monkeypatch.setattr("slashcompute.launcher.controller.process_alive",
                        lambda pid: any(p.pid == pid for p in launcher._spawned))  # type: ignore[attr-defined]
    with pytest.raises(LauncherError, match="Port 8765 is already in use"):
        launcher.start(LauncherSettings(mode="host", contribute=False))
    snap = launcher.snapshot()
    assert snap.coordinator_pid is None and not (tmp_path / "coordinator.pid").exists()
    assert snap.last_error == ("Port 8765 is already in use — quit the other /compute or coordinator, "
                               "then Start hosting.")
    assert "address already in use" in (tmp_path / "logs" / "coordinator.log").read_text()

    launcher._popen = spawn   # the other app quit: Start hosting works again
    launcher.poll_health = lambda url: {"ok": True} if len(launcher._spawned) == 2 else None  # type: ignore
    snap = launcher.start(LauncherSettings(mode="host", contribute=False))
    assert len(launcher._spawned) == 2  # type: ignore[attr-defined]
    assert snap.coordinator_pid == launcher._spawned[1].pid and snap.last_error == ""  # type: ignore


def test_coordinator_exit_reports_the_end_of_its_log_but_a_stop_does_not(tmp_path, monkeypatch):
    monkeypatch.setattr("slashcompute.launcher.controller.process_alive", lambda pid: True)
    monkeypatch.setattr("slashcompute.launcher.controller.os.kill", lambda pid, sig: None)
    launcher = _launcher(tmp_path)
    (launcher.log_dir / "coordinator.log").write_bytes(BIND_ERROR)   # an earlier run's failure
    launcher.poll_health = lambda url: {"ok": True} if launcher._spawned else None  # type: ignore
    launcher.start(LauncherSettings(mode="host", contribute=False))
    with open(tmp_path / "logs" / "coordinator.log", "ab") as fh:
        fh.write(b"Traceback (most recent call last):\nModuleNotFoundError: No module named 'zeroconf'\n")
    launcher._spawned[0].returncode = 1  # type: ignore[attr-defined]
    launcher.poll_health = lambda url: None  # type: ignore
    snap = launcher.snapshot()
    assert snap.coordinator_pid is None
    assert snap.last_error.startswith("The coordinator exited with code 1: ")
    assert "No module named 'zeroconf'" in snap.last_error and "Port" not in snap.last_error

    launcher.poll_health = lambda url: {"ok": True} if len(launcher._spawned) == 2 else None  # type: ignore
    assert launcher.start(LauncherSettings(mode="host", contribute=False)).last_error == ""
    launcher.poll_health = lambda url: None  # type: ignore
    launcher.stop()
    launcher._spawned[-1].returncode = -signal.SIGTERM  # type: ignore[attr-defined]
    assert launcher.snapshot().last_error == ""


def test_start_hosting_on_a_taken_port(tmp_path):
    # Something else answers no /health on :8765: say so instead of spawning a coordinator that dies.
    launcher = _launcher(tmp_path, port_free_fn=lambda host, port: False)
    with pytest.raises(LauncherError, match="Port 8765 is already in use"):
        launcher.start(LauncherSettings(mode="host", contribute=False))
    assert launcher._spawned == []  # type: ignore[attr-defined]
    # A coordinator that answers on loopback (not the LAN address) is used as it is.
    launcher.poll_health = lambda url: {"ok": True} if "127.0.0.1" in url else None  # type: ignore
    snap = launcher.start(LauncherSettings(mode="host", contribute=False))
    assert launcher._spawned == [] and snap.last_error == ""  # type: ignore[attr-defined]


@pytest.fixture
def stray_coordinator(tmp_path):
    """A process that looks like our coordinator (serve --home tmp_path) but has no coordinator.pid."""
    import subprocess
    import sys

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)",
                             "slashcompute.coordinator.main", "serve", "--home", str(tmp_path)])
    yield proc
    proc.kill()
    proc.wait()


def test_hosting_asks_loopback_and_stop_stops_our_coordinator_without_its_pid_file(
        tmp_path, stray_coordinator):
    # Our coordinator answered only on loopback (bound to 127.0.0.1, or the LAN reply was slow):
    # Start spawned a duplicate that overwrote coordinator.pid and died, then Stop left ours up.
    http = FakeHTTP({"ok": True})
    launcher = _launcher(tmp_path, http=http)
    snap = launcher.start(LauncherSettings(mode="host", contribute=False))
    assert http.urls[0] == "http://127.0.0.1:8765/health"
    assert launcher._spawned == []  # type: ignore[attr-defined]
    assert snap.coordinator_pid == stray_coordinator.pid   # re-recorded from its command line
    (tmp_path / "coordinator.pid").unlink()
    launcher.stop()
    assert stray_coordinator.wait(timeout=10) == -signal.SIGTERM


def test_start_never_spawns_over_our_coordinator_that_is_slow_to_answer(tmp_path, stray_coordinator):
    launcher = _launcher(tmp_path)
    launcher.wait_health = lambda url, timeout=8.0: False  # type: ignore[method-assign]
    launcher.start(LauncherSettings(mode="host", contribute=False))
    assert launcher._spawned == []  # type: ignore[attr-defined]
    assert launcher.read_coordinator_pid() == stray_coordinator.pid


def test_coordinator_pid_file_of_a_zombie_or_reused_pid_is_not_hosting(tmp_path):
    import os
    import subprocess
    import sys
    import time

    import psutil

    launcher = _launcher(tmp_path)
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        deadline = time.monotonic() + 10
        while psutil.Process(child.pid).status() != psutil.STATUS_ZOMBIE and time.monotonic() < deadline:
            time.sleep(0.02)
        (tmp_path / "coordinator.pid").write_text(f"{child.pid}\n")
        assert launcher.read_coordinator_pid() is None
        assert not (tmp_path / "coordinator.pid").exists()
    finally:
        child.wait()
    (tmp_path / "coordinator.pid").write_text(f"{os.getpid()}\n")
    assert launcher.read_coordinator_pid() == os.getpid()
    os.utime(tmp_path / "coordinator.pid", (0, 0))   # recorded long before this process began
    assert launcher.read_coordinator_pid() is None


def test_port_free_sees_a_listener():
    import socket

    from slashcompute.launcher.controller import port_free

    with socket.socket() as s:
        s.bind(("0.0.0.0", 0))
        s.listen()
        port = s.getsockname()[1]
        assert port_free("0.0.0.0", port) is False
    assert port_free("0.0.0.0", port) is True


def test_start_host_without_contribute_skips_agent(tmp_path):
    http = FakeHTTP()
    launcher = _launcher(tmp_path, http=http)
    launcher.poll_health = lambda url: {"ok": True} if launcher._spawned else None  # type: ignore
    launcher.start(LauncherSettings(mode="host", contribute=False))
    kinds = [p.argv[2] for p in launcher._spawned]  # type: ignore[attr-defined]
    assert kinds == ["slashcompute.coordinator.main"]


def test_start_hosting_keeps_this_macs_agent_and_llm_node(tmp_path, monkeypatch):
    # "Start hosting" re-posts the running contribution with mode=host: the coordinator comes up
    # and the training agent and LLM node already lending to it keep running.
    monkeypatch.delenv("SLASHCOMPUTE_SESSION", raising=False)
    launcher = _launcher(tmp_path, http=FakeHTTP({"ok": True, "inference_transport": "direct"}))
    settings = LauncherSettings(mode="host", contribute=True, training=True, inference=True)
    local = launcher.proxy_url(settings)
    launcher.paths.pid_file.write_text("77\n")
    (tmp_path / "agent.args").write_text(json.dumps(launch_record(launcher.agent_argv(local, 50), "")))
    launcher.inference_pid_path.parent.mkdir(parents=True, exist_ok=True)
    launcher.inference_pid_path.write_text("88\n")
    (tmp_path / "inference.args").write_text(json.dumps(launch_record(launcher.inference_argv(local, settings), "")))
    monkeypatch.setattr("slashcompute.launcher.controller.process_alive", lambda pid: True)
    monkeypatch.setattr("slashcompute.launcher.controller.request_stop",
                        lambda paths: pytest.fail("hosting should not stop the training agent"))
    monkeypatch.setattr("slashcompute.launcher.controller.os.kill",
                        lambda pid, sig: pytest.fail("hosting should not stop the LLM node"))

    snap = launcher.start(settings)

    assert launcher._spawned == []  # type: ignore[attr-defined]
    assert snap.agent_running and snap.inference_running
    saved = launcher.load_settings()
    assert (saved.mode, saved.contribute, saved.training, saved.inference) == ("host", True, True, True)


@pytest.mark.parametrize(("old_token", "new_url", "new_gpu", "new_token"), [
    ("tok", "http://10.0.0.2:8765", 50, "tok"),
    ("tok", "http://10.0.0.1:8765", 75, "tok"),
    ("tok", "http://10.0.0.1:8765", 50, "newtok"),
    ("", "http://10.0.0.1:8765", 50, "tok"),
    ("tok", "http://10.0.0.1:8765", 50, ""),
])
def test_start_restarts_agent_when_effective_arguments_change(
    tmp_path, monkeypatch, old_token, new_url, new_gpu, new_token,
):
    monkeypatch.delenv("SLASHCOMPUTE_SESSION", raising=False)
    http = FakeHTTP({"ok": True, "nodes": 0, "jobs": 0})
    launcher = _launcher(tmp_path, http=http)
    (tmp_path / "agent" / "agent.pid").write_text("77\n")
    old_args = launch_record(launcher.agent_argv("http://10.0.0.1:8765", 50), old_token)
    (tmp_path / "agent.args").write_text(json.dumps(old_args))
    running = {77: True}
    stopped = []

    def alive(pid: int) -> bool:
        return running.get(pid, False)

    def stop(paths):
        stopped.append(paths.read_pid())
        running[77] = False
        paths.clear_pid()
        return True

    monkeypatch.setattr("slashcompute.launcher.controller.process_alive", alive)
    monkeypatch.setattr("slashcompute.launcher.controller.request_stop", stop)
    snap = launcher.start(_signed_in(launcher, new_token, mode="join", url=new_url, gpu_percent=new_gpu))
    spawned = launcher._spawned  # type: ignore[attr-defined]
    assert stopped == [77]
    assert len(spawned) == 1
    assert spawned[0].argv == launcher.agent_argv(new_url, new_gpu)
    assert spawned[0].session == new_token
    assert json.loads((tmp_path / "agent.args").read_text()) == launch_record(spawned[0].argv, new_token)
    assert snap.last_error == ""


@pytest.mark.parametrize(("explicit_token", "expected_token", "should_restart"), [
    ("", "new-environment-token", True),
    ("explicit-token", "explicit-token", False),
])
def test_agent_restart_compares_effective_environment_session(
    tmp_path, monkeypatch, explicit_token, expected_token, should_restart,
):
    launcher = _launcher(tmp_path, http=FakeHTTP({"ok": True}))
    settings = _signed_in(launcher, explicit_token, mode="join", url="http://10.0.0.1:8765")
    monkeypatch.setenv("SLASHCOMPUTE_SESSION", "old-environment-token")
    launcher.start(settings)
    assert launcher._spawned[0].session == (explicit_token or "old-environment-token")
    launcher.paths.pid_file.write_text("77\n")
    monkeypatch.setattr("slashcompute.launcher.controller.process_alive", lambda pid: True)
    stopped = []

    def stop(paths):
        stopped.append(paths.read_pid())
        paths.clear_pid()

    monkeypatch.setattr("slashcompute.launcher.controller.request_stop", stop)
    monkeypatch.setenv("SLASHCOMPUTE_SESSION", "new-environment-token")
    # Ensure rewriting an existing args file also repairs overly broad permissions.
    (tmp_path / "agent.args").chmod(0o644 if should_restart else 0o600)

    launcher.start(settings)

    assert stopped == ([77] if should_restart else [])
    assert len(launcher._spawned) == (2 if should_restart else 1)
    assert launcher._spawned[-1].session == expected_token
    assert "--session-token" not in launcher._spawned[-1].argv   # argv shows in ps
    assert json.loads((tmp_path / "agent.args").read_text()) == launch_record(
        launcher._spawned[-1].argv, expected_token)
    assert stat.S_IMODE((tmp_path / "agent.args").stat().st_mode) == 0o600


def test_start_when_already_up_is_noop(tmp_path, monkeypatch):
    launcher = _launcher(tmp_path, http=FakeHTTP({"ok": True, "nodes": 1, "jobs": 0}))
    (tmp_path / "agent" / "agent.pid").write_text("77\n")
    argv = launcher.agent_argv("http://10.0.0.1:8765", 100)
    (tmp_path / "agent.args").write_text(json.dumps(launch_record(argv, "tok")))
    monkeypatch.setattr("slashcompute.launcher.controller.process_alive", lambda pid: True)
    monkeypatch.setattr("slashcompute.launcher.controller.request_stop",
                        lambda paths: pytest.fail("unchanged agent should not stop"))
    snap = launcher.start(_signed_in(launcher, "tok", mode="join", url="http://10.0.0.1:8765/", gpu_percent=999))
    assert launcher._spawned == []  # type: ignore[attr-defined]
    assert snap.agent_running and snap.last_error == ""


@pytest.mark.parametrize("args_text", [None, "not-json", "argv"])
def test_start_restarts_agent_with_unknown_previous_arguments(tmp_path, monkeypatch, args_text):
    launcher = _launcher(tmp_path, http=FakeHTTP({"ok": True}))
    launcher.paths.pid_file.write_text("77\n")
    settings = _signed_in(launcher, "tok", mode="host")
    # Launchers before agent.args recorded only the bound session; later ones the bare argv, token included.
    (tmp_path / "agent.session").write_text("tok")
    if args_text == "argv":
        args_text = json.dumps(launcher.agent_argv(launcher.proxy_url(settings), 50) + ["--session-token", "tok"])
    if args_text is not None:
        (tmp_path / "agent.args").write_text(args_text)
    monkeypatch.setattr("slashcompute.launcher.controller.process_alive", lambda pid: True)
    monkeypatch.setattr("slashcompute.launcher.controller.request_stop",
                        lambda paths: paths.clear_pid())
    launcher.start(settings)
    assert len(launcher._spawned) == 1
    assert json.loads((tmp_path / "agent.args").read_text()) == launch_record(launcher._spawned[0].argv, "tok")


def test_agent_restart_that_cannot_stop_yet_completes_on_a_later_poll(tmp_path, monkeypatch):
    # The agent drains its step before it exits (minutes, mid-download). Start used to give up
    # after 3 s with "Settings have not been applied" and never restart it: the Mac kept running
    # the old settings, or stopped contributing for good when the drain finished.
    launcher = _launcher(tmp_path, http=FakeHTTP({"ok": True}))
    launcher.paths.pid_file.write_text("77\n")
    old_args = launch_record(launcher.agent_argv("http://10.0.0.1:8765", 50), "tok")
    (tmp_path / "agent.args").write_text(json.dumps(old_args))
    running = {77: True}
    stopped = []
    monkeypatch.setattr("slashcompute.launcher.controller.process_alive", lambda pid: running.get(pid, False))
    monkeypatch.setattr("slashcompute.launcher.controller.request_stop",
                        lambda paths: stopped.append(paths.read_pid()))   # it drains: still up
    _fake_clock(monkeypatch)
    settings = _signed_in(launcher, "tok", mode="join", url="http://10.0.0.2:8765")

    snap = launcher.start(settings)

    assert stopped == [77]
    assert snap.agent_running and snap.agent_pid == 77 and snap.agent_draining
    assert "Restarting the training agent" in snap.last_error
    assert launcher._spawned == []
    assert json.loads((tmp_path / "agent.args").read_text()) == old_args

    # Still draining on the next poll: nothing changes, and it is not asked to stop again.
    snap = launcher.snapshot()
    assert snap.agent_draining and launcher._spawned == [] and stopped == [77]

    # It finished and exited: the next poll starts the new one with the saved settings.
    running[77] = False
    snap = launcher.snapshot()
    assert len(launcher._spawned) == 1
    assert launcher._spawned[0].argv == launcher.agent_argv(settings.url, 50)
    assert launcher._spawned[0].session == "tok"
    assert json.loads((tmp_path / "agent.args").read_text()) == launch_record(launcher._spawned[0].argv, "tok")
    assert snap.last_error == "" and not snap.agent_draining
    assert launcher.snapshot().last_error == "" and len(launcher._spawned) == 1   # done once


def test_sign_in_while_the_agent_drains_restarts_it_once_it_is_gone(tmp_path, monkeypatch):
    launcher = _launcher(tmp_path, http=FakeHTTP({"ok": True, "nodes": 0, "jobs": 0}))
    alive = _running(launcher, monkeypatch)
    launcher.start(LauncherSettings(mode="join", url="http://10.0.0.1:8765"))
    [agent] = launcher._spawned  # type: ignore[attr-defined]
    monkeypatch.setattr("slashcompute.launcher.controller.request_stop", lambda paths: None)  # mid-step
    _fake_clock(monkeypatch)
    launcher.save_settings(launcher.with_session(launcher.load_settings(), "tok"))

    snap = launcher.rebind_session()
    assert snap.agent_running and snap.agent_draining and len(launcher._spawned) == 1
    assert "Restarting the training agent" in snap.last_error

    alive[agent.pid] = False      # drained and exited
    snap = launcher.snapshot()
    rebound = launcher._spawned[1]  # type: ignore[attr-defined]
    assert rebound.argv == agent.argv and rebound.session == "tok"
    assert snap.agent_running and not snap.agent_draining and snap.last_error == ""


def test_a_sign_in_during_a_pending_restart_reaches_the_restarted_agent(tmp_path, monkeypatch):
    launcher = _launcher(tmp_path, http=FakeHTTP({"ok": True}))
    launcher.paths.pid_file.write_text("77\n")
    (tmp_path / "agent.args").write_text(json.dumps(launch_record(launcher.agent_argv("http://10.0.0.1:8765", 50), "")))
    running = {77: True}
    monkeypatch.setattr("slashcompute.launcher.controller.process_alive", lambda pid: running.get(pid, False))
    monkeypatch.setattr("slashcompute.launcher.controller.request_stop", lambda paths: None)
    monkeypatch.delenv("SLASHCOMPUTE_SESSION", raising=False)
    _fake_clock(monkeypatch)
    launcher.start(LauncherSettings(mode="join", url="http://10.0.0.1:8765", gpu_percent=80))   # pending
    launcher.save_settings(launcher.with_session(launcher.load_settings(), "tok"))
    snap = launcher.rebind_session()
    assert snap.agent_draining and launcher._spawned == []  # type: ignore[attr-defined]
    running[77] = False
    launcher.snapshot()
    [agent] = launcher._spawned  # type: ignore[attr-defined]
    assert agent.argv == launcher.agent_argv("http://10.0.0.1:8765", 80) and agent.session == "tok"


def test_stopping_drops_a_pending_restart(tmp_path, monkeypatch):
    launcher = _launcher(tmp_path, http=FakeHTTP({"ok": True}))
    launcher.paths.pid_file.write_text("77\n")
    (tmp_path / "agent.args").write_text(json.dumps(launch_record(launcher.agent_argv("http://10.0.0.1:8765", 50), "")))
    running = {77: True}
    monkeypatch.setattr("slashcompute.launcher.controller.process_alive", lambda pid: running.get(pid, False))
    monkeypatch.setattr("slashcompute.launcher.controller.request_stop", lambda paths: None)
    _fake_clock(monkeypatch)
    launcher.start(LauncherSettings(mode="join", url="http://10.0.0.1:8765", gpu_percent=80))
    assert launcher._pending_agent is not None
    launcher.stop_agent()
    running[77] = False
    launcher.snapshot()
    assert launcher._spawned == [] and launcher._pending_agent is None  # type: ignore[attr-defined]


def test_start_join_requires_url(tmp_path):
    launcher = _launcher(tmp_path)
    with pytest.raises(LauncherError, match="coordinator URL"):
        launcher.start(LauncherSettings(mode="join", url=""))
    assert launcher._spawned == []  # type: ignore[attr-defined]


def test_start_join_without_health_fails(tmp_path):
    launcher = _launcher(tmp_path, http=FakeHTTP(None))
    with pytest.raises(LauncherError, match="No coordinator"):
        launcher.start(LauncherSettings(mode="join", url="http://10.0.0.8:8765"))


def test_unreachable_error_clears_once_the_coordinator_answers(tmp_path):
    http = FakeHTTP(None)
    launcher = _launcher(tmp_path, http=http)
    with pytest.raises(LauncherError, match="No coordinator"):
        launcher.start(LauncherSettings(mode="join", url="http://127.0.0.1:9399"))
    # Still down: the error stays.
    assert launcher.snapshot().last_error == "No coordinator at http://127.0.0.1:9399."

    # Connect saves a fixed address (no start); the next status poll must drop the stale banner.
    launcher.save_settings(LauncherSettings(mode="join", url="http://10.0.0.8:8765"))
    http.health = {"ok": True}
    snap = launcher.snapshot()
    assert snap.coordinator_up and snap.last_error == ""
    assert launcher.last_error == ""


def test_switching_pool_drops_the_old_pools_error(tmp_path):
    launcher = _launcher(tmp_path, port_free_fn=lambda host, port: False)
    with pytest.raises(LauncherError, match="Port 8765 is already in use"):
        launcher.start(LauncherSettings(mode="host", contribute=False))
    launcher.save_settings(LauncherSettings(mode="host", gpu_percent=80))   # same pool: still true
    assert launcher.snapshot().last_error.startswith("Port 8765 is already in use")
    launcher.save_settings(LauncherSettings(mode="join", url="http://10.0.0.8:8765"))
    assert launcher.snapshot().last_error == ""   # nothing answers there yet, but it was never hosting


def test_join_stops_the_pool_hosted_here_and_moves_the_agent(tmp_path, monkeypatch):
    kills: list[tuple[int, int]] = []
    monkeypatch.setattr("slashcompute.launcher.controller.os.kill", lambda pid, sig: kills.append((pid, sig)))
    monkeypatch.setattr("slashcompute.launcher.controller.process_alive", lambda pid: True)
    monkeypatch.setattr("slashcompute.launcher.controller.request_stop", lambda paths: paths.clear_pid())
    launcher = _launcher(tmp_path)
    launcher.poll_health = lambda url: {"ok": True} if launcher._spawned else None  # type: ignore
    launcher.start(LauncherSettings(mode="host", contribute=True))
    coord, agent = launcher._spawned  # type: ignore[attr-defined]
    launcher.paths.pid_file.write_text(f"{agent.pid}\n")

    snap = launcher.start(LauncherSettings(mode="join", url="http://10.0.0.8:8765", training=True))
    assert (coord.pid, signal.SIGTERM) in kills
    moved = launcher._spawned[-1].argv  # type: ignore[attr-defined]
    assert moved[2] == "slashcompute.agent.main" and moved[moved.index("--url") + 1] == "http://10.0.0.8:8765"
    coord.returncode = -signal.SIGTERM   # the stop we asked for is not an error
    assert launcher.snapshot().last_error == snap.last_error == ""


def test_reachable_coordinator_keeps_other_errors(tmp_path):
    launcher = _launcher(tmp_path, http=FakeHTTP({"ok": True}))
    launcher.last_error = "Training agent is still stopping. Settings have not been applied."
    assert launcher.snapshot().last_error == launcher.last_error != ""


def test_start_public_requires_url_and_token(tmp_path):
    launcher = _launcher(tmp_path, http=FakeHTTP({"ok": True}))
    with pytest.raises(LauncherError, match="public coordinator URL"):
        launcher.start(_signed_in(launcher, "tok", mode="public", url=""))
    with pytest.raises(LauncherError, match="Sign in first"):
        launcher.start(LauncherSettings(mode="public", url="https://pool.example.com"))
    assert launcher._spawned == []  # type: ignore[attr-defined]


def test_start_public_does_not_spawn_coordinator(tmp_path, monkeypatch):
    launcher = _launcher(tmp_path, http=FakeHTTP({"ok": True, "nodes": 0, "jobs": 0}))
    monkeypatch.setattr("slashcompute.launcher.controller.process_alive",
                        lambda pid: any(p.pid == pid for p in launcher._spawned))
    snap = launcher.start(_signed_in(launcher, "tok", mode="public", url="https://pool.example.com",
                                     contribute=True))
    kinds = [p.argv[2] for p in launcher._spawned]  # type: ignore[attr-defined]
    assert kinds == ["slashcompute.agent.main"]
    assert snap.coordinator_pid is None
    agent = launcher._spawned[0].argv  # type: ignore[attr-defined]
    assert agent[agent.index("--url") + 1] == "https://pool.example.com"
    assert launcher._spawned[0].session == "tok"  # type: ignore[attr-defined]


def _running(launcher: Launcher, monkeypatch) -> dict:
    """Spawned agents and LLM nodes stay up until stopped through the launcher."""
    alive: dict[int, bool] = {}
    monkeypatch.setattr("slashcompute.launcher.controller.process_alive", lambda pid: alive.get(pid, False))

    def popen(argv, **kw):
        proc = FakeProc(4000 + len(launcher._spawned) + 1, argv, kw.get("env"))  # type: ignore[attr-defined]
        launcher._spawned.append(proc)  # type: ignore[attr-defined]
        alive[proc.pid] = True
        if argv[2] == "slashcompute.agent.main":
            launcher.paths.pid_file.write_text(f"{proc.pid}\n")
        else:
            launcher.inference_pid_path.parent.mkdir(parents=True, exist_ok=True)
            launcher.inference_pid_path.write_text(f"{proc.pid}\n")
        return proc

    def stop_agent(paths):
        alive[paths.read_pid()] = False
        paths.clear_pid()

    launcher._popen = popen
    monkeypatch.setattr("slashcompute.launcher.controller.request_stop", stop_agent)
    monkeypatch.setattr(launcher, "_stop_inference",
                        lambda wait=0.0: alive.update({launcher.read_inference_pid(): False}))
    monkeypatch.delenv("SLASHCOMPUTE_SESSION", raising=False)
    return alive


def test_sign_in_rebinds_the_running_agent_and_llm_node(tmp_path, monkeypatch):
    # Signing in while contributing must reach the processes already running, not the next Start.
    launcher = _launcher(tmp_path, http=FakeHTTP({"ok": True, "nodes": 0, "jobs": 0}))
    _running(launcher, monkeypatch)
    pool = "http://10.0.0.1:8765"
    launcher.start(LauncherSettings(mode="join", url=pool, inference=True))
    first = list(launcher._spawned)  # type: ignore[attr-defined]
    assert [p.session for p in first] == ["", ""]

    launcher.save_settings(launcher.with_session(launcher.load_settings(), "tok"))
    launcher.rebind_session()
    rebound = launcher._spawned[2:]  # type: ignore[attr-defined]
    assert [p.argv for p in rebound] == [p.argv for p in first]   # same pool and settings, new session
    assert [p.session for p in rebound] == ["tok", "tok"]
    assert all("tok" not in arg for p in rebound for arg in p.argv)  # never on the command line
    for name in ("agent.args", "inference.args", "launcher.json"):
        assert stat.S_IMODE((tmp_path / name).stat().st_mode) == 0o600, name

    launcher.rebind_session()   # nothing changed: nothing restarts
    assert len(launcher._spawned) == 4  # type: ignore[attr-defined]

    launcher.save_settings(launcher.with_session(launcher.load_settings(), ""))   # sign out
    launcher.rebind_session()
    assert [p.session for p in launcher._spawned[4:]] == ["", ""]  # type: ignore[attr-defined]


def test_a_session_is_only_ever_sent_to_the_pool_that_issued_it(tmp_path, monkeypatch):
    launcher = _launcher(tmp_path, http=FakeHTTP({"ok": True, "nodes": 0, "jobs": 0}))
    _running(launcher, monkeypatch)
    a, b = "https://a.example.com", "https://b.example.com"
    launcher.save_settings(_signed_in(launcher, "tok-a", mode="public", url=a))
    launcher.start(launcher.load_settings())
    assert launcher._spawned[-1].session == "tok-a"  # type: ignore[attr-defined]

    to_b = launcher.load_settings()
    to_b.url = b
    with pytest.raises(LauncherError, match="Sign in first"):
        launcher.start(to_b)
    to_b.mode, to_b.url = "join", "http://10.0.0.2:8765"
    launcher.start(to_b)
    assert launcher._spawned[-1].session == ""  # type: ignore[attr-defined]

    # Signing out of B leaves A's session; going back to A uses it again.
    launcher.save_settings(launcher.with_session(launcher.load_settings(), ""))
    back = launcher.load_settings()
    back.mode, back.url = "public", a
    launcher.start(back)
    assert launcher._spawned[-1].session == "tok-a"  # type: ignore[attr-defined]


def test_a_session_stored_before_pools_were_tracked_stays_with_its_pool(tmp_path):
    launcher = _launcher(tmp_path)
    (tmp_path / "launcher.json").write_text(json.dumps(
        {"mode": "public", "url": "https://a.example.com", "session_token": "tok"}))
    s = launcher.load_settings()
    assert launcher.session_for(s) == "tok"
    s.url = "https://b.example.com"
    assert launcher.session_for(s) == ""


def test_find_on_lan(tmp_path):
    launcher = _launcher(
        tmp_path,
        discover_fn=lambda timeout=5.0: "http://192.168.0.4:8765/",
    )
    assert launcher.find_on_lan() == "http://192.168.0.4:8765"


def test_stop_signals_agent_and_coordinator(tmp_path, monkeypatch):
    kills: list[tuple[int, int]] = []

    def fake_kill(pid, sig):
        kills.append((pid, sig))

    monkeypatch.setattr("os.kill", fake_kill)
    monkeypatch.setattr("slashcompute.agent.daemon._alive", lambda pid: True)
    launcher = _launcher(tmp_path)
    (tmp_path / "coordinator.pid").write_text("111\n")
    (tmp_path / "agent" / "agent.pid").write_text("222\n")
    launcher.stop()
    assert (111, signal.SIGTERM) in kills
    assert (222, signal.SIGTERM) in kills


def test_snapshot_reads_health_and_agent_status(tmp_path, monkeypatch):
    monkeypatch.setattr("slashcompute.launcher.controller.process_alive", lambda pid: pid == 9)
    launcher = _launcher(
        tmp_path,
        http=FakeHTTP({"ok": True, "nodes": 2, "jobs": 1}),
        lan_ip_fn=lambda: "192.168.9.9",
    )
    launcher.paths.write_status(status="idle", job_id=None)
    launcher.paths.pid_file.write_text("9\n")
    snap = launcher.snapshot(LauncherSettings(mode="host"))
    assert snap.coordinator_up is True
    assert snap.nodes == 2
    assert snap.jobs == 1
    assert snap.agent_running is True
    assert snap.agent_status == "idle"
    assert snap.lan_ip == "192.168.9.9"


def test_stop_agent_leaves_coordinator_running(tmp_path, monkeypatch):
    kills: list[tuple[int, int]] = []
    monkeypatch.setattr("os.kill", lambda pid, sig: kills.append((pid, sig)))
    monkeypatch.setattr("slashcompute.agent.daemon._alive", lambda pid: True)
    launcher = _launcher(tmp_path)
    (tmp_path / "coordinator.pid").write_text("111\n")
    (tmp_path / "agent" / "agent.pid").write_text("222\n")
    launcher.stop_agent()
    assert (222, signal.SIGTERM) in kills
    assert (111, signal.SIGTERM) not in kills  # only liveness probes (signal 0)


def test_my_node_id_never_creates_one(tmp_path):
    launcher = _launcher(tmp_path)
    assert launcher.my_node_id() is None
    assert not launcher.paths.node_id_file.exists()
    launcher.paths.node_id_file.write_text("abc123\n")
    assert launcher.my_node_id() == "abc123"


def test_fetch_pool_down_and_partial(tmp_path):
    assert _launcher(tmp_path, http=FakeHTTP(None)).fetch_pool("http://x:8765").online is False
    # Health answers but list endpoints return a dict: tolerated as empty lists.
    pool = _launcher(tmp_path, http=FakeHTTP({"ok": True})).fetch_pool("http://x:8765")
    assert pool.online is True and (pool.nodes, pool.jobs, pool.ledger) == ([], [], [])


# ------------------------------------------------------------ memory lent

def test_memory_setting_round_trips_clamps_and_reaches_the_agent(tmp_path):
    launcher = _launcher(tmp_path)
    launcher.save_settings(LauncherSettings(memory_gb=6, inference_memory_gb=10))
    s = launcher.load_settings()
    assert (s.memory_gb, s.inference_memory_gb) == (6, 10)
    assert LauncherSettings(memory_gb="lots").clamp().memory_gb == 0
    assert LauncherSettings(memory_gb=-3).clamp().memory_gb == 0
    assert "--max-memory-gb" not in launcher.agent_argv("http://10.0.0.1:8765", 50)       # 0 = automatic
    assert launcher.agent_argv("http://10.0.0.1:8765", 50, 6)[-2:] == ["--max-memory-gb", "6"]


def test_changing_memory_restarts_the_running_agent(tmp_path, monkeypatch):
    launcher = _launcher(tmp_path, http=FakeHTTP({"ok": True, "nodes": 0, "jobs": 0}))
    (tmp_path / "agent" / "agent.pid").write_text("77\n")
    (tmp_path / "agent.args").write_text(json.dumps(launch_record(launcher.agent_argv("http://10.0.0.1:8765", 50), "")))
    running = {77: True}
    stopped = []

    def stop(paths):
        stopped.append(paths.read_pid())
        running[77] = False
        paths.clear_pid()
        return True

    monkeypatch.setattr("slashcompute.launcher.controller.process_alive", lambda pid: running.get(pid, False))
    monkeypatch.setattr("slashcompute.launcher.controller.request_stop", stop)
    launcher.start(LauncherSettings(mode="join", url="http://10.0.0.1:8765", memory_gb=6))
    assert stopped == [77]
    assert launcher._spawned[0].argv == launcher.agent_argv("http://10.0.0.1:8765", 50, 6)  # type: ignore


def test_status_reports_this_macs_memory(tmp_path):
    launcher = _launcher(tmp_path)
    launcher._memory = lambda: (16 << 30, 5 << 30)
    snap = launcher.snapshot()
    assert (snap.memory_total_bytes, snap.memory_available_bytes) == (16 << 30, 5 << 30)


# ------------------------------------------------------------ one launcher, many threads

def test_settings_are_rewritten_atomically(tmp_path):
    # Every status poll loads launcher.json on its own thread. A truncate-then-write let a poll
    # read an empty file (defaults, no session) which the next POST /api/settings saved back.
    launcher = _launcher(tmp_path)
    a = launcher.with_session(LauncherSettings(mode="join", url="http://10.0.0.1:8765"), "tok")
    b = replace(a, gpu_percent=75)
    launcher.save_settings(a)
    path = tmp_path / "launcher.json"
    seen: list = []
    stop = threading.Event()

    def read_the_file():   # the file itself, not load_settings: the lock is not what protects this
        while not stop.is_set():
            try:
                seen.append(json.loads(path.read_text()).get("gpu_percent"))
            except (OSError, ValueError) as e:
                seen.append(repr(e))

    reader = threading.Thread(target=read_the_file)
    reader.start()
    try:
        for i in range(300):
            write_private(path, json.dumps(asdict(b if i % 2 else a), indent=2) + "\n")
    finally:
        stop.set()
        reader.join(5)
    assert seen and all(v in (50, 75) for v in seen), [v for v in seen if v not in (50, 75)][:3]
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".launcher.json")] == []
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert launcher.load_settings().session_token == "tok"


def test_load_settings_never_waits_for_the_lock(tmp_path):
    # start()/set_inference()/rebind_session() hold the lock for up to ~25 s while the shell's
    # async handlers load the settings on the event loop: a locked read froze the whole shell.
    launcher = _launcher(tmp_path)
    launcher.save_settings(LauncherSettings(mode="join", url="http://10.0.0.1:8765", gpu_percent=75))
    loaded: list = []
    saved = threading.Event()

    def read():
        loaded.append(launcher.load_settings())

    def write():
        launcher.save_settings(LauncherSettings(mode="join", url="http://10.0.0.1:8765", gpu_percent=80))
        saved.set()

    with launcher._lock:                       # a Start in progress on another thread
        reader = threading.Thread(target=read)
        reader.start()
        reader.join(2)
        assert not reader.is_alive() and loaded[0].gpu_percent == 75
        writer = threading.Thread(target=write)
        writer.start()
        assert not saved.wait(0.3)             # writes still queue behind the state changes
    assert saved.wait(2)
    assert launcher.load_settings().gpu_percent == 80


def test_concurrent_starts_spawn_one_coordinator(tmp_path, monkeypatch):
    # Two Start clicks reach the shared launcher on two threads: both used to pass the
    # "no coordinator.pid yet" check while the first coordinator was still being spawned.
    launcher = _launcher(tmp_path)
    monkeypatch.setattr("slashcompute.launcher.controller.process_alive",
                        lambda pid: any(p.pid == pid for p in launcher._spawned))  # type: ignore[attr-defined]
    launcher.poll_health = lambda url: {"ok": True} if launcher._spawned else None  # type: ignore
    spawn = launcher._popen
    entered = threading.Event()

    def slow_popen(argv, **kw):
        entered.set()
        time.sleep(0.3)        # a second Start arrives while this one is still spawning
        return spawn(argv, **kw)

    launcher._popen = slow_popen
    errors: list = []

    def run():
        try:
            launcher.start(LauncherSettings(mode="host", contribute=False))
        except Exception as e:   # noqa: BLE001
            errors.append(e)

    first = threading.Thread(target=run)
    first.start()
    assert entered.wait(5)
    second = threading.Thread(target=run)
    second.start()
    first.join(10)
    second.join(10)
    assert errors == []
    assert [p.argv[2] for p in launcher._spawned] == ["slashcompute.coordinator.main"]  # type: ignore
    assert launcher.snapshot().coordinator_pid == launcher._spawned[0].pid  # type: ignore[attr-defined]


# ------------------------------------------------------------ pid files that name a stranger

def test_a_pid_file_naming_another_program_is_not_our_agent_or_node(tmp_path, monkeypatch):
    # After a crash or a reboot the pid in agent.pid / node.pid can be any other program: the
    # UI said "Contributing" and Stop sent it SIGTERM.
    kills: list[tuple[int, int]] = []
    monkeypatch.setattr("os.kill", lambda pid, sig: kills.append((pid, sig)))
    monkeypatch.setattr("slashcompute.launcher.controller.process_alive", lambda pid: True)
    monkeypatch.setattr("slashcompute.agent.daemon._alive", lambda pid: True)
    launcher = _launcher(tmp_path)
    cmdlines = {
        77: ["/usr/bin/python3", "some_other_app.py"],
        88: ["/opt/venv/bin/python", "-m", "slashcompute.inference.node", "start", "--home", "/elsewhere"],
    }
    monkeypatch.setattr("slashcompute.launcher.controller.process_cmdline", lambda pid: cmdlines.get(pid))

    def stale_pid_files():
        launcher.paths.pid_file.write_text("77\n")
        launcher.inference_pid_path.parent.mkdir(parents=True, exist_ok=True)
        launcher.inference_pid_path.write_text("88\n")

    stale_pid_files()
    snap = launcher.snapshot()
    assert not snap.agent_running and snap.agent_pid is None and not launcher.paths.pid_file.exists()
    assert not snap.inference_running and not launcher.inference_pid_path.exists()
    assert not launcher._agent_running() and launcher.read_inference_pid() is None

    stale_pid_files()
    launcher.stop()
    assert [k for k in kills if k[1] == signal.SIGTERM] == []

    # Our own agent and LLM node (the module and this home on their command lines) are ours.
    cmdlines[77] = launcher.agent_argv("http://10.0.0.1:8765", 50)
    cmdlines[88] = launcher.inference_argv("http://10.0.0.1:8765", LauncherSettings())
    stale_pid_files()
    snap = launcher.snapshot()
    assert snap.agent_running and snap.agent_pid == 77 and snap.inference_running and snap.inference_pid == 88
    assert launcher.paths.pid_file.exists() and launcher.inference_pid_path.exists()
    launcher.stop()
    assert {k[0] for k in kills if k[1] == signal.SIGTERM} == {77, 88}

    # One started by hand without --home runs in the default home: ours only when that is this home.
    cmdlines[77] = ["/usr/bin/python3", "-m", "slashcompute.agent.main", "start", "--url", "http://10.0.0.1:8765"]
    stale_pid_files()
    assert not launcher.snapshot().agent_running
    monkeypatch.setenv("SLASHCOMPUTE_HOME", str(tmp_path))
    stale_pid_files()
    assert launcher.snapshot().agent_running


def test_a_coordinator_pid_file_naming_another_program_is_not_ours_and_never_signalled(tmp_path, monkeypatch):
    # coordinator.pid survives a crash or a reboot; the pid in it can then be any other program,
    # which the UI called our pool and Stop (or joining another pool) SIGTERMed.
    kills: list[tuple[int, int]] = []
    monkeypatch.setattr("slashcompute.launcher.controller.os.kill", lambda pid, sig: kills.append((pid, sig)))
    monkeypatch.setattr("slashcompute.launcher.controller.process_alive", lambda pid: True)
    monkeypatch.setattr("slashcompute.launcher.controller.started_after", lambda pid, when: False)
    launcher = _launcher(tmp_path, http=FakeHTTP({"ok": True}))
    cmdlines = {111: ["/usr/bin/python3", "some_other_app.py"]}
    monkeypatch.setattr("slashcompute.launcher.controller.process_cmdline", lambda pid: cmdlines.get(pid))

    (tmp_path / "coordinator.pid").write_text("111\n")
    assert launcher.read_coordinator_pid() is None and not (tmp_path / "coordinator.pid").exists()
    (tmp_path / "coordinator.pid").write_text("111\n")
    assert launcher.snapshot().coordinator_pid is None
    (tmp_path / "coordinator.pid").write_text("111\n")
    launcher.stop()
    (tmp_path / "coordinator.pid").write_text("111\n")
    launcher.start(LauncherSettings(mode="join", url="http://10.0.0.8:8765", training=False))
    assert [k for k in kills if k[1] == signal.SIGTERM] == []

    # A coordinator serving another home is not ours either; ours (this home) is.
    cmdlines[111] = launcher.coordinator_argv()[:-1] + ["/elsewhere"]
    (tmp_path / "coordinator.pid").write_text("111\n")
    assert launcher.read_coordinator_pid() is None
    cmdlines[111] = launcher.coordinator_argv()
    (tmp_path / "coordinator.pid").write_text("111\n")
    assert launcher.read_coordinator_pid() == 111 and launcher.snapshot().coordinator_pid == 111
    launcher.stop()
    assert (111, signal.SIGTERM) in kills


# ------------------------------------------------------------ draining and the running share

def test_status_reports_a_draining_agent_and_its_share(tmp_path, monkeypatch):
    monkeypatch.setattr("slashcompute.launcher.controller.process_alive", lambda pid: pid == 9)
    launcher = _launcher(tmp_path, http=FakeHTTP({"ok": True}))
    launcher.paths.pid_file.write_text("9\n")
    launcher.paths.write_status(status="running", draining=False, gpu_percent=35, job_id="j1")
    snap = launcher.snapshot()
    assert snap.agent_running and not snap.agent_draining and snap.agent_gpu_percent == 35

    launcher.paths.write_status(draining=True)
    assert launcher.snapshot().agent_draining

    # Asked to stop but still finishing its step (status.json not updated yet): draining too.
    launcher.paths.write_status(draining=False, gpu_percent="lots")
    monkeypatch.setattr("slashcompute.launcher.controller.request_stop", lambda paths: None)
    snap = launcher.stop_agent()
    assert snap.agent_running and snap.agent_draining and snap.agent_gpu_percent is None

    monkeypatch.setattr("slashcompute.launcher.controller.process_alive", lambda pid: False)
    snap = launcher.snapshot()
    assert not snap.agent_running and not snap.agent_draining and snap.agent_gpu_percent is None


def test_a_hung_agent_is_force_stopped_after_the_grace_period(tmp_path, monkeypatch):
    # daemon.shutdown drains the step for grace_period_s and exits; one that is still alive well
    # past that is hung, and the shell said "Stopping…" (or "Restarting…") for ever.
    launcher = _launcher(tmp_path, http=FakeHTTP({"ok": True}))
    launcher.paths.pid_file.write_text("77\n")
    (tmp_path / "agent.args").write_text(json.dumps(launch_record(launcher.agent_argv("http://10.0.0.1:8765", 50), "")))
    running = {77: True}
    kills: list[tuple[int, int]] = []
    monkeypatch.setattr("slashcompute.launcher.controller.process_alive", lambda pid: running.get(pid, False))
    monkeypatch.setattr("slashcompute.launcher.controller.os.kill", lambda pid, sig: kills.append((pid, sig)))
    monkeypatch.setattr("slashcompute.launcher.controller.request_stop", lambda paths: None)   # SIGTERM ignored
    clock = _fake_clock(monkeypatch)
    limit = launcher.cfg.grace_period_s + AGENT_STOP_MARGIN_S

    snap = launcher.start(LauncherSettings(mode="join", url="http://10.0.0.1:8765", gpu_percent=80))
    assert snap.agent_draining and launcher._pending_agent is not None and kills == []

    clock["t"] += limit - 5                      # still within what a drain may take
    snap = launcher.snapshot()
    assert snap.agent_draining and kills == [] and launcher._spawned == []

    clock["t"] += 10
    snap = launcher.snapshot()
    assert kills == [(77, signal.SIGKILL)]
    assert "force-stopped" in snap.last_error and f"{limit:.0f} s" in snap.last_error
    assert not snap.agent_draining and launcher._pending_agent is None

    running[77] = False                          # killed: the restart it waited on is not attempted
    snap = launcher.snapshot()
    assert not snap.agent_running and launcher._spawned == [] and "force-stopped" in snap.last_error
    assert kills == [(77, signal.SIGKILL)]       # once


# ------------------------------------------------------------ our own pool is not one to join

def test_discovery_skips_the_coordinator_hosted_here(tmp_path):
    ours = _launcher(tmp_path, discover_fn=lambda timeout=5.0: "http://192.168.1.20:8765",
                     lan_ip_fn=lambda: "192.168.1.20")
    assert ours.find_on_lan() is None
    assert ours.is_own_url("http://127.0.0.1:8765") and ours.is_own_url("192.168.1.20")
    assert not ours.is_own_url("http://192.168.1.21:8765") and not ours.is_own_url("http://192.168.1.20:9000")
    other = _launcher(tmp_path, discover_fn=lambda timeout=5.0: "http://192.168.1.21:8765/")
    assert other.find_on_lan() == "http://192.168.1.21:8765"


def test_find_on_lan_while_hosting_listens_past_our_own_advertisement(tmp_path, monkeypatch):
    # Discovery used to stop at the first advertisement, ours while hosting, so Find on LAN never
    # found another pool. The launcher now asks it to skip ours and take the next.
    ads = ["http://192.168.1.20:8765", "http://192.168.1.21:8765"]
    asked: list = []

    def fake_discover(timeout, exclude=None):
        asked.append((timeout, exclude))
        return next((url for url in ads if exclude is None or not exclude(url)), None)

    monkeypatch.setattr("slashcompute.launcher.controller.discover", fake_discover)
    launcher = _launcher(tmp_path, discover_fn=None, lan_ip_fn=lambda: "192.168.1.20")
    assert launcher.find_on_lan(timeout=2.0) == "http://192.168.1.21:8765"
    assert asked[0][0] == 2.0 and asked[0][1]("http://192.168.1.20:8765") and not asked[0][1]("http://192.168.1.21:8765")
    del ads[1]
    assert launcher.find_on_lan() is None                      # only ours is advertised


@pytest.mark.parametrize("own_url", [
    "http://192.168.1.20:8765",     # the LAN ip
    "http://127.0.0.1:8765",
    "http://0.0.0.0:8765",          # what the coordinator is bound to
    "http://test-mac.local:8765",   # the Bonjour name another Mac would have used for us
    "http://10.8.0.5:8765",         # a VPN interface
    "http://[fd00::20]:8765",       # an IPv6 interface address
])
def test_joining_our_own_pool_does_not_stop_hosting_it(tmp_path, monkeypatch, own_url):
    # From the default home this is the user's live :8765 pool: start(join) on any address of it
    # used to SIGTERM the coordinator just joined when the address was not the LAN ip or loopback.
    _this_mac(monkeypatch, interfaces={"en0": ["192.168.1.20", "fd00::20"], "utun3": ["10.8.0.5"]})
    kills: list[tuple[int, int]] = []
    monkeypatch.setattr("slashcompute.launcher.controller.os.kill", lambda pid, sig: kills.append((pid, sig)))
    monkeypatch.setattr("slashcompute.launcher.controller.process_alive", lambda pid: True)
    launcher = _launcher(tmp_path)
    launcher.poll_health = lambda url: {"ok": True} if launcher._spawned else None  # type: ignore
    launcher.start(LauncherSettings(mode="host", contribute=False))
    [coord] = launcher._spawned  # type: ignore[attr-defined]

    snap = launcher.start(LauncherSettings(mode="join", url=own_url, training=False))
    assert (coord.pid, signal.SIGTERM) not in kills
    assert snap.coordinator_pid == coord.pid and snap.last_error == ""


def test_is_own_url_knows_every_address_this_mac_answers_on(tmp_path, monkeypatch):
    _this_mac(monkeypatch, interfaces={"lo0": ["127.0.0.1", "::1", "fe80::1%lo0"],
                                       "en0": ["192.168.1.20", "fe80::8a1:5af6%en0"],
                                       "utun4": ["100.68.31.99"]},
              hostname="My-Mac.local",
              names={"my-mac.local": ["192.168.1.20", "fe80::8a1:5af6"],
                     "pool.lan": ["192.168.1.30"],
                     "alias.lan": ["10.9.9.9", "100.68.31.99"]})   # a DNS name for our VPN address
    launcher = _launcher(tmp_path, lan_ip_fn=lambda: "192.168.1.20")
    for url in ("http://my-mac.local:8765", "http://MY-MAC.local:8765", "my-mac", "My-Mac.local",
                "http://0.0.0.0:8765", "http://[::]:8765", "http://[fe80::8a1:5af6%en0]:8765",
                "http://100.68.31.99:8765", "http://alias.lan:8765", "http://[0:0:0:0:0:0:0:1]:8765"):
        assert launcher.is_own_url(url), url
    for url in ("http://pool.lan:8765", "http://192.168.1.30:8765", "http://my-mac.local:9000",
                "http://100.68.31.98:8765", "https://pool.example.com", "http://nowhere.lan:8765", ""):
        assert not launcher.is_own_url(url), url


def test_is_own_url_gives_up_on_a_name_that_is_slow_to_resolve(tmp_path, monkeypatch):
    # getaddrinfo has no timeout of its own: a dead DNS must not hang Start or Find on LAN.
    def hangs(host, port, *a, **kw):
        time.sleep(0.5)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.20", 0))]

    monkeypatch.setattr("slashcompute.launcher.controller.socket.getaddrinfo", hangs)
    monkeypatch.setattr("slashcompute.launcher.controller.RESOLVE_TIMEOUT_S", 0.05)
    launcher = _launcher(tmp_path)
    began = time.monotonic()
    assert not launcher.is_own_url("http://slow.lan:8765")
    assert time.monotonic() - began < 0.4


# ------------------------------------------------------------ odds and ends

def test_malformed_health_counts_are_unhealthy_not_a_crash(tmp_path):
    launcher = _launcher(tmp_path, http=FakeHTTP({"ok": True, "nodes": "many", "jobs": [1]}))
    launcher.save_settings(LauncherSettings(mode="join", url="http://10.0.0.8:8765"))
    snap = launcher.snapshot()
    assert not snap.coordinator_up and (snap.nodes, snap.jobs) == (0, 0)
    launcher.poll_health = lambda url: {"ok": True, "nodes": None, "jobs": "2", "inference_nodes": {}}  # type: ignore
    snap = launcher.snapshot()
    assert snap.coordinator_up and (snap.nodes, snap.jobs, snap.inference_nodes) == (0, 2, 0)


def test_the_spawned_agent_is_kept_and_reaped_when_it_exits(tmp_path):
    launcher = _launcher(tmp_path, http=FakeHTTP({"ok": True}))
    launcher.start(LauncherSettings(mode="join", url="http://10.0.0.1:8765"))
    [agent] = launcher._spawned  # type: ignore[attr-defined]
    assert launcher._agent_proc is agent
    agent.returncode = 0
    launcher.snapshot()
    assert launcher._agent_proc is None


def test_a_just_spawned_agent_counts_as_running_before_it_writes_its_pid(tmp_path, monkeypatch):
    # The agent writes agent.pid only once its imports are done (seconds): a second Start in that
    # window spawned a second agent, and Stop or a sign-in could not reach the first.
    monkeypatch.delenv("SLASHCOMPUTE_SESSION", raising=False)
    launcher = _launcher(tmp_path, http=FakeHTTP({"ok": True}))
    launcher.paths.write_status(status="draining", draining=True)   # the previous run's last words
    settings = LauncherSettings(mode="join", url="http://10.0.0.1:8765")
    snap = launcher.start(settings)
    [agent] = launcher._spawned  # type: ignore[attr-defined]
    assert snap.agent_running and snap.agent_pid == agent.pid and not snap.agent_draining
    assert snap.agent_status == ""                                   # not the previous run's

    launcher.start(settings)
    assert launcher._spawned == [agent]                              # no second agent

    launcher.save_settings(launcher.with_session(settings, "tok"))  # signed in: restart it with the session
    launcher.rebind_session()
    assert agent.signals == [signal.SIGTERM] and launcher._pending_agent is None
    assert [p.session for p in launcher._spawned[1:]] == ["tok"]
    rebound = launcher._spawned[1]
    agent.returncode = -signal.SIGTERM

    snap = launcher.stop()
    assert rebound.signals == [signal.SIGTERM] and snap.agent_pid is None
    assert len(launcher._spawned) == 2 and launcher._agent_proc is None
