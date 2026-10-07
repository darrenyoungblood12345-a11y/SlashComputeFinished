import logging
import os
import stat
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from slashcompute.launcher.main import (
    MIN_H, MIN_W, SCREEN_MARGIN_H, SCREEN_MARGIN_W, SHELL_GENERATION, WINDOW_H, WINDOW_W,
    _stop_listener, confirm_close_when_busy, ensure_shell, main, shell_busy, ui_ready,
    watch_busy, window_size,
)
from slashcompute.web.server import asset_version


ROOT = Path(__file__).resolve().parents[1]
INSTALL = ROOT / "scripts" / "macos" / "install_app.sh"
URL = "http://127.0.0.1:8766"


class FakeResponse:
    def __init__(self, payload=None, status_code: int = 200, text: str = "<h1>COMPUTE</h1>") -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._payload


def shell_answering(info: dict):
    """An httpx.get that serves the /compute page and ``info`` as /api/shell."""
    def get(url, timeout=None):
        return FakeResponse(info if url.endswith("/api/shell") else {})
    return get


def test_ui_ready_accepts_compute_page(monkeypatch):
    class R:
        status_code = 200
        text = "<title>/compute</title>\n<h1>COMPUTE</h1>"

    monkeypatch.setattr("slashcompute.launcher.main.httpx.get", lambda *a, **k: R())
    assert ui_ready("http://127.0.0.1:8766") is True


def test_ui_ready_rejects_foreign_port(monkeypatch):
    class R:
        status_code = 200
        text = "ok"

    monkeypatch.setattr("slashcompute.launcher.main.httpx.get", lambda *a, **k: R())
    assert ui_ready("http://127.0.0.1:8766") is False


def test_ensure_shell_reuses_only_an_identical_shell(monkeypatch):
    info = {"ok": True, "generation": SHELL_GENERATION, "assets": asset_version(), "proxy": "coord"}
    monkeypatch.setattr("slashcompute.launcher.main.httpx.get", shell_answering(info))
    monkeypatch.setattr("slashcompute.launcher.main._stop_listener",
                        lambda port: pytest.fail("stopped the shell we should reuse"))
    monkeypatch.setattr("slashcompute.launcher.main._serve", lambda: pytest.fail("started a second shell"))
    assert ensure_shell() == URL


@pytest.mark.parametrize("info", [
    {"ok": True, "generation": SHELL_GENERATION + 1, "assets": asset_version()},   # a dev shell ahead of us
    {"ok": True, "generation": SHELL_GENERATION - 1, "assets": asset_version()},
    {"ok": True, "generation": SHELL_GENERATION, "assets": "0ther8uild00"},       # same shell, other app.js
    {"ok": True, "generation": SHELL_GENERATION},                                  # too old to say
    {"ok": True, "generation": "seven", "assets": asset_version()},
])
def test_ensure_shell_replaces_a_shell_that_is_not_ours(monkeypatch, info):
    stopped: list[int] = []
    served = threading.Event()
    monkeypatch.setattr("slashcompute.launcher.main.httpx.get", shell_answering(info))
    monkeypatch.setattr("slashcompute.launcher.main._stop_listener", stopped.append)
    monkeypatch.setattr("slashcompute.launcher.main._port_open", lambda host, port: False)
    monkeypatch.setattr("slashcompute.launcher.main._serve", served.set)
    monkeypatch.setattr("slashcompute.launcher.main.time.sleep", lambda s: None)
    assert ensure_shell() == URL
    assert stopped == [8766]
    assert served.wait(2), "our own shell was not started"


def test_ensure_shell_refuses_a_port_held_by_something_else(monkeypatch):
    monkeypatch.setattr("slashcompute.launcher.main.httpx.get",
                        lambda url, timeout=None: FakeResponse({}, text="ok"))
    monkeypatch.setattr("slashcompute.launcher.main._port_open", lambda host, port: True)
    monkeypatch.setattr("slashcompute.launcher.main._stop_listener", lambda port: None)
    monkeypatch.setattr("slashcompute.launcher.main._serve", lambda: pytest.fail("bound a busy port"))
    monkeypatch.setattr("slashcompute.launcher.main.time.sleep", lambda s: None)
    with pytest.raises(SystemExit, match="Port 8766 is in use"):
        ensure_shell()


OLD_SHELL = ["/Applications/compute.app/Contents/Resources/python/bin/python3", "-m",
             "slashcompute.launcher.main"]
OTHER_SERVER = ["uvicorn", "myapp:app", "--port", "8766"]


class FakeProc:
    """A psutil.Process stand-in: what process_iter yields and what Process(pid) returns.

    ``listening`` are the ports it serves, ``connected`` those it is a client of; with
    ``sockets_visible=False`` its sockets are off limits, as another user's are on macOS."""

    def __init__(self, pid: int, cmdline: list[str], listening=(), connected=(),
                 sockets_visible: bool = True) -> None:
        self.pid = pid
        self._cmdline = cmdline
        self._listening = listening
        self._connected = connected
        self._sockets_visible = sockets_visible
        self.terminated = False

    def ppid(self) -> int:
        return 1

    def cmdline(self) -> list[str]:
        return self._cmdline

    def net_connections(self, kind="inet"):
        import psutil

        if not self._sockets_visible:
            raise psutil.AccessDenied(pid=self.pid)

        def conn(port: int, status: str) -> SimpleNamespace:
            return SimpleNamespace(laddr=SimpleNamespace(ip="127.0.0.1", port=port),
                                   status=status, pid=self.pid)

        return [*(conn(p, "LISTEN") for p in self._listening),
                *(conn(p, "ESTABLISHED") for p in self._connected)]

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout=None) -> int:
        return 0


def non_root_psutil(monkeypatch, procs: list[FakeProc]) -> None:
    """psutil as a non-root user sees it on macOS: the system-wide socket table is denied, each
    process answers for its own sockets, ``Process(pid)`` knows only the processes listed."""
    import psutil

    by_pid = {p.pid: p for p in procs}

    def denied(kind="inet"):
        raise psutil.AccessDenied(pid=os.getpid())

    def process(pid: int) -> FakeProc:
        if pid not in by_pid:
            raise psutil.NoSuchProcess(pid)
        return by_pid[pid]

    monkeypatch.setattr(psutil, "net_connections", denied)
    monkeypatch.setattr(psutil, "process_iter", lambda attrs=None, ad_value=None: iter(procs))
    monkeypatch.setattr(psutil, "Process", process)


def no_lsof(monkeypatch) -> None:
    def missing(cmd, **kwargs):
        raise FileNotFoundError(cmd[0])
    monkeypatch.setattr("slashcompute.launcher.main.subprocess.run", missing)


def test_stop_listener_quits_only_the_old_shell_without_the_system_socket_table(monkeypatch):
    other = FakeProc(11, OTHER_SERVER, listening=[8000])
    old_shell = FakeProc(12, OLD_SHELL, listening=[8766])
    client = FakeProc(13, ["python", "-m", "slashcompute.cli", "status"], connected=[8766])
    hidden = FakeProc(14, ["launchd"], sockets_visible=False)
    non_root_psutil(monkeypatch, [other, hidden, old_shell, client])
    no_lsof(monkeypatch)   # psutil found the listener: lsof is not needed
    _stop_listener(8766)
    assert [p.pid for p in (other, old_shell, client, hidden) if p.terminated] == [12]


def test_stop_listener_leaves_an_unrelated_server_alone(monkeypatch):
    other = FakeProc(11, OTHER_SERVER, listening=[8766])
    non_root_psutil(monkeypatch, [other])
    no_lsof(monkeypatch)
    _stop_listener(8766)
    assert other.terminated is False


def test_stop_listener_asks_lsof_when_psutil_cannot_see_the_listener(monkeypatch):
    old_shell = FakeProc(12, OLD_SHELL, listening=[8766], sockets_visible=False)
    other = FakeProc(11, OTHER_SERVER, listening=[8000])
    non_root_psutil(monkeypatch, [other, old_shell])
    calls: list[tuple[list[str], dict]] = []

    def lsof(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="12\n12\n", stderr="")

    monkeypatch.setattr("slashcompute.launcher.main.subprocess.run", lsof)
    _stop_listener(8766)
    assert old_shell.terminated is True and other.terminated is False
    [(cmd, kwargs)] = calls
    assert cmd[0] == "lsof" and {"-iTCP:8766", "-sTCP:LISTEN", "-t"} <= set(cmd)
    assert kwargs["timeout"] == 3


@pytest.mark.parametrize("lsof_outcome", ["missing", "timeout", "nothing", "unknown pid"])
def test_stop_listener_gives_up_quietly_when_nothing_of_ours_is_found(monkeypatch, lsof_outcome):
    hidden = FakeProc(99, ["someone-elses-server"], listening=[8766], sockets_visible=False)
    non_root_psutil(monkeypatch, [hidden])

    def lsof(cmd, **kwargs):
        if lsof_outcome == "missing":
            raise FileNotFoundError("lsof")
        if lsof_outcome == "timeout":
            raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
        if lsof_outcome == "nothing":
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="4242\n", stderr="")   # psutil: NoSuchProcess

    monkeypatch.setattr("slashcompute.launcher.main.subprocess.run", lsof)
    _stop_listener(8766)   # no exception: ensure_shell goes on to report the busy port
    assert hidden.terminated is False


def test_window_size_fits_small_screens_but_never_below_the_layout_minimum():
    assert window_size(None) == (WINDOW_W, WINDOW_H)
    assert window_size((2560, 1440)) == (WINDOW_W, WINDOW_H)
    assert window_size((1280, 800)) == (1280 - SCREEN_MARGIN_W, 800 - SCREEN_MARGIN_H)
    assert window_size((800, 600)) == (MIN_W, MIN_H)
    assert MIN_W >= 960 and MIN_H >= 640


@pytest.mark.parametrize("status, busy", [
    ({"agent_running": True}, True),
    ({"inference_running": True}, True),
    ({"coordinator_pid": 4242}, True),
    # joined to someone else's pool, nothing running here
    ({"agent_running": False, "inference_running": False, "coordinator_pid": None,
      "coordinator_up": True}, False),
])
def test_shell_busy_follows_api_status(monkeypatch, status, busy):
    monkeypatch.setattr("slashcompute.launcher.main.httpx.get",
                        lambda url, timeout=None: FakeResponse(status))
    assert shell_busy(URL) is busy


@pytest.mark.parametrize("failure", [httpx.ConnectError("refused"), httpx.ReadTimeout("slow")])
def test_shell_busy_is_unknown_not_idle_when_the_shell_does_not_answer(monkeypatch, failure):
    def gone(url, timeout=None):
        raise failure

    monkeypatch.setattr("slashcompute.launcher.main.httpx.get", gone)
    assert shell_busy(URL) is None


def wait_for(condition, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting for the busy watch"
        time.sleep(0.005)


def test_watch_busy_polls_in_the_background_and_keeps_the_last_answer(monkeypatch):
    answer = {"status": {"agent_running": True}}
    calls: list[tuple[threading.Thread, str, float]] = []

    def status(url, timeout=None):
        calls.append((threading.current_thread(), url, timeout))
        if isinstance(answer["status"], Exception):
            raise answer["status"]
        return FakeResponse(answer["status"])

    monkeypatch.setattr("slashcompute.launcher.main.httpx.get", status)
    stop = threading.Event()
    is_busy = watch_busy(URL, interval=0.01, stop=stop)
    try:
        wait_for(lambda: is_busy() is True)
        answer["status"] = httpx.ReadTimeout("slow")   # a slow shell: keep "busy", do not skip the prompt
        time.sleep(0.05)
        assert is_busy() is True
        answer["status"] = {"agent_running": False, "inference_running": False, "coordinator_pid": None}
        wait_for(lambda: is_busy() is False)
    finally:
        stop.set()
    assert calls, "the shell was never asked"
    assert all(t is not threading.main_thread() for t, _, _ in calls), "asked on the GUI thread"
    assert all(url == f"{URL}/api/status" and timeout == 2.0 for _, url, timeout in calls)


@pytest.mark.parametrize("busy", [True, False])
def test_closing_asks_only_while_the_pool_works_here(busy):
    from webview.event import Event   # pywebview's dispatcher, without a window or a display

    window = SimpleNamespace(confirm_close=False)
    closing = Event(window, should_lock=True)
    closing += confirm_close_when_busy(lambda: busy)
    cancelled = closing.set()
    assert cancelled is False          # the close goes on, to pywebview's confirm_close prompt
    assert window.confirm_close is busy


def run_main_expecting_failure(monkeypatch, tmp_path) -> list[str]:
    """``main()`` with the window and the native alert stubbed; returns the alerts shown."""
    alerts: list[str] = []
    monkeypatch.setattr("slashcompute.launcher.main.launcher_home", lambda: tmp_path)
    monkeypatch.setattr("slashcompute.launcher.main.open_window",
                        lambda url: pytest.fail("opened a window without a shell"))
    monkeypatch.setattr("slashcompute.launcher.main.show_alert", alerts.append)
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        with pytest.raises(SystemExit) as exc:
            main()
    finally:
        for h in [h for h in root.handlers if h not in before]:
            root.removeHandler(h)
            h.close()
    assert exc.value.code == 1
    return alerts


def test_main_logs_and_alerts_when_the_shell_cannot_start(monkeypatch, tmp_path):
    def no_shell() -> str:
        raise SystemExit("UI did not start at http://127.0.0.1:8766")

    monkeypatch.setattr("slashcompute.launcher.main.ensure_shell", no_shell)
    alerts = run_main_expecting_failure(monkeypatch, tmp_path)
    assert alerts == ["UI did not start at http://127.0.0.1:8766"]
    assert "UI did not start" in (tmp_path / "logs" / "shell.log").read_text()


def test_main_alerts_when_another_program_holds_the_port(monkeypatch, tmp_path):
    """End to end as a non-root user: the system socket table is denied, the only listener on
    :8766 is somebody else's server, so it is left running and the alert says who to quit."""
    other = FakeProc(11, OTHER_SERVER, listening=[8766])
    non_root_psutil(monkeypatch, [other])
    no_lsof(monkeypatch)
    monkeypatch.setattr("slashcompute.launcher.main.httpx.get",
                        lambda url, timeout=None: FakeResponse({}, text="ok"))
    monkeypatch.setattr("slashcompute.launcher.main._port_open", lambda host, port: True)
    monkeypatch.setattr("slashcompute.launcher.main._serve", lambda: pytest.fail("bound a busy port"))
    monkeypatch.setattr("slashcompute.launcher.main.time.sleep", lambda s: None)
    alerts = run_main_expecting_failure(monkeypatch, tmp_path)
    assert alerts == ["Port 8766 is in use by another program. Quit it and open /compute again."]
    assert other.terminated is False
    assert "Port 8766 is in use" in (tmp_path / "logs" / "shell.log").read_text()


def test_tk_window_module_is_gone():
    assert not (ROOT / "src" / "slashcompute" / "launcher" / "window.py").exists()
    needles = ("launcher.window", "launcher import window", "tkinter")
    sources = [*ROOT.glob("src/**/*.py"), *ROOT.glob("tests/**/*.py"), *ROOT.glob("scripts/**/*.sh"),
               ROOT / "pyproject.toml"]
    for path in sources:
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text()
        assert not any(n in text for n in needles), path


def test_install_app_writes_plist_and_launcher(tmp_path):
    repo = tmp_path / "repo"
    py = repo / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("#!/bin/sh\n")
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    dest = tmp_path / "out" / "compute.app"
    dest.parent.mkdir()
    subprocess.run(
        ["bash", str(INSTALL), "--repo", str(repo), "--dest", str(dest)],
        check=True,
        env={**os.environ, "HOME": str(tmp_path)},
    )
    plist = (dest / "Contents" / "Info.plist").read_text()
    assert "com.slashcompute.app" in plist
    assert "<string>/compute</string>" in plist
    assert "<string>compute</string>" in plist
    launch = (dest / "Contents" / "MacOS" / "compute").read_text()
    assert str(repo) in launch
    assert "-m slashcompute.launcher.main" in launch
    assert os.access(dest / "Contents" / "MacOS" / "compute", os.X_OK)
    icon = dest / "Contents" / "Resources" / "icon.png"
    assert icon.is_file()
    assert icon.read_bytes() == (ROOT / "scripts" / "macos" / "icon.png").read_bytes()


def _fake_repo(repo: Path, marker: Path) -> Path:
    py = repo / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text(f"#!/bin/sh\npwd > '{marker}'\n")
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    return repo


def test_install_app_refuses_non_app_dest(tmp_path):
    repo = _fake_repo(tmp_path / "repo", tmp_path / "marker")
    home = tmp_path / "home"
    home.mkdir()
    keep = home / "keep.txt"
    keep.write_text("precious")
    # Only temp paths here: a regression would rm -rf whatever is listed.
    for dest in (str(home), f"{home}/", str(tmp_path / "somedir")):
        r = subprocess.run(
            ["bash", str(INSTALL), "--repo", str(repo), "--dest", dest],
            env={**os.environ, "HOME": str(home)},
            capture_output=True,
            text=True,
        )
        assert r.returncode != 0, dest
        assert "Refusing --dest" in r.stderr
    assert keep.read_text() == "precious"


def test_install_app_launcher_quotes_hostile_repo_path(tmp_path):
    marker = tmp_path / "marker"
    repo = _fake_repo(tmp_path / 're"po $(touch pwned) `touch pwned` $HOME', marker)
    dest = tmp_path / "compute.app"
    subprocess.run(
        ["bash", str(INSTALL), "--repo", str(repo), "--dest", str(dest)],
        check=True,
        env={**os.environ, "HOME": str(tmp_path)},
    )
    launch = dest / "Contents" / "MacOS" / "compute"
    subprocess.run(["bash", "-n", str(launch)], check=True)
    subprocess.run([str(launch)], check=True, cwd=tmp_path)
    assert not (tmp_path / "pwned").exists()
    assert Path(marker.read_text().strip()).resolve() == repo.resolve()
