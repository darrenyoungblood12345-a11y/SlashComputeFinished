"""``slashcompute`` — native window around the local shell."""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import httpx
import uvicorn

from slashcompute.common.config import EngineConfig
from slashcompute.common.logging import setup_logging
from slashcompute.web.server import (
    SHELL_GENERATION, SHELL_HOST, SHELL_PORT, asset_version, create_shell,
)

WINDOW_W = 1280
WINDOW_H = 820
# The UI lays out for at least this much; pywebview's default minimum (200x100) lets it crumple.
MIN_W = 960
MIN_H = 640
# Kept free when the start size is clamped to the screen: menu bar, Dock, title bar, some air.
SCREEN_MARGIN_W = 64
SCREEN_MARGIN_H = 128
# The Quit/Cancel prompt when the window closes while this Mac still works for the pool.
BUSY_MESSAGE = (
    "This Mac is still training, serving LLMs or hosting the pool. Everything keeps running in "
    "the background after the window closes; open /compute again to watch or stop it."
)

log = logging.getLogger("slashcompute.launcher")


def shell_url() -> str:
    return f"http://127.0.0.1:{SHELL_PORT}"


def launcher_home() -> Path:
    """Where launcher.json and logs/ live (``SLASHCOMPUTE_HOME``)."""
    return Path(EngineConfig.from_env().home).expanduser()


def _port_open(host: str, port: int) -> bool:
    s = socket.socket()
    s.settimeout(0.2)
    try:
        s.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _shell_info(url: str, timeout: float = 0.6) -> dict:
    """``/api/shell`` of whatever listens at ``url``; {} when nothing /compute-like answers."""
    try:
        r = httpx.get(f"{url.rstrip('/')}/api/shell", timeout=timeout)
    except httpx.RequestError:
        return {}
    if r.status_code != 200:
        return {}
    try:
        info = r.json()
    except ValueError:
        return {}
    return info if isinstance(info, dict) else {}


def _shell_is_ours(url: str) -> bool:
    """A running shell is reused only when it serves exactly this build: same generation and the
    same app.js/app.css. A dev shell or an older build at the same generation serves its own."""
    info = _shell_info(url)
    try:
        generation = int(info.get("generation", 0))
    except (TypeError, ValueError):
        return False
    return generation == SHELL_GENERATION and info.get("assets") == asset_version()


def _lsof_listener_pids(port: int) -> list[int]:
    """PIDs lsof sees listening on TCP ``port``; [] when lsof is missing, slow or finds none."""
    cmd = ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        log.warning("lsof could not look at port %s: %s", port, e)
        return []
    return [int(pid) for pid in r.stdout.split() if pid.isdigit()]   # exit 1 = nothing found


def _listener_pids(port: int) -> list[int]:
    """PIDs listening on TCP ``port``.

    ``psutil.net_connections()`` wants root on macOS (AccessDenied for anyone else), so each
    process is asked for its own sockets instead, which works for the user's own processes.
    When that finds nothing, lsof gets a look: it sees listeners psutil is blind to.
    """
    import psutil

    pids: list[int] = []
    for proc in psutil.process_iter():
        try:
            conns = proc.net_connections(kind="tcp")
        except psutil.Error:
            continue
        if any(c.status == "LISTEN" and c.laddr and c.laddr.port == port for c in conns):
            pids.append(proc.pid)
    return list(dict.fromkeys(pids or _lsof_listener_pids(port)))


def _stop_listener(port: int) -> None:
    """Quit a leftover /compute shell (never an unrelated server) so this launch can bind :8766.

    An unrelated or unknown listener is left alone: ``ensure_shell`` then reports it.
    """
    import psutil

    for pid in _listener_pids(port):
        if pid == os.getpid():
            continue
        try:
            proc = psutil.Process(pid)
            if proc.ppid() == os.getpid():
                continue
            cmd = " ".join(proc.cmdline())
        except (psutil.Error, OSError):
            continue
        if "slashcompute" not in cmd:
            continue
        try:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except psutil.TimeoutExpired:
                proc.kill()
        except (psutil.Error, OSError) as e:
            log.warning("could not stop the old shell (pid %s): %s", pid, e)


def ui_ready(url: str, timeout: float = 0.6) -> bool:
    try:
        r = httpx.get(url, timeout=timeout)
    except httpx.RequestError:
        return False
    if r.status_code != 200:
        return False
    text = r.text
    return "COMPUTE" in text or "/compute" in text


def _serve() -> None:
    # log_config=None leaves uvicorn's loggers propagating to the root, i.e. into shell.log.
    uvicorn.run(create_shell(), host=SHELL_HOST, port=SHELL_PORT, log_level="warning",
                log_config=None)


def ensure_shell(wait: float = 8.0) -> str:
    """Bind the local UI if needed. Returns the URL the window should load."""
    url = shell_url()
    if ui_ready(url) and _shell_is_ours(url):
        return url
    if ui_ready(url) or _port_open("127.0.0.1", SHELL_PORT):
        _stop_listener(SHELL_PORT)
        time.sleep(0.2)
        if ui_ready(url) and _shell_is_ours(url):
            return url
        if _port_open("127.0.0.1", SHELL_PORT):
            raise SystemExit(
                f"Port {SHELL_PORT} is in use by another program. Quit it and open /compute again."
            )
    threading.Thread(target=_serve, daemon=True).start()
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if ui_ready(url):
            return url
        time.sleep(0.1)
    raise SystemExit(f"UI did not start at {url}")


def window_size(screen: Optional[tuple[int, int]]) -> tuple[int, int]:
    """Start size: the default, shrunk to fit the primary screen, never below the minimum."""
    width, height = WINDOW_W, WINDOW_H
    if screen:
        width = min(width, screen[0] - SCREEN_MARGIN_W)
        height = min(height, screen[1] - SCREEN_MARGIN_H)
    return max(width, MIN_W), max(height, MIN_H)


def primary_screen() -> Optional[tuple[int, int]]:
    import webview

    try:
        screens = webview.screens
    except Exception:   # noqa: BLE001 — no display or toolkit trouble: keep the default size
        return None
    if not screens:
        return None
    return int(screens[0].width), int(screens[0].height)


def shell_busy(url: str, timeout: float = 2.0) -> Optional[bool]:
    """Whether this Mac trains, serves LLMs or hosts the pool, per the shell's /api/status.

    None when the shell did not answer in time (or not with status), so a caller can keep
    the last answer it had instead of taking silence for idle.
    """
    try:
        status = httpx.get(f"{url.rstrip('/')}/api/status", timeout=timeout).json()
    except (httpx.HTTPError, ValueError):
        return None
    if not isinstance(status, dict):
        return None
    return any(status.get(k) for k in ("agent_running", "inference_running", "coordinator_pid"))


def watch_busy(url: str, interval: float = 3.0, timeout: float = 2.0,
               stop: Optional[threading.Event] = None) -> Callable[[], bool]:
    """Keep a cached answer to ``shell_busy`` fresh from a daemon thread; the returned function
    reads it without blocking. Asking the shell while the window closes would freeze the GUI
    thread and, when /api/status is slow, answer "idle" exactly when something is running; here
    a failed poll keeps the last answer. ``stop`` ends the thread (tests)."""
    if stop is None:
        stop = threading.Event()
    last = [False]

    def poll() -> None:
        while True:
            try:
                busy = shell_busy(url, timeout=timeout)
            except Exception as e:   # noqa: BLE001 — a surprise must not end the watch
                log.warning("busy check failed: %s", e)
                busy = None
            if busy is not None:
                last[0] = busy
            if stop.wait(interval):
                return

    threading.Thread(target=poll, daemon=True, name="slashcompute-busy").start()
    return lambda: last[0]


def confirm_close_when_busy(is_busy: Callable[[], bool]) -> Callable[[Any], None]:
    """``window.events.closing`` handler: ask before closing while the pool still works here.

    It runs on the GUI thread as the window closes (Cmd-Q included), where
    ``window.create_confirmation_dialog`` would deadlock: it waits on the main run loop we are
    on. So flip pywebview's own ``confirm_close`` and let it show the native Quit/Cancel prompt
    with BUSY_MESSAGE. Returning None lets the close go on to that prompt. ``is_busy`` must not
    block: ``watch_busy`` provides a cached answer.
    """
    def closing(window: Any) -> None:
        window.confirm_close = is_busy()
    return closing


def open_window(url: str) -> None:
    import webview

    width, height = window_size(primary_screen())
    window = webview.create_window(
        "/compute", url, width=width, height=height, min_size=(MIN_W, MIN_H),
        zoomable=True, text_select=True,
        localization={"global.quitConfirmation": BUSY_MESSAGE},
    )
    window.events.closing += confirm_close_when_busy(watch_busy(url))
    # Not private, so the page's cookies survive relaunches. On macOS pywebview ignores
    # storage_path and keeps them in WebKit's shared default data store (one jar for every
    # home on this Mac); the shell identifies the caller by its stored token as a Bearer and
    # never relies on that cookie.
    webview.start(private_mode=False)


def log_to(path: Path) -> Optional[logging.Handler]:
    """Launcher and uvicorn messages (500 tracebacks included) land in logs/shell.log."""
    setup_logging("slashcompute.launcher")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
    except OSError as e:
        log.warning("not logging to %s: %s", path, e)
        return None
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname).1s [%(name)s] %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler


def show_alert(message: str) -> None:
    """Native alert for a launch that failed: the .app has no terminal to print to."""
    script = [
        "on run argv",
        'display alert "/compute could not start" message (item 1 of argv) as critical',
        "end run",
    ]
    cmd = ["osascript", *(arg for line in script for arg in ("-e", line)), "--", message]
    try:
        subprocess.run(cmd, capture_output=True, timeout=120, check=False)
    except (OSError, subprocess.SubprocessError):
        pass


def fail(message: str) -> None:
    """Record why the launch failed, tell the user natively, exit non-zero."""
    log.error("%s", message)
    show_alert(message)
    raise SystemExit(1)


def main() -> None:
    home = launcher_home()
    log_to(home / "logs" / "shell.log")
    try:
        open_window(ensure_shell())
    except SystemExit as e:
        if not e.code:   # a clean exit (None or 0)
            raise
        fail(e.code if isinstance(e.code, str) else f"exit status {e.code}")
    except Exception as e:   # noqa: BLE001 — anything else would just bounce the .app icon
        log.exception("launcher failed")
        fail(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
