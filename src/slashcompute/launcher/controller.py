"""Start/stop the coordinator, the MLX agent and the inference node; persist launcher settings."""

from __future__ import annotations

import contextlib
import errno
import hashlib
import ipaddress
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from http.cookiejar import CookieJar, DefaultCookiePolicy
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlsplit

import httpx
import psutil

from slashcompute.agent.daemon import request_stop
from slashcompute.agent.paths import AgentPaths
from slashcompute.common.config import EngineConfig
from slashcompute.common.discovery import discover, lan_ip
from slashcompute.launcher.dashboard import PoolData


class LauncherError(Exception):
    """User-facing start/stop failure."""


FINISHES = ("carbon", "poster", "signal", "thermal", "void")
TRANSPORTS = ("direct", "relay")
# Errors that only say the coordinator could not be reached: stale once it answers.
UNREACHABLE_ERRORS = ("No coordinator at ", "Coordinator started but is not answering ")
OUTDATED_COORDINATOR = ("This pool's coordinator has no LLM inference: it runs an older /compute. "
                        "Ask whoever hosts it to update and restart it, or host a pool on this Mac.")
PORT_IN_USE = "Port {port} is already in use — quit the other /compute or coordinator, then Start hosting."
# A restart that waits for the running process to finish its step; cleared once the new one is up.
AGENT_RESTARTING = ("Restarting the training agent after the current step finishes; "
                    "the new settings apply then.")
NODE_RESTARTING = "Restarting the LLM node after the current step finishes; the new settings apply then."
# An agent asked to stop drains its step for the grace period (daemon.shutdown waits that long),
# then has this much longer to exit before it counts as hung and is killed.
AGENT_STOP_MARGIN_S = 30.0
AGENT_FORCE_STOPPED = ("The training agent did not stop within {seconds:.0f} s and was force-stopped; "
                       "Start again to keep contributing.")
RESOLVE_TIMEOUT_S = 1.5   # a name that takes longer to resolve (DNS or mDNS away) is not one of ours
AGENT_MODULE = "slashcompute.agent.main"
COORDINATOR_MODULE = "slashcompute.coordinator.main"
INFERENCE_MODULE = "slashcompute.inference.node"
HEALTH_COUNTS = ("nodes", "jobs", "inference_nodes")
# Where a server bound to 0.0.0.0 (as our coordinator is) answers besides its interfaces.
LOCAL_HOSTS = frozenset({"0.0.0.0", "::", "127.0.0.1", "::1", "localhost"})


def stateless_http(**kw: Any) -> httpx.Client:
    """Client for the shared shell: it proxies many browsers, so it must never keep a cookie (a
    stored Set-Cookie would sign every cookie-less request in as the last user). Each request
    carries only the caller's own Cookie/Authorization headers."""
    jar = CookieJar(policy=DefaultCookiePolicy(allowed_domains=[]))
    return httpx.Client(follow_redirects=False, cookies=jar, **kw)


def supports_inference(health: Optional[dict]) -> Optional[bool]:
    """Whether the coordinator serves LLMs (every build since inference reports its transport in
    /health). None while it is unreachable."""
    return None if not health else "inference_transport" in health


def health_count(health: dict, key: str) -> int:
    """A count from /health, 0 when it is missing or not a number."""
    try:
        return int(health.get(key, 0) or 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def agent_gpu_percent(status: dict) -> Optional[int]:
    """The GPU share the agent reports in its status.json, None when absent or malformed."""
    try:
        value = status.get("gpu_percent")
        return None if value is None else int(value)
    except (TypeError, ValueError, OverflowError, AttributeError):
        return None


def valid_health(data: Any) -> bool:
    """Whether a /health body is one the shell can use: a coordinator whose counts are not numbers
    is treated as unhealthy rather than crashing every status poll."""
    if not isinstance(data, dict):
        return False
    for key in HEALTH_COUNTS:
        value = data.get(key, 0)
        try:
            int(value or 0)
        except (TypeError, ValueError, OverflowError):
            return False
    return True


@dataclass
class LauncherSettings:
    mode: str = "host"
    url: str = ""
    gpu_percent: int = 50
    contribute: bool = True
    finish: str = "carbon"
    session_token: str = ""
    session_url: str = ""              # the pool that issued session_token: it is never sent to another
    grant_split: int = 0
    training: bool = True              # lend this Mac to MLX fine-tunes
    memory_gb: int = 0                 # GiB lent to fine-tunes; 0 = automatic (what is free at start)
    inference: bool = False            # also host llama.cpp layers for the pool's LLMs
    inference_memory_gb: int = 0       # 0 = automatic (75% of RAM minus 4 GiB)
    inference_head: bool = True        # may run llama-server (needs the model file; uploads are pushed)
    models_dir: str = "~/models"
    transport: str = "direct"          # host only: direct (LAN) or relay (internet, RPC via the coordinator)

    def clamp(self) -> "LauncherSettings":
        mode = self.mode if self.mode in ("host", "join", "public") else "host"
        finish = self.finish if self.finish in FINISHES else "carbon"
        transport = self.transport if self.transport in TRANSPORTS else "direct"
        try:
            gpu = max(1, min(100, int(self.gpu_percent)))
        except (TypeError, ValueError, OverflowError):
            gpu = 50
        try:
            split = max(0, min(100, int(self.grant_split)))
        except (TypeError, ValueError, OverflowError):
            split = 0
        try:
            mem = max(0, min(1024, int(self.inference_memory_gb)))
        except (TypeError, ValueError, OverflowError):
            mem = 0
        try:
            train_mem = max(0, min(1024, int(self.memory_gb)))
        except (TypeError, ValueError, OverflowError):
            train_mem = 0
        return LauncherSettings(
            mode=mode, url=str(self.url or ""), gpu_percent=gpu,
            contribute=bool(self.contribute), finish=finish,
            session_token=str(self.session_token or ""), session_url=str(self.session_url or ""),
            grant_split=split,
            training=bool(self.training), memory_gb=train_mem, inference=bool(self.inference),
            inference_memory_gb=mem,
            inference_head=bool(self.inference_head), models_dir=str(self.models_dir or "~/models"),
            transport=transport,
        )


# What the LLMs tab may change when it starts serving (never anything training uses).
INFERENCE_SETTINGS = ("inference_memory_gb", "inference_head", "models_dir", "transport")


@dataclass
class StatusSnapshot:
    coordinator_up: bool = False
    nodes: int = 0
    jobs: int = 0
    agent_running: bool = False
    agent_status: str = ""
    agent_job_id: Optional[str] = None
    agent_fetch_done_bytes: Optional[int] = None    # the agent is downloading a job's model
    agent_fetch_total_bytes: Optional[int] = None
    agent_draining: bool = False                    # it was asked to stop and finishes its step first
    agent_gpu_percent: Optional[int] = None         # the share the running agent was started with
    coordinator_pid: Optional[int] = None
    agent_pid: Optional[int] = None
    lan_ip: str = ""
    last_error: str = ""
    inference_running: bool = False
    inference_pid: Optional[int] = None
    inference_status: dict = field(default_factory=dict)   # the node's status.json
    inference_nodes: int = 0
    inference_transport: str = ""
    inference_supported: Optional[bool] = None
    memory_total_bytes: int = 0        # this Mac's unified memory, for the memory sliders
    memory_available_bytes: int = 0


PopenFn = Callable[..., Any]


def normalize_url(url: str, port: int = 8765, scheme: str = "http") -> str:
    u = (url or "").strip()
    if not u:
        return ""
    if "://" not in u:
        if ":" not in u.split("/")[0]:
            u = f"{u}:{port}"
        u = f"{scheme}://{u}"
    return u.rstrip("/")


def write_private(path: Path, text: str) -> None:
    """Write a file only this user can read: it holds a session or a fingerprint of one.

    Written beside the file and renamed over it: a reader on another thread (every status poll
    loads launcher.json) sees the old or the new content, never a truncated file that loads as
    defaults — and could then be saved back with the session gone."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")   # created 0600
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def launch_record(argv: list[str], session: str) -> dict:
    """What an agent or LLM node was started with. Its session goes through the environment (argv
    shows in ps), so only a digest of it is kept to notice when it changes."""
    return {"argv": argv, "session": hashlib.sha256(session.encode()).hexdigest() if session else ""}


def recorded_argv(record: Any) -> Optional[list[str]]:
    """The argv a running agent or node was started with, from its launch record (or the bare argv,
    session included, that launchers before records wrote). None when unknown."""
    if isinstance(record, dict):
        record = record.get("argv")
    if not isinstance(record, list) or "--url" not in record:
        return None
    if "--session-token" in record:
        i = record.index("--session-token")
        record = record[:i] + record[i + 2:]
    return record


def health_timeout(url: str) -> float:
    return 5.0 if (url or "").lower().startswith("https://") else 1.0


def system_memory() -> tuple[int, int]:
    vm = psutil.virtual_memory()
    return int(vm.total), int(vm.available)


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:   # an exited child its parent has not reaped yet still answers kill(0)
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return True


def started_after(pid: int, when: float) -> bool:
    """Whether process `pid` began after `when`, i.e. the pid was reused since we recorded it."""
    try:
        return psutil.Process(pid).create_time() > when + 1.0
    except psutil.Error:
        return False


def process_cmdline(pid: int) -> Optional[list[str]]:
    """The command line of process `pid`; None when it cannot be read (it is gone, or not ours)."""
    try:
        return list(psutil.Process(pid).cmdline() or [])
    except psutil.Error:
        return None


def same_path(a: str, b: Path) -> bool:
    try:
        return Path(a).expanduser().resolve() == Path(b).resolve()
    except (OSError, RuntimeError, ValueError):
        return str(a) == str(b)


def port_free(host: str, port: int) -> bool:
    """Whether a server could bind host:port (uvicorn binds with SO_REUSEADDR too)."""
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError as e:
            return e.errno != errno.EADDRINUSE
    return True


def canonical_ip(text: str) -> Optional[str]:
    """`text` as the one spelling of an IP address (`::1`, not `0:0:0:0:0:0:0:1`; a link-local
    `%en0` scope dropped), or None when it is not one."""
    try:
        return str(ipaddress.ip_address(text.strip("[]").split("%", 1)[0]))
    except ValueError:
        return None


def local_addresses() -> set[str]:
    """Every name and address this Mac answers on: each interface's addresses (LAN, VPN, link-local),
    loopback, the wildcard, and its hostname with the `.local` form Bonjour gives it. Lower-cased."""
    own = set(LOCAL_HOSTS)
    try:
        for entries in psutil.net_if_addrs().values():
            for entry in entries:
                if entry.family in (socket.AF_INET, socket.AF_INET6) and (ip := canonical_ip(entry.address)):
                    own.add(ip)
    except (psutil.Error, OSError):
        pass
    try:
        name = socket.gethostname().lower()
    except OSError:
        name = ""
    if name:
        short = name.split(".")[0]
        own |= {name, short, f"{short}.local"}
    return own


def resolve_host(host: str, timeout: Optional[float] = None) -> set[str]:
    """The addresses `host` names: an IP literal is itself; a name is looked up on a thread we stop
    waiting for after `timeout` (getaddrinfo has none of its own), so a dead DNS or a `.local`
    name nobody answers for cannot hang a Start."""
    if (ip := canonical_ip(host)) is not None:
        return {ip}
    resolved: list[set[str]] = []

    def lookup() -> None:
        try:
            infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
        except (OSError, UnicodeError):
            infos = []
        resolved.append({ip for info in infos if (ip := canonical_ip(str(info[4][0])))})

    thread = threading.Thread(target=lookup, name="resolve-host", daemon=True)
    thread.start()
    thread.join(RESOLVE_TIMEOUT_S if timeout is None else timeout)
    return resolved[0] if resolved else set()


def signal_child(proc: Any, sig: int) -> None:
    """Signal a process through the Popen we spawned it with: it knows the pid is still its
    child's (never reused until reaped), where a bare os.kill on a recorded pid does not. A
    stand-in without send_signal cannot be signalled."""
    send = getattr(proc, "send_signal", None)
    if send is not None:
        with contextlib.suppress(OSError):
            send(sig)


class Launcher:
    def __init__(
        self,
        home: Optional[Path] = None,
        python: Optional[str] = None,
        popen: PopenFn = subprocess.Popen,
        http: Optional[httpx.Client] = None,
        discover_fn: Optional[Callable[[float], Optional[str]]] = None,
        lan_ip_fn: Callable[[], str] = lan_ip,
        memory_fn: Callable[[], tuple[int, int]] = system_memory,
        port_free_fn: Callable[[str, int], bool] = port_free,
    ) -> None:
        self.cfg = EngineConfig.from_env(home=home)
        if home is not None:
            self.cfg.home = Path(home)
        self.home = Path(self.cfg.home)
        self.home.mkdir(parents=True, exist_ok=True)
        self.python = python or sys.executable
        self._popen = popen
        self._http = http or stateless_http()
        # While hosting, the first advertisement to answer is our own coordinator: take the first
        # that is not (``discover`` is looked up when called, so tests can stand one in).
        self._discover = discover_fn or (lambda timeout: discover(timeout, exclude=self.is_own_url))
        self._lan_ip = lan_ip_fn
        self._memory = memory_fn
        self._port_free = port_free_fn
        self.last_error = ""
        # One launcher serves every request thread of the shell: starts, stops and the status polls
        # that reap and restart processes take this lock (re-entrant: start() ends in snapshot()).
        # Reading the settings never does: a start holds the lock for up to ~25 s while the async
        # handlers (the proxy) load them on the event loop, and launcher.json is replaced atomically.
        self._lock = threading.RLock()
        self._coordinator_proc: Any = None      # the coordinator this launcher spawned, while it runs
        self._coordinator_log_at = 0            # coordinator.log size when it was spawned
        self._stopping: list[Any] = []          # processes we stopped, reaped once they exit
        self._agent_proc: Any = None            # the training agent we spawned, reaped when it exits
        self._agent_stop_pid: Optional[int] = None   # the agent we asked to stop, until it has
        self._agent_stop_at: Optional[float] = None  # when we asked (monotonic), for the hung check
        self._inference_proc: Any = None        # the LLM node we spawned, to report its exit
        self._inference_log_at = 0
        # A restart that could not happen yet (the old process still finishes its step): what to
        # start it as, done from snapshot() once the old one is gone.
        self._pending_agent: Optional[tuple[list[str], str]] = None
        self._pending_inference: Optional[tuple[list[str], str]] = None
        self.paths = AgentPaths(self.home)

    @property
    def settings_path(self) -> Path:
        return self.home / "launcher.json"

    @property
    def coordinator_pid_path(self) -> Path:
        return self.home / "coordinator.pid"

    @property
    def log_dir(self) -> Path:
        d = self.home / "logs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def load_settings(self) -> LauncherSettings:
        """The saved settings. Lock-free: write_private replaces launcher.json atomically, so a
        reader sees the old or the new file, never a partial one — and the shell's event loop
        must never wait behind a Start holding the lock."""
        if not self.settings_path.exists():
            return LauncherSettings()
        try:
            raw = json.loads(self.settings_path.read_text())
        except (OSError, json.JSONDecodeError):
            return LauncherSettings()
        if not isinstance(raw, dict):
            return LauncherSettings()
        s = LauncherSettings(
            mode=raw.get("mode", "host"),
            url=raw.get("url", ""),
            gpu_percent=raw.get("gpu_percent", 50),
            contribute=raw.get("contribute", True),
            finish=raw.get("finish", "carbon"),
            session_token=raw.get("session_token", ""),
            session_url=raw.get("session_url", ""),
            grant_split=raw.get("grant_split", 0),
            training=raw.get("training", True),
            memory_gb=raw.get("memory_gb", 0),
            inference=raw.get("inference", False),
            inference_memory_gb=raw.get("inference_memory_gb", 0),
            inference_head=raw.get("inference_head", True),
            models_dir=raw.get("models_dir", "~/models"),
            transport=raw.get("transport", "direct"),
        ).clamp()
        if s.session_token and "session_url" not in raw:   # stored before sessions were tied to a pool
            s.session_url = self.proxy_url(s)
        return s

    def save_settings(self, settings: LauncherSettings) -> None:
        s = settings.clamp()
        with self._lock:
            before = self.load_settings()
            if (s.mode, s.url) != (before.mode, before.url):
                self.last_error = ""   # it was about the pool we just left (e.g. the port taken while hosting)
            write_private(self.settings_path, json.dumps(asdict(s), indent=2) + "\n")

    def session_for(self, settings: LauncherSettings, url: Optional[str] = None) -> str:
        """The stored session, but only for the pool that issued it (`url`, default the pool the
        settings point at): pool A's token is never presented to pool B."""
        s = settings.clamp()
        url = self.proxy_url(s) if url is None else url
        return s.session_token if s.session_token and s.session_url == url else ""

    def with_session(self, settings: LauncherSettings, token: str) -> LauncherSettings:
        """Settings holding a session just issued by the pool they point at ("" = signed out of it)."""
        s = settings.clamp()
        url = self.proxy_url(s)
        if not token and s.session_url != url:
            return s   # signing out of this pool leaves another pool's session alone
        return replace(s, session_token=token, session_url=url if token else "")

    def coordinator_url(self, settings: LauncherSettings) -> str:
        if settings.mode == "host":
            return f"http://{self._lan_ip()}:{self.cfg.coordinator_port}"
        raw = settings.url or (self.cfg.public_url if settings.mode == "public" else "")
        scheme = "https" if settings.mode == "public" else "http"
        return normalize_url(raw, self.cfg.coordinator_port, scheme=scheme)

    def proxy_url(self, settings: Optional[LauncherSettings] = None) -> str:
        """Coordinator URL the local shell should dial (loopback when hosting)."""
        s = settings.clamp() if settings is not None else self.load_settings()
        if s.mode == "host":
            return f"http://127.0.0.1:{self.cfg.coordinator_port}"
        return self.coordinator_url(s)

    def coordinator_argv(self, transport: str = "direct") -> list[str]:
        argv = [self.python, "-m", COORDINATOR_MODULE, "serve",
                "--home", str(self.home)]
        if transport != "direct":
            argv.extend(["--inference-transport", transport])
        return argv

    def agent_argv(self, url: str, gpu_percent: int, memory_gb: int = 0) -> list[str]:
        argv = [
            self.python, "-m", "slashcompute.agent.main", "start",
            "--url", url, "--gpu-percent", str(int(gpu_percent)),
            "--home", str(self.home),
        ]
        if memory_gb:
            argv.extend(["--max-memory-gb", str(int(memory_gb))])
        return argv

    def inference_argv(self, url: str, settings: LauncherSettings) -> list[str]:
        s = settings.clamp()
        return [
            self.python, "-m", "slashcompute.inference.node", "start",
            "--url", url, "--home", str(self.home), "--models-dir", s.models_dir,
            "--memory-gb", str(s.inference_memory_gb), "--head" if s.inference_head else "--no-head",
        ]

    @property
    def inference_pid_path(self) -> Path:
        return self.home / "inference" / "node.pid"

    def read_inference_pid(self) -> Optional[int]:
        try:
            pid = int(self.inference_pid_path.read_text().strip())
        except (OSError, ValueError):
            return None
        if not process_alive(pid):
            return None
        if not self._owned_process(pid, INFERENCE_MODULE):
            self.inference_pid_path.unlink(missing_ok=True)   # another program has the pid now
            return None
        return pid

    def _owned_process(self, pid: Optional[int], module: str) -> bool:
        """Whether `pid` is alive and is our `module` serving this home. After a crash or a reboot
        the pid in a left-over pid file can belong to any other program: the UI must not call it
        ours and Stop must not SIGTERM it."""
        if not pid or not process_alive(pid):
            return False
        argv = process_cmdline(pid)
        if argv is None:
            return True   # alive, but its command line is out of reach: the pid file is all we have
        if not any(module in arg for arg in argv):
            return False
        if "--home" in argv[:-1]:
            return same_path(argv[argv.index("--home") + 1], self.home)
        return same_path(str(EngineConfig.from_env().home), self.home)   # started with the default home

    def inference_status(self) -> dict:
        try:
            data = json.loads((self.home / "inference" / "status.json").read_text())
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _stop_inference(self, wait: float = 0.0) -> None:
        if self._inference_proc is not None:   # its exit is expected now: only reap it
            self._stopping.append(self._inference_proc)
            self._inference_proc = None
        pid = self.read_inference_pid()
        if pid is None:
            return
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline and process_alive(pid):
            time.sleep(0.05)

    def _inference_args_path(self) -> Path:
        return self.home / "inference.args"

    def _write_inference_pid(self, pid: int) -> None:
        self.inference_pid_path.parent.mkdir(parents=True, exist_ok=True)
        self.inference_pid_path.write_text(f"{pid}\n")

    def _check_inference(self) -> bool:
        """Reap the LLM node we started once it exits on its own, and say why in last_error
        (the shell used to keep saying "Serving…" or just "Not serving"). True when it had exited."""
        proc = self._inference_proc
        if proc is None or (code := proc.poll()) is None:
            return False
        self._inference_proc = None
        if code != 0:
            self.last_error = self._inference_exit_reason(code)
        return True

    def _inference_exit_reason(self, code: int) -> str:
        log = self.log_dir / "inference.log"
        try:
            with open(log, "rb") as fh:
                fh.seek(self._inference_log_at)   # only what this run wrote
                lines = [ln.strip() for ln in fh.read().decode("utf-8", "replace").splitlines() if ln.strip()]
        except OSError:
            lines = []
        how = f"was stopped by signal {-code}" if code < 0 else f"exited with code {code}"
        # The last lines say why (a SystemExit message, or a traceback's final line).
        why = " | ".join(ln for ln in lines[-2:] if not ln.startswith("Traceback")) or f"see {log}"
        return f"The LLM node {how}: {why}"

    def _await_node(self, proc: Any, timeout: float = 8.0) -> None:
        """After starting the LLM node: wait until it reached the coordinator or gave up, so
        Start serving reports what happened instead of claiming success."""
        if proc is None:
            return                                # it was already running
        deadline = time.monotonic() + timeout
        st: dict = {}
        while time.monotonic() < deadline:
            if self._check_inference():
                return                            # exited: last_error says why
            st = self.inference_status()
            if st.get("pid") == proc.pid and st.get("reason") not in (None, "connecting"):
                if st.get("reason") == "unsupported":
                    self.last_error = st.get("last_error") or "The pool's coordinator has no LLM inference."
                return                            # registered (or told why it can't)
            time.sleep(0.2)
        if st.get("pid") == proc.pid and st.get("last_error"):
            self.last_error = f"The LLM node is still trying to join: {st['last_error']}"

    def set_inference(self, on: bool, **changes: Any) -> StatusSnapshot:
        """Start or stop only this Mac's LLM node. The training agent is never touched.

        Start serving used to resend every setting through ``start``. A training setting saved
        since the agent started (its memory, say) then restarted the training agent, which can't
        stop while it downloads a model, so the click failed with "still stopping"."""
        with self._lock:
            return self._set_inference(on, **changes)

    def _set_inference(self, on: bool, **changes: Any) -> StatusSnapshot:
        allowed = {k: v for k, v in changes.items() if k in INFERENCE_SETTINGS}
        s = replace(self.load_settings(), **allowed)
        # Lending to LLMs is contributing; fine-tuning stays as it is right now.
        s = replace(s, inference=True, contribute=True, training=self._agent_running()) if on \
            else replace(s, inference=False)
        s = s.clamp()
        self.save_settings(s)
        self.last_error = ""
        if not on:
            self._pending_inference = None
            self._stop_inference()
            return self.snapshot(s)
        url = self.coordinator_url(s)
        try:
            if s.mode == "host":
                self._ensure_coordinator(s, url)
            elif s.mode == "public" and not self.session_for(s):
                raise LauncherError("Sign in first.")
            elif not self.poll_health(url):
                raise LauncherError(f"No coordinator at {url}." if url else "Enter a coordinator URL first.")
            node_url = self.proxy_url(s) if s.mode == "host" else url
            self._await_node(self._ensure_inference(self.inference_argv(node_url, s),
                                                    self.session_for(s, node_url)))
        except LauncherError as e:
            self.last_error = str(e)
        return self.snapshot(s)

    def _restart_coordinator(self, settings: LauncherSettings) -> None:
        """The inference transport is fixed when the coordinator starts: restart ours to change it."""
        pid = self.read_coordinator_pid()
        if pid is None:
            self.last_error = ("Inference transport can only change when this app started the coordinator; "
                               "restart it with --inference-transport " + settings.transport + ".")
            return
        self._forget_coordinator()
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline and process_alive(pid):
            time.sleep(0.05)
        if process_alive(pid):   # never start a second one over it
            self.last_error = "The coordinator did not stop in time; Stop it, then Start hosting again."
            return
        self.coordinator_pid_path.unlink(missing_ok=True)
        self._spawn_coordinator(settings.transport)
        self.wait_health(self.proxy_url(settings))

    def read_coordinator_pid(self) -> Optional[int]:
        """Our coordinator's pid from coordinator.pid, or None (and no file) when the process it
        names is gone, began after the file was written, or is not our coordinator for this home:
        a stale pid reused by another program is neither reported as ours nor ever signalled."""
        if not self.coordinator_pid_path.exists():
            return None
        try:
            pid = int(self.coordinator_pid_path.read_text().strip())
            written = self.coordinator_pid_path.stat().st_mtime
        except (OSError, ValueError):
            return None
        if (process_alive(pid) and not started_after(pid, written)
                and self._owned_process(pid, COORDINATOR_MODULE)):
            return pid
        self.coordinator_pid_path.unlink(missing_ok=True)
        return None

    def find_coordinators(self) -> list[int]:
        """Coordinators serving this home, found by their command line: ours even if coordinator.pid
        was lost or overwritten."""
        pids = []
        for proc in psutil.process_iter(["cmdline", "status"]):
            argv = proc.info["cmdline"] or []
            if (proc.info["status"] != psutil.STATUS_ZOMBIE and COORDINATOR_MODULE in argv
                    and "serve" in argv and "--home" in argv[:-1]
                    and argv[argv.index("--home") + 1] == str(self.home)):
                pids.append(proc.pid)
        return pids

    def own_coordinator_pid(self) -> Optional[int]:
        """Our running coordinator, re-recording its pid when coordinator.pid was lost."""
        pid = self.read_coordinator_pid()
        if pid is None and (found := self.find_coordinators()):
            pid = found[0]
            self.write_coordinator_pid(pid)
        return pid

    def _spawn_coordinator(self, transport: str) -> None:
        log = self.log_dir / "coordinator.log"
        self._coordinator_log_at = log.stat().st_size if log.exists() else 0
        self._coordinator_proc = self._spawn(self.coordinator_argv(transport), log,
                                             pid_writer=self.write_coordinator_pid)

    def _forget_coordinator(self) -> None:
        """We are stopping our coordinator: its exit is expected, so only reap it."""
        if self._coordinator_proc is not None:
            self._stopping.append(self._coordinator_proc)
            self._coordinator_proc = None

    def _check_coordinator(self) -> bool:
        """Reap the coordinator we spawned once it exits (e.g. its port was taken): forget its pid and
        say why in last_error. True when it had exited."""
        self._stopping = [p for p in self._stopping if p.poll() is None]
        proc = self._coordinator_proc
        if proc is None or (code := proc.poll()) is None:
            return False
        self._coordinator_proc = None
        self.coordinator_pid_path.unlink(missing_ok=True)
        if self.load_settings().mode == "host":   # after a switch to another pool it is not news
            self.last_error = self._coordinator_exit_reason(code)
        return True

    def _coordinator_exit_reason(self, code: int) -> str:
        log = self.log_dir / "coordinator.log"
        try:
            with open(log, "rb") as fh:
                fh.seek(self._coordinator_log_at)   # only what this run wrote
                lines = [ln.strip() for ln in fh.read().decode("utf-8", "replace").splitlines() if ln.strip()]
        except OSError:
            lines = []
        if code == 3 or any("address already in use" in ln.lower() for ln in lines):
            return PORT_IN_USE.format(port=self.cfg.coordinator_port)
        how = f"was stopped by signal {-code}" if code < 0 else f"exited with code {code}"
        return f"The coordinator {how}: " + (" | ".join(lines[-3:]) if lines else f"see {log}.")

    def write_coordinator_pid(self, pid: int) -> None:
        self.coordinator_pid_path.write_text(str(pid) + "\n")

    def poll_health(self, url: str) -> Optional[dict]:
        if not url:
            return None
        try:
            r = self._http.get(f"{url.rstrip('/')}/health", timeout=health_timeout(url))
            if r.status_code == 200:
                data = r.json()
                if not isinstance(data, dict):
                    return {"ok": True}
                return data if valid_health(data) else None   # malformed counts: unhealthy
        except Exception:
            return None
        return None

    def wait_health(self, url: str, timeout: float = 8.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.poll_health(url):
                return True
            if self._check_coordinator():
                return False
            time.sleep(0.2)
        return False

    def find_on_lan(self, timeout: float = 5.0) -> Optional[str]:
        """The first coordinator advertised on the LAN that is not the one hosted on this Mac
        (Connect would join it only for start(join) to stop hosting it). While hosting, ours is
        the first to answer: discovery keeps listening past it until `timeout`."""
        found = self._discover(timeout)
        if not found:
            return None
        url = normalize_url(found)
        return None if self.is_own_url(url) else url

    def is_own_url(self, url: str) -> bool:
        """Whether `url` is this Mac's own coordinator: on our port, at any address or name this
        Mac answers on — an interface (LAN, VPN), loopback, 0.0.0.0, `my-mac.local` or another
        name resolving to one of them. The LAN ip and loopback alone missed the rest, and
        start(join) then stopped the very pool it had just joined."""
        try:
            parts = urlsplit(normalize_url(url, self.cfg.coordinator_port))
            port = parts.port or {"https": 443}.get(parts.scheme, 80)
        except ValueError:
            return False
        host = (parts.hostname or "").lower()
        if not host or port != self.cfg.coordinator_port:
            return False
        own = local_addresses() | {self._lan_ip()}
        return (canonical_ip(host) or host) in own or bool(resolve_host(host) & own)

    def _ensure_coordinator(self, s: LauncherSettings, url: str) -> None:
        """Hosting: run our coordinator (with the chosen LLM transport) and wait for it to answer."""
        # Ask on loopback first, the LAN address may not answer (bound to 127.0.0.1, slow Wi-Fi).
        health = self.poll_health(self.proxy_url(s)) or self.poll_health(url)
        self._check_coordinator()
        self.own_coordinator_pid()
        if not health:
            if self.read_coordinator_pid() is None:   # ours may just be slow to answer: never start two
                if not self._port_free(self.cfg.coordinator_host, self.cfg.coordinator_port):
                    self.last_error = PORT_IN_USE.format(port=self.cfg.coordinator_port)
                    raise LauncherError(self.last_error)
                self.last_error = ""
                self._spawn_coordinator(s.transport)
            if not (self.wait_health(self.proxy_url(s)) or self.poll_health(url)):
                if self._coordinator_proc is None and self.last_error:
                    raise LauncherError(self.last_error)   # it exited: nothing to lend to
                self.last_error = (
                    f"Coordinator started but is not answering {url}/health yet. "
                    f"Watch {self.log_dir / 'coordinator.log'}."
                )
        elif health.get("inference_transport", "direct") != s.transport:
            self._restart_coordinator(s)

    def start(self, settings: LauncherSettings) -> StatusSnapshot:
        with self._lock:
            return self._start(settings)

    def _start(self, settings: LauncherSettings) -> StatusSnapshot:
        s = settings.clamp()
        self.save_settings(s)
        self.last_error = ""
        url = self.coordinator_url(s)
        if s.mode == "join" and not url:
            self.last_error = "Enter a coordinator URL, or find one on the LAN."
            raise LauncherError(self.last_error)
        if s.mode == "public" and not url:
            self.last_error = "Enter the public coordinator URL."
            raise LauncherError(self.last_error)
        if s.mode == "public" and not self.session_for(s):
            self.last_error = "Sign in first."
            raise LauncherError(self.last_error)

        want_coord = s.mode == "host"
        lend = s.mode == "join" or s.contribute
        want_agent = lend and s.training
        want_inference = lend and s.inference

        if want_coord:
            self._ensure_coordinator(s, url)

        agent_url = self.proxy_url(s) if s.mode == "host" else url
        if s.mode in ("join", "public") and not self.poll_health(url):
            self.last_error = f"No coordinator at {url}."
            raise LauncherError(self.last_error)
        if s.mode in ("join", "public") and not self.is_own_url(url):
            self._stop_coordinator()   # the new pool answers: stop hosting ours, nothing here dials it now
        if not want_agent:
            self._pending_agent = None
            if self._agent_running():
                self._request_agent_stop()
        if want_agent:
            self._ensure_agent(self.agent_argv(agent_url, s.gpu_percent, s.memory_gb),
                               self._agent_session(s, agent_url))

        if want_inference:
            if s.mode == "join" and not self.poll_health(url):
                self.last_error = f"No coordinator at {url}."
                raise LauncherError(self.last_error)
            self._ensure_inference(self.inference_argv(agent_url, s), self.session_for(s, agent_url))
        else:
            self._stop_inference()

        return self.snapshot(s)

    def rebind_session(self) -> StatusSnapshot:
        """After a sign-in or sign-out: restart the agent and LLM node running here with the session
        now stored for their pool, so they earn for the signed-in account without another Start.
        One that cannot stop yet is restarted from a later snapshot(): this Mac never silently
        stops contributing over a sign-in."""
        with self._lock:
            return self._rebind_session()

    def _rebind_session(self) -> StatusSnapshot:
        s = self.load_settings()
        try:
            if self._pending_agent is not None:   # restarting anyway: with the session stored now
                argv, _ = self._pending_agent
                self._pending_agent = (argv, self._agent_session(s, argv[argv.index("--url") + 1]))
            elif self._agent_running() and (argv := recorded_argv(self._read_agent_args())):
                self._ensure_agent(argv, self._agent_session(s, argv[argv.index("--url") + 1]))
            if self._pending_inference is not None:
                argv, _ = self._pending_inference
                self._pending_inference = (argv, self.session_for(s, argv[argv.index("--url") + 1]))
            elif self.read_inference_pid() is not None and (argv := recorded_argv(self._read_inference_args())):
                self._ensure_inference(argv, self.session_for(s, argv[argv.index("--url") + 1]))
        except LauncherError:
            pass   # last_error says why
        return self.snapshot(s)

    def _agent_session(self, s: LauncherSettings, url: str) -> str:
        return self.session_for(s, url) or os.environ.get("SLASHCOMPUTE_SESSION", "")

    def _request_agent_stop(self) -> None:
        """Ask our training agent to stop (it drains its step first); its exit is expected now."""
        proc, self._agent_proc = self._agent_proc, None
        if proc is not None:
            self._stopping.append(proc)
        self._agent_stop_pid = self.paths.read_pid()
        self._agent_stop_at = time.monotonic()
        if self._agent_stop_pid is None and proc is not None and proc.poll() is None:
            # Spawned moments ago, agent.pid not written yet: only its Popen can name it.
            self._agent_stop_pid = int(proc.pid)
            signal_child(proc, signal.SIGTERM)
            return
        request_stop(self.paths)

    def _ensure_agent(self, argv: list[str], session: str) -> None:
        """Run the training agent as `argv` with `session`, restarting one started differently.
        One still draining after a short wait is restarted later, from snapshot(), once it is gone:
        a settings change or sign-in must never leave this Mac quietly not contributing."""
        record = launch_record(argv, session)
        if self._agent_running() and self._read_agent_args() != record:
            self._request_agent_stop()
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and self._agent_running():
                time.sleep(0.05)
            if self._agent_running():
                self._pending_agent = (argv, session)
                self.last_error = AGENT_RESTARTING
                return
        self._pending_agent = None
        if not self._agent_running():
            self._agent_proc = self._spawn(argv, self.log_dir / "agent.log", session=session)
            write_private(self._agent_args_path(), json.dumps(record))

    def _ensure_inference(self, argv: list[str], session: str) -> Any:
        """Run the LLM node as `argv` with `session`, restarting one started differently.
        Returns the process when it started one. One still draining after the wait is restarted
        later, from snapshot()."""
        record = launch_record(argv, session)
        if self.read_inference_pid() is not None and self._read_inference_args() != record:
            self._stop_inference(wait=10.0)   # settings changed: drain, then rejoin with the new ones
            if self.read_inference_pid() is not None:
                self._pending_inference = (argv, session)
                self.last_error = NODE_RESTARTING
                return None
        self._pending_inference = None
        if self.read_inference_pid() is not None:
            return None
        log = self.log_dir / "inference.log"
        self._inference_log_at = log.stat().st_size if log.exists() else 0
        # Its pid is recorded now: the node writes it only after probing the hardware (seconds),
        # and a second click in between used to start a second node.
        self._inference_proc = self._spawn(argv, log, pid_writer=self._write_inference_pid, session=session)
        write_private(self._inference_args_path(), json.dumps(record))
        return self._inference_proc

    def stop(self) -> StatusSnapshot:
        with self._lock:
            self.last_error = ""
            self._pending_agent = self._pending_inference = None
            if self._agent_running():   # never SIGTERM a stranger that was given a stale pid
                self._request_agent_stop()
            self._stop_inference()
            self._stop_coordinator()
            return self.snapshot()

    def _stop_coordinator(self) -> None:
        """Stop the coordinator hosted from this home, and only that: a pid from the pid file or
        a command line is signalled only once it is confirmed ours (another home's pool, or a
        program that inherited a stale pid, is never SIGTERMed)."""
        self._forget_coordinator()
        # Also those found by command line: Stop must not leave ours up when its pid file was lost.
        for pid in dict.fromkeys([self.read_coordinator_pid(), *self.find_coordinators()]):
            if pid is None or not self._owned_process(pid, COORDINATOR_MODULE):
                continue
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                self.coordinator_pid_path.unlink(missing_ok=True)

    def stop_agent(self) -> StatusSnapshot:
        """Stop contributing; a coordinator hosted here keeps running."""
        with self._lock:
            self.last_error = ""
            self._pending_agent = None
            if self._agent_running():
                self._request_agent_stop()
            return self.snapshot()

    def my_node_id(self) -> Optional[str]:
        """This Mac's agent id, once it has registered. Never creates one."""
        try:
            return self.paths.node_id_file.read_text().strip() or None
        except OSError:
            return None

    def fetch_pool(self, url: str) -> PoolData:
        """Nodes, jobs and ledger from the coordinator, or empty when it is down."""
        if not url or not self.poll_health(url):
            return PoolData()
        return PoolData(online=True, nodes=self._get_list(url, "/nodes"),
                        jobs=self._get_list(url, "/jobs"), ledger=self._get_list(url, "/ledger"))

    def _get_list(self, url: str, path: str) -> list:
        try:
            r = self._http.get(f"{url.rstrip('/')}{path}", timeout=1.5)
            data = r.json() if r.status_code == 200 else []
        except Exception:
            return []
        return data if isinstance(data, list) else []

    def snapshot(self, settings: Optional[LauncherSettings] = None) -> StatusSnapshot:
        with self._lock:   # what reaps, restarts or forgets processes; the network polls run outside
            s = settings.clamp() if settings is not None else self.load_settings()
            self._check_coordinator()
            self._check_inference()
            self._check_agent()
            self._finish_pending()
            agent = self.paths.read_status()
            agent_pid = self._agent_pid()
            agent_running = agent_pid is not None
            if agent_running and self.paths.read_pid() != agent_pid:
                agent = {}   # spawned moments ago: status.json is still the last run's
            draining = agent_running and (
                bool(agent.get("draining")) or self._pending_agent is not None
                or self._agent_stop_pid == agent_pid)
            inference_pid = self.read_inference_pid()
            inference_status = self.inference_status() if inference_pid is not None else {}
            coordinator_pid = self.read_coordinator_pid()
        url = self.coordinator_url(s)
        health = self.poll_health(self.proxy_url(s)) if s.mode == "host" else None
        health = health or self.poll_health(url) or {}
        mem_total, mem_free = self._memory()
        with self._lock:
            if health and self.last_error.startswith(UNREACHABLE_ERRORS):
                self.last_error = ""   # the coordinator answers now (e.g. after Connect fixed the address)
            last_error = self.last_error
        return StatusSnapshot(
            coordinator_up=bool(health),
            nodes=health_count(health, "nodes"),
            jobs=health_count(health, "jobs"),
            agent_running=agent_running,
            agent_status=str(agent.get("status", "") or ""),
            agent_job_id=agent.get("job_id"),
            agent_fetch_done_bytes=agent.get("fetch_done_bytes") if agent_running else None,
            agent_fetch_total_bytes=agent.get("fetch_total_bytes") if agent_running else None,
            agent_draining=draining,
            agent_gpu_percent=agent_gpu_percent(agent) if agent_running else None,
            coordinator_pid=coordinator_pid,
            agent_pid=agent_pid,
            lan_ip=self._lan_ip(),
            last_error=last_error,
            inference_running=inference_pid is not None,
            inference_pid=inference_pid,
            inference_status=inference_status,
            inference_nodes=health_count(health, "inference_nodes"),
            inference_transport=str(health.get("inference_transport", "") or ""),
            inference_supported=supports_inference(health),
            memory_total_bytes=mem_total,
            memory_available_bytes=mem_free,
        )

    def _check_agent(self) -> None:
        """Reap the training agent we spawned once it exits (a zombie until then, and the
        coordinators and nodes we stopped along with it)."""
        self._stopping = [p for p in self._stopping if p.poll() is None]
        if self._agent_proc is not None and self._agent_proc.poll() is not None:
            self._agent_proc = None
        self._force_stop_hung_agent()

    def _finish_pending(self) -> None:
        """Complete a restart that waited for the old agent or LLM node to finish its step."""
        if self._pending_agent is not None and not self._agent_running():
            argv, session = self._pending_agent
            self._pending_agent = None   # one attempt: if it fails, last_error says why
            try:
                self._ensure_agent(argv, session)
            except LauncherError:
                return
            if self.last_error == AGENT_RESTARTING:
                self.last_error = ""
        if self._pending_inference is not None and self.read_inference_pid() is None:
            argv, session = self._pending_inference
            self._pending_inference = None
            try:
                self._ensure_inference(argv, session)
            except LauncherError:
                return
            if self.last_error == NODE_RESTARTING:
                self.last_error = ""

    def _read_inference_args(self) -> Any:
        try:
            return json.loads(self._inference_args_path().read_text())
        except (OSError, ValueError):
            return []

    def _agent_args_path(self) -> Path:
        return self.home / "agent.args"

    def _read_agent_args(self) -> Any:
        try:
            return json.loads(self._agent_args_path().read_text())
        except (OSError, ValueError):
            return []

    def _agent_pid(self) -> Optional[int]:
        """The pid of our running training agent: the one agent.pid names when that is ours, else
        the one we spawned and have not asked to stop while it still runs (it writes agent.pid
        only once its imports are done, seconds in which a second Start used to spawn another).
        A pid file left by one that is gone, or that now names another program, is removed."""
        pid = self.paths.read_pid()
        if pid is not None and self._owned_process(pid, AGENT_MODULE):
            return pid
        if pid is not None:
            self.paths.clear_pid()
            if self._agent_stop_pid == pid:
                self._agent_stop_pid = self._agent_stop_at = None
        proc = self._agent_proc
        return int(proc.pid) if proc is not None and proc.poll() is None else None

    def _agent_running(self) -> bool:
        return self._agent_pid() is not None

    def _force_stop_hung_agent(self) -> None:
        """An agent asked to stop drains its step for the grace period (daemon.shutdown waits that
        long) and then exits; one still alive a margin later is hung and is killed, so the shell
        does not say "Stopping…" for ever. Only our own agent is signalled (_agent_pid vouches for
        the pid), and a restart that waited on it is dropped: last_error says what happened."""
        pid, asked = self._agent_stop_pid, self._agent_stop_at
        if pid is None or asked is None:
            return
        if self._agent_pid() != pid:                  # it is gone (or another agent runs now)
            self._agent_stop_pid = self._agent_stop_at = None
            return
        limit = self.cfg.grace_period_s + AGENT_STOP_MARGIN_S
        if time.monotonic() - asked < limit:
            return
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)
        self._agent_stop_pid = self._agent_stop_at = None
        self._pending_agent = None
        self.last_error = AGENT_FORCE_STOPPED.format(seconds=limit)

    def _spawn(self, argv: list[str], log_path: Path,
               pid_writer: Optional[Callable[[int], None]] = None, session: Optional[str] = None) -> Any:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        if session is not None:   # through the environment: argv is visible to every user in ps
            env.pop("SLASHCOMPUTE_SESSION", None)
            if session:
                env["SLASHCOMPUTE_SESSION"] = session
        # Append, never truncate: a failed start's reason must survive the next attempt. Unbuffered,
        # so a child that dies at once still leaves its last words in the log.
        with open(log_path, "ab") as fh:
            try:
                proc = self._popen(
                    argv, stdout=fh, stderr=subprocess.STDOUT,
                    start_new_session=True, env=env,
                )
            except OSError as e:
                self.last_error = f"Could not start process: {e}"
                raise LauncherError(self.last_error) from e
        if pid_writer is not None:
            pid_writer(int(proc.pid))
        return proc
