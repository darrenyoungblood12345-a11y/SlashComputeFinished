"""``python -m slashcompute.inference.node`` — host llama.cpp layers for the pool's LLMs."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import socket
from pathlib import Path
from typing import Optional

import httpx
import typer

from slashcompute.common.config import EngineConfig
from slashcompute.common.logging import setup_logging
from slashcompute.inference import PREFIX
from slashcompute.inference.node.config import Commitment, NodeConfig

app = typer.Typer(no_args_is_help=True, help="/compute inference node (llama.cpp RPC)")
GIB = 1024 ** 3


def _home(home: Optional[Path]) -> Path:
    return Path(home).expanduser() if home else EngineConfig.from_env().home


def pid_file(home: Path) -> Path:
    return home / "inference" / "node.pid"


def read_pid(home: Path) -> Optional[int]:
    try:
        pid = int(pid_file(home).read_text().strip())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def coordinator_url(url: str) -> str:
    """The inference routes live under ``/inference`` on the main coordinator."""
    url = url.rstrip("/")
    return url if url.endswith(PREFIX) else url + PREFIX


def connect_problem(e: httpx.HTTPError) -> tuple[str, str]:
    """(status reason, message) for a failed join. A 404 means the coordinator answers but predates
    LLM inference: waiting will not help until it is updated."""
    if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 404:
        return "unsupported", ("the coordinator has no LLM inference (it runs an older /compute); "
                               "update and restart it")
    return "connecting", f"coordinator not reachable yet ({e})"


async def connect(agent, shutdown: asyncio.Event, log, max_delay: float = 15.0) -> bool:
    """Reach the coordinator, retrying with backoff (it may still be starting). False if told to stop."""
    delay = 1.0
    while not shutdown.is_set():
        try:
            await agent.measure_rtt()
            await agent.register()
            agent.last_available, agent.last_reason, agent.last_error = False, "joining", ""
            agent.write_status()
            return True
        except httpx.HTTPError as e:
            reason, msg = connect_problem(e)
            # The shell shows status.json: never leave an older run's "available" there while not joined.
            agent.last_available, agent.last_reason, agent.last_error = False, reason, msg
            agent.write_status()
            log.warning("%s; retrying in %.0fs", msg, delay)
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(shutdown.wait(), delay)
        delay = min(delay * 2, max_delay)
    return False


def auto_memory_gb(total_bytes: int) -> float:
    total = (total_bytes or 16 * GIB) / GIB
    return float(max(2, round(total * 0.75 - 4)))


def build_config(url: str, home: Path, name: Optional[str], models_dir: str, memory_gb: float, head: bool,
                 probe_port: int, benchmark: str, session_token: str, inference_token: str, ip: str,
                 total_mem_bytes: int, trash_removed: bool = False) -> NodeConfig:
    from slashcompute.inference.node import binaries

    return NodeConfig(
        coordinator_url=coordinator_url(url), name=name or socket.gethostname().split(".")[0] or "mac",
        llama_server=binaries.find_llama_server(), rpc_server=binaries.find_rpc_server(),
        models_dir=models_dir, download_dir=str(home / "models"), probe_port=probe_port,
        state_file=str(home / "inference" / "node_state.json"), status_file=str(home / "inference" / "status.json"),
        agent_status_file=str(home / "agent" / "status.json"), ip=ip, benchmark=benchmark,
        session_token=session_token, inference_token=inference_token, trash_removed=trash_removed,
        commitment=Commitment(memory_gb=memory_gb or auto_memory_gb(total_mem_bytes), may_be_head=head),
    )


async def run_node(cfg: NodeConfig, home: Path, fake: bool = False) -> None:
    from slashcompute.inference.node import hardware, latency
    from slashcompute.inference.node.agent import Agent, load_state, model_folders, save_state, scan_models

    log = setup_logging("slashcompute.inference.node")
    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, shutdown.set)
    info = hardware.detect()
    if fake:
        from slashcompute.inference.node.fake_engine import FakeCluster, FakeEngine

        engine = FakeEngine("", FakeCluster(), ip=cfg.ip or "127.0.0.1")
        build = "fake"
    else:
        from slashcompute.inference.node.engine import LlamaCppRpcEngine, reap_orphans
        from slashcompute.inference.node.head import build_string

        pids = home / "inference" / "llama.pids"
        if killed := reap_orphans(pids):
            log.warning("stopped %d llama.cpp process(es) left over from a previous run: %s", len(killed), killed)
        if not hardware.supports_rpc(cfg.rpc_server):
            raise SystemExit(f"llama.cpp RPC server not found ({cfg.rpc_server}): run scripts/build_llama.sh "
                             "or set SLASHCOMPUTE_LLAMA_DIR")
        build = await build_string(cfg.llama_server)
        engine = LlamaCppRpcEngine("", cfg.llama_server, cfg.rpc_server, cfg.model_dirs, cfg.ip or "127.0.0.1",
                                   device=cfg.commitment.device or None, basket_model=cfg.basket_model,
                                   basket_ref=(cfg.basket_ref_prompt_tps, cfg.basket_ref_gen_tps))
        engine.pid_file = pids
    folders = model_folders(cfg.model_dirs)   # before the scan: a file added meanwhile is picked up after it
    files = scan_models(cfg.model_dirs)
    state = load_state(cfg.state_file)
    agent = Agent(cfg, engine, info=info, build=build, ip=cfg.ip, gguf_files=files, latency_fn=latency.measure,
                  state=state, probe_fn=latency.serve_probe, folders=folders)
    if not await connect(agent, shutdown, log):
        return
    engine.node_id = agent.node_id
    log.info("%s on %s, llama.cpp %s, %s transport (RTT %.0f ms), commits %.0f GiB, %d model(s) on disk",
             cfg.name, info["chip"], build, agent.transport, agent.rtt_ms or 0, cfg.commitment.memory_gb, len(files))
    bench_key = f"{build}|{info['chip']}"
    bench = state.get("benchmark") if state.get("benchmark_key") == bench_key else None
    basket_present = any((Path(d).expanduser() / cfg.basket_model).exists() for d in cfg.model_dirs)
    if bench is None and (fake or cfg.benchmark == "skip" or not basket_present):
        bench = {"prompt_score": 1.0 if fake else cfg.assumed_prompt_score,
                 "gen_score": 1.0 if fake else cfg.assumed_gen_score, "skipped": True}
    if bench is None:
        log.info("running join benchmark on %s ...", cfg.basket_model)
        bench = await engine.benchmark()
        state.update(benchmark=bench, benchmark_key=bench_key)
    save_state(cfg.state_file, agent.state)
    await agent.start(bench)
    agent.write_status()
    try:
        await shutdown.wait()
        log.info("shutting down: leaving the network and stopping llama.cpp processes")
    finally:
        await agent.leave()
        await agent.stop()
        status = Path(cfg.status_file).expanduser()
        with contextlib.suppress(OSError, ValueError):
            data = json.loads(status.read_text())
            data.update(available=False, reason="stopped", pipelines=[])
            status.write_text(json.dumps(data))


@app.command()
def start(
    url: str = typer.Option(..., help="Coordinator URL, e.g. http://192.168.1.10:8765"),
    home: Optional[Path] = typer.Option(None, help="State directory (default ~/.slashcompute)"),
    name: Optional[str] = typer.Option(None, help="Display name (default: hostname)"),
    models_dir: str = typer.Option("~/models", help="Folder with GGUF models"),
    memory_gb: float = typer.Option(0.0, help="GiB of memory lent to inference (0 = 75% of RAM minus 4)"),
    head: bool = typer.Option(True, "--head/--no-head", help="May run llama-server for models on its disk"),
    probe_port: int = typer.Option(0, help="Latency probe port for direct transport (0 = any)"),
    benchmark: str = typer.Option("auto", help="auto: benchmark once if the basket model is on disk; skip"),
    session_token: Optional[str] = typer.Option(None, envvar="SLASHCOMPUTE_SESSION",
                                                help="Account session so this node earns credits"),
    inference_token: Optional[str] = typer.Option(None, envvar="SLASHCOMPUTE_INF_TOKEN",
                                                  help="Shared secret if the coordinator requires one"),
    localhost: bool = typer.Option(False, help="Advertise 127.0.0.1 (same-machine cluster)"),
    trash_removed: bool = typer.Option(False, "--trash-removed/--keep-removed",
                                       help="When the pool removes a model, move a copy it did not send here (one "
                                            "in --models-dir, for example) to the Trash; the copies it sent are "
                                            "always deleted"),
    fake: bool = typer.Option(False, hidden=True, help="Simulated engine (demos and tests)"),
):
    """Join the pool's LLM network. Runs in the foreground; SIGTERM leaves gracefully."""
    from slashcompute.common.discovery import lan_ip
    from slashcompute.inference.node import hardware

    h = _home(home)
    # The launcher records this process's pid as it spawns it: only another live pid means a node runs.
    if (pid := read_pid(h)) is not None and pid != os.getpid():
        typer.echo(f"inference node already running (pid {pid})", err=True)
        raise typer.Exit(1)
    info_mem = hardware.detect().get("total_mem_bytes") or 0
    cfg = build_config(url, h, name, models_dir, memory_gb, head, probe_port, benchmark, session_token or "",
                       inference_token or "", "127.0.0.1" if localhost else lan_ip(), info_mem, trash_removed)
    pf = pid_file(h)
    pf.parent.mkdir(parents=True, exist_ok=True)
    pf.write_text(f"{os.getpid()}\n")
    try:
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(run_node(cfg, h, fake=fake))
    finally:
        with contextlib.suppress(FileNotFoundError):
            pf.unlink()


@app.command()
def stop(home: Optional[Path] = typer.Option(None)):
    """Ask a running inference node to drain and exit."""
    pid = read_pid(_home(home))
    if pid is None:
        typer.echo("no running inference node", err=True)
        raise typer.Exit(1)
    os.kill(pid, signal.SIGTERM)
    typer.echo(f"sent SIGTERM to pid {pid}")


@app.command()
def status(home: Optional[Path] = typer.Option(None)):
    """Show the last-known inference node status."""
    h = _home(home)
    try:
        data = json.loads((h / "inference" / "status.json").read_text())
    except (OSError, ValueError):
        data = {}
    data["running"] = read_pid(h) is not None
    typer.echo(json.dumps(data, indent=2))


if __name__ == "__main__":
    app()
