import asyncio
from pathlib import Path

import pytest

from slashcompute.agent.paths import resolve_data_host
from slashcompute.agent.throttle import sleep_s
from slashcompute.transport import Frame, LinkServer, TcpLink


def test_sleep_s_duty_cycle():
    assert sleep_s(2.0, 100) == 0.0
    assert sleep_s(2.0, 50) == pytest.approx(2.0)
    assert sleep_s(1.0, 25) == pytest.approx(3.0)
    assert sleep_s(1.0, 0) == 3600.0


def test_resolve_data_host_localhost():
    assert resolve_data_host(True) == "127.0.0.1"
    assert resolve_data_host(False) != ""


def test_benchmark_profile_shape():
    from slashcompute.agent.benchmark import benchmark
    from slashcompute.common.protocol import DeviceProfile

    d = benchmark(max_memory_bytes=512 * 1024 * 1024)
    assert isinstance(d, DeviceProfile)
    assert d.memory_contrib_bytes <= 512 * 1024 * 1024
    assert d.matmul_tflops > 0 and d.mem_bandwidth_gbps > 0
    assert d.chip


async def test_hello_reject_does_not_leak_expect():
    server = await LinkServer("127.0.0.1", 0, {"job_id": "secret", "epoch": 9}).start()
    reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
    link = TcpLink(reader, writer)
    await link.send(Frame("hello", {"job_id": "nope", "epoch": 0}))
    ack = await link.recv(5)
    assert ack.kind == "hello_reject"
    assert "expect" not in ack.meta and "secret" not in str(ack.meta)
    assert "job_id" in ack.meta["reason"]               # says what's wrong, not the value
    await link.close()
    await server.close()


def test_agent_options_localhost_host(tmp_path):
    from slashcompute.agent.daemon import AgentOptions

    opt = AgentOptions(url="http://127.0.0.1:8765", home=tmp_path, localhost=True)
    assert opt.data_host == "127.0.0.1"
    assert opt.node_id == AgentOptions(url="http://127.0.0.1:8765", home=tmp_path).node_id


def test_claim_data_port_skips_a_busy_port():
    import socket

    from slashcompute.agent.paths import claim_data_port

    busy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    busy.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    busy.bind(("127.0.0.1", 0))
    port = busy.getsockname()[1]
    try:
        claimed = claim_data_port(port, host="127.0.0.1")
        assert claimed != port
        assert 1024 <= claimed <= 65535
    finally:
        busy.close()


# ------------------------------------------------------------ coordinator outages


def _fake_profile(_max_bytes=None):
    from slashcompute.common.protocol import DeviceProfile

    return DeviceProfile(chip="fake", memory_total_bytes=16 << 30, memory_available_bytes=8 << 30,
                         working_set_bytes=12 << 30, memory_contrib_bytes=8 << 30, matmul_tflops=1.0,
                         mem_bandwidth_gbps=100.0)


async def _fake_coordinator(on_register):
    """A /ws/agent endpoint that calls ``on_register(ws, n)`` for the n-th registration."""
    import websockets

    count = {"n": 0}

    async def handler(ws):
        await ws.recv()                                    # Register
        count["n"] += 1
        await on_register(ws, count["n"])

    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, f"http://127.0.0.1:{port}", count


async def test_agent_reconnects_after_the_coordinator_restarts(tmp_path, monkeypatch):
    """A coordinator restart (close 1012) used to end the agent with a traceback; it must rejoin."""
    from slashcompute.agent.daemon import AgentOptions, Daemon
    from slashcompute.common.protocol import Welcome, dump

    monkeypatch.setattr("slashcompute.agent.daemon.benchmark", _fake_profile)
    rejoined = asyncio.Event()

    async def on_register(ws, n):
        await ws.send(dump(Welcome(node_id="x", heartbeat_interval_s=30)))
        if n == 1:
            await ws.close(code=1012, reason="service restart")
            return
        rejoined.set()
        await ws.wait_closed()

    server, url, count = await _fake_coordinator(on_register)
    daemon = Daemon(AgentOptions(url=url, home=tmp_path, localhost=True))
    running = asyncio.create_task(daemon.run())
    try:
        await asyncio.wait_for(rejoined.wait(), 10)
        assert count["n"] == 2 and not running.done()
    finally:
        await daemon.shutdown()
        await asyncio.wait_for(running, 5)
        server.close()
    assert daemon.status == "stopped"


async def test_agent_stops_when_the_coordinator_refuses_it(tmp_path, monkeypatch):
    from slashcompute.agent.daemon import AgentOptions, Daemon

    monkeypatch.setattr("slashcompute.agent.daemon.benchmark", _fake_profile)

    async def on_register(ws, n):
        await ws.close(code=4003, reason="banned")

    server, url, count = await _fake_coordinator(on_register)
    daemon = Daemon(AgentOptions(url=url, home=tmp_path, localhost=True))
    try:
        with pytest.raises(SystemExit, match="banned"):
            await asyncio.wait_for(daemon.run(), 10)
    finally:
        server.close()
    assert count["n"] == 1                                 # no retry loop against a refusal


async def test_agent_survives_a_message_handler_raising_an_http_error(tmp_path, monkeypatch):
    """A blob upload/download failing (httpx error, not OSError) used to kill the daemon
    outright; it must not even cost the session (a reconnect would drop a running stage)."""
    import httpx

    from slashcompute.agent.daemon import AgentOptions, Daemon
    from slashcompute.common.protocol import Drain, Welcome, dump

    monkeypatch.setattr("slashcompute.agent.daemon.benchmark", _fake_profile)
    handled = []
    second = asyncio.Event()

    async def on_register(ws, n):
        await ws.send(dump(Welcome(node_id="x", heartbeat_interval_s=30)))
        await ws.send(dump(Drain(job_id="j", epoch=1)))
        await ws.send(dump(Drain(job_id="j", epoch=2)))
        await ws.wait_closed()

    async def boom(msg):
        handled.append(msg.epoch)
        if msg.epoch == 1:
            raise httpx.ConnectError("coordinator blob store unreachable")
        second.set()

    server, url, count = await _fake_coordinator(on_register)
    daemon = Daemon(AgentOptions(url=url, home=tmp_path, localhost=True))
    monkeypatch.setattr(daemon, "_handle", boom)
    running = asyncio.create_task(daemon.run())
    try:
        await asyncio.wait_for(second.wait(), 10)
        assert handled == [1, 2] and count["n"] == 1 and not running.done()
    finally:
        await daemon.shutdown()
        await asyncio.wait_for(running, 5)
        server.close()


# ------------------------------------------------------------ verification


class _HeldBundle:
    def save_bundle(self, step, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"bundle")
        return True


def _assignment(job_id="j", epoch=1, stage_idx=0):
    from slashcompute.common.protocol import LoraFinetuneSpec, StageAssignment

    return StageAssignment(
        job_id=job_id, epoch=epoch, stage_idx=stage_idx, num_stages=1, layer_start=0, layer_end=8,
        num_layers=8, spec=LoraFinetuneSpec(dataset_path="dataset.jsonl", steps=2),
        checkpoint_every=25, verify_ring_size=8,
    )


def _held_stage(job_id="j", epoch=1):
    from slashcompute.agent.daemon import _Stage

    return _Stage(_assignment(job_id, epoch), session=_HeldBundle())


def _verify_daemon(tmp_path):
    from slashcompute.agent.daemon import AgentOptions, Daemon

    daemon = Daemon(AgentOptions(url="http://127.0.0.1:9930", home=tmp_path, localhost=True))
    sent = []

    async def send(msg):
        sent.append(msg)

    daemon.send = send
    return daemon, sent


@pytest.mark.parametrize("status", [None, 413])
async def test_failed_bundle_upload_is_reported_not_raised(tmp_path, status):
    """The stage must answer VerifyFetch with an error instead of crashing on a failed upload."""
    import httpx

    from slashcompute.common.protocol import VerifyBundleReady, VerifyFetch

    daemon, sent = _verify_daemon(tmp_path)
    daemon._stage = _held_stage()

    def put_bytes(path, data, params=None):
        req = httpx.Request("POST", "http://127.0.0.1:9931" + path)
        if status is None:
            raise httpx.ConnectError("connection refused", request=req)
        httpx.Response(status, request=req).raise_for_status()

    daemon.opt.http.put_bytes = put_bytes
    await daemon._on_fetch(VerifyFetch(verify_id="v1", job_id="j", epoch=1, stage_idx=0, step=3))
    [reply] = sent
    assert isinstance(reply, VerifyBundleReady) and reply.verify_id == "v1"
    assert reply.path is None and "upload failed" in reply.error


async def test_replay_does_not_block_the_event_loop(tmp_path, monkeypatch):
    """A replay longer than the heartbeat timeout used to starve heartbeats and get the verifier dropped."""
    import time

    from slashcompute.common.protocol import VerifyRequest, VerifyResult

    daemon, sent = _verify_daemon(tmp_path)
    daemon.opt.http.get_file = lambda path, dest: (dest.write_bytes(b"bundle"), dest)[1]
    daemon.opt.http.put_bytes = lambda *a, **k: None

    def slow_replay(req, bundle, dest):
        time.sleep(0.5)
        dest.write_bytes(b"out")
        return {"wall_s": 0.5}

    monkeypatch.setattr("slashcompute.agent.daemon.run_replay", slow_replay)
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.05)
            ticks += 1

    t = asyncio.create_task(ticker())
    try:
        await daemon._on_verify(VerifyRequest(verify_id="v2", kind="replay", model="m",
                                              bundle_url="/verify/v2/bundle"))
    finally:
        t.cancel()
    assert ticks >= 5                                     # the loop kept running during the replay
    [reply] = sent
    assert isinstance(reply, VerifyResult) and reply.error is None and reply.stats == {"wall_s": 0.5}


def test_chosen_memory_is_lent_even_above_what_is_free_but_not_past_the_working_set(monkeypatch):
    from slashcompute.agent import benchmark as bm

    monkeypatch.setattr(bm, "_memory", lambda: (16 << 30, 5 << 30))   # 16 GB Mac, 5 GB free
    monkeypatch.setattr(bm, "_matmul_tflops", lambda: 1.0)
    monkeypatch.setattr(bm, "_mem_bandwidth_gbps", lambda: 100.0)
    assert bm.benchmark().memory_contrib_bytes == 5 << 30            # automatic: what is free
    assert bm.benchmark(10 << 30).memory_contrib_bytes == 10 << 30   # the owner's choice
    assert bm.benchmark(3 << 30).memory_contrib_bytes == 3 << 30
    assert bm.benchmark(15 << 30).memory_contrib_bytes == 12 << 30   # capped at 75% of RAM


def test_sandbox_profile_ships_inside_the_package():
    import slashcompute.agent as agent_pkg
    from slashcompute.agent.sandbox import profile_path

    profile = profile_path()
    assert profile.is_file()
    assert profile.parent == Path(agent_pkg.__file__).resolve().parent
    assert "(deny default)" in profile.read_text()


def test_wrap_command_fails_closed_without_profile(monkeypatch, tmp_path):
    from slashcompute.agent import sandbox

    monkeypatch.setattr(sandbox.shutil, "which", lambda _: "/usr/bin/sandbox-exec")
    monkeypatch.setattr(sandbox, "profile_path", lambda: tmp_path / "missing.sb")
    with pytest.raises(RuntimeError, match="unsandboxed"):
        sandbox.wrap_command(["python", "-m", "worker"], tmp_path, tmp_path)


def test_wrap_command_fails_closed_without_sandbox_exec(monkeypatch, tmp_path):
    from slashcompute.agent import sandbox

    monkeypatch.setattr(sandbox.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="unsandboxed"):
        sandbox.wrap_command(["python", "-m", "worker"], tmp_path, tmp_path)


def test_wrap_command_prefixes_sandbox_exec(monkeypatch, tmp_path):
    from slashcompute.agent import sandbox

    monkeypatch.setattr(sandbox.shutil, "which", lambda _: "/usr/bin/sandbox-exec")
    cmd = sandbox.wrap_command(["python", "-m", "worker"], tmp_path / "job", tmp_path)
    assert cmd[:3] == ["/usr/bin/sandbox-exec", "-f", str(sandbox.profile_path())]
    assert cmd[-4:] == ["--", "python", "-m", "worker"]


async def test_bad_dataset_reports_a_fatal_stage_error(tmp_path, tiny_model):
    from slashcompute.agent.worker import WorkerContext, run_stage
    from slashcompute.common.protocol import StageAssignment, StageFinished
    from slashcompute.jobs import LoraFinetuneSpec

    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"nope": 1}\n')

    class Http:
        def get_file(self, url, dest):
            dest.write_bytes(bad.read_bytes())
            return dest

    spec = LoraFinetuneSpec(model=str(tiny_model), dataset_path=str(bad), steps=2, batch_size=2,
                            microbatches=1, lora_rank=4)
    asg = StageAssignment(job_id="j", epoch=1, stage_idx=0, num_stages=1, layer_start=0,
                          layer_end=6, num_layers=6, spec=spec, dataset_url="/jobs/j/dataset",
                          checkpoint_every=10, verify_ring_size=2)
    ctx = WorkerContext(assignment=asg, http=Http(), job_dir=tmp_path / "job", data_bind="127.0.0.1",
                        data_port=0, gpu_percent=100, node_id="n")
    sent = []

    async def emit(msg):
        sent.append(msg)

    with pytest.raises(ValueError):
        await run_stage(ctx, emit)
    assert len(sent) == 1 and isinstance(sent[0], StageFinished)
    assert sent[0].reason == "error" and sent[0].fatal
    assert "unrecognised dataset row keys" in sent[0].detail


# A sandboxed worker whose stage runs until drained: the real stdio CLI around a fake stage.


_DRAINABLE_WORKER = """
import asyncio, sys
from types import SimpleNamespace
from slashcompute.agent import worker
from slashcompute.common.protocol import StageFinished, StageReady


async def run_stage(ctx, emit):
    asg = ctx.assignment
    ctx.session.runner = SimpleNamespace(drain=asyncio.Event())
    await emit(StageReady(job_id=asg.job_id, epoch=asg.epoch, stage_idx=asg.stage_idx))
    await ctx.session.runner.drain.wait()
    await emit(StageFinished(job_id=asg.job_id, epoch=asg.epoch, stage_idx=asg.stage_idx,
                             reason="drained", last_step=7))

worker.run_stage = run_stage
worker._stdio_app()
"""


async def test_stop_drains_a_sandboxed_worker_before_disconnecting(tmp_path, monkeypatch):
    """`stop` used to drop the websocket under a sandboxed worker, so the job rolled back."""
    import json
    import os
    import sys

    import slashcompute
    from slashcompute.agent.daemon import AgentOptions, Daemon
    from slashcompute.common.protocol import LoraFinetuneSpec, StageAssignment

    src = str(Path(slashcompute.__file__).resolve().parents[1])
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [src, os.environ.get("PYTHONPATH")])))
    monkeypatch.setattr("slashcompute.agent.daemon.wrap_command",
                        lambda cmd, *_: [sys.executable, "-c", _DRAINABLE_WORKER, *cmd[3:]])
    monkeypatch.setattr("slashcompute.agent.daemon.fetch_command",
                        lambda model: [sys.executable, "-c", _INSTANT_FETCH])

    sent, closed = [], asyncio.Event()

    class FakeWS:
        async def send(self, raw):
            assert not closed.is_set(), "sent after the websocket closed"
            sent.append(json.loads(raw))

        async def close(self):
            closed.set()

    opt = AgentOptions(url="http://127.0.0.1:9920", home=tmp_path, localhost=True, sandbox=True)
    opt.cfg.grace_period_s = 20
    daemon = Daemon(opt)
    daemon._ws = FakeWS()
    asg = StageAssignment(
        job_id="j", epoch=1, stage_idx=0, num_stages=1, layer_start=0, layer_end=8,
        num_layers=8, spec=LoraFinetuneSpec(dataset_path="dataset.jsonl", steps=100),
        checkpoint_every=25, verify_ring_size=8,
    )
    stage = await daemon._start_stage(asg)
    await asyncio.wait_for(stage.starter, 30)
    proc = stage.proc
    try:
        for _ in range(200):
            if any(m["type"] == "stage_ready" for m in sent):
                break
            await asyncio.sleep(0.05)
        assert daemon.status == "running"
        await asyncio.wait_for(daemon.shutdown(), 15)
        finished = [m for m in sent if m["type"] == "stage_finished"]
        assert finished and finished[0]["reason"] == "drained" and finished[0]["last_step"] == 7
        assert closed.is_set() and proc.returncode == 0
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()


def _run_sandboxed(code, tmp_path):
    import shutil
    import subprocess
    import sys

    from slashcompute.agent import sandbox

    if shutil.which("sandbox-exec") is None:
        pytest.skip("sandbox-exec not available")
    cmd = sandbox.wrap_command([sys.executable, "-c", code], tmp_path / "job", tmp_path)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)


def test_sandboxed_worker_can_chmod_and_touch_in_user_cache_but_not_write_elsewhere(tmp_path):
    outside = Path(__file__).resolve().parent / f"sbx-probe-{tmp_path.name}.txt"
    code = f"""
import os, subprocess, tempfile


cache = subprocess.check_output(["getconf", "DARWIN_USER_CACHE_DIR"], text=True).strip()
with tempfile.TemporaryDirectory(prefix="slashcompute-sbx-", dir=cache) as d:
    p = os.path.join(d, "monolithic_metal.pcm")
    with open(p + ".tmp", "w") as f:
        f.write("x")
    os.chmod(p + ".tmp", 0o644)
    os.utime(p + ".tmp", None)
    os.rename(p + ".tmp", p)
try:
    open({str(outside)!r}, "w").close()
except PermissionError:
    print("outside denied")
"""
    try:
        proc = _run_sandboxed(code, tmp_path)
    finally:
        outside.unlink(missing_ok=True)
    assert proc.returncode == 0, proc.stderr
    assert "outside denied" in proc.stdout


def test_sandboxed_worker_can_build_a_metal_kernel_from_source(tmp_path):
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Metal not available")
    # A never-seen kernel source forces a runtime compile through MTLCompilerService.
    code = f"""
import mlx.core as mx


k = mx.fast.metal_kernel(name="sbx_probe", input_names=["inp"], output_names=["out"],
                         source="out[thread_position_in_grid.x] = inp[thread_position_in_grid.x] + 1.0f; // {tmp_path.name}")


a = mx.zeros(4)
(o,) = k(inputs=[a], grid=(4, 1, 1), threadgroup=(4, 1, 1), output_shapes=[a.shape], output_dtypes=[a.dtype])
print(o.tolist())
"""
    proc = _run_sandboxed(code, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "[1.0, 1.0, 1.0, 1.0]" in proc.stdout


async def test_sandboxed_worker_resolves_the_model_the_daemon_fetched(tmp_path, monkeypatch):
    """The worker used to snapshot_download inside the sandbox, which can't write the HF cache
    (~/.cache/huggingface locks/refs/blobs): "Operation not permitted" on a model not yet cached."""
    import json
    import os
    import shutil
    import sys
    import tempfile

    import slashcompute
    from slashcompute.agent import sandbox
    from slashcompute.agent.daemon import AgentOptions, Daemon
    from slashcompute.common.protocol import LoraFinetuneSpec, StageAssignment

    if shutil.which("sandbox-exec") is None:
        pytest.skip("sandbox-exec not available")
    # The cache must sit where the profile denies writes, like ~/.cache (tmp dirs are writable).
    home = Path(tempfile.mkdtemp(prefix=".sbx-home-", dir=Path(__file__).resolve().parent))
    hf_home = home / ".cache" / "huggingface"
    model, sha = "slashcompute-test/tiny", "0" * 40
    src = str(Path(slashcompute.__file__).resolve().parents[1])
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [src, os.environ.get("PYTHONPATH")])))
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:9720")    # nothing listens: no real network

    # The daemon's unsandboxed download (agent.fetch), writing HF's cache layout.
    fetch = f"""
import json, os
from pathlib import Path
repo = Path(os.environ["HF_HOME"]) / "hub" / "models--{model.replace('/', '--')}"
snap = repo / "snapshots" / "{sha}"
snap.mkdir(parents=True)
(snap / "config.json").write_text('{{"model_type": "tiny"}}')
(repo / "refs").mkdir()
(repo / "refs" / "main").write_text("{sha}")
print(json.dumps({{"type": "done", "path": str(snap)}}))
"""

    opt = AgentOptions(url="http://127.0.0.1:9720", home=tmp_path, localhost=True, sandbox=True)
    job_dir = opt.paths.job_dir("j", 1)
    probe = f"""
import json, os
from slashcompute.pipeline.model_profile import resolve_model_path
out = {{}}
try:
    out["config"] = json.loads((resolve_model_path({model!r}) / "config.json").read_text())
except Exception as e:
    out["error"] = f"{{type(e).__name__}}: {{e}}"
try:
    locks = os.path.join(os.environ["HF_HOME"], "hub", ".locks")
    os.makedirs(locks, exist_ok=True)
    open(os.path.join(locks, "probe.lock"), "w").close()
    out["cache"] = "writable"
except PermissionError:
    out["cache"] = "read-only"
open({str(job_dir / "probe.json")!r}, "w").write(json.dumps(out))
"""
    monkeypatch.setattr("slashcompute.agent.daemon.fetch_command",
                        lambda name: [sys.executable, "-c", fetch])
    monkeypatch.setattr("slashcompute.agent.daemon.wrap_command",
                        lambda cmd, *a: sandbox.wrap_command([sys.executable, "-c", probe], *a))
    daemon = Daemon(opt)
    asg = StageAssignment(
        job_id="j", epoch=1, stage_idx=0, num_stages=1, layer_start=0, layer_end=8,
        num_layers=8, spec=LoraFinetuneSpec(model=model, dataset_path="dataset.jsonl", steps=2),
        checkpoint_every=25, verify_ring_size=8,
    )
    try:
        stage = await daemon._start_stage(asg)
        await asyncio.wait_for(stage.starter, 60)
        await asyncio.wait_for(stage.pump, 120)
        out = json.loads((job_dir / "probe.json").read_text())
    finally:
        shutil.rmtree(home, ignore_errors=True)
    assert out.get("config") == {"model_type": "tiny"}, out.get("error")
    assert out["cache"] == "read-only"


# ------------------------------------------------------------ stage lifecycle races


_INSTANT_FETCH = 'import json; print(json.dumps({"type": "done", "path": "x"}))'
# A download that never ends: notes its pid, then reports progress until it is killed.
_SLOW_FETCH = """
import json, os, sys, time
open(sys.argv[1], "a").write(f"{os.getpid()}\\n")
for i in range(1, 600):
    print(json.dumps({"type": "progress", "done": i * 1000, "total": 10**9}), flush=True)
    time.sleep(0.1)
"""


def _patch_agent_children(tmp_path, monkeypatch, worker="import time; time.sleep(30)",
                          fetch=_INSTANT_FETCH):
    """Stand-ins for the sandboxed worker and the model download (``fetch`` is a script)."""
    import sys

    monkeypatch.setattr("slashcompute.agent.daemon.wrap_command",
                        lambda cmd, *_: [sys.executable, "-c", worker])
    monkeypatch.setattr("slashcompute.agent.daemon.fetch_command",
                        lambda model: [sys.executable, "-c", fetch, str(tmp_path / "fetch-pids")])


def _sandboxed_daemon(tmp_path, monkeypatch, worker="import time; time.sleep(30)", fetch=_INSTANT_FETCH):
    from slashcompute.agent.daemon import AgentOptions, Daemon

    _patch_agent_children(tmp_path, monkeypatch, worker, fetch)
    daemon = Daemon(AgentOptions(url="http://127.0.0.1:9940", home=tmp_path, localhost=True,
                                 sandbox=True))
    sent = []

    async def send(msg):
        sent.append(msg)

    daemon.send = send
    return daemon, sent


async def _started(daemon, asg):
    """Assign ``asg`` and wait until its worker is up (the start runs on its own)."""
    stage = await daemon._start_stage(asg)
    await asyncio.wait_for(stage.starter, 30)
    return stage


async def test_finished_worker_does_not_clobber_the_next_stage(tmp_path, monkeypatch):
    daemon, _ = _sandboxed_daemon(tmp_path, monkeypatch)
    a = await _started(daemon, _assignment(epoch=1))
    b = await _started(daemon, _assignment(epoch=2))  # replaces (and stops) epoch 1
    try:
        await asyncio.wait_for(a.pump, 10)  # epoch 1's pump ran its cleanup after epoch 2 began
        assert a.proc.returncode is not None
        assert daemon._stage is b and daemon._proc is b.proc and b.proc.returncode is None
        assert (daemon.status, daemon.job_id, daemon.epoch) == ("loading", "j", 2)
    finally:
        await daemon._cancel_stage()
    assert b.proc.returncode is not None and daemon._stage is None and daemon.status == "idle"


async def test_stale_stage_commands_are_ignored(tmp_path):
    from slashcompute.agent.daemon import AgentOptions, Daemon, _Stage
    from slashcompute.common.protocol import CancelStage, Drain, VerifyFetch

    class Session(_HeldBundle):
        drained = cancelled = False

        def request_drain(self):
            self.drained = True

        async def cancel(self):
            self.cancelled = True

    daemon = Daemon(AgentOptions(url="http://127.0.0.1:9941", home=tmp_path, localhost=True))
    sent = []

    async def send(msg):
        sent.append(msg)

    daemon.send = send
    session = Session()
    stage = daemon._stage = _Stage(_assignment(epoch=2), session=session)

    await daemon._handle(Drain(job_id="j", epoch=1))
    await daemon._handle(CancelStage(job_id="j", epoch=1))
    await daemon._handle(CancelStage(job_id="other", epoch=2))
    await daemon._on_fetch(VerifyFetch(verify_id="v", job_id="j", epoch=1, stage_idx=0, step=3))
    assert not session.drained and not session.cancelled and daemon._stage is stage
    assert sent[-1].error == "bundle no longer held"  # epoch 1's ring is not this one

    await daemon._handle(Drain(job_id="j", epoch=2))
    assert session.drained
    await daemon._handle(CancelStage(job_id="j", epoch=2))
    assert session.cancelled and daemon._stage is None


async def test_repeated_assignment_keeps_the_running_stage(tmp_path, monkeypatch):
    daemon, _ = _sandboxed_daemon(tmp_path, monkeypatch)
    first = await _started(daemon, _assignment(epoch=1))
    try:
        again = await daemon._start_stage(_assignment(epoch=1))  # e.g. replayed after a reconnect
        assert again is first and daemon._stage is first and first.proc.returncode is None
    finally:
        await daemon._cancel_stage()


async def test_worker_that_dies_silently_is_reported(tmp_path, monkeypatch):
    from slashcompute.common.protocol import StageFinished

    daemon, sent = _sandboxed_daemon(tmp_path, monkeypatch, worker="raise SystemExit(3)")
    stage = await _started(daemon, _assignment(epoch=1))
    await asyncio.wait_for(stage.pump, 10)
    [finished] = [m for m in sent if isinstance(m, StageFinished)]
    assert finished.reason == "error" and "exited with code 3" in finished.detail
    assert daemon._stage is None and daemon.status == "idle"


async def test_second_shutdown_is_a_no_op_and_blocks_new_stages(tmp_path, monkeypatch):
    from slashcompute.agent.daemon import _Stage

    daemon, _ = _sandboxed_daemon(tmp_path, monkeypatch)
    daemon.opt.cfg.grace_period_s = 0.5
    session = _HeldBundle()
    session.request_drain = lambda: None
    session.task = asyncio.create_task(asyncio.sleep(30))  # a stage that never drains

    async def cancel():
        session.task.cancel()

    session.cancel = cancel
    daemon._stage = _Stage(_assignment(epoch=1), session=session, ready=True)
    first = asyncio.create_task(daemon.shutdown())
    await asyncio.sleep(0.05)
    await asyncio.wait_for(daemon.shutdown(), 0.1)  # returns at once instead of draining again
    await daemon._start_stage(_assignment(epoch=2))
    assert daemon._stage is None or daemon._stage.asg.epoch == 1
    await asyncio.wait_for(first, 5)
    assert daemon._stage is None and session.task.cancelled()


# ------------------------------------------------------------ reliable sessions across reconnects


class _FakeSession(_HeldBundle):
    task = None
    drained = cancelled = False

    def request_drain(self):
        self.drained = True

    async def cancel(self):
        self.cancelled = True


async def _scripted_coordinator(on_register):
    """Like _fake_coordinator, but passes the parsed Register too."""
    import websockets

    from slashcompute.common.protocol import parse_agent_message

    count = {"n": 0}

    async def handler(ws):
        reg = parse_agent_message(await ws.recv())
        count["n"] += 1
        await on_register(ws, count["n"], reg)

    server = await websockets.serve(handler, "127.0.0.1", 0)
    return server, f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}", count


async def _until(cond, timeout=10.0):
    for _ in range(int(timeout / 0.01)):
        if cond():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met")


def _step(step=1):
    from slashcompute.common.protocol import StepMetrics, UsageSample

    return StepMetrics(job_id="j", epoch=1, stage_idx=0, step=step, in_digest="a", out_digest="b",
                       usage=UsageSample(flops=1.0, tokens=1, peak_mem_bytes=1, resident_mem_bytes=1,
                                         mem_byte_seconds=1.0, wall_s=1.0, busy_s=1.0))


async def _drop_then(second_welcome, tmp_path, monkeypatch, grace=30.0):
    """A coordinator that welcomes a reliable session, drops it, then answers the
    reconnect with ``second_welcome`` (None: never accepts another connection)."""
    from slashcompute.agent.daemon import AgentOptions, Daemon, _Stage
    from slashcompute.common.protocol import Heartbeat, Welcome, dump, parse_agent_message

    monkeypatch.setattr("slashcompute.agent.daemon.benchmark", _fake_profile)
    registers, got = [], []

    async def on_register(ws, n, reg):
        registers.append(reg)
        if n == 1:
            await ws.send(dump(Welcome(node_id="x", heartbeat_interval_s=30, session=True,
                                       reconnect_grace_s=grace)))
            await ws.close(code=1012, reason="network blip")
            return
        await ws.send(dump(second_welcome))
        async for raw in ws:
            m = parse_agent_message(raw)
            if not isinstance(m, Heartbeat):
                got.append(m)

    server, url, _ = await _scripted_coordinator(on_register)
    daemon = Daemon(AgentOptions(url=url, home=tmp_path, localhost=True))
    running = asyncio.create_task(daemon.run())
    await _until(lambda: registers and daemon._reliable and daemon._ws is None)
    session = _FakeSession()
    daemon._stage = _Stage(_assignment(), session=session)  # assigned in the first session
    if second_welcome is None:
        server.close()
    await daemon.send(_step())  # sent while the connection is down
    return daemon, running, server, session, registers, got


async def test_reconnect_resumes_the_session_and_replays_what_was_missed(tmp_path, monkeypatch):
    from slashcompute.common.protocol import StepMetrics, Welcome

    resumed = Welcome(node_id="x", heartbeat_interval_s=30, session=True, resumed=True, last_seq=0,
                      reconnect_grace_s=30)
    daemon, running, server, session, registers, got = await _drop_then(resumed, tmp_path, monkeypatch)
    try:
        await _until(lambda: got)
        assert isinstance(got[0], StepMetrics) and got[0].seq == 1  # replayed, not lost
        assert registers[1].session_id == registers[0].session_id
        assert daemon._stage is not None and not session.cancelled  # the stage kept running
    finally:
        await daemon.shutdown()
        await asyncio.wait_for(running, 5)
        server.close()


async def test_fresh_session_after_reconnect_releases_the_stage(tmp_path, monkeypatch):
    from slashcompute.common.protocol import Welcome

    fresh = Welcome(node_id="x", heartbeat_interval_s=30, session=True, reconnect_grace_s=30)
    daemon, running, server, session, _, got = await _drop_then(fresh, tmp_path, monkeypatch)
    try:
        await _until(lambda: daemon._stage is None)
        assert session.cancelled  # the coordinator restarted: it knows nothing of this stage
        await asyncio.sleep(0.2)
        assert got == []  # nor of the old session's messages
    finally:
        await daemon.shutdown()
        await asyncio.wait_for(running, 5)
        server.close()


async def test_stage_is_released_once_the_grace_window_passes(tmp_path, monkeypatch):
    daemon, running, server, session, _, _ = await _drop_then(None, tmp_path, monkeypatch, grace=0.2)
    try:
        await _until(lambda: daemon._stage is None, timeout=10)
        assert session.cancelled
    finally:
        await daemon.shutdown()
        await asyncio.wait_for(running, 5)


async def test_replayed_coordinator_message_is_handled_once(tmp_path, monkeypatch):
    from slashcompute.agent.daemon import AgentOptions, Daemon
    from slashcompute.common.protocol import Ack, Drain, Welcome, dump, parse_agent_message

    monkeypatch.setattr("slashcompute.agent.daemon.benchmark", _fake_profile)
    acks = []

    async def on_register(ws, n, reg):
        await ws.send(dump(Welcome(node_id="x", heartbeat_interval_s=30, session=True,
                                   reconnect_grace_s=30)))
        drain = Drain(job_id="j", epoch=1, seq=1)
        await ws.send(dump(drain))
        await ws.send(dump(drain))
        async for raw in ws:
            m = parse_agent_message(raw)
            if isinstance(m, Ack):
                acks.append(m.upto)

    server, url, _ = await _scripted_coordinator(on_register)
    daemon = Daemon(AgentOptions(url=url, home=tmp_path, localhost=True))
    handled = []

    async def handle(msg):
        handled.append(msg)

    monkeypatch.setattr(daemon, "_handle", handle)
    running = asyncio.create_task(daemon.run())
    try:
        await _until(lambda: len(acks) == 2)
        assert len(handled) == 1 and acks == [1, 1]
    finally:
        await daemon.shutdown()
        await asyncio.wait_for(running, 5)
        server.close()


async def test_checkpoint_upload_retries_connection_failures_only(tmp_path, monkeypatch):
    import httpx

    from slashcompute.agent import worker

    monkeypatch.setattr(worker, "UPLOAD_RETRY_DELAYS_S", (0.01, 0.01, 0.01))
    path = tmp_path / "ckpt.safetensors"
    path.write_bytes(b"weights")
    calls = []

    class Http:
        def put_bytes(self, url, data, params=None):
            calls.append(data)
            req = httpx.Request("POST", "http://c" + url)
            if len(calls) < 3:
                raise httpx.ConnectError("coordinator unreachable", request=req)
            if url.endswith("/stale"):
                httpx.Response(409, request=req).raise_for_status()

    await worker.upload_checkpoint(Http(), "/jobs/j/checkpoints/5", path, {"epoch": 1})
    assert calls == [b"weights"] * 3  # two blips, then through

    calls.clear()
    with pytest.raises(httpx.HTTPStatusError):  # the coordinator said no: retrying won't help
        await worker.upload_checkpoint(Http(), "/stale", path, {})
    assert len(calls) == 3


async def test_worker_that_cannot_start_is_reported_at_once(tmp_path, monkeypatch):
    from slashcompute.common.protocol import StageFinished

    daemon, sent = _sandboxed_daemon(tmp_path, monkeypatch)

    async def no_sandbox(*args, **kwargs):
        raise FileNotFoundError("sandbox-exec")

    async def fetched(stage):
        pass

    monkeypatch.setattr(daemon, "_fetch_model", fetched)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", no_sandbox)
    await _started(daemon, _assignment(epoch=1))
    [finished] = [m for m in sent if isinstance(m, StageFinished)]
    # Not a stage stuck "loading" until the coordinator's 15-minute start timeout.
    assert finished.reason == "error" and "could not start the worker" in finished.detail
    assert daemon._stage is None and daemon.status == "idle"


# ------------------------------------------------------------ a model download never blocks the agent


def _fetch_pids(tmp_path):
    f = tmp_path / "fetch-pids"
    return [int(x) for x in f.read_text().split()] if f.exists() else []


def _gone(pid):
    import os

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def _finished(sent):
    from slashcompute.common.protocol import StageFinished

    return [(m.reason, m.epoch) for m in sent if isinstance(m, StageFinished)]


async def test_cancel_during_model_fetch_returns_quickly_and_kills_the_fetch(tmp_path, monkeypatch):
    """A 7B download used to hold the agent's message loop for 10 minutes: the cancel (and
    every job assigned after it) sat unread, so the jobs showed "starting" and never ran."""
    import time

    from slashcompute.common.protocol import CancelStage

    daemon, sent = _sandboxed_daemon(tmp_path, monkeypatch, fetch=_SLOW_FETCH)
    stage = await daemon._start_stage(_assignment(epoch=1))
    await _until(lambda: stage.fetch_done)
    beat = daemon._heartbeat_msg()
    assert (beat.status, beat.phase) == ("loading", "fetching") and beat.fetch_total_bytes == 10**9
    assert daemon.opt.paths.read_status()["fetch_done_bytes"]
    [pid] = _fetch_pids(tmp_path)
    t0 = time.monotonic()
    await asyncio.wait_for(daemon._handle(CancelStage(job_id="j", epoch=1)), 10)
    assert time.monotonic() - t0 < 3
    assert _gone(pid) and stage.proc is None and stage.starter.done()
    assert _finished(sent) == [("cancelled", 1)]
    assert daemon._stage is None and daemon.status == "idle"
    assert daemon._heartbeat_msg().phase is None
    assert daemon.opt.paths.read_status()["fetch_done_bytes"] is None


async def test_cancel_right_after_assignment_never_starts_the_fetch(tmp_path, monkeypatch):
    from slashcompute.common.protocol import CancelStage

    daemon, sent = _sandboxed_daemon(tmp_path, monkeypatch, fetch=_SLOW_FETCH)
    await daemon._handle(_assignment(epoch=1))
    await daemon._handle(CancelStage(job_id="j", epoch=1))   # no chance for the start to run
    await asyncio.sleep(0.5)
    assert _fetch_pids(tmp_path) == [] and daemon._stage is None
    assert _finished(sent) == [("cancelled", 1)]


async def test_new_assignment_during_fetch_preempts_it(tmp_path, monkeypatch):
    import time

    daemon, sent = _sandboxed_daemon(tmp_path, monkeypatch, fetch=_SLOW_FETCH)
    first = await daemon._start_stage(_assignment(epoch=1))
    await _until(lambda: first.fetch_done)
    [pid] = _fetch_pids(tmp_path)
    t0 = time.monotonic()
    second = await asyncio.wait_for(daemon._start_stage(_assignment(epoch=2)), 10)
    try:
        assert time.monotonic() - t0 < 3
        assert _gone(pid) and daemon._stage is second
        assert (daemon.status, daemon.job_id, daemon.epoch) == ("loading", "j", 2)
        assert _finished(sent) == [("cancelled", 1)]
        await _until(lambda: second.fetch_done)               # the new one downloads
    finally:
        await daemon._cancel_stage()
    assert _finished(sent) == [("cancelled", 1), ("cancelled", 2)]


async def test_fetch_failure_is_reported_as_a_stage_error(tmp_path, monkeypatch):
    failing = ('import json, sys; print(json.dumps({"type": "error", '
               '"detail": "RuntimeError: CAS Client Error"})); sys.exit(1)')
    daemon, sent = _sandboxed_daemon(tmp_path, monkeypatch, fetch=failing)
    stage = await _started(daemon, _assignment(epoch=1))
    from slashcompute.common.protocol import StageFinished

    [finished] = [m for m in sent if isinstance(m, StageFinished)]
    assert finished.reason == "error" and "model fetch failed: RuntimeError: CAS Client Error" in finished.detail
    assert stage.proc is None and daemon._stage is None and daemon.status == "idle"


async def _agent_against(tmp_path, monkeypatch, on_message):
    """A real daemon.run() against a coordinator that welcomes it, assigns epoch 1 and hands
    every message it gets to ``on_message(ws, msg)``."""
    from slashcompute.agent.daemon import AgentOptions, Daemon
    from slashcompute.common.protocol import Welcome, dump, parse_agent_message

    monkeypatch.setattr("slashcompute.agent.daemon.benchmark", _fake_profile)
    _patch_agent_children(tmp_path, monkeypatch, fetch=_SLOW_FETCH)
    got = []

    async def on_register(ws, n, reg):
        await ws.send(dump(Welcome(node_id="x", heartbeat_interval_s=0.1)))
        await ws.send(dump(_assignment(epoch=1)))
        async for raw in ws:
            msg = parse_agent_message(raw)
            got.append(msg)
            await on_message(ws, msg)

    server, url, _ = await _scripted_coordinator(on_register)
    daemon = Daemon(AgentOptions(url=url, home=tmp_path, localhost=True, sandbox=True))
    return daemon, asyncio.create_task(daemon.run()), server, got


async def test_shutdown_during_fetch_exits_promptly(tmp_path, monkeypatch):
    """Stop used to wait out the download (36 minutes on Oct 3), so the app said the training
    agent was "still stopping" and would not start it again."""
    import time

    from slashcompute.common.protocol import Heartbeat, StageFinished

    async def nothing(ws, msg):
        pass

    daemon, running, server, got = await _agent_against(tmp_path, monkeypatch, nothing)
    try:
        await _until(lambda: any(isinstance(m, Heartbeat) and m.phase == "fetching"
                                 and m.fetch_done_bytes for m in got))
        [pid] = _fetch_pids(tmp_path)
        t0 = time.monotonic()
        await asyncio.wait_for(daemon.shutdown(), 10)
        await asyncio.wait_for(running, 10)
        assert time.monotonic() - t0 < 4
        assert _gone(pid)
        assert [m.reason for m in got if isinstance(m, StageFinished)] == ["cancelled"]
        assert daemon.opt.paths.read_status()["status"] == "stopped"
    finally:
        if not running.done():
            running.cancel()
        server.close()


async def test_cancel_over_the_socket_is_handled_mid_fetch(tmp_path, monkeypatch):
    import time

    from slashcompute.common.protocol import CancelStage, Heartbeat, StageFinished, dump

    cancelled_at = []

    async def cancel_once_fetching(ws, msg):
        if not cancelled_at and isinstance(msg, Heartbeat) and msg.phase == "fetching":
            cancelled_at.append(time.monotonic())
            await ws.send(dump(CancelStage(job_id="j", epoch=1)))

    daemon, running, server, got = await _agent_against(tmp_path, monkeypatch, cancel_once_fetching)
    try:
        await _until(lambda: any(isinstance(m, StageFinished) for m in got))
        assert time.monotonic() - cancelled_at[0] < 3
        [finished] = [m for m in got if isinstance(m, StageFinished)]
        assert finished.reason == "cancelled"
        after = got.index(finished)
        await _until(lambda: any(isinstance(m, Heartbeat) and m.status == "idle" and m.phase is None
                                 for m in got[after:]))
        assert all(_gone(pid) for pid in _fetch_pids(tmp_path))
    finally:
        await daemon.shutdown()
        await asyncio.wait_for(running, 10)
        server.close()


async def test_in_process_stage_runs_only_after_prefetch(tmp_path, monkeypatch):
    from slashcompute.agent.daemon import AgentOptions, Daemon
    from slashcompute.common.protocol import CancelStage

    calls = []

    async def run_stage(ctx, emit):
        calls.append(ctx.assignment.epoch)
        await asyncio.sleep(30)

    monkeypatch.setattr("slashcompute.agent.daemon.run_stage", run_stage)
    _patch_agent_children(tmp_path, monkeypatch, fetch=_SLOW_FETCH)
    daemon = Daemon(AgentOptions(url="http://127.0.0.1:9942", home=tmp_path, localhost=True,
                                 sandbox=False))
    sent = []

    async def send(msg):
        sent.append(msg)

    daemon.send = send
    stage = await daemon._start_stage(_assignment(epoch=1))
    await _until(lambda: stage.fetch_done)
    assert calls == []                                    # not on this loop, mid-download
    await daemon._handle(CancelStage(job_id="j", epoch=1))
    await asyncio.sleep(0.2)
    assert calls == [] and _finished(sent) == [("cancelled", 1)]

    _patch_agent_children(tmp_path, monkeypatch, fetch=_INSTANT_FETCH)
    await _started(daemon, _assignment(epoch=2))
    await _until(lambda: calls == [2])
    await daemon._cancel_stage()
    assert _finished(sent) == [("cancelled", 1), ("cancelled", 2)]


def test_fetch_progress_counts_complete_and_partial_files(tmp_path):
    from slashcompute.agent.fetch import downloaded_bytes

    blobs = tmp_path / "blobs"
    blobs.mkdir()
    expected = {"a": 100, "b": 50_000, "c": 7}
    (blobs / "a").write_bytes(b"x" * 100)                          # finished
    (blobs / "b.1234abcd.incomplete").write_bytes(b"x" * 4096)     # ours, half way
    (blobs / "c.0ld0ld00.incomplete").write_bytes(b"x" * 7)        # another process's
    (blobs / "zz.99999999.incomplete").write_bytes(b"x" * 4096)    # not a file we fetch
    done = downloaded_bytes(blobs, expected, ignore=frozenset({"c.0ld0ld00.incomplete"}))
    assert done == 100 + 4096
    (blobs / "b.1234abcd.incomplete").rename(blobs / "b")
    (blobs / "b").write_bytes(b"x" * 50_000)
    assert downloaded_bytes(blobs, expected, ignore=frozenset({"c.0ld0ld00.incomplete"})) == 50_100


def test_sweep_removes_only_unlocked_orphans(tmp_path, monkeypatch):
    from filelock import FileLock

    from slashcompute.agent import fetch

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path / "hub"))
    repo, locks = fetch._cache_dirs("org/model")
    (repo / "blobs").mkdir(parents=True)
    locks.mkdir(parents=True)
    orphan = repo / "blobs" / "aaa.11111111.incomplete"     # its download was killed
    live = repo / "blobs" / "bbb.22222222.incomplete"       # another process is writing it
    orphan.write_bytes(b"x")
    live.write_bytes(b"y")
    with FileLock(str(locks / "bbb.lock")):
        removed = fetch.sweep_orphans("org/model")
    assert removed == [orphan] and not orphan.exists() and live.exists()


def test_fetch_child_dies_promptly_on_sigterm(tmp_path):
    """Stopping a download that is stuck on the network is a SIGTERM, not a 40-minute wait."""
    import os
    import socket
    import subprocess
    import sys
    import time

    import slashcompute

    hub = socket.socket()
    hub.bind(("127.0.0.1", 0))
    hub.listen(16)                       # accepts connections, never answers: a stalled hub
    env = {**os.environ, "HF_HOME": str(tmp_path / "hf"),
           "HF_ENDPOINT": f"http://127.0.0.1:{hub.getsockname()[1]}",
           "PYTHONPATH": str(Path(slashcompute.__file__).resolve().parents[1])}
    for k in ("HF_HUB_OFFLINE", "HF_HUB_CACHE"):
        env.pop(k, None)
    proc = subprocess.Popen([sys.executable, "-m", "slashcompute.agent.fetch", "org/model"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    try:
        time.sleep(1.5)
        assert proc.poll() is None, proc.stderr.read().decode()
        t0 = time.monotonic()
        proc.terminate()
        assert proc.wait(5) == 143 and time.monotonic() - t0 < 2
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        hub.close()
