"""In-process fake inference cluster: a real coordinator (uvicorn on a random port) plus fake node
agents that use FakeEngine. Shared by the inference pipeline, integration, relay and upload tests."""

from __future__ import annotations

import asyncio
import socket
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import httpx
import uvicorn
from fastapi import FastAPI

from slashcompute.inference import PREFIX
from slashcompute.inference.config import InferenceSettings
from slashcompute.inference.coordinator import registry
from slashcompute.inference.coordinator.layers import ModelLayout
from slashcompute.inference.coordinator.service import create_inference_app
from slashcompute.inference.node.agent import Agent
from slashcompute.inference.node.config import Commitment, NodeConfig
from slashcompute.inference.node.fake_engine import FakeCluster, FakeEngine


@dataclass
class FakeNode:
    name: str
    memory_gb: float
    gen_score: float = 1.0
    prompt_score: float = 1.0
    may_be_head: bool = False
    files: tuple[str, ...] = ()
    allowed_models: tuple[str, ...] = ()
    hours: tuple[tuple[int, int], ...] = ((0, 24),)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def isolated_paths(home: str) -> dict:
    """Every NodeConfig path under ``home``: the defaults point at the real ~/.slashcompute and ~/models."""
    return dict(models_dir=f"{home}/models", download_dir=f"{home}/downloads",
                state_file=f"{home}/node_state.json", status_file=f"{home}/status.json",
                agent_status_file=f"{home}/agent/status.json")


def fast_settings(**overrides) -> InferenceSettings:
    base = dict(DB_PATH=":memory:", HEARTBEAT_SECONDS=0.05, OFFLINE_AFTER_SECONDS=0.4, TICK_SECONDS=0.2,
                LATENCY_INTERVAL_SECONDS=3600, COMMAND_LONG_POLL_SECONDS=1.0, JOB_TIMEOUT_SECONDS=15,
                PIPELINE_FORM_TIMEOUT_SECONDS=15,
                # fake clusters set their own latency matrix; relay mode is tested in test_inf_relay.py
                TRANSPORT="direct")
    return InferenceSettings(**{**base, **overrides})


@dataclass
class Harness:
    settings: InferenceSettings
    cluster: FakeCluster
    time_scale: float = 0.0
    url: str = ""                 # server root (``/v1/*`` lives here)
    app: Optional[FastAPI] = None
    server: Optional[uvicorn.Server] = None
    agents: dict[str, Agent] = field(default_factory=dict)
    ids: dict[str, str] = field(default_factory=dict)
    app_factory: Optional[Callable[[], FastAPI]] = None
    tmp: Optional[str] = None     # per-node home directories (downloads, status files)
    _task: Optional[asyncio.Task] = None

    @property
    def svc(self):
        return self.app.state.inference

    @property
    def conn(self):
        return self.svc.conn

    @property
    def node_url(self) -> str:
        return self.url + PREFIX

    async def start(self, port: Optional[int] = None) -> "Harness":
        port = port or free_port()
        self.app = self.app_factory() if self.app_factory else create_inference_app(self.settings)
        self.server = uvicorn.Server(uvicorn.Config(self.app, host="127.0.0.1", port=port, log_level="warning",
                                                    timeout_graceful_shutdown=1))
        self._task = asyncio.create_task(self.server.serve())
        while not self.server.started:
            await asyncio.sleep(0.01)
        self.url = f"http://127.0.0.1:{port}"
        return self

    def add_model(self, layout: ModelLayout) -> None:
        registry.store_layout(self.conn, layout.name, layout, "synthetic", size_bytes=layout.file_bytes)

    def node_config(self, spec: FakeNode, **kw) -> NodeConfig:
        if self.tmp is None:
            self.tmp = tempfile.mkdtemp(prefix="inf-harness-")
        extra = isolated_paths(f"{self.tmp}/{spec.name}")
        return NodeConfig(coordinator_url=self.node_url, name=spec.name, **{**extra, **kw}, commitment=Commitment(
            memory_gb=spec.memory_gb, hours=spec.hours, allowed_models=spec.allowed_models,
            may_be_head=spec.may_be_head))

    async def add_node(self, spec: FakeNode, index: int, engine_cls=FakeEngine, gguf_files=None,
                       busy_fn=None, **cfg_kw) -> Agent:
        ip = f"100.64.0.{index + 2}"
        engine = engine_cls(spec.name, self.cluster, spec.gen_score, spec.prompt_score, self.time_scale, ip=ip)
        cfg = self.node_config(spec, **cfg_kw)

        async def latency_fn(peers):
            return {p["node_id"]: self.cluster.one_way(agent.node_id, p["node_id"]) for p in peers}

        files = gguf_files if gguf_files is not None else [{"name": f, "size": 0} for f in spec.files]
        agent = Agent(cfg, engine, info={"os": "fake", "chip": f"fake-{spec.name}", "total_mem_bytes": 0},
                      build="b11160-fake", ip=ip, gguf_files=files, latency_fn=latency_fn,
                      client=httpx.AsyncClient(base_url=self.node_url, timeout=30),
                      busy_fn=busy_fn or (lambda: False))
        await agent.register()
        engine.node_id = agent.node_id  # the fake head reports its device order by node id
        await agent.start()
        self.agents[spec.name] = agent
        self.ids[spec.name] = agent.node_id
        return agent

    async def add_nodes(self, specs: list[FakeNode], latency_ms: Optional[dict[tuple[str, str], float]] = None,
                        default_ms: float = 2.0, engine_cls=FakeEngine) -> None:
        for i, spec in enumerate(specs):
            await self.add_node(spec, i, engine_cls)
        names = [s.name for s in specs]
        lat = latency_ms or {}
        for a in names:
            for b in names:
                if a != b:
                    self.cluster.latency_ms[(self.ids[a], self.ids[b])] = lat.get((a, b), lat.get((b, a), default_ms))
        # store the matrix directly (the agents would measure the same values)
        now = time.time()
        for (a, b), ms in self.cluster.latency_ms.items():
            self.conn.execute("INSERT OR REPLACE INTO latency (a, b, one_way_ms, measured_at) VALUES (?,?,?,?)",
                              (a, b, ms, now))

    async def drop(self, name: str) -> None:
        """Unplug a node: it stops heartbeating and any pipeline using it breaks."""
        self.cluster.dead.add(self.ids[name])
        await self.agents[name].stop()

    async def stop(self) -> None:
        for a in self.agents.values():
            if not a.stopped.is_set():
                await a.stop()
        for a in self.agents.values():
            await a.client.aclose()
        if self.server:
            self.server.should_exit = True
            await self._task


async def start_harness(settings: Optional[InferenceSettings] = None, time_scale: float = 0.0,
                        app_factory: Optional[Callable[[], FastAPI]] = None, tmp: Optional[str] = None) -> Harness:
    h = Harness(settings or fast_settings(), FakeCluster(), time_scale, app_factory=app_factory, tmp=tmp)
    return await h.start()


async def chat(h: Harness, model: str, content: str = "hi", max_tokens: int = 8, stream: bool = False,
               headers: Optional[dict] = None) -> httpx.Response:
    async with httpx.AsyncClient(base_url=h.url, timeout=30) as c:
        return await c.post("/v1/chat/completions", headers=headers or {}, json={
            "model": model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "stream": stream})


async def upload(h: Harness, name: str, data: bytes) -> httpx.Response:
    async with httpx.AsyncClient(base_url=h.node_url, timeout=30) as c:
        return await c.post("/models/upload", params={"name": name}, content=data)


async def wait_for(cond, timeout: float = 5.0) -> None:
    for _ in range(int(timeout / 0.05)):
        if cond():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("condition not met")
