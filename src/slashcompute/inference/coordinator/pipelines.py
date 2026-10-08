"""Pipelines: a planned group of devices that together hold one model.

Lifecycle: planned -> starting -> loading -> active -> draining -> stopped | broken
- starting: each worker agent starts its RPC server (LAN IP or 127.0.0.1 behind the relay)
- loading:  the head starts llama-server --rpc ... --tensor-split ... and streams weights out
- active:   requests for the model are routed here (one at a time, -np 1)
- draining: finish the in-flight job, then stop (idle timeout, training took the Mac, a member left, or
            the model was unloaded, stopped or removed in the LLMs tab)
- broken:   a member missed heartbeats or the head lost an RPC worker; in-flight jobs fail with a
            retryable error and the request is re-planned without that node
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import AsyncIterator, Optional

from slashcompute.inference import flops as fl
from slashcompute.inference.accounting import Accounting, NullAccounting
from slashcompute.inference.config import InferenceSettings
from slashcompute.inference.coordinator import nodes, registry
from slashcompute.inference.coordinator.bus import CommandBus, CommandFailed, JobStreams
from slashcompute.inference.coordinator.db import tx
from slashcompute.inference.coordinator.planner import NoPlan, Plan, PlanMember, plan, reassign_for_device_order

log = logging.getLogger(__name__)

TRANSITIONS: dict[str, set[str]] = {
    "planned": {"starting", "broken", "stopped"},
    "starting": {"loading", "broken", "stopped"},
    "loading": {"active", "broken", "stopped"},
    "active": {"draining", "broken"},
    "draining": {"stopped", "broken"},
    "stopped": set(),
    "broken": set(),
}
TERMINAL = ("stopped", "broken")


class IllegalTransition(RuntimeError):
    pass


class NoCapacity(RuntimeError):
    pass


class ModelUnavailable(NoCapacity):
    """The model was stopped or removed in the LLMs tab (or never was ready): a 404, not a busy pool."""
    status = 404


class PipelineFailed(RuntimeError):
    pass


class JobFailed(RuntimeError):
    def __init__(self, message: str, retryable: bool = True, status: Optional[int] = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status  # set when the head rejected the request itself (a 4xx for the client)


def check_transition(old: str, new: str) -> None:
    if new not in TRANSITIONS.get(old, set()):
        raise IllegalTransition(f"{old} -> {new}")


def transition(conn, pipeline_id: str, new: str, reason: Optional[str] = None, now: Optional[float] = None) -> str:
    now = now or time.time()
    with tx(conn):
        old = conn.execute("SELECT state FROM pipelines WHERE id=?", (pipeline_id,)).fetchone()["state"]
        check_transition(old, new)
        extra = {"active": ", active_at=?, last_used_at=?", "stopped": ", stopped_at=?",
                 "broken": ", stopped_at=?, broken_reason=?"}.get(new, "")
        args = {"active": (now, now), "stopped": (now,), "broken": (now, reason)}.get(new, ())
        conn.execute(f"UPDATE pipelines SET state=?{extra} WHERE id=?", (new, *args, pipeline_id))
    log.info("pipeline %s: %s -> %s%s", pipeline_id, old, new, f" ({reason})" if reason else "")
    return old


@dataclass
class Runtime:
    id: str
    model_id: str
    ctx: int
    members: tuple[PlanMember, ...]
    state: str = "planned"
    error: Optional[str] = None
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)   # set once stopped or broken
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    inflight: set = field(default_factory=set)
    last_used: float = field(default_factory=time.time)
    broken_at: Optional[float] = None
    prompt_tps_ema: Optional[float] = None   # for the generation time-weight

    @property
    def head(self) -> PlanMember:
        return next(m for m in self.members if m.role == "head")

    @property
    def node_ids(self) -> list[str]:
        return [m.node_id for m in self.members]


def _member_rows(pipeline_id: str, members, endpoints: Optional[dict] = None):
    endpoints = endpoints or {}
    return [(pipeline_id, m.node_id, m.position, m.role, m.layer_start, m.layer_end, m.full_bytes,
             m.active_bytes, m.kv_bytes, repr(m.share), endpoints.get(m.node_id)) for m in members]


class PipelineManager:
    def __init__(self, conn, settings: InferenceSettings, bus: CommandBus, streams: JobStreams,
                 accounting: Optional[Accounting] = None):
        self.conn = conn
        self.s = settings
        self.bus = bus
        self.streams = streams
        self.accounting = accounting or NullAccounting()
        self.runtimes: dict[str, Runtime] = {}
        self._model_locks: dict[str, asyncio.Lock] = {}
        self._tasks: set[asyncio.Task] = set()
        self.teardown_hooks: list = []   # async fn(pipeline_id), e.g. the relay closing streams

    def _spawn(self, coro) -> asyncio.Task:
        t = asyncio.create_task(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)
        return t

    # ------------------------------------------------------------ planning

    def plan_for(self, model_id: str, ctx: int, exclude=(), freeing=()) -> Plan | NoPlan:
        """Plan on the memory free now, or as if the `freeing` pipelines had already given theirs back."""
        row = registry.model_row(self.conn, model_id)
        if row is None or row["status"] != "ready":
            return NoPlan(f"model {model_id} is not ready (status {row['status'] if row else 'unknown'})")
        layout = registry.model_layout(row)
        freed = Counter()
        for r in freeing:
            for m in r.members:
                freed[m.node_id] += m.full_bytes + m.kv_bytes
        cands = [replace(c, reserved_bytes=max(0, c.reserved_bytes - freed[c.node_id]))
                 for c in nodes.candidates(self.conn, self.s, model_id)]
        return plan(layout, cands, registry.est_out_s(layout, self.s),
                    self.s, ctx=ctx, latency=nodes.latency_matrix(self.conn), model_key=model_id, exclude=exclude)

    def live(self, model_id: Optional[str] = None) -> list[Runtime]:
        return [r for r in self.runtimes.values()
                if r.state not in TERMINAL and (model_id is None or r.model_id == model_id)]

    def _check_model(self, model_id: str) -> None:
        why = registry.unavailable(registry.model_row(self.conn, model_id), model_id)
        if why:
            raise ModelUnavailable(why)

    # ------------------------------------------------------------ routing

    async def get_pipeline(self, model_id: str, ctx: int, exclude=()) -> Runtime:
        lock = self._model_locks.setdefault(model_id, asyncio.Lock())
        excluded = set(exclude)
        async with lock:
            self._check_model(model_id)  # stopped while this request waited for the lock: don't reuse a pipeline
            usable = [r for r in self.live(model_id) if r.ctx >= ctx and not excluded & set(r.node_ids)]
            rt = next((r for r in usable if r.state == "active"), None)
            if rt is None:
                rt = next((r for r in usable if r.state in ("planned", "starting", "loading")), None)
            if rt is None:
                rt = await self._create(model_id, ctx, exclude)
        try:
            await asyncio.wait_for(rt.ready.wait(), self.s.PIPELINE_FORM_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            await self.break_pipeline(rt.id, "formation timed out")
        if rt.state != "active":
            raise PipelineFailed(rt.error or f"pipeline {rt.id} is {rt.state}")
        return rt

    async def _create(self, model_id: str, ctx: int, exclude=()) -> Runtime:
        p = self.plan_for(model_id, ctx, exclude)
        deadline = time.time() + self.s.PIPELINE_FORM_TIMEOUT_SECONDS
        waiting_for = None   # logged once per change, not on every one-second pass
        while isinstance(p, NoPlan):
            self._check_model(model_id)  # Stop serving / Remove while waiting: refuse now, not at the deadline
            draining = [r for r in self.live() if r.state == "draining"]
            forming = [r for r in self.live() if r.state in ("planned", "starting", "loading")]
            # an idle pipeline of the same model is fair game too when its ctx is too small for this request
            idle = sorted((r for r in self.live() if r.state == "active" and not r.inflight
                           and (r.model_id != model_id or r.ctx < ctx)),
                          key=lambda r: r.last_used)
            if (draining or forming) and time.time() < deadline:
                # memory is held by pipelines that are about to stop (drain) or that may still fail to
                # form (e.g. planned with a node that just dropped): wait for one to settle, then re-plan
                waits = [asyncio.ensure_future(r.done.wait()) for r in draining]
                waits += [asyncio.ensure_future(r.ready.wait()) for r in forming]
                if waiting_for != (now_waiting := ({r.id for r in draining}, {r.id for r in forming})):
                    waiting_for = now_waiting
                    log.info("waiting for %d draining / %d forming pipeline(s) before planning %s",
                             len(draining), len(forming), model_id)
                try:  # at most a second: the model may be stopped meanwhile
                    await asyncio.wait(waits, timeout=max(0.1, min(1.0, deadline - time.time())),
                                       return_when=asyncio.FIRST_COMPLETED)
                finally:
                    for w in waits:
                        w.cancel()
            elif idle and not isinstance(self.plan_for(model_id, ctx, exclude, freeing=idle + draining), NoPlan):
                # evict only when the request fits once the idle pipelines are gone: one that can never
                # fit (e.g. a huge max_tokens) must not tear down everyone else's warm pipelines first
                log.info("evicting idle pipeline %s (%s) to make room for %s", idle[0].id, idle[0].model_id, model_id)
                await self.stop_pipeline(idle[0].id, "evicted for another model" if idle[0].model_id != model_id
                                         else f"evicted for a larger ctx ({ctx})")
            else:
                raise NoCapacity(p.reason)
            p = self.plan_for(model_id, ctx, exclude)
        return self.create_from_plan(model_id, p)

    def create_from_plan(self, model_id: str, p: Plan) -> Runtime:
        pid = "p-" + uuid.uuid4().hex[:8]
        now = time.time()
        with tx(self.conn):
            self.conn.execute(
                "INSERT INTO pipelines (id, model_id, ctx, state, head_node_id, plan_json, explanation, tensor_split, "
                "est_tok_s, created_at) VALUES (?,?,?,'planned',?,?,?,?,?,?)",
                (pid, model_id, p.ctx, p.head.node_id,
                 json.dumps({"members": [m.__dict__ for m in p.members],
                             "alternatives": [a.__dict__ for a in p.alternatives]}),
                 p.explanation, json.dumps(list(p.tensor_split)), p.est_tok_s, now))
            self.conn.executemany("INSERT INTO pipeline_members VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                                  _member_rows(pid, p.members))
        rt = Runtime(pid, model_id, p.ctx, p.members)
        self.runtimes[pid] = rt
        self._spawn(self._form(rt, p))
        return rt

    def _set_state(self, rt: Runtime, new: str, reason: Optional[str] = None) -> None:
        transition(self.conn, rt.id, new, reason)
        rt.state = new
        if new in TERMINAL:
            rt.done.set()

    async def _form(self, rt: Runtime, p: Plan) -> None:
        s = self.s
        row = registry.model_row(self.conn, rt.model_id)
        layout = registry.model_layout(row)
        out_s = registry.est_out_s(layout, s)
        try:
            self._set_state(rt, "starting")
            workers = [m for m in rt.members if m.role == "worker"]
            results = await asyncio.gather(*[
                self.bus.send(m.node_id, "start_worker", {
                    "pipeline_id": rt.id, "model": rt.model_id, "mem_bytes": m.memory_bytes,
                }, timeout=60) for m in workers])
            endpoints = {m.node_id: r["endpoint"] for m, r in zip(workers, results)}
            with tx(self.conn):
                for nid, ep in endpoints.items():
                    self.conn.execute("UPDATE pipeline_members SET endpoint=? WHERE pipeline_id=? AND node_id=?",
                                      (ep, rt.id, nid))
            if rt.state != "starting":
                return
            self._set_state(rt, "loading")
            head = rt.head
            spec = {
                "pipeline_id": rt.id, "model": rt.model_id, "ctx": rt.ctx, "head_layers": head.n_layers,
                "mem_bytes": head.memory_bytes,
                "workers": [{"node_id": m.node_id, "endpoint": endpoints[m.node_id], "n_layers": m.n_layers}
                            for m in workers],
                # only the FakeEngine reads this: how long tokens take on the reference machine
                "sim": {"out_s": out_s, "in_s": out_s / 30,
                        "members": [{"node_id": m.node_id, "share": m.share, "gen_score": m.gen_score}
                                    for m in rt.members]},
            }
            res = await self.bus.send(head.node_id, "start_head", spec, timeout=s.PIPELINE_FORM_TIMEOUT_SECONDS)
            order = res.get("order") or rt.node_ids
            if order != rt.node_ids:
                log.warning("pipeline %s: llama.cpp device order %s differs from plan %s; remapping layers",
                            rt.id, order, rt.node_ids)
                rt.members = reassign_for_device_order(layout, rt.ctx, rt.members, order)
                with tx(self.conn):
                    self.conn.execute("DELETE FROM pipeline_members WHERE pipeline_id=?", (rt.id,))
                    self.conn.executemany("INSERT INTO pipeline_members VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                                          _member_rows(rt.id, rt.members, endpoints))
            with tx(self.conn):
                self.conn.execute("UPDATE pipelines SET device_list=?, tensor_split=?, load_seconds=? WHERE id=?",
                                  (json.dumps(res.get("devices", [])), json.dumps(res.get("tensor_split") or
                                                                                  list(p.tensor_split)),
                                   res.get("load_seconds"), rt.id))
            if rt.state != "loading":
                return
            self._set_state(rt, "active")
            rt.last_used = time.time()
        except Exception as e:  # noqa: BLE001 - any failure while forming breaks the pipeline
            log.warning("pipeline %s formation failed: %s", rt.id, e)
            await self.break_pipeline(rt.id, f"formation failed: {e}")
        finally:
            rt.ready.set()

    # ------------------------------------------------------------ teardown

    def _stop_processes(self, rt: Runtime, skip: frozenset | set = frozenset()) -> None:
        for hook in self.teardown_hooks:
            self._spawn(hook(rt.id))
        for m in rt.members:
            if m.node_id in skip:
                continue
            self.bus.post(m.node_id, "stop_head" if m.role == "head" else "stop_worker", {"pipeline_id": rt.id})

    async def break_pipeline(self, pipeline_id: str, reason: str, dropped_node: Optional[str] = None) -> None:
        rt = self.runtimes.get(pipeline_id)
        if rt is None or rt.state in TERMINAL:
            return
        self._set_state(rt, "broken", reason)
        rt.error = reason
        rt.broken_at = time.time()
        rt.ready.set()
        for job_id in list(rt.inflight):
            self.streams.fail(job_id, f"pipeline broken: {reason}", retryable=True)
        self._stop_processes(rt, skip={dropped_node} if dropped_node else set())

    async def stop_pipeline(self, pipeline_id: str, reason: str = "idle") -> None:
        rt = self.runtimes.get(pipeline_id)
        if rt is None or rt.state in TERMINAL:
            return
        if rt.state == "active":
            self._set_state(rt, "draining", reason)
        if rt.state in ("planned", "starting", "loading"):
            await self.break_pipeline(pipeline_id, f"stopped while forming: {reason}")
            return
        async with rt.lock:  # wait for the in-flight job to finish
            if rt.state != "draining":
                return
            self._set_state(rt, "stopped", reason)
        self._stop_processes(rt)

    async def unload(self, model_id: str, reason: str) -> int:
        """Stop every live pipeline of a model: active ones drain (the reply in progress finishes),
        forming ones break. Returns how many it stopped (draining ones are already on their way)."""
        n = 0
        for rt in self.live(model_id):
            if rt.state == "active":
                self._set_state(rt, "draining", reason)  # now, so the next status poll shows it
                self._spawn(self.stop_pipeline(rt.id, reason))
            elif rt.state in ("planned", "starting", "loading"):
                await self.break_pipeline(rt.id, f"stopped while forming: {reason}")
            else:
                continue
            n += 1
        return n

    def recover(self, reason: str = "coordinator restarted") -> None:
        """At startup: pipelines a previous run left live have no runtime here and would hold their
        members' memory forever. Mark them stopped and tell the members to stop their processes."""
        live = ",".join("?" * len(nodes.LIVE_STATES))
        rows = self.conn.execute(
            f"SELECT m.pipeline_id, m.node_id, m.role FROM pipeline_members m JOIN pipelines p ON p.id = m.pipeline_id "
            f"WHERE p.state IN ({live})", nodes.LIVE_STATES).fetchall()
        with tx(self.conn):
            n = self.conn.execute(f"UPDATE pipelines SET state='stopped', stopped_at=?, broken_reason=? "
                                  f"WHERE state IN ({live})", (time.time(), reason, *nodes.LIVE_STATES)).rowcount
        if n:
            log.info("stopped %d pipeline(s) left live by a previous run (%s)", n, reason)
        for r in rows:
            self.bus.post(r["node_id"], f"stop_{r['role']}", {"pipeline_id": r["pipeline_id"]})

    def reconcile(self, node_id: str, held: list[str]) -> None:
        """A node reports the pipelines it holds: stop the ones not running here (left from before a
        restart, or broken while the node was unreachable so its stop never arrived)."""
        for pid in held:
            rt = self.runtimes.get(pid)
            if rt is not None and rt.state not in TERMINAL:
                continue
            row = self.conn.execute("SELECT role FROM pipeline_members WHERE pipeline_id=? AND node_id=?",
                                    (pid, node_id)).fetchone()
            for kind in [f"stop_{row['role']}"] if row else ["stop_head", "stop_worker"]:
                self.bus.post(node_id, kind, {"pipeline_id": pid})

    async def on_node_offline(self, node_id: str, name: str = "") -> None:
        """A node missed its heartbeats: break its pipelines and lower its reliability if it dropped
        out of a pipeline (live, or broken moments ago by the head losing it)."""
        now = time.time()
        recent = now - 2 * self.s.OFFLINE_AFTER_SECONDS
        dropped = False
        for rt in list(self.runtimes.values()):
            if node_id not in rt.node_ids:
                continue
            if rt.state not in TERMINAL:
                await self.break_pipeline(rt.id, f"node {name or node_id} missed heartbeats", dropped_node=node_id)
                dropped = True
            elif rt.state == "broken" and (rt.broken_at or 0) >= recent:
                dropped = True
        if dropped:
            with tx(self.conn):
                self.conn.execute("UPDATE nodes SET reliability = MAX(0, reliability - ?) WHERE id=?",
                                  (self.s.RELIABILITY_PENALTY, node_id))

    async def suspects(self, rt: Runtime) -> set[str]:
        """After a pipeline breaks: members that haven't heartbeated since. Waits up to the offline
        timeout so the next plan doesn't reuse a dead node."""
        since = rt.broken_at or time.time()
        deadline = since + self.s.OFFLINE_AFTER_SECONDS
        ids = rt.node_ids
        while True:
            rows = self.conn.execute(f"SELECT id, last_heartbeat FROM nodes WHERE id IN ({','.join('?' * len(ids))})",
                                     ids).fetchall()
            seen = {r["id"]: r["last_heartbeat"] or 0 for r in rows}
            pending = {n for n in ids if seen.get(n, 0) <= since}
            if not pending or time.time() >= deadline:
                return pending
            await asyncio.sleep(min(0.25, self.s.HEARTBEAT_SECONDS / 4))

    async def drain_node(self, node_id: str, reason: str) -> None:
        for rt in self.live():
            if node_id in rt.node_ids and rt.state == "active":
                self._spawn(self.stop_pipeline(rt.id, reason))

    async def tick(self, now: Optional[float] = None) -> None:
        """Tear down pipelines idle too long."""
        now = now or time.time()
        for rt in self.live():
            if rt.state == "active" and not rt.inflight and now - rt.last_used > self.s.PIPELINE_IDLE_SECONDS:
                self._spawn(self.stop_pipeline(rt.id, "idle timeout"))

    # ------------------------------------------------------------ jobs

    async def execute(self, rt: Runtime, job_id: str, body: dict) -> AsyncIterator[dict]:
        """Run one request on the pipeline's head, yielding engine events (chunk ... final)."""
        q = self.streams.open(job_id)
        try:
            async with rt.lock:
                if rt.state != "active":
                    raise JobFailed(f"pipeline {rt.id} is {rt.state}", retryable=True)
                rt.inflight.add(job_id)
                rt.last_used = time.time()
                self.conn.execute("UPDATE jobs SET state='running', pipeline_id=? WHERE id=?", (rt.id, job_id))
                self.bus.post(rt.head.node_id, "run_job", {"job_id": job_id, "pipeline_id": rt.id, "body": body})
                done = False  # the head stopped by itself (final or error)
                try:
                    while True:
                        ev = await asyncio.wait_for(q.get(), self.s.JOB_TIMEOUT_SECONDS)
                        done = ev["type"] in ("final", "error")
                        if ev["type"] == "error":
                            if ev.get("pipeline_broken"):
                                await self.break_pipeline(rt.id, ev["error"])
                            raise JobFailed(ev["error"], ev.get("retryable", True), ev.get("status"))
                        yield ev
                        if done:
                            return
                except asyncio.TimeoutError:
                    await self.break_pipeline(rt.id, "job timed out")
                    raise JobFailed("job timed out", retryable=True)
                finally:
                    if not done:  # the requester went away: stop the head generating for nobody
                        self.bus.post(rt.head.node_id, "cancel_job", {"job_id": job_id})
                    rt.inflight.discard(job_id)
                    rt.last_used = time.time()
                    self.conn.execute("UPDATE pipelines SET last_used_at=? WHERE id=?", (rt.last_used, rt.id))
        finally:
            self.streams.close(job_id)

    def gen_weight_for(self, model_id: str) -> float:
        """The generation weight a request for this model would get now (shown by /plan; reservations
        use GEN_WEIGHT_MAX, the only bound on what a request can be charged)."""
        rt = next((r for r in self.live(model_id) if r.prompt_tps_ema), None)
        if rt is None:
            return self.s.GEN_WEIGHT_DEFAULT
        row = self.conn.execute("SELECT live_tok_s FROM pipelines WHERE id=?", (rt.id,)).fetchone()
        return fl.gen_weight(rt.prompt_tps_ema, row["live_tok_s"] if row else None, self.s.GEN_WEIGHT_DEFAULT,
                             self.s.GEN_WEIGHT_MAX)

    def finish_job(self, rt: Runtime, job_id: str, final: dict, account_id: Optional[str],
                   output_head: list[str], started_at: float, state: str = "done") -> dict:
        """Credit a finished request: FLOPs per member from the head's token counts and timings."""
        s = self.s
        t = final.get("timings") or {}
        prompt_n, cache_n, predicted_n = int(t.get("prompt_n", 0)), int(t.get("cache_n", 0)), int(t.get("predicted_n", 0))
        p_tps, g_tps = fl.prompt_tps(t), fl.gen_tps(t)
        if p_tps and prompt_n >= s.GEN_WEIGHT_MIN_PROMPT:
            a = s.GEN_WEIGHT_EMA
            rt.prompt_tps_ema = p_tps if rt.prompt_tps_ema is None else (1 - a) * rt.prompt_tps_ema + a * p_tps
        w = fl.gen_weight(rt.prompt_tps_ema, g_tps, s.GEN_WEIGHT_DEFAULT, s.GEN_WEIGHT_MAX)
        mf = registry.model_flops(registry.model_row(self.conn, rt.model_id))
        per_node = fl.request_flops(mf, rt.members, cache_n, prompt_n, predicted_n, w)
        total = sum(per_node.values())
        wall_s = max(time.time() - started_at, 0.0)
        if account_id:
            try:
                self.accounting.record(account_id, per_node, prompt_n + predicted_n, wall_s)
            except Exception:  # noqa: BLE001 - never lose the reply over bookkeeping
                log.exception("recording credits for %s failed", job_id)
        with tx(self.conn):
            self.conn.execute(
                "UPDATE jobs SET state=?, prompt_n=?, cache_n=?, predicted_n=?, flops=?, gen_weight=?, tok_s=?, "
                "output_head=?, finished_at=? WHERE id=?",
                (state, prompt_n, cache_n, predicted_n, total, w, g_tps, json.dumps(output_head), time.time(), job_id))
            if g_tps and predicted_n >= s.GEN_WEIGHT_MIN_PREDICTED:  # a 1-token reply "runs" at ~1e6 tok/s
                self.conn.execute("UPDATE pipelines SET live_tok_s=? WHERE id=?", (g_tps, rt.id))
        names = nodes.node_names(self.conn)
        return {
            "job_id": job_id, "pipeline_id": rt.id, "prompt_n": prompt_n, "cache_n": cache_n,
            "predicted_n": predicted_n, "tok_s": g_tps, "prompt_tok_s": p_tps, "flops": total, "gen_weight": w,
            "members": [{"node": names.get(m.node_id, m.node_id), "node_id": m.node_id, "role": m.role,
                         "layers": [m.layer_start, m.layer_end - 1] if m.n_layers else [],
                         "share": round(m.share, 4), "flops": per_node.get(m.node_id, 0.0)}
                        for m in rt.members],
        }

    def abandon_job(self, rt: Optional[Runtime], job_id: str, account_id: Optional[str], output_head: list[str],
                    started_at: float, prompt_n: int, predicted_n: int) -> None:
        """The requester went away mid-request: credit the tokens generated so far and mark the job cancelled.
        Synchronous, so it completes even inside a cancelled task."""
        if rt is None or not predicted_n:
            self.conn.execute("UPDATE jobs SET state='cancelled', finished_at=? WHERE id=?", (time.time(), job_id))
            return
        final = {"timings": {"prompt_n": prompt_n, "predicted_n": predicted_n}}
        self.finish_job(rt, job_id, final, account_id, output_head, started_at, state="cancelled")

    async def shutdown(self) -> None:
        for t in list(self._tasks):
            t.cancel()


def estimate_prompt_tokens(body: dict) -> int:
    """Typical prompt length (~4 bytes per token): what a disconnected requester is charged for its prompt."""
    return max(8, len(json.dumps(body.get("messages", ""))) // 4)


TEMPLATE_TOKENS_PER_MESSAGE = 16   # role markers, separators, a tokenizer's leading-space token
TEMPLATE_TOKENS_FIXED = 64         # generation prompt, a template's default system prompt


def prompt_token_bound(body: dict) -> int:
    """Upper bound on the prompt's tokens (sizes the context and the credit reservation): every token of a
    byte-level BPE or byte-fallback vocabulary is at least one byte, so the UTF-8 length of the messages
    and tools (as JSON, which only adds bytes) plus the chat template's markers can't be exceeded."""
    messages = body.get("messages") or []
    text = json.dumps([messages, body.get("tools") or []], ensure_ascii=False)
    n = len(messages) if isinstance(messages, list) else 1
    return len(text.encode("utf-8", "surrogatepass")) + TEMPLATE_TOKENS_PER_MESSAGE * n + TEMPLATE_TOKENS_FIXED


def new_job(conn, model_id: str, requester_id: Optional[str], body: dict, stream: bool, attempt: int = 1,
            retry_of: Optional[str] = None) -> str:
    job_id = "job-" + uuid.uuid4().hex[:10]
    conn.execute(
        "INSERT INTO jobs (id, requester_id, model_id, state, attempt, retry_of, stream, request_json, created_at) "
        "VALUES (?,?,?,'queued',?,?,?,?,?)",
        (job_id, requester_id, model_id, attempt, retry_of, int(stream), json.dumps(body), time.time()))
    return job_id


def fail_job(conn, job_id: str, error: str, retryable: bool) -> None:
    conn.execute("UPDATE jobs SET state='failed', error=?, retryable=?, finished_at=? WHERE id=?",
                 (error, int(retryable), time.time(), job_id))


def chunk_text(ev: dict) -> str:
    try:
        return ev["data"]["choices"][0]["delta"].get("content") or ""
    except (KeyError, IndexError):
        return ""


def is_token(ev: dict) -> bool:
    """A chunk carrying generated output (not llama-server's opening role-only delta)."""
    try:
        return any(v for k, v in ev["data"]["choices"][0]["delta"].items() if k != "role")
    except (KeyError, IndexError, AttributeError):
        return False


async def serve(mgr: PipelineManager, model_id: str, body: dict, ctx: int, requester_id: Optional[str],
                stream: bool, account_id: Optional[str] = None, head_tokens: int = 32) -> AsyncIterator[dict]:
    """Route a request to a pipeline (forming one if needed) and run it; on a retryable failure
    re-plan without the failed node and retry once (if nothing was streamed to the requester yet).
    Closed or cancelled mid-request (the requester disconnected): the head is told to stop and the
    tokens generated so far are credited, before the caller settles the reservation.

    Yields: chunk events, an optional {'type': 'reset'} before a retry, then 'final' (with the
    credit summary) or 'error'."""
    retry_of = None
    exclude: set[str] = set()
    for attempt in (1, 2):
        job_id = new_job(mgr.conn, model_id, requester_id, body, stream, attempt, retry_of)
        sent = False
        head: list[str] = []
        generated = 0
        rt = None
        started = time.time()
        try:
            rt = await mgr.get_pipeline(model_id, ctx, exclude)
            async with contextlib.aclosing(mgr.execute(rt, job_id, body)) as events:
                async for ev in events:
                    if ev["type"] == "chunk":
                        if len(head) < head_tokens:
                            head.append(chunk_text(ev))
                        generated += is_token(ev)
                        sent = sent or stream
                        yield ev
                    elif ev["type"] == "final":
                        final = ev
                        summary = mgr.finish_job(rt, job_id, ev, account_id, head, started)
                        break
        except (asyncio.CancelledError, GeneratorExit):
            mgr.abandon_job(rt, job_id, account_id, head, started,
                            estimate_prompt_tokens(body) if generated else 0, generated)
            raise
        except (JobFailed, PipelineFailed, CommandFailed) as e:
            retryable = getattr(e, "retryable", True)
            fail_job(mgr.conn, job_id, str(e), retryable)
            if retryable and attempt == 1 and not sent:
                if rt is not None and rt.state == "broken":
                    exclude |= await mgr.suspects(rt)
                log.info("job %s failed (%s); re-planning without %s and retrying once", job_id, e, exclude or "-")
                retry_of = job_id
                yield {"type": "reset"}
                continue
            yield {"type": "error", "status": getattr(e, "status", None) or 503, "error": str(e),
                   "retryable": retryable, "job_id": job_id}
            return
        except NoCapacity as e:
            fail_job(mgr.conn, job_id, str(e), False)
            yield {"type": "error", "status": getattr(e, "status", None) or 503, "error": str(e), "retryable": False,
                   "job_id": job_id}
            return
        yield {"type": "final", "summary": summary, "finish_reason": final.get("finish_reason"),
               "usage": final.get("usage"), "output_head": head}
        return
