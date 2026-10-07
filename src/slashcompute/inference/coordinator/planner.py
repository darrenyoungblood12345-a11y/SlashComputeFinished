"""The planner: which devices hold which layers of a model.

Rules (latency mode, the MVP default):
  1. Filter out nodes that are offline, outside their hours, on the wrong build, not allowed this
     model, unreliable, or more than MAX_HOP_MS from the head. Workers must run the head's
     llama.cpp build (RPC breaks across builds).
  2. Usable memory = committed - (10% of committed + 1 GiB compute buffer) - memory already used by
     the node's other pipelines. Each layer costs its full bytes + its KV reserve at the requested
     context. The head also holds the embedding + output tensors.
  3. If one eligible head can hold everything, use the fastest such head. No split.
  4. Otherwise brute-force subsets (head + up to MAX_PIPELINE_NODES-1 workers from the fastest
     PLANNER_TOP_CANDIDATES), fill the fastest nodes first with contiguous blocks, and pick the
     lowest estimated seconds per output token:
        sum_i (active_i / total_active * out_s) / gen_score_i  +  n_devices * avg one-way delay
Memory is planned with full bytes; speed and shares with active bytes.
Device order follows llama.cpp: RPC workers first (in `--rpc` order), the head's local GPU last.
"""
from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from slashcompute.inference.config import GIB, InferenceSettings
from slashcompute.inference.coordinator.layers import ModelLayout

log = logging.getLogger(__name__)
GB = 1e9


@dataclass(frozen=True)
class NodeCandidate:
    node_id: str
    name: str
    committed_bytes: int
    reserved_bytes: int = 0             # memory already used by this node's other pipelines
    gen_score: float = 1.0              # output-token speed relative to the reference machine
    prompt_score: float = 1.0
    can_head: bool = False
    has_file: bool = False              # has this model's GGUF on disk (heads only)
    build: str = ''
    online: bool = True
    hours: tuple[tuple[int, int], ...] = ((0, 24),)   # allowed UTC hour ranges [start, end)
    allowed_models: tuple[str, ...] | None = None     # None = any model
    reliability: float = 1.0


@dataclass(frozen=True)
class PlanMember:
    node_id: str
    name: str
    role: str                 # head | worker
    position: int             # index in llama.cpp device order
    layer_start: int          # [start, end)
    layer_end: int
    full_bytes: int           # weights held (incl. embedding/output for the head)
    active_bytes: int         # bytes read per token
    kv_bytes: int             # KV reserve at the planned context
    share: float              # active_bytes / total active bytes
    gen_score: float

    @property
    def n_layers(self) -> int:
        return self.layer_end - self.layer_start

    @property
    def memory_bytes(self) -> int:
        return self.full_bytes + self.kv_bytes


@dataclass(frozen=True)
class Alternative:
    names: tuple[str, ...]
    n_devices: int
    est_tok_s: float


@dataclass(frozen=True)
class Plan:
    model: str
    ctx: int
    members: tuple[PlanMember, ...]       # llama.cpp device order (workers..., head)
    tensor_split: tuple[int, ...]          # per-device layer counts, +1 on the head (output layer)
    sec_per_token: float
    explanation: str
    alternatives: tuple[Alternative, ...] = ()

    @property
    def est_tok_s(self) -> float:
        return 1.0 / self.sec_per_token if self.sec_per_token > 0 else 0.0

    @property
    def head(self) -> PlanMember:
        return next(m for m in self.members if m.role == 'head')

    @property
    def workers(self) -> tuple[PlanMember, ...]:
        return tuple(m for m in self.members if m.role == 'worker')

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(m.node_id for m in self.members)


@dataclass(frozen=True)
class NoPlan:
    reason: str
    excluded: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------------------------
# helpers

def latency_ms(latency: dict, a: str, b: str, unknown: float) -> float:
    if a == b:
        return 0.0
    v = latency.get((a, b), latency.get((b, a)))
    return unknown if v is None else v


def in_hours(hours, now: datetime) -> bool:
    h = now.astimezone(timezone.utc).hour
    for start, end in hours:
        if start <= end and start <= h < end:
            return True
        if start > end and (h >= start or h < end):  # wraps midnight
            return True
    return False


def build_matches(node_build: str, pinned: str) -> bool:
    """An empty pin accepts any build (pipelines still require one build across members)."""
    if not pinned:
        return True
    return node_build == pinned or node_build.startswith(pinned + '-')


def usable_bytes(n: NodeCandidate, s: InferenceSettings) -> int:
    buffer = int(n.committed_bytes * s.COMPUTE_BUFFER_FRACTION) + s.COMPUTE_BUFFER_FIXED_BYTES
    return n.committed_bytes - buffer - n.reserved_bytes


def single_node_bytes(layout: ModelLayout, s: InferenceSettings, ctx: int | None = None) -> int:
    """Memory one Mac needs to run ``layout`` alone: weights plus the KV cache at ``ctx``."""
    return sum(_layer_costs(layout, ctx or s.DEFAULT_CTX)) + layout.head_full_bytes


def commit_for(need: int, s: InferenceSettings) -> int:
    """Memory a Mac must lend so that what is usable after the compute buffer holds ``need``
    (the inverse of ``usable_bytes`` for a node with nothing reserved)."""
    return math.ceil((need + s.COMPUTE_BUFFER_FIXED_BYTES) / (1 - s.COMPUTE_BUFFER_FRACTION))


def _gb(b: float) -> str:
    return f'{b / GB:.1f} GB'


def filter_nodes(nodes, model_key: str, now: datetime, s: InferenceSettings, exclude=()) -> tuple[list, dict]:
    ok, excluded = [], {}
    for n in nodes:
        reason = None
        if n.node_id in exclude:
            reason = 'excluded'
        elif not n.online:
            reason = 'offline'
        elif not in_hours(n.hours, now):
            reason = 'outside allowed hours'
        elif not build_matches(n.build, s.PINNED_LLAMA_BUILD):
            reason = f'llama.cpp build {n.build!r} != pinned {s.PINNED_LLAMA_BUILD!r}'
        elif n.allowed_models is not None and model_key not in n.allowed_models:
            reason = 'model not allowed by commitment'
        elif n.reliability < s.MIN_RELIABILITY:
            reason = f'reliability {n.reliability:.2f} too low'
        elif usable_bytes(n, s) <= 0:
            reason = 'no free committed memory'
        if reason:
            excluded[n.node_id] = reason
        else:
            ok.append(n)
    return ok, excluded


def _layer_costs(layout: ModelLayout, ctx: int) -> list[int]:
    return [l.full_bytes + l.kv_bytes(ctx) for l in layout.layers]


def _fill(order, caps, costs, remainder_idx):
    """Max-fill every device except `remainder_idx` with contiguous layers (devices before it take
    from the front, devices after it from the back); the remainder device takes what's left.
    Returns per-device [start, end) or None if it doesn't fit."""
    n = len(costs)
    ranges = [None] * len(order)
    pos = 0
    for i in range(remainder_idx):
        start, used = pos, 0
        while pos < n and used + costs[pos] <= caps[i]:
            used += costs[pos]
            pos += 1
        ranges[i] = (start, pos)
    end = n
    for i in range(len(order) - 1, remainder_idx, -1):
        stop, used = end, 0
        while end > pos and used + costs[end - 1] <= caps[i]:
            used += costs[end - 1]
            end -= 1
        ranges[i] = (end, stop)
    if sum(costs[pos:end]) > caps[remainder_idx]:
        return None
    ranges[remainder_idx] = (pos, end)
    return ranges


def _members_for(layout, ctx, order, ranges, head_id) -> tuple[PlanMember, ...]:
    total_active = layout.total_active_bytes
    out = []
    for pos, (node, (a, b)) in enumerate(zip(order, ranges)):
        layers = layout.layers[a:b]
        full = sum(l.full_bytes for l in layers)
        active = sum(l.active_bytes for l in layers)
        kv = sum(l.kv_bytes(ctx) for l in layers)
        is_head = node.node_id == head_id
        if is_head:
            full += layout.head_full_bytes
            active += layout.head_active_bytes
        out.append(PlanMember(
            node_id=node.node_id, name=node.name, role='head' if is_head else 'worker', position=pos,
            layer_start=a, layer_end=b, full_bytes=full, active_bytes=active, kv_bytes=kv,
            share=active / total_active if total_active else 1.0 / len(order), gen_score=node.gen_score,
        ))
    return tuple(out)


def sec_per_token(members, out_s: float, latency: dict, s: InferenceSettings) -> float:
    compute = sum(m.share * out_s / max(m.gen_score, 1e-6) for m in members)
    if len(members) == 1:
        return compute
    ids = [m.node_id for m in members]
    hops = [latency_ms(latency, a, b, s.UNKNOWN_LATENCY_MS) for a, b in zip(ids, ids[1:] + ids[:1])]
    return compute + len(members) * (sum(hops) / len(hops)) / 1000.0


def tensor_split_vector(members) -> tuple[int, ...]:
    return tuple(m.n_layers + (1 if m.role == 'head' else 0) for m in members)


def _validate(members, caps) -> bool:
    return all(m.memory_bytes <= c for m, c in zip(members, caps))


# ----------------------------------------------------------------------------------------------
# main entry point

def plan(layout: ModelLayout, nodes: list[NodeCandidate], out_s: float, settings: InferenceSettings,
         ctx: int | None = None, latency: dict | None = None, now: datetime | None = None,
         model_key: str | None = None, exclude=()) -> Plan | NoPlan:
    s = settings
    ctx = ctx or s.DEFAULT_CTX
    latency = latency or {}
    now = now or datetime.now(timezone.utc)
    model_key = model_key or layout.name
    eligible, excluded = filter_nodes(nodes, model_key, now, s, exclude)
    costs = _layer_costs(layout, ctx)
    need = single_node_bytes(layout, s, ctx)
    need_txt = (f'{layout.name} needs {_gb(need)} at ctx {ctx} '
                f'(weights {_gb(layout.total_full_bytes)} + KV {_gb(layout.kv_bytes(ctx))})')

    heads = [n for n in eligible if n.can_head and n.has_file]
    if not heads:
        blocked = [f'{n.name}: {excluded[n.node_id]}' for n in nodes
                   if n.can_head and n.has_file and n.node_id in excluded]
        why = f' (heads with the file were excluded: {"; ".join(blocked)})' if blocked else ''
        return NoPlan(f'no eligible head node has {model_key} on disk{why}', excluded)

    # 3. one node if possible: fastest head that holds everything
    single = [h for h in heads if usable_bytes(h, s) >= need]
    if single:
        h = max(single, key=lambda n: (n.gen_score, n.reliability, n.name))
        members = _members_for(layout, ctx, [h], [(0, layout.n_layers)], h.node_id)
        spt = sec_per_token(members, out_s, latency, s)
        others = sorted((x for x in single if x is not h), key=lambda n: -n.gen_score)[:3]
        alts = tuple(Alternative((o.name,), 1, 1 / sec_per_token(
            _members_for(layout, ctx, [o], [(0, layout.n_layers)], o.node_id), out_s, latency, s))
            for o in others)
        why = (f'{need_txt}. Fits on {h.name} alone ({_gb(usable_bytes(h, s))} usable): one device, '
               f'no network hops, ~{1 / spt:.1f} tok/s.')
        result = Plan(layout.name, ctx, members, tensor_split_vector(members), spt, why, alts)
        _log(result, excluded)
        return result

    total_usable = sum(max(usable_bytes(n, s), 0) for n in eligible)
    if total_usable < need:
        return NoPlan(f'{need_txt}; only {_gb(total_usable)} usable among {len(eligible)} eligible nodes '
                      f'({_gb(sum(n.committed_bytes for n in eligible))} committed)', excluded)

    # 4. split: as few devices as possible, fastest first, scored by estimated seconds per token
    candidates = []
    unknown = s.UNKNOWN_LATENCY_MS
    for h in sorted(heads, key=lambda n: -n.gen_score)[: s.PLANNER_TOP_CANDIDATES]:
        pool = [n for n in eligible if n.node_id != h.node_id and n.build == h.build
                and latency_ms(latency, h.node_id, n.node_id, unknown) <= s.max_hop_ms]
        pool = sorted(pool, key=lambda n: (-n.gen_score, n.name))[: s.PLANNER_TOP_CANDIDATES]
        head_cap = usable_bytes(h, s) - layout.head_full_bytes
        if head_cap < 0:
            continue  # can't even hold the embedding + output layer
        for k in range(1, s.MAX_PIPELINE_NODES):
            for workers in itertools.combinations(pool, k):
                order = list(workers) + [h]  # llama.cpp order: RPC workers first, head last
                caps = [usable_bytes(n, s) for n in workers] + [head_cap]
                slowest = min(range(len(order)), key=lambda i: (order[i].gen_score, -i))
                ranges = _fill(order, caps, costs, slowest)
                if ranges is None:
                    continue
                if any(a == b for (a, b), n in zip(ranges, order) if n is not h):
                    continue  # a worker with no layers: a smaller subset does the same job
                members = _members_for(layout, ctx, order, ranges, h.node_id)
                full_caps = [usable_bytes(n, s) for n in order]
                if not _validate(members, full_caps):
                    continue
                spt = sec_per_token(members, out_s, latency, s)
                min_rel = min(n.reliability for n in order)
                candidates.append((spt, len(order), -min_rel, tuple(n.name for n in order), members))

    far = [n.name for n in eligible
           if (others := [h for h in heads if h.node_id != n.node_id])
           and all(latency_ms(latency, h.node_id, n.node_id, unknown) > s.max_hop_ms for h in others)]
    if not candidates:
        extra = f'; too far from every head (> {s.max_hop_ms:.0f} ms): {", ".join(far)}' if far else ''
        return NoPlan(f'{need_txt}; no combination of up to {s.MAX_PIPELINE_NODES} eligible nodes fits{extra}',
                      excluded)

    candidates.sort(key=lambda c: c[:4])
    best = candidates[0]
    members = best[4]
    seen, alts = {best[3]}, []
    for c in candidates[1:]:
        if c[3] in seen:
            continue
        seen.add(c[3])
        alts.append(Alternative(c[3], c[1], 1 / c[0]))
        if len(alts) == 3:
            break
    biggest = max(heads, key=lambda n: usable_bytes(n, s))
    layout_txt = ', '.join(
        f'{m.name} ({m.role}, layers {m.layer_start}-{m.layer_end - 1}, {m.share:.0%} of work)' if m.n_layers
        else f'{m.name} ({m.role}, embedding+output only, {m.share:.0%} of work)' for m in members)
    alt_txt = '; '.join(f'{" + ".join(a.names)}: {a.est_tok_s:.1f} tok/s' for a in alts) or 'none'
    why = (f'{need_txt}. No single eligible head can hold it (largest: {biggest.name} with '
           f'{_gb(usable_bytes(biggest, s))} usable). Split over {len(members)} devices, fastest filled first: '
           f'{layout_txt}. Estimated {1 / best[0]:.1f} tok/s including {len(members)} network hops. '
           f'Alternatives considered: {alt_txt}.')
    if far:
        why += f' Too far from the head (> {s.max_hop_ms:.0f} ms): {", ".join(far)}.'
    result = Plan(layout.name, ctx, members, tensor_split_vector(members), best[0], why, tuple(alts))
    _log(result, excluded)
    return result


def _log(p: Plan, excluded: dict) -> None:
    log.info('plan %s ctx=%d -> %s est %.1f tok/s | alternatives: %s | excluded: %s',
             p.model, p.ctx, [(m.name, m.layer_start, m.layer_end) for m in p.members], p.est_tok_s,
             [(a.names, round(a.est_tok_s, 2)) for a in p.alternatives], excluded)


def reassign_for_device_order(layout: ModelLayout, ctx: int, members: tuple[PlanMember, ...],
                              node_order: list[str]) -> tuple[PlanMember, ...]:
    """Recompute contiguous ranges and shares if llama.cpp reports a different device order than
    planned. Layer counts per node stay the same."""
    by_id = {m.node_id: m for m in members}
    pos = 0
    out = []
    total_active = layout.total_active_bytes
    for i, node_id in enumerate(node_order):
        m = by_id[node_id]
        a, b = pos, pos + m.n_layers
        pos = b
        layers = layout.layers[a:b]
        full = sum(l.full_bytes for l in layers) + (layout.head_full_bytes if m.role == 'head' else 0)
        active = sum(l.active_bytes for l in layers) + (layout.head_active_bytes if m.role == 'head' else 0)
        out.append(PlanMember(m.node_id, m.name, m.role, i, a, b, full, active,
                              sum(l.kv_bytes(ctx) for l in layers), active / total_active, m.gen_score))
    return tuple(out)
