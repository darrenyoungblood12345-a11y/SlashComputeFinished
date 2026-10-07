import random
from datetime import datetime, timezone

import pytest

from slashcompute.inference.devices import layer_ranges_from_vector, parse_list_devices, simulate_llamacpp_assignment, tensor_split_for
from slashcompute.inference.config import GIB, InferenceSettings
from slashcompute.inference.coordinator.layers import synthetic_layout
from slashcompute.inference.coordinator.planner import NodeCandidate, NoPlan, Plan, in_hours, plan, reassign_for_device_order, usable_bytes

GB = 10 ** 9
S = InferenceSettings(TRANSPORT='direct')  # the 30 ms direct-mode hop limit; relay mode tested below
OUT_S = 0.0335  # reference seconds per output token (Qwen3.8-27B set value)
NOON = datetime(2026, 10, 3, 12, tzinfo=timezone.utc)


def node(node_id, gib, gen=1.0, head=False, file=False, **kw):
    return NodeCandidate(node_id=node_id, name=node_id, committed_bytes=int(gib * GIB), gen_score=gen,
                         can_head=head, has_file=file, build='b11160-a3c12db', **kw)


def run(layout, nodes, **kw):
    kw.setdefault('now', NOON)
    return plan(layout, nodes, OUT_S, kw.pop('settings', S), **kw)


def all_pairs(nodes, ms):
    return {(a.node_id, b.node_id): ms for a in nodes for b in nodes if a.node_id != b.node_id}


def assert_valid(p: Plan, layout, nodes):
    by_id = {n.node_id: n for n in nodes}
    assert abs(sum(m.share for m in p.members) - 1.0) < 1e-9
    # contiguous, complete, in device order
    assert p.members[0].layer_start == 0 and p.members[-1].layer_end == layout.n_layers
    for a, b in zip(p.members, p.members[1:]):
        assert a.layer_end == b.layer_start
    for m in p.members:
        assert m.memory_bytes <= usable_bytes(by_id[m.node_id], S)
    assert sum(m.full_bytes for m in p.members) == layout.total_full_bytes
    assert p.members[-1].role == 'head'
    # the --tensor-split vector reproduces the planned ranges under llama.cpp's assignment
    ranges = layer_ranges_from_vector(list(p.tensor_split), layout.n_layers)
    for m, (a, b) in zip(p.members, ranges):
        if m.n_layers:
            assert (m.layer_start, m.layer_end) == (a, b)


# --- required tests -----------------------------------------------------------------------------

def test_model_that_fits_on_one_node_gets_a_one_node_plan():
    layout = synthetic_layout('qwen27b', n_layers=64, total_bytes=int(17.56 * GB), head_bytes=1 * GB)
    nodes = [node('studio', 64, gen=1.0, head=True, file=True),
             node('fast', 32, gen=1.5, head=True, file=True),
             node('big-worker', 128, gen=3.0)]
    p = run(layout, nodes)
    assert isinstance(p, Plan)
    assert [m.name for m in p.members] == ['fast']  # fastest eligible head that fits
    assert p.tensor_split == (65,)
    assert p.members[0].share == 1.0
    assert 'alone' in p.explanation
    assert_valid(p, layout, nodes)


def test_42gb_80_layer_model_splits_about_50_50_over_two_24gb_nodes_with_kv_reserved():
    layout = synthetic_layout('llama70b', n_layers=80, total_bytes=int(42.5 * GB), head_bytes=1 * GB,
                              kv_bytes_per_token_per_layer=4096)
    nodes = [node('a', 24, head=True, file=True), node('b', 24, head=True)]
    p = run(layout, nodes, ctx=4096)
    assert isinstance(p, Plan), getattr(p, 'reason', '')
    assert len(p.members) == 2
    assert p.head.node_id == 'a'
    counts = [m.n_layers for m in p.members]
    assert sum(counts) == 80 and min(counts) >= 37
    assert all(0.45 <= m.share <= 0.55 for m in p.members)
    assert all(m.kv_bytes == m.n_layers * 4096 * 4096 for m in p.members)
    assert_valid(p, layout, nodes)


def _cost_per_layer(layout, ctx=4096):
    l = layout.layers[0]
    return l.full_bytes + l.kv_bytes(ctx)


def test_fastest_nodes_are_filled_first():
    layout = synthetic_layout('m55', n_layers=100, total_bytes=55 * GB, head_bytes=1 * GB)
    nodes = [node('head', 24, gen=1.0, head=True, file=True), node('fast', 24, gen=2.0), node('slow', 24, gen=0.5)]
    p = run(layout, nodes, latency=all_pairs(nodes, 1.0))
    assert isinstance(p, Plan) and len(p.members) == 3
    by = {m.name: m for m in p.members}
    cost = _cost_per_layer(layout)
    assert by['fast'].n_layers == usable_bytes(nodes[1], S) // cost  # filled to capacity
    assert by['head'].n_layers == (usable_bytes(nodes[0], S) - layout.head_full_bytes) // cost
    assert by['slow'].n_layers < usable_bytes(nodes[2], S) // cost  # gets the remainder
    assert by['slow'].n_layers == 100 - by['fast'].n_layers - by['head'].n_layers
    assert_valid(p, layout, nodes)


def test_moe_plans_memory_with_full_bytes_and_shares_with_active_bytes():
    layout = synthetic_layout('moe', n_layers=48, total_bytes=120 * GB, head_bytes=2 * GB,
                              expert_fraction=0.9, expert_count=64, expert_used_count=8)
    assert layout.total_active_bytes < 0.25 * layout.total_full_bytes
    nodes = [node('head', 64, head=True, file=True), node('w', 96)]
    p = run(layout, nodes)
    # active bytes (~26 GB) would fit on the head alone; full bytes (120 GB) don't
    assert isinstance(p, Plan) and len(p.members) == 2
    assert sum(m.full_bytes for m in p.members) == layout.total_full_bytes
    for m in p.members:
        assert m.share == pytest.approx(m.active_bytes / layout.total_active_bytes)
    head = p.head
    assert head.share != pytest.approx(head.full_bytes / layout.total_full_bytes, rel=1e-3)
    assert_valid(p, layout, nodes)


def test_per_hop_delay_prefers_two_big_nodes_over_four_small_ones():
    layout = synthetic_layout('m40', n_layers=80, total_bytes=40 * GB, head_bytes=1 * GB)
    nodes = [node('head', 24, gen=1.0, head=True, file=True), node('big', 24, gen=1.0),
             node('s1', 9, gen=1.3), node('s2', 9, gen=1.3), node('s3', 9, gen=1.3)]
    p = run(layout, nodes, latency=all_pairs(nodes, 10.0))
    assert isinstance(p, Plan)
    assert [m.name for m in p.members] == ['big', 'head']
    assert any(len(a.names) >= 3 for a in p.alternatives)
    # without network delay the faster small nodes would win: the hop cost is what decides
    p0 = run(layout, nodes, latency=all_pairs(nodes, 0.0))
    assert len(p0.members) > 2
    assert_valid(p, layout, nodes)


def test_distant_nodes_are_excluded():
    layout = synthetic_layout('llama70b', n_layers=80, total_bytes=int(42.5 * GB), head_bytes=1 * GB)
    nodes = [node('head', 24, head=True, file=True), node('far', 64, gen=3.0), node('near', 24, gen=1.0)]
    lat = {('head', 'far'): 45.0, ('head', 'near'): 4.0, ('near', 'far'): 45.0}
    p = run(layout, nodes, latency=lat)
    assert isinstance(p, Plan)
    assert 'far' not in p.node_ids and 'near' in p.node_ids
    assert 'far' in p.explanation  # explained as too far from the head
    # if only the far node had the room, there is no plan, with a clear reason
    q = run(layout, [nodes[0], nodes[1]], latency=lat)
    assert isinstance(q, NoPlan) and 'too far' in q.reason


def test_tensor_split_order_matches_mocked_list_devices():
    layout = synthetic_layout('m55', n_layers=100, total_bytes=55 * GB, head_bytes=1 * GB)
    nodes = [node('head', 24, gen=1.0, head=True, file=True), node('w1', 24, gen=2.0), node('w2', 24, gen=0.5)]
    p = run(layout, nodes, latency=all_pairs(nodes, 1.0))
    endpoints = {'w1': '100.64.0.2:50052', 'w2': '100.64.0.3:50053'}
    workers = [m for m in p.members if m.role == 'worker']
    text = 'Available devices:\n' + ''.join(
        f'  RPC{i}: {endpoints[m.node_id]} (24576 MiB, 24000 MiB free)\n' for i, m in enumerate(workers)
    ) + '  MTL0: Apple M3 Ultra (24576 MiB, 24000 MiB free)\n'
    devices = parse_list_devices(text)
    assert [d.endpoint for d in devices] == [endpoints[m.node_id] for m in workers] + [None]
    counts = {endpoints[m.node_id]: m.n_layers for m in workers}
    vec = tensor_split_for(devices, counts, p.head.n_layers)
    assert tuple(vec) == p.tensor_split
    assert layer_ranges_from_vector(vec, 100) == [(m.layer_start, m.layer_end) for m in p.members]
    # output layer lands on the head's local device
    assert simulate_llamacpp_assignment(vec, 100)[-1] == len(devices) - 1

    # llama.cpp reports the RPC devices in the other order: the vector and ranges follow it
    swapped = parse_list_devices('\n'.join(reversed(text.splitlines()[1:3])) + '\n' + text.splitlines()[3])
    vec2 = tensor_split_for(swapped, counts, p.head.n_layers)
    order = [next(m.node_id for m in workers if endpoints[m.node_id] == d.endpoint) for d in swapped[:2]] + ['head']
    moved = reassign_for_device_order(layout, p.ctx, p.members, order)
    assert layer_ranges_from_vector(vec2, 100) == [(m.layer_start, m.layer_end) for m in moved]
    assert abs(sum(m.share for m in moved) - 1) < 1e-9


def test_lowering_a_commitment_forces_a_split():
    layout = synthetic_layout('llama70b', n_layers=80, total_bytes=int(42.5 * GB), head_bytes=1 * GB)
    nodes = [node('studio', 64, head=True, file=True), node('pc', 64)]
    assert len(run(layout, nodes).members) == 1
    lowered = [node('studio', 24, head=True, file=True), node('pc', 64)]
    p = run(layout, lowered)
    assert len(p.members) == 2 and {m.name for m in p.members} == {'studio', 'pc'}


# --- extra coverage -------------------------------------------------------------------------------

def test_not_enough_memory_gives_a_clear_reason():
    layout = synthetic_layout('huge', n_layers=60, total_bytes=420 * GB, head_bytes=2 * GB)
    nodes = [node('a', 128, head=True, file=True), node('b', 96), node('c', 96)]
    p = run(layout, nodes)
    assert isinstance(p, NoPlan)
    assert 'needs' in p.reason and 'usable' in p.reason and 'committed' in p.reason


def test_filters_offline_wrong_build_hours_and_model():
    layout = synthetic_layout('small', n_layers=32, total_bytes=8 * GB, head_bytes=1 * GB)
    pinned = InferenceSettings(TRANSPORT='direct', PINNED_LLAMA_BUILD='b11160')
    base = dict(head=True, file=True)
    nodes = [
        node('offline', 64, online=False, **base),
        NodeCandidate('oldbuild', 'oldbuild', 64 * GIB, can_head=True, has_file=True, build='b9999'),
        node('night', 64, hours=((22, 6),), **base),
        node('other-model', 64, allowed_models=('x.gguf',), **base),
        node('ok', 16, **base),
    ]
    p = run(layout, nodes, settings=pinned)
    assert isinstance(p, Plan) and p.node_ids == ('ok',)
    none = run(layout, nodes[:-1], settings=pinned)
    assert isinstance(none, NoPlan)
    assert set(none.excluded) == {'offline', 'oldbuild', 'night', 'other-model'}


def test_unpinned_build_accepts_any_build_but_splits_only_within_one():
    # the model needs two nodes; the only other node runs a different llama.cpp build than the head
    layout = synthetic_layout('mid', n_layers=40, total_bytes=40 * GB, head_bytes=1 * GB)
    head = NodeCandidate('head', 'head', 32 * GIB, can_head=True, has_file=True, build='b11160')
    other = NodeCandidate('other', 'other', 32 * GIB, build='b12000')
    assert isinstance(run(layout, [head, other]), NoPlan)
    same = NodeCandidate('same', 'same', 32 * GIB, build='b11160')
    p = run(layout, [head, other, same])
    assert isinstance(p, Plan) and set(p.node_ids) == {'head', 'same'}


def test_hours_wrap_midnight():
    assert in_hours(((22, 6),), datetime(2026, 1, 1, 23, tzinfo=timezone.utc))
    assert in_hours(((22, 6),), datetime(2026, 1, 1, 3, tzinfo=timezone.utc))
    assert not in_hours(((22, 6),), datetime(2026, 1, 1, 12, tzinfo=timezone.utc))


def test_tensor_split_vector_is_exact_for_many_layer_counts():
    rng = random.Random(7)
    for _ in range(500):
        k = rng.randint(1, 5)
        counts = [rng.randint(1, 40) for _ in range(k)]
        n = sum(counts)
        vec = counts[:-1] + [counts[-1] + 1]
        ranges = layer_ranges_from_vector(vec, n)
        pos = 0
        for c, (a, b) in zip(counts, ranges):
            assert (a, b) == (pos, pos + c)
            pos += c


def test_head_can_hold_only_embedding_and_output():
    layout = synthetic_layout('m', n_layers=40, total_bytes=30 * GB, head_bytes=1 * GB)
    nodes = [node('tiny-head', 4, head=True, file=True), node('w', 48)]
    p = run(layout, nodes)
    assert isinstance(p, Plan)
    assert p.head.n_layers == 0 and p.tensor_split == (40, 1)
    assert p.head.share > 0  # still runs the output layer


def test_accelerator_devices_are_left_out_of_the_vector():
    # real `llama-server --list-devices` output on an M1 Pro (b11160), plus one RPC worker
    text = ('Available devices:\n'
            '  RPC0: 100.64.0.2:50052 (24576 MiB, 24000 MiB free)\n'
            '  MTL0: Apple M1 Pro (12124 MiB, 12123 MiB free)\n'
            '  BLAS: Accelerate (0 MiB, 0 MiB free)\n')
    devices = parse_list_devices(text)
    assert [d.name for d in devices] == ['RPC0', 'MTL0', 'BLAS'] and devices[2].is_accel
    assert tensor_split_for(devices, {'100.64.0.2:50052': 20}, 12) == [20, 13]


def test_rpc_server_exposing_two_gpus_gets_its_block_on_the_first():
    text = ('Available devices:\n'
            '  RPC0: 100.64.0.2:50052 (24576 MiB, 24000 MiB free)\n'
            '  RPC1: 100.64.0.2:50052 (24576 MiB, 24000 MiB free)\n'
            '  RPC2: 100.64.0.3:50060 (16384 MiB, 16000 MiB free)\n'
            '  CUDA0: NVIDIA RTX 4090 (24564 MiB, 23000 MiB free)\n')
    vec = tensor_split_for(parse_list_devices(text), {'100.64.0.2:50052': 30, '100.64.0.3:50060': 20}, 10)
    assert vec == [30, 0, 20, 11]
    assert layer_ranges_from_vector(vec, 60) == [(0, 30), (0, 0), (30, 50), (50, 60)]


def test_list_devices_registry_order_is_reordered_rpc_first():
    # exact `llama-server --rpc 127.0.0.1:59026 --list-devices` output from b11160 on an M1 Pro:
    # the local GPU is printed first, but the model puts RPC devices at the front of its list
    text = ('Available devices:\n'
            '  MTL0: Apple M1 Pro (12124 MiB, 12123 MiB free)\n'
            '  BLAS: Accelerate (0 MiB, 0 MiB free)\n'
            '  RPC0: 127.0.0.1:59026 (12124 MiB, 12123 MiB free)\n')
    devices = parse_list_devices(text)
    assert tensor_split_for(devices, {'127.0.0.1:59026': 15}, 13) == [15, 14]


def test_relay_mode_uses_the_relay_hop_limit():
    layout = synthetic_layout('llama70b', n_layers=80, total_bytes=int(42.5 * GB), head_bytes=1 * GB)
    nodes = [node('head', 24, head=True, file=True), node('over-internet', 24), node('too-far', 64, gen=3.0)]
    lat = {('head', 'over-internet'): 45.0, ('head', 'too-far'): 150.0}
    relay = InferenceSettings(TRANSPORT='relay')
    p = run(layout, nodes, latency=lat, settings=relay)
    assert isinstance(p, Plan) and set(p.node_ids) == {'head', 'over-internet'}
    assert '100 ms' in p.explanation and 'too-far' in p.explanation


def test_memory_a_mac_must_lend_to_run_a_model_alone():
    """The LLMs tab says how much to lend before a chat fails for lack of memory."""
    import math

    from slashcompute.inference.coordinator.planner import commit_for, single_node_bytes

    layout = synthetic_layout('m', 28, int(6.0 * GB), int(0.5 * GB))
    gib = math.ceil(commit_for(single_node_bytes(layout, S), S) / GIB)
    assert isinstance(run(layout, [node('a', gib, head=True, file=True)]), Plan)
    assert isinstance(run(layout, [node('a', gib - 1, head=True, file=True)]), NoPlan)
