import asyncio
import json
import os
import subprocess
import sys

import httpx
import pytest

from inf_harness import isolated_paths
from slashcompute.inference.node import agent as agent_mod
from slashcompute.inference.node import head
from slashcompute.inference.node.agent import (Agent, move_to_trash, remove_model_files, save_json, scan_models,
                                               still_ours)
from slashcompute.inference.node.config import Commitment, NodeConfig
from slashcompute.inference.node.engine import EngineError, reap_orphans
from slashcompute.inference.node.fake_engine import FakeCluster, FakeEngine

GIB = 1024 ** 3


def test_reaper_only_kills_llama_processes(tmp_path):
    other = subprocess.Popen(['sleep', '30'])
    try:
        pid_file = tmp_path / 'node.pids'
        pid_file.write_text(json.dumps([other.pid, 999999]))
        assert reap_orphans(pid_file) == []      # not llama-server / rpc-server: left alone
        assert other.poll() is None
        assert not pid_file.exists()
    finally:
        other.kill()


# Error bodies as llama-server (b11160) returns them.
JINJA_500 = {'error': {'code': 500, 'type': 'server_error', 'message': (
    "\n------------\nWhile executing CallExpression at line 43, column 24 in source:\n... "
    "raise_exception('No messages provided.') ...\nError: Jinja Exception: No messages provided.")}}
CTX_400 = {'error': {'code': 400, 'type': 'exceed_context_size_error', 'n_prompt_tokens': 20010, 'n_ctx': 4096,
                     'message': 'request (20010 tokens) exceeds the available context size (4096 tokens), '
                                'try increasing it'}}
LOADING_503 = {'error': {'code': 503, 'type': 'unavailable_error', 'message': 'Loading model'}}
PORT = 9982


async def stub_llama_server(status: int, body: dict):
    payload = json.dumps(body).encode()

    async def handle(reader, writer):
        headers = (await reader.readuntil(b'\r\n\r\n')).decode().split('\r\n')
        length = next((int(h.split(':')[1]) for h in headers if h.lower().startswith('content-length')), 0)
        await reader.readexactly(length)
        writer.write(f'HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {len(payload)}\r\n'
                     'Connection: close\r\n\r\n'.encode() + payload)
        await writer.drain()
        writer.close()

    return await asyncio.start_server(handle, '127.0.0.1', PORT)


async def run_chat(status: int, body: dict) -> BaseException:
    server = await stub_llama_server(status, body)
    try:
        with pytest.raises(RuntimeError) as e:
            async for _ in head.stream_chat(PORT, {'messages': []}):
                pass
        return e.value
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.parametrize('status,body,message', [(500, JINJA_500, 'No messages provided'),
                                                 (400, CTX_400, 'exceeds the available context size')])
async def test_llama_server_request_errors_are_client_errors_not_broken_pipelines(status, body, message):
    err = await run_chat(status, body)
    assert isinstance(err, EngineError) and err.status == 400 and err.pipeline_broken is False
    assert message in str(err)


async def test_llama_server_failures_still_break_the_pipeline():
    err = await run_chat(503, LOADING_503)
    assert getattr(err, 'status', None) is None  # engine wraps it as EngineError(pipeline_broken=True)


# ------------------------------------------------------------ removing a model from this Mac


@pytest.fixture
def trash(tmp_path, monkeypatch):
    """A Trash under tmp (the real one is off limits: see conftest.no_real_trash)."""
    bin_ = tmp_path / 'Trash'
    bin_.mkdir()
    monkeypatch.setattr(agent_mod, 'move_to_trash', lambda path: os.replace(path, bin_ / path.name))
    return bin_


def files(d, *names):
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / n).write_bytes(b'GGUF' + n.encode())
    return d


SPLIT = ['big-00001-of-00002.gguf', 'big-00002-of-00002.gguf']
OTHERS = ['big-x-00001-of-00002.gguf', 'big-x-00002-of-00002.gguf', 'other.gguf']   # look alike, not the model


def test_removal_trashes_the_models_folder_copy_and_deletes_the_one_the_pool_sent(tmp_path, trash):
    user = files(tmp_path / 'models', *SPLIT, *OTHERS)
    app = files(tmp_path / 'downloads', *SPLIT, '.big-00001-of-00002.gguf.part', 'other.gguf')
    out = remove_model_files('big-00001-of-00002.gguf', str(user), str(app), trash=True, deletable=set(SPLIT))
    assert sorted(out['trashed']) == SPLIT and out['kept'] == []
    assert sorted(out['deleted']) == ['.big-00001-of-00002.gguf.part', *SPLIT]
    assert sorted(p.name for p in trash.iterdir()) == SPLIT                    # every shard, Put Back works
    assert sorted(p.name for p in user.iterdir()) == sorted(OTHERS)            # other models untouched
    assert [p.name for p in app.iterdir()] == ['other.gguf']


def test_without_trash_the_models_folder_copy_stays(tmp_path, trash):
    user = files(tmp_path / 'models', 'm.gguf')
    app = files(tmp_path / 'downloads', 'm.gguf')
    out = remove_model_files('m.gguf', str(user), str(app), trash=False, deletable={'m.gguf'})
    assert out == {'trashed': [], 'deleted': ['m.gguf'], 'kept': [{'name': 'm.gguf', 'folder': 'models'}]}
    assert (user / 'm.gguf').exists() and not (app / 'm.gguf').exists()
    assert not any(trash.iterdir())


def test_a_copy_in_the_apps_folder_the_pool_did_not_send_is_trashed_or_kept_never_deleted(tmp_path, trash):
    """This Mac's own upload when it hosts a pool, another pool's push, or one pushed before the app kept
    a record: on a LAN pool it goes to the Trash, on a public pool it stays (and says where)."""
    app = files(tmp_path / 'downloads', 'm.gguf', '.m.gguf.part')
    user = tmp_path / 'models'
    out = remove_model_files('m.gguf', str(user), str(app), trash=False, deletable={'other.gguf'})
    assert out == {'trashed': [], 'deleted': [], 'kept': [{'name': 'm.gguf', 'folder': 'app'}]}
    assert sorted(p.name for p in app.iterdir()) == ['.m.gguf.part', 'm.gguf']   # nor its partial download
    out = remove_model_files('m.gguf', str(user), str(app), trash=True)
    assert out == {'trashed': ['m.gguf'], 'deleted': [], 'kept': []}
    assert [p.name for p in trash.iterdir()] == ['m.gguf']


def test_one_folder_for_both_is_the_users_trash_never_delete(tmp_path, trash):
    d = files(tmp_path / 'models', 'm.gguf')
    out = remove_model_files('m.gguf', str(d), str(d) + '/', trash=True, deletable={'m.gguf'})
    assert out['trashed'] == ['m.gguf'] and out['deleted'] == []
    assert [p.name for p in trash.iterdir()] == ['m.gguf']
    files(d, 'm.gguf')
    assert remove_model_files('m.gguf', str(d), str(d), trash=False)['kept'] == [{'name': 'm.gguf', 'folder': 'models'}]
    assert (d / 'm.gguf').exists()


def test_removal_matches_exact_names_only(tmp_path, trash):
    """APFS opens 'Model.gguf' for 'model.gguf': a different model with a name that differs in case only."""
    user = files(tmp_path / 'models', 'llama.gguf')
    app = files(tmp_path / 'downloads', 'model.gguf')
    for name in ('Model.gguf', 'Llama.gguf', 'MODEL.gguf'):
        out = remove_model_files(name, str(user), str(app), trash=True, deletable={name})
        assert out == {'trashed': [], 'deleted': [], 'kept': []}
    assert (user / 'llama.gguf').exists() and (app / 'model.gguf').exists() and not any(trash.iterdir())


@pytest.mark.parametrize('name', ['../x.gguf', '.x.gguf', 'x.txt', 'a/x.gguf', '', None])
def test_removal_only_takes_a_bare_gguf_name(tmp_path, name):
    with pytest.raises(ValueError):
        remove_model_files(name, str(tmp_path), str(tmp_path / 'downloads'), trash=True)


def test_a_failed_trash_names_the_file_and_leaves_it(tmp_path):
    user = files(tmp_path / 'models', 'm.gguf')            # conftest's Trash refuses every move
    app = files(tmp_path / 'downloads', 'm.gguf')
    with pytest.raises(OSError) as e:
        remove_model_files('m.gguf', str(user), str(app), trash=True, deletable={'m.gguf'})
    assert 'm.gguf' in str(e.value) and str(tmp_path) not in str(e.value)   # file names only, never folders
    assert (user / 'm.gguf').exists() and not (app / 'm.gguf').exists()     # the copy the pool sent still went


def test_a_file_written_over_since_it_was_downloaded_is_not_the_pools_any_more(tmp_path):
    app = files(tmp_path / 'downloads', 'sent.gguf', 'mine.gguf')
    sent = {n: {'size': (app / n).stat().st_size, 'mtime_ns': (app / n).stat().st_mtime_ns}
            for n in ('sent.gguf', 'mine.gguf', 'gone.gguf') if (app / n).exists()}
    sent['gone.gguf'] = {'size': 1, 'mtime_ns': 1}
    (app / 'mine.gguf').write_bytes(b'GGUFMINE.GGUF')        # the same size: this Mac's own upload since
    os.utime(app / 'mine.gguf', ns=(1, 1))                 # whatever the clock: not the time it was downloaded
    assert still_ours(str(app), sent) == {'sent.gguf'}


def test_the_record_of_downloads_is_written_atomically(tmp_path, monkeypatch):
    path = tmp_path / 'inference' / 'downloaded.json'
    save_json(path, {'http://pool/inference': {'m.gguf': {'size': 1, 'mtime_ns': 2}}})
    assert json.loads(path.read_text()) == {'http://pool/inference': {'m.gguf': {'size': 1, 'mtime_ns': 2}}}

    def crash(*a):
        raise OSError(28, 'No space left on device')

    monkeypatch.setattr(agent_mod.os, 'replace', crash)
    with pytest.raises(OSError):
        save_json(path, {'half': 'written'})
    assert json.loads(path.read_text())['http://pool/inference']    # the previous record, whole
    assert [p.name for p in path.parent.iterdir()] == ['downloaded.json']   # no temporary file left


def test_the_trash_needs_pyobjc_and_never_falls_back_to_deleting(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'objc', None)         # PyObjC missing: no Trash, so an OSError
    monkeypatch.setitem(sys.modules, 'Foundation', None)
    f = files(tmp_path, 'm.gguf') / 'm.gguf'
    with pytest.raises(OSError):
        move_to_trash(f)
    assert f.exists()


def test_scan_skips_hidden_files(tmp_path):
    from inf_gguf_fixtures import tiny_gguf

    (tmp_path / 'm.gguf').write_bytes(tiny_gguf())
    (tmp_path / '._m.gguf').write_bytes(tiny_gguf())       # Finder's AppleDouble files on other volumes
    assert [f['name'] for f in scan_models([str(tmp_path)])] == ['m.gguf']


def test_a_model_still_being_copied_in_is_left_out_until_it_is_whole(tmp_path):
    from inf_gguf_fixtures import tiny_gguf

    data = tiny_gguf()
    f = tmp_path / 'm.gguf'
    f.write_bytes(data[:-100])                              # the header is there, the last tensor is not yet
    assert scan_models([str(tmp_path)]) == []
    f.write_bytes(data)
    assert [x['name'] for x in scan_models([str(tmp_path)])] == ['m.gguf']


def test_a_split_model_with_parts_still_to_come_is_left_out_until_all_are_there(tmp_path):
    """The coordinator keeps the first layer table it gets: one from part 1 alone would stay wrong."""
    from inf_gguf_fixtures import tiny_gguf

    (tmp_path / 'm-00001-of-00002.gguf').write_bytes(tiny_gguf())
    assert scan_models([str(tmp_path)]) == []
    (tmp_path / 'm-00002-of-00002.gguf').write_bytes(tiny_gguf())
    assert [(x['name'], len(x['headers'])) for x in scan_models([str(tmp_path)])] == [('m-00001-of-00002.gguf', 2)]


def test_the_node_keeps_removed_models_unless_started_with_trash_removed(tmp_path):
    from typer.testing import CliRunner

    from slashcompute.inference.node.__main__ import app, build_config

    assert NodeConfig().trash_removed is False
    args = ('http://127.0.0.1:8765', tmp_path, 'mac', '~/models', 8.0, True, 0, 'skip', '', '', '127.0.0.1', 0)
    assert build_config(*args).trash_removed is False and build_config(*args, True).trash_removed is True
    help_text = CliRunner().invoke(app, ['start', '--help'], env={'COLUMNS': '200'}).output
    assert '--trash-removed' in help_text and '--keep-removed' in help_text


@pytest.mark.parametrize('flag, want', [(['--trash-removed'], True), (['--keep-removed'], False), ([], False)])
def test_start_hands_the_trash_flag_to_the_node(tmp_path, monkeypatch, flag, want):
    from typer.testing import CliRunner

    from slashcompute.inference.node import __main__ as node_main
    from slashcompute.inference.node import hardware

    seen = {}

    async def run_node(cfg, home, fake=False):              # what start() would run, given its config
        seen['cfg'] = cfg

    monkeypatch.setattr(node_main, 'run_node', run_node)
    monkeypatch.setattr(hardware, 'detect', lambda: {'total_mem_bytes': 0})
    r = CliRunner().invoke(node_main.app, ['start', '--url', 'http://127.0.0.1:9', '--home', str(tmp_path),
                                           '--localhost', '--memory-gb', '8', *flag])
    assert r.exit_code == 0, r.output
    assert seen['cfg'].trash_removed is want
    assert seen['cfg'].download_dir == str(tmp_path / 'models')


# ------------------------------------------------------------ stopped while starting


class BlockingEngine(FakeEngine):
    """Starts block until `release` is set: a stop arrives while llama.cpp is still loading."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.release = asyncio.Event()

    async def start_worker(self, pipeline_id, spec):
        await self.release.wait()
        return await super().start_worker(pipeline_id, spec)

    async def start_head(self, pipeline_id, spec):
        await self.release.wait()
        return await super().start_head(pipeline_id, spec)


@pytest.mark.parametrize('role', ['worker', 'head'])
async def test_a_pipeline_stopped_while_starting_leaves_nothing_running(tmp_path, role):
    engine = BlockingEngine('n-1', FakeCluster())
    cfg = NodeConfig(coordinator_url='http://127.0.0.1:9/inference', **isolated_paths(str(tmp_path)),
                     commitment=Commitment(memory_gb=16, may_be_head=True))
    agent = Agent(cfg, engine, info={}, build='fake', ip='127.0.0.1', gguf_files=[], latency_fn=None,
                  busy_fn=lambda: False)
    spec = {'pipeline_id': 'p-1', 'model': 'm.gguf', 'mem_bytes': GIB, 'ctx': 4096, 'head_layers': 2,
            'workers': [], 'sim': {'out_s': 0.01, 'in_s': 0.001, 'members': []}}
    try:
        start = asyncio.create_task(agent.dispatch(f'start_{role}', spec))
        for _ in range(100):
            await asyncio.sleep(0)
        assert agent.in_use == {'p-1': GIB} and not start.done()     # inside the engine's start
        await agent.dispatch(f'stop_{role}', {'pipeline_id': 'p-1'})
        engine.release.set()
        with pytest.raises(EngineError, match='stopped while loading') as e:
            await start
        assert e.value.pipeline_broken is False
        assert engine.workers == {} and engine.heads == {} and agent.in_use == {}
    finally:
        await agent.client.aclose()


def node_agent(tmp_path, engine, client=None) -> Agent:
    cfg = NodeConfig(coordinator_url='http://coordinator/inference', **isolated_paths(str(tmp_path)),
                     commitment=Commitment(memory_gb=16, may_be_head=True))
    agent = Agent(cfg, engine, info={}, build='fake', ip='127.0.0.1', gguf_files=[], latency_fn=None,
                  client=client, busy_fn=lambda: False)
    agent.token = 't'
    return agent


def coordinator(batches: list[list[dict]], results: list[dict]) -> httpx.AsyncClient:
    """A coordinator that hands out ``batches`` one poll at a time and records the command results."""
    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith('/agent/commands'):
            if batches:
                return httpx.Response(200, json={'commands': batches.pop(0)})
            await asyncio.sleep(0.05)
            return httpx.Response(200, json={'commands': []})
        if request.url.path.endswith('/result'):
            results.append({'id': request.url.path.split('/')[-2], **json.loads(request.content)})
        return httpx.Response(200, json={'ok': True})

    return httpx.AsyncClient(base_url='http://coordinator/inference', transport=httpx.MockTransport(handle))


async def until(cond, timeout: float = 5.0) -> None:
    for _ in range(int(timeout / 0.01)):
        if cond():
            return
        await asyncio.sleep(0.01)
    raise AssertionError('condition not met')


SPEC = {'pipeline_id': 'p-1', 'model': 'm.gguf', 'mem_bytes': GIB, 'ctx': 4096, 'head_layers': 2,
        'workers': [], 'sim': {'out_s': 0.01, 'in_s': 0.001, 'members': []}}


@pytest.mark.parametrize('role', ['worker', 'head'])
async def test_a_stop_that_comes_with_its_start_in_one_poll_keeps_it_from_loading(tmp_path, role):
    """The coordinator queued start and stop before this Mac polled: stops run first, inline."""
    results = []
    client = coordinator([[{'id': 'start', 'kind': f'start_{role}', 'payload': SPEC},
                           {'id': 'stop', 'kind': f'stop_{role}', 'payload': {'pipeline_id': 'p-1'}}]], results)
    engine = FakeEngine('n-1', FakeCluster())
    agent = node_agent(tmp_path, engine, client)
    loop = asyncio.create_task(agent.command_loop())
    try:
        await until(lambda: len(results) == 2)
        await asyncio.sleep(0.1)                              # nothing loads late either
        assert [r['id'] for r in results] == ['stop', 'start']
        assert results[0]['ok'] is True
        assert results[1]['ok'] is False and 'StoppedWhileLoading' in results[1]['error']
        assert agent.in_use == {} and engine.workers == {} and engine.heads == {}
        assert agent.last_error == ''                         # expected, not shown as this Mac's error
    finally:
        loop.cancel()
        await asyncio.gather(loop, return_exceptions=True)
        await client.aclose()


class KilledWhileLoading(FakeEngine):
    """Like llama.cpp: a stop during the load kills llama-server, and the start fails with its exit code."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.loading = asyncio.Event()
        self.killed = asyncio.Event()

    async def start_head(self, pipeline_id, spec):
        self.loading.set()
        await asyncio.wait_for(self.killed.wait(), 5)
        raise RuntimeError('llama-server exited with code -15 while loading')

    async def stop_head(self, pipeline_id):
        self.killed.set()
        await super().stop_head(pipeline_id)


async def test_unloading_a_model_that_is_loading_leaves_no_error_on_this_mac(tmp_path):
    results = []
    client = coordinator([], results)
    engine = KilledWhileLoading('n-1', FakeCluster())
    agent = node_agent(tmp_path, engine, client)
    try:
        start = asyncio.create_task(agent.handle({'id': 'start', 'kind': 'start_head', 'payload': SPEC}))
        await asyncio.wait_for(engine.loading.wait(), 5)
        await agent.handle({'id': 'stop', 'kind': 'stop_head', 'payload': {'pipeline_id': 'p-1'}})
        await start
        assert results[-1]['id'] == 'start' and results[-1]['ok'] is False
        assert 'StoppedWhileLoading' in results[-1]['error']   # the coordinator still learns the start failed
        assert agent.last_error == '' and agent.in_use == {}
        agent.write_status()
        assert json.loads((tmp_path / 'status.json').read_text())['last_error'] == ''

        # a load that fails by itself is still this Mac's error
        engine.killed.set()
        await agent.handle({'id': 'again', 'kind': 'start_head', 'payload': {**SPEC, 'pipeline_id': 'p-2'}})
        assert agent.last_error == 'start_head: llama-server exited with code -15 while loading'
    finally:
        await client.aclose()


async def test_stop_ends_a_task_whose_cancel_was_swallowed(tmp_path):
    """anyio takes a cancel that lands while it opens a connection (it cancels its other connect attempts
    then) for its own, and swallows it: the heartbeat loop then carried on and the node never stopped."""
    agent = node_agent(tmp_path, FakeEngine('n-1', FakeCluster()), coordinator([], []))
    swallowed = asyncio.Event()

    async def loop():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            swallowed.set()                                   # as anyio's connect_tcp does
        await asyncio.sleep(10)                               # and on it goes

    agent._spawn(loop())
    await asyncio.sleep(0)
    try:
        await asyncio.wait_for(agent.stop(), 5)
        assert swallowed.is_set() and not agent._tasks and agent.stopped.is_set()
    finally:
        await agent.client.aclose()


async def test_stop_also_ends_a_task_spawned_while_it_stops(tmp_path):
    """A command handled after stop() listed the tasks (its loop swallowed the first cancel) is ended too:
    otherwise it could start llama-server after the engine stopped everything."""
    agent = node_agent(tmp_path, FakeEngine('n-1', FakeCluster()), coordinator([], []))
    late = []

    async def loop():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            late.append(agent._spawn(asyncio.sleep(10)))      # a command it took meanwhile
        await asyncio.sleep(10)

    agent._spawn(loop())
    await asyncio.sleep(0)
    try:
        await asyncio.wait_for(agent.stop(), 5)
        assert late and late[0].done() and not agent._tasks
    finally:
        await agent.client.aclose()
