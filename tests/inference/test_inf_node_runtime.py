import asyncio
import json
import os
import subprocess
import sys

import pytest

from inf_harness import isolated_paths
from slashcompute.inference.node import agent as agent_mod
from slashcompute.inference.node import head
from slashcompute.inference.node.agent import Agent, move_to_trash, remove_model_files, scan_models
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


def test_removal_trashes_the_models_folder_copy_and_deletes_the_apps(tmp_path, trash):
    user = files(tmp_path / 'models', *SPLIT, *OTHERS)
    app = files(tmp_path / 'downloads', *SPLIT, '.big-00001-of-00002.gguf.part', 'other.gguf')
    out = remove_model_files('big-00001-of-00002.gguf', str(user), str(app), trash=True)
    assert sorted(out['trashed']) == SPLIT and out['kept'] == []
    assert sorted(out['deleted']) == ['.big-00001-of-00002.gguf.part', *SPLIT]
    assert sorted(p.name for p in trash.iterdir()) == SPLIT                    # every shard, Put Back works
    assert sorted(p.name for p in user.iterdir()) == sorted(OTHERS)            # other models untouched
    assert [p.name for p in app.iterdir()] == ['other.gguf']


def test_without_trash_the_models_folder_copy_stays(tmp_path, trash):
    user = files(tmp_path / 'models', 'm.gguf')
    app = files(tmp_path / 'downloads', 'm.gguf')
    out = remove_model_files('m.gguf', str(user), str(app), trash=False)
    assert out == {'trashed': [], 'deleted': ['m.gguf'], 'kept': ['m.gguf']}
    assert (user / 'm.gguf').exists() and not (app / 'm.gguf').exists()
    assert not any(trash.iterdir())


def test_one_folder_for_both_is_the_users_trash_never_delete(tmp_path, trash):
    d = files(tmp_path / 'models', 'm.gguf')
    assert remove_model_files('m.gguf', str(d), str(d) + '/', trash=True)['trashed'] == ['m.gguf']
    assert [p.name for p in trash.iterdir()] == ['m.gguf']
    files(d, 'm.gguf')
    assert remove_model_files('m.gguf', str(d), str(d), trash=False)['kept'] == ['m.gguf']
    assert (d / 'm.gguf').exists()


@pytest.mark.parametrize('name', ['../x.gguf', '.x.gguf', 'x.txt', 'a/x.gguf', '', None])
def test_removal_only_takes_a_bare_gguf_name(tmp_path, name):
    with pytest.raises(ValueError):
        remove_model_files(name, str(tmp_path), str(tmp_path / 'downloads'), trash=True)


def test_a_failed_trash_names_the_file_and_leaves_it(tmp_path):
    user = files(tmp_path / 'models', 'm.gguf')            # conftest's Trash refuses every move
    app = files(tmp_path / 'downloads', 'm.gguf')
    with pytest.raises(OSError) as e:
        remove_model_files('m.gguf', str(user), str(app), trash=True)
    assert 'm.gguf' in str(e.value) and str(tmp_path) not in str(e.value)   # file names only, never folders
    assert (user / 'm.gguf').exists() and not (app / 'm.gguf').exists()     # the app's copy still went


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


def test_the_node_keeps_removed_models_unless_started_with_trash_removed(tmp_path):
    from typer.testing import CliRunner

    from slashcompute.inference.node.__main__ import app, build_config

    assert NodeConfig().trash_removed is False
    args = ('http://127.0.0.1:8765', tmp_path, 'mac', '~/models', 8.0, True, 0, 'skip', '', '', '127.0.0.1', 0)
    assert build_config(*args).trash_removed is False and build_config(*args, True).trash_removed is True
    help_text = CliRunner().invoke(app, ['start', '--help'], env={'COLUMNS': '200'}).output
    assert '--trash-removed' in help_text and '--keep-removed' in help_text


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
