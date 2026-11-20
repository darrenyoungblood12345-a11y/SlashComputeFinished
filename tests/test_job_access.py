"""Public-pool private data, ownership checks and upload cleanup."""

import asyncio

import pytest

from accounts import sign_in
from fastapi.testclient import TestClient

from slashcompute.common.config import EngineConfig
from slashcompute.common.protocol import DeviceProfile, Register
from slashcompute.coordinator.app import create_app
from slashcompute.coordinator.db import Verification
from slashcompute.pipeline.local import build_compute


@pytest.fixture
def pool(tmp_path, tiny_model, tiny_dataset, monkeypatch):
    app = create_app(EngineConfig(home=tmp_path / 'home', public_pool=True))
    core = app.state.core
    users, tokens = {}, {}
    for role in ('admin', 'owner', 'other', 'worker', 'verifier'):
        user, token = sign_in(core.auth, f'{role}@example.test', role)
        user = core.auth.accept_terms(user)
        core.credits.contribute(user.id, 1e12)
        users[role], tokens[role] = user, token
    # No background scheduling: each test controls the current assignment explicitly.
    client = TestClient(app, raise_server_exceptions=True)
    try:
        response = client.post('/jobs/upload', headers=_headers(tokens['owner']),
                               files={'dataset': ('train.jsonl', tiny_dataset.read_bytes())},
                               data={'model': str(tiny_model), 'steps': 2, 'batch_size': 2,
                                     'microbatches': 1, 'max_flops': 1e9})
        assert response.status_code == 200, response.text
        job = core.jobs[response.json()['id']]
        yield client, core, job, users, tokens
    finally:
        client.close()


def _headers(token):
    return {'Authorization': f'Bearer {token}'}


def _register(core, token, name):
    async def send(_):
        pass

    asyncio.run(core.on_register(Register(
        node_id=name, name=name, data_host='127.0.0.1', data_port=9700, gpu_percent=50,
        session_token=token,
        device=DeviceProfile(chip='test', memory_total_bytes=8 << 30,
                             memory_available_bytes=8 << 30, working_set_bytes=8 << 30,
                             memory_contrib_bytes=8 << 30, matmul_tflops=1,
                             mem_bandwidth_gbps=100),
    ), send))
    return core.registry.get(name)


@pytest.mark.parametrize('role,expected', [(None, 401), ('invalid', 401), ('other', 403),
                                          ('owner', 200), ('admin', 200)])
def test_private_job_files_require_owner_or_admin(pool, role, expected):
    client, core, job, _, tokens = pool
    checkpoint = core.checkpoints.merged_path(job.id, 1)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b'checkpoint')
    adapter = core.checkpoints.job_dir(job.id) / 'adapter'
    adapter.mkdir()
    (adapter / 'adapter_config.json').write_text('{}')
    headers = _headers(tokens.get(role, 'invalid')) if role else {}
    for suffix in ('dataset', 'checkpoints/1', 'adapter/adapter_config.json'):
        response = client.get(f'/jobs/{job.id}/{suffix}', headers=headers)
        assert response.status_code == expected, response.text
    response = client.post(f'/jobs/{job.id}/cancel', headers=headers)
    assert response.status_code == expected
    assert job.row.status == ('cancelled' if expected == 200 else 'queued')


def test_owner_cookie_works_but_banned_owner_cannot_access_job(pool):
    client, core, job, users, tokens = pool
    client.cookies.set('slashcompute_session', tokens['owner'])
    assert client.get(f'/jobs/{job.id}/dataset').status_code == 200
    core.auth.set_banned(users['owner'], True)
    assert client.get(f'/jobs/{job.id}/dataset').status_code == 403
    assert client.post(f'/jobs/{job.id}/cancel').status_code == 403


def test_assigned_agent_transfers_checkpoint_but_cannot_cancel_or_keep_access(pool, tmp_path):
    client, core, job, _, tokens = pool
    node = _register(core, tokens['worker'], 'worker-node')
    asyncio.run(core.scheduler.try_start(job))
    assert node.assignment.job_id == job.id
    url = f'/jobs/{job.id}'
    headers = _headers(tokens['worker'])
    assert client.get(f'{url}/dataset', headers=headers).status_code == 200
    assert client.post(f'{url}/cancel', headers=headers).status_code == 403
    checkpoint = tmp_path / 'stage.safetensors'
    compute = build_compute(job.spec, 0, job.profile.num_layers, job.profile.num_layers)
    compute.save_checkpoint(checkpoint, 1)
    for role, epoch, stage in [('other', 1, 0), ('owner', 1, 0), ('worker', 0, 0), ('worker', 1, 1)]:
        response = client.post(f'{url}/checkpoints/1', params={'epoch': epoch, 'stage': stage},
                               headers=_headers(tokens[role]), content=checkpoint.read_bytes())
        assert response.status_code == 403
    assert client.post(f'{url}/checkpoints/1', params={'epoch': 1, 'stage': 0},
                       content=checkpoint.read_bytes()).status_code == 401
    response = client.post(f'{url}/checkpoints/1', params={'epoch': 1, 'stage': 0},
                           headers=headers, content=checkpoint.read_bytes())
    assert response.status_code == 200, response.text
    assert client.get(f'{url}/checkpoints/1', headers=headers).status_code == 200
    node.assignment = None
    assert client.get(f'{url}/dataset', headers=headers).status_code == 403
    assert client.get(f'{url}/checkpoints/1', headers=headers).status_code == 403


def test_public_user_cannot_copy_another_users_server_dataset_path(pool):
    client, core, job, _, tokens = pool
    spec = job.spec.model_dump(mode='json')
    spec['dataset_path'] = str(core.dataset_path(job.id))
    response = client.post('/jobs', headers=_headers(tokens['other']),
                           json={**spec, 'max_flops': 1e9})
    assert response.status_code == 403
    assert len(core.jobs) == 1


@pytest.mark.parametrize('role,model,budget,expected', [
    (None, 'valid', '1000', 401),
    ('invalid', 'valid', '1000', 401),
    ('other', 'not/allowed', '1000', 400),
    ('other', 'valid', '1e30', 400),
    ('other', 'valid', '1000', 200),
])
def test_uploads_never_leave_temporary_or_rejected_datasets(pool, role, model, budget, expected):
    client, core, job, _, tokens = pool
    headers = _headers(tokens.get(role, 'invalid')) if role else {}
    response = client.post('/jobs/upload', headers=headers,
                           files={'dataset': ('rejected.jsonl', b'{"tokens":[1,2,3]}\n')},
                           data={'model': job.spec.model if model == 'valid' else model,
                                 'max_flops': budget})
    assert response.status_code == expected, response.text
    assert list((core.cfg.home / 'uploads').iterdir()) == []
    datasets = list(core.cfg.coordinator_dir.rglob('dataset.jsonl'))
    assert len(datasets) == (2 if expected == 200 else 1)
    if expected == 200:
        assert client.get(f"/jobs/{response.json()['id']}/dataset", headers=headers).content == (
            b'{"tokens":[1,2,3]}\n'
        )


def test_verification_transfers_require_assigned_users(pool):
    client, core, job, _, tokens = pool
    _register(core, tokens['worker'], 'worker-node')
    verifier = _register(core, tokens['verifier'], 'verifier-node')
    row = Verification(id='review', kind='replay', job_id=job.id, status='fetching',
                       target_node_id='worker-node', verifier_node_id='verifier-node')
    core.db.add(row)
    for role, expected in [(None, 401), ('other', 403), ('worker', 200)]:
        headers = _headers(tokens[role]) if role else {}
        assert client.post('/verify/review/bundle', headers=headers,
                           content=b'private tensors').status_code == expected
    row.status = 'running'
    core.db.save(row)
    verifier.verifying = row.id
    for role, expected in [(None, 401), ('other', 403), ('worker', 403), ('verifier', 200)]:
        headers = _headers(tokens[role]) if role else {}
        assert client.get('/verify/review/bundle', headers=headers).status_code == expected
    for role, expected in [(None, 401), ('other', 403), ('worker', 403), ('verifier', 200)]:
        headers = _headers(tokens[role]) if role else {}
        assert client.post('/verify/review/result', headers=headers,
                           content=b'replay').status_code == expected
    row.status = 'passed'
    core.db.save(row)
    verifier.verifying = None
    assert client.get('/verify/review/bundle', headers=_headers(tokens['verifier'])).status_code == 403
    with pytest.raises(PermissionError):
        _register(core, tokens['other'], 'verifier-node')
    assert client.get('/verify/review/bundle', headers=_headers(tokens['other'])).status_code == 403
