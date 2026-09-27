"""A recovered action may remove its own Docker container, never another action's."""
from contextvars import Context
import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

import core.sandbox as sandbox


def completed(returncode=0, stdout=b'', stderr=b''):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def inspection(name, identity='a' * 64, *, owner=None):
    return completed(stdout=json.dumps([{
        'Id': identity, 'Name': '/' + name,
        'Config': {'Labels': {sandbox.ACTION_CONTAINER_LABEL: name if owner is None else owner}},
    }]).encode())


def test_action_names_are_deterministic_and_isolated():
    first = sandbox.action_container_name('run_1', 'execute_1')
    assert first == sandbox.action_container_name('run_1', 'execute_1')
    assert first != sandbox.action_container_name('run_2', 'execute_1')
    assert first != sandbox.action_container_name('run_1', 'execute_2')
    assert '/' not in sandbox.action_container_name('../run', 'unsafe/action')
    args = sandbox._run_args('docker', first, sandbox.SandboxLimits())
    assert args[args.index('--name') + 1] == first
    assert args[args.index('--label') + 1] == sandbox.ACTION_CONTAINER_LABEL + '=' + first


def test_container_identity_context_does_not_leak():
    name = sandbox.action_container_name('run', 'execute')
    token = sandbox.container_name.set(name)
    try:
        assert sandbox.container_name.get() == name
        assert Context().run(sandbox.container_name.get) is None
    finally:
        sandbox.container_name.reset(token)
    assert sandbox.container_name.get() is None


def test_recovery_removes_verified_container_by_id_not_reusable_name(monkeypatch):
    name = sandbox.action_container_name('run', 'execute')
    calls = []
    monkeypatch.setattr(sandbox, '_docker_binary', lambda: 'docker')
    def run(args, **kwargs):
        calls.append((args, kwargs['timeout']))
        return inspection(name) if args[1] == 'inspect' else completed()
    monkeypatch.setattr(sandbox.subprocess, 'run', run)
    sandbox.cleanup_action_container(name)
    assert calls[0][0] == ['docker', 'inspect', '--type', 'container', name]
    assert calls[1][0] == ['docker', 'rm', '-f', 'a' * 64]
    assert all(0 < timeout <= 10 for _, timeout in calls)


@pytest.mark.parametrize('difference', ['owner', 'name', 'id'])
def test_recovery_refuses_unrelated_or_malformed_container(monkeypatch, difference):
    name = sandbox.action_container_name('run', 'execute')
    calls = []
    monkeypatch.setattr(sandbox, '_docker_binary', lambda: 'docker')
    record = json.loads(inspection(name).stdout)
    if difference == 'owner':
        record[0]['Config']['Labels'][sandbox.ACTION_CONTAINER_LABEL] = 'another-action'
    elif difference == 'name':
        record[0]['Name'] = '/another-container'
    else:
        record[0]['Id'] = '--all'
    def run(args, **kwargs):
        calls.append(args)
        return completed(stdout=json.dumps(record).encode())
    monkeypatch.setattr(sandbox.subprocess, 'run', run)
    with pytest.raises(sandbox.SandboxExecutionError) as error:
        sandbox.cleanup_action_container(name)
    assert error.value.code == 'cleanup_failed'
    assert len(calls) == 1 and calls[0][1] == 'inspect'


def test_recovery_treats_already_removed_container_as_complete(monkeypatch):
    calls = []
    monkeypatch.setattr(sandbox, '_docker_binary', lambda: 'docker')
    def run(args, **kwargs):
        calls.append(args)
        return completed(returncode=1, stderr=b'Error: No such object: gone')
    monkeypatch.setattr(sandbox.subprocess, 'run', run)
    sandbox.cleanup_action_container(sandbox.action_container_name('run', 'execute'))
    assert len(calls) == 1


def test_recovery_cleanup_has_one_shared_deadline(monkeypatch):
    name = sandbox.action_container_name('run', 'execute')
    now = [0]
    calls = []
    monkeypatch.setattr(sandbox, '_docker_binary', lambda: 'docker')
    monkeypatch.setattr(sandbox.time, 'monotonic', lambda: now[0])
    def run(args, **kwargs):
        calls.append((args, kwargs['timeout']))
        if args[1] == 'inspect':
            now[0] = 4
            return inspection(name)
        raise subprocess.TimeoutExpired(args, kwargs['timeout'])
    monkeypatch.setattr(sandbox.subprocess, 'run', run)
    with pytest.raises(sandbox.SandboxExecutionError) as error:
        sandbox.cleanup_action_container(name, timeout_seconds=6)
    assert error.value.code == 'cleanup_failed'
    assert calls[0][1] == 5
    assert calls[1][1] == 2


def test_execute_uses_persisted_identity_and_cleans_it_after_cancellation(monkeypatch):
    from core.request_control import RequestCancelled
    name = sandbox.action_container_name('run', 'execute')
    calls = []
    class Pipe:
        def close(self):
            pass
    process = SimpleNamespace(stdout=Pipe(), stderr=Pipe(), poll=lambda: 0, wait=lambda: 0)
    monkeypatch.setattr(sandbox, '_docker_binary', lambda: 'docker')
    monkeypatch.setattr(sandbox.subprocess, 'Popen', lambda *args, **kwargs: process)
    def cancel(*args):
        raise RequestCancelled('cancelled during sandbox output')
    monkeypatch.setattr(sandbox, '_bounded_output', cancel)
    def run(args, **kwargs):
        calls.append(args)
        return inspection(name) if args[1] == 'inspect' else completed()
    monkeypatch.setattr(sandbox.subprocess, 'run', run)
    token = sandbox.container_name.set(name)
    try:
        with pytest.raises(RequestCancelled):
            sandbox.execute_pipeline_sandbox(np.ones((2, 2)), {'steps': [
                {'id': 'mask', 'op': 'global_threshold', 'input': 'image', 'params': {}}]})
    finally:
        sandbox.container_name.reset(token)
    assert calls[0][calls[0].index('--name') + 1] == name
    assert calls[-1] == ['docker', 'rm', '-f', 'a' * 64]


@pytest.mark.parametrize('name', ['other-container', '--all', '../task', 'liangce-sandbox-action-bad'])
def test_invalid_recovery_identity_does_not_call_docker(monkeypatch, name):
    def forbidden():
        pytest.fail('invalid identity reached Docker')
    monkeypatch.setattr(sandbox, '_docker_binary', forbidden)
    with pytest.raises(ValueError):
        sandbox.cleanup_action_container(name)


def test_recovery_removes_real_orphaned_action_container(docker_sandbox):
    import uuid
    name = sandbox.action_container_name(uuid.uuid4().hex, 'execute_1')
    args = sandbox._run_args(docker_sandbox, name, sandbox.SandboxLimits())
    args[-1:-1] = ['--entrypoint', 'python']
    args.extend(['-c', 'import time; time.sleep(30)'])
    subprocess.run(args, capture_output=True, timeout=15, check=True)
    try:
        subprocess.run([docker_sandbox, 'start', name], capture_output=True, timeout=10, check=True)
        sandbox.cleanup_action_container(name)
        probe = subprocess.run([docker_sandbox, 'inspect', '--type', 'container', name],
                               capture_output=True, timeout=5)
        assert probe.returncode and any(message in probe.stderr for message in
            (b'No such object', b'No such container'))
        sandbox.cleanup_action_container(name)
    finally:
        sandbox.cleanup_action_container(name)
