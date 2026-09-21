import json
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest

import core.sandbox as sandbox


def pipeline(source=None):
    if source is None:
        return {'steps': [{'id': 'final_mask', 'op': 'global_threshold', 'input': 'image', 'params': {}}]}
    return {'generated_operators': [{'name': 'custom', 'source': source}],
            'steps': [{'id': 'final_mask', 'op': 'custom', 'input': 'image', 'params': {}}]}


def test_container_configuration_has_no_host_mounts_or_privilege():
    args = sandbox._run_args('docker', 'test', sandbox.SandboxLimits())
    for flag in ['--network=none', '--read-only', '--user=65534:65534', '--cap-drop=ALL',
                 '--security-opt=no-new-privileges:true', '--pull=never', '--ipc=none']:
        assert flag in args
    assert not {'--privileged', '-v', '--volume', '--mount', '--env-file'} & set(args)
    assert args[args.index('--memory') + 1] == args[args.index('--memory-swap') + 1]


def test_docker_unavailable_fails_without_running_generated_source(monkeypatch, tmp_path):
    sentinel = tmp_path / 'sentinel'
    def unavailable():
        raise sandbox.SandboxExecutionError('Docker is required', code='sandbox_unavailable')
    monkeypatch.setattr(sandbox, '_docker_binary', unavailable)
    with pytest.raises(sandbox.SandboxExecutionError) as error:
        sandbox.execute_pipeline_sandbox(np.ones((3, 3)), pipeline(
            f"open({str(sentinel)!r}, 'w').write('oops')\ndef apply(data, params): return data > 0"))
    assert error.value.code == 'sandbox_unavailable'
    assert not sentinel.exists()


@pytest.mark.parametrize('script,timeout,max_bytes,code', [
    ('import time; time.sleep(10)', 0.1, 1024, 'timeout'),
    ('import os; os.write(1, b"x" * 10000)', 2, 128, 'resource_limit'),
    ('import os; os.write(2, b"x" * 10000)', 2, 128, 'resource_limit'),
])
def test_output_and_time_are_bounded(script, timeout, max_bytes, code):
    process = subprocess.Popen([sys.executable, '-c', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with pytest.raises(sandbox.SandboxExecutionError) as error:
            sandbox._bounded_output(process, timeout, max_bytes)
        assert error.value.code == code
    finally:
        process.kill()
        process.wait()
        process.stdout.close()
        process.stderr.close()


@pytest.mark.parametrize('output,expected', [
    (b'not json', 'invalid_output'),
    (json.dumps({'ok': True, 'result': {'mask': [[1]]}}).encode(), 'invalid_output'),
    (json.dumps({'ok': True, 'result': {'mask': [[2, 2], [2, 2]]}}).encode(), 'invalid_output'),
])
def test_untrusted_output_rejected_and_container_removed(monkeypatch, output, expected):
    calls = []
    class Pipe:
        def close(self): pass
    process = SimpleNamespace(returncode=0, stdout=Pipe(), stderr=Pipe(), poll=lambda: 0, wait=lambda: 0)
    monkeypatch.setattr(sandbox, '_docker_binary', lambda: 'docker')
    monkeypatch.setattr(sandbox.subprocess, 'Popen', lambda *args, **kwargs: process)
    monkeypatch.setattr(sandbox, '_bounded_output', lambda *args: (output, b''))
    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stderr=b'')
    monkeypatch.setattr(sandbox.subprocess, 'run', run)
    with pytest.raises(sandbox.SandboxExecutionError) as error:
        sandbox.execute_pipeline_sandbox(np.ones((2, 2)), pipeline())
    assert error.value.code == expected
    assert calls[-1][:3] == ['docker', 'rm', '-f']


def test_timeout_also_removes_container(monkeypatch):
    calls = []
    class Pipe:
        def close(self): pass
    process = SimpleNamespace(stdout=Pipe(), stderr=Pipe(), poll=lambda: None,
                              kill=lambda: calls.append('kill-cli'), wait=lambda: 0)
    monkeypatch.setattr(sandbox, '_docker_binary', lambda: 'docker')
    monkeypatch.setattr(sandbox.subprocess, 'Popen', lambda *args, **kwargs: process)
    def timeout(*args):
        raise sandbox.SandboxExecutionError('timeout', code='timeout')
    monkeypatch.setattr(sandbox, '_bounded_output', timeout)
    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stderr=b'')
    monkeypatch.setattr(sandbox.subprocess, 'run', run)
    with pytest.raises(sandbox.SandboxExecutionError) as error:
        sandbox.execute_pipeline_sandbox(np.ones((2, 2)), pipeline())
    assert error.value.code == 'timeout'
    assert any(isinstance(call, list) and call[:3] == ['docker', 'rm', '-f'] for call in calls)
    assert 'kill-cli' in calls


def test_docker_isolation(docker_sandbox):
    source = '''import os
import socket
from pathlib import Path
def apply(data, params):
    assert os.getuid() != 0
    assert not Path('/var/run/docker.sock').exists()
    assert not Path('/Users/ss').exists()
    assert 'DASHSCOPE_API_KEY' not in os.environ
    try:
        Path('/app/escape').write_text('oops')
    except OSError:
        pass
    else:
        raise AssertionError('root filesystem is writable')
    # Docker Desktop may expose dormant tunnel interfaces even with network=none.
    routes = Path('/proc/net/route').read_text().splitlines()[1:]
    assert not any(line.split()[1] == '00000000' for line in routes), 'default route exists'
    with socket.socket() as sock:
        sock.settimeout(0.3)
        try:
            sock.connect(('1.1.1.1', 443))
        except OSError:
            pass
        else:
            raise AssertionError('network is available')
    Path('/tmp/test').write_text('allowed')
    return data > 0
'''
    result = sandbox.execute_pipeline_sandbox(np.ones((3, 3)), pipeline(source))
    assert result.mask.data.all()


def test_docker_kills_infinite_loop_and_removes_container(docker_sandbox, monkeypatch):
    names = []
    original = sandbox._run_args
    def capture(docker, name, limits):
        names.append(name)
        return original(docker, name, limits)
    monkeypatch.setattr(sandbox, '_run_args', capture)
    with pytest.raises(sandbox.SandboxExecutionError) as error:
        sandbox.execute_pipeline_sandbox(np.ones((3, 3)), pipeline(
            'def apply(data, params):\n    while True: pass'), sandbox.SandboxLimits(timeout_seconds=3))
    assert error.value.code == 'timeout'
    remaining = subprocess.run([docker_sandbox, 'ps', '-aq', '--filter', 'name=' + names[0]],
                               capture_output=True, timeout=10, check=True)
    assert not remaining.stdout.strip()


def test_docker_memory_limit(docker_sandbox):
    with pytest.raises(sandbox.SandboxExecutionError):
        sandbox.execute_pipeline_sandbox(np.ones((3, 3)), pipeline(
            'def apply(data, params):\n    data = bytearray(2 * 1024**3)\n    return data'),
            sandbox.SandboxLimits(memory_mb=256))


def test_docker_process_limit(docker_sandbox):
    source = '''import subprocess
def apply(data, params):
    children = []
    limited = False
    try:
        for _ in range(100):
            try:
                children.append(subprocess.Popen(['sleep', '10']))
            except OSError:
                limited = True
                break
    finally:
        for child in children:
            child.kill()
            child.wait()
    assert limited, 'PID limit not enforced'
    return data > 0
'''
    result = sandbox.execute_pipeline_sandbox(np.ones((3, 3)), pipeline(source),
                                              sandbox.SandboxLimits(pids=24))
    assert result.mask.data.all()


def test_docker_output_flood_is_bounded(docker_sandbox, monkeypatch):
    original = sandbox._bounded_output
    monkeypatch.setattr(sandbox, '_bounded_output', lambda proc, timeout: original(proc, timeout, 4096))
    with pytest.raises(sandbox.SandboxExecutionError) as error:
        sandbox.execute_pipeline_sandbox(np.ones((3, 3)), pipeline(
            'import os\ndef apply(data, params):\n    while True: os.write(2, b"x" * 8192)'))
    assert error.value.code == 'resource_limit'


def test_valid_result_uses_original_pipeline_not_container_claim(monkeypatch):
    original = pipeline()
    output = json.dumps({'ok': True, 'result': {
        'mask': [[1, 0], [0, 1]], 'pipeline': {'name': 'forged'},
        'trace': [], 'artifacts': {}}}).encode()
    class Pipe:
        def close(self): pass
    process = SimpleNamespace(returncode=0, stdout=Pipe(), stderr=Pipe(), poll=lambda: 0, wait=lambda: 0)
    monkeypatch.setattr(sandbox, '_docker_binary', lambda: 'docker')
    monkeypatch.setattr(sandbox.subprocess, 'Popen', lambda *args, **kwargs: process)
    monkeypatch.setattr(sandbox, '_bounded_output', lambda *args: (output, b''))
    monkeypatch.setattr(sandbox.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stderr=b''))
    result = sandbox.execute_pipeline_sandbox(np.ones((2, 2)), original)
    assert result.pipeline == original
    assert result.mask.data.sum() == 2


def test_cleanup_failure_is_reported(monkeypatch):
    class Pipe:
        def close(self): pass
    process = SimpleNamespace(returncode=0, stdout=Pipe(), stderr=Pipe(), poll=lambda: 0, wait=lambda: 0)
    monkeypatch.setattr(sandbox, '_docker_binary', lambda: 'docker')
    monkeypatch.setattr(sandbox.subprocess, 'Popen', lambda *args, **kwargs: process)
    monkeypatch.setattr(sandbox, '_bounded_output', lambda *args: (b'{}', b''))
    monkeypatch.setattr(sandbox.subprocess, 'run', lambda args, **kwargs:
                        SimpleNamespace(returncode=1 if args[1] == 'rm' else 0, stderr=b'daemon unavailable'))
    with pytest.raises(sandbox.SandboxExecutionError) as error:
        sandbox.execute_pipeline_sandbox(np.ones((2, 2)), pipeline())
    assert error.value.code == 'execution_failed'
    assert any('Docker cleanup failed' in note for note in error.value.__notes__)


def test_auto_remove_race_requires_confirmed_absence(monkeypatch):
    class Pipe:
        def close(self): pass
    process = SimpleNamespace(returncode=0, stdout=Pipe(), stderr=Pipe(), poll=lambda: 0, wait=lambda: 0)
    monkeypatch.setattr(sandbox, '_docker_binary', lambda: 'docker')
    monkeypatch.setattr(sandbox.subprocess, 'Popen', lambda *args, **kwargs: process)
    def flooded(*args):
        raise sandbox.SandboxExecutionError('output limit', code='resource_limit')
    monkeypatch.setattr(sandbox, '_bounded_output', flooded)
    def run(args, **kwargs):
        if args[1] == 'rm':
            return SimpleNamespace(returncode=1, stderr=b'removal is already in progress')
        if args[1] == 'inspect':
            return SimpleNamespace(returncode=1, stderr=b'Error: No such object: container')
        return SimpleNamespace(returncode=0, stderr=b'')
    monkeypatch.setattr(sandbox.subprocess, 'run', run)
    with pytest.raises(sandbox.SandboxExecutionError) as error:
        sandbox.execute_pipeline_sandbox(np.ones((2, 2)), pipeline())
    assert error.value.code == 'resource_limit'
