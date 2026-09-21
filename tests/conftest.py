"""Docker integration tests require an explicitly built local image."""
import os
import subprocess
import pytest


@pytest.fixture
def docker_sandbox():
    from core.sandbox import _docker_binary, DEFAULT_IMAGE, SandboxExecutionError
    try:
        docker = _docker_binary()
        probe = subprocess.run([docker, 'image', 'inspect',
                                os.environ.get('LIANGCE_SANDBOX_IMAGE', DEFAULT_IMAGE)],
                               capture_output=True, timeout=10)
        if probe.returncode:
            raise RuntimeError('Docker daemon or sandbox image is unavailable')
    except (SandboxExecutionError, RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        if os.environ.get('LIANGCE_REQUIRE_DOCKER_TESTS') == '1':
            pytest.fail(str(exc))
        pytest.skip(str(exc))
    return docker


@pytest.fixture(autouse=True)
def isolated_graph_checkpoints(tmp_path, monkeypatch):
    """Persistent graph IDs must not leak across tests or into app data."""
    monkeypatch.setenv('LIANGCE_CHECKPOINT_PATH', str(tmp_path / 'checkpoints.sqlite3'))
    yield
    from core.memory.checkpoints import close_checkpointers
    close_checkpointers()
