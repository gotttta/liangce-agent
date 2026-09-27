"""Accepting the same persisted run can resume without duplicate algorithm records."""
from pathlib import Path

import pytest

from core.algorithm_registry import AlgorithmRegistry


def state():
    return {'description': 'Find bright regions', 'pipeline': {
        'steps': [{'id': 'mask', 'op': 'global_threshold', 'input': 'image', 'params': {}}]}}


def test_stable_publication_returns_original_record_without_republishing(tmp_path, monkeypatch):
    registry = AlgorithmRegistry(tmp_path)
    first = registry.publish('task_1', state(), publication_id='run_1:accept')
    original = Path(first['path']).read_bytes()
    def forbidden(*args):
        pytest.fail('completed publication was written again')
    monkeypatch.setattr(AlgorithmRegistry, '_write_json', forbidden)
    repeated = AlgorithmRegistry(tmp_path).publish('task_1', {'pipeline': {'changed': True}},
                                                  note='different retry note', publication_id='run_1:accept')
    assert repeated == first
    assert Path(first['path']).read_bytes() == original
    assert len(list(tmp_path.glob('algorithm_*/algorithm.json'))) == 1


def test_publication_can_resume_after_failure_before_json_commit(tmp_path, monkeypatch):
    registry = AlgorithmRegistry(tmp_path)
    def failed_write(*args):
        raise OSError('disk unavailable')
    with monkeypatch.context() as patch:
        patch.setattr(AlgorithmRegistry, '_write_json', failed_write)
        with pytest.raises(OSError, match='disk unavailable'):
            registry.publish('task_1', state(), publication_id='run_1:accept')
    assert len(list(tmp_path.glob('algorithm_*'))) == 1
    published = AlgorithmRegistry(tmp_path).publish('task_1', state(), publication_id='run_1:accept')
    assert Path(published['path']).is_file()
    assert len(list(tmp_path.glob('algorithm_*'))) == 1


def test_incomplete_json_is_repaired_in_same_publication_directory(tmp_path):
    registry = AlgorithmRegistry(tmp_path)
    first = registry.publish('task_1', state(), publication_id='run_1:accept')
    path = Path(first['path'])
    path.write_text('{"schema_version":', encoding='utf-8')
    restored = registry.publish('task_1', state(), publication_id='run_1:accept')
    assert restored['id'] == first['id']
    assert restored['pipeline'] == state()['pipeline']
    assert registry.list_algorithms()[0]['id'] == first['id']


def test_stable_publication_cannot_be_reused_by_another_task(tmp_path):
    registry = AlgorithmRegistry(tmp_path)
    first = registry.publish('task_1', state(), publication_id='run_1:accept')
    before = Path(first['path']).read_bytes()
    with pytest.raises(ValueError, match='source_task_id'):
        registry.publish('task_2', state(), publication_id='run_1:accept')
    assert Path(first['path']).read_bytes() == before


def test_publication_keys_cannot_escape_registry_and_legacy_calls_remain_unique(tmp_path):
    registry = AlgorithmRegistry(tmp_path)
    published = registry.publish('task_1', state(), publication_id='../../elsewhere/action')
    assert Path(published['path']).parent.parent == tmp_path
    assert registry.publish('task_1', state())['id'] != registry.publish('task_1', state())['id']


@pytest.mark.parametrize('publication_id', ['', ' ', 123])
def test_invalid_publication_key_is_rejected_before_creating_record(tmp_path, publication_id):
    registry = AlgorithmRegistry(tmp_path)
    with pytest.raises(ValueError, match='publication_id'):
        registry.publish('task_1', state(), publication_id=publication_id)
    assert not list(tmp_path.iterdir())
