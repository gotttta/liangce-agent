"""Recovery boundaries use disk records, with no model service or sandbox required."""
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import sys
import threading

import pytest

from core.experiments.drafts import DraftStore, atomic_json
from core.orchestration_runtime import (
    ActionConflictError, ActionStore, LinkedCancellation, RunDeadline, RunLimits, TaskBusyError,
    initial_budget, task_lock,
)
from core.tools.contracts import ToolError


def pipeline():
    return {'steps': [{'id': 'mask', 'op': 'global_threshold', 'input': 'image',
                       'params': {'polarity': 'bright', 'sensitivity': 1}}]}


def action(action_id='proposal_1', input_hash='inputs-v1'):
    return {'id': action_id, 'kind': 'propose', 'input_hash': input_hash,
            'input_refs': {'draft_id': 'draft_1'}}


def test_budget_settings_are_validated_and_persisted(monkeypatch):
    settings = {
        'LIANGCE_RUN_TIMEOUT_SECONDS': '300',
        'LIANGCE_RUN_MAX_MODEL_CALLS': '8',
        'LIANGCE_RUN_MAX_EXECUTIONS': '2',
        'LIANGCE_MODEL_CALL_TIMEOUT_SECONDS': '45',
    }
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    limits = RunLimits.from_env()
    budget = initial_budget(limits, now=1000)
    assert budget == {
        'limits': {'timeout_seconds': 300, 'max_model_calls': 8, 'max_executions': 2,
                   'model_call_timeout_seconds': 45},
        'usage': {'model_calls': 0, 'executions': 0}, 'started_at': 1000, 'deadline_at': 1300,
    }
    budget['usage']['model_calls'] = 4
    restored = json.loads(json.dumps(budget))
    assert restored['usage']['model_calls'] == 4
    assert restored['deadline_at'] == 1300


@pytest.mark.parametrize('field,value', [
    ('timeout_seconds', 0), ('timeout_seconds', float('nan')),
    ('model_call_timeout_seconds', float('inf')), ('max_model_calls', -1),
    ('max_executions', 1.5), ('max_executions', True),
])
def test_invalid_limits_fail_before_starting(field, value):
    with pytest.raises(ValueError):
        RunLimits(**{field: value})


def test_deadline_does_not_extend_on_wall_clock_rollback_and_restore_uses_remaining_time():
    clock = {'wall': 1000, 'mono': 5}
    budget = initial_budget(RunLimits(timeout_seconds=60), now=clock['wall'])
    deadline = RunDeadline(budget['deadline_at'], wall_clock=lambda: clock['wall'],
                           monotonic=lambda: clock['mono'])
    clock.update(wall=500, mono=35)
    assert deadline.remaining() == 30
    clock.update(wall=1045, mono=1)
    restored = RunDeadline(budget['deadline_at'], wall_clock=lambda: clock['wall'],
                           monotonic=lambda: clock['mono'])
    assert restored.remaining() == 15
    clock['mono'] = 17
    assert restored.remaining() == 0


def test_reserved_but_not_started_action_can_be_dispatched(tmp_path):
    pending = json.loads(json.dumps(action()))
    store = ActionStore(tmp_path)
    assert store.load(pending['id'], pending['input_hash']) is None
    assert not store.has_started(pending['id'], pending['input_hash'])


def test_started_without_receipt_remains_unknown_after_restart(tmp_path):
    pending = action()
    store = ActionStore(tmp_path)
    store.prepare(pending)
    restored = ActionStore(tmp_path)
    assert restored.has_started(pending['id'], pending['input_hash'])
    assert restored.load(pending['id'], pending['input_hash']) is None
    before = (restored.root / pending['id'] / 'started.json').read_bytes()
    restored.prepare(pending)
    assert (restored.root / pending['id'] / 'started.json').read_bytes() == before


def test_completed_action_replays_after_state_checkpoint_was_lost(tmp_path):
    pending = action()
    outcome = {'status': 'success', 'draft_ref': {'draft_id': 'draft_1', 'revision': 1}}
    store = ActionStore(tmp_path)
    store.prepare(pending)
    store.complete(pending, outcome)
    restored = ActionStore(tmp_path)
    assert restored.load(pending['id'], pending['input_hash']) == outcome
    restored.complete(pending, outcome)
    changed = restored.load(pending['id'], pending['input_hash'])
    changed['draft_ref']['revision'] = 100
    assert restored.load(pending['id'], pending['input_hash']) == outcome
    with pytest.raises(ActionConflictError, match='another outcome'):
        restored.complete(pending, changed)


def test_receipt_can_be_repaired_from_verified_artifact(tmp_path):
    store = ActionStore(tmp_path)
    store.complete(action(), {'artifact_ref': 'experiments/e1/manifest.json'})
    assert store.load('proposal_1', 'inputs-v1')['artifact_ref'].endswith('/manifest.json')


def test_changed_inputs_or_corrupt_receipt_stop_replay(tmp_path):
    store = ActionStore(tmp_path)
    store.complete(action(), {'status': 'success'})
    with pytest.raises(ActionConflictError):
        store.load('proposal_1', 'changed')
    with pytest.raises(ActionConflictError):
        store.has_started('proposal_1', 'changed')
    with pytest.raises(ActionConflictError):
        store.prepare({**action(), 'kind': 'execute'})
    path = store.root / 'proposal_1' / 'outcome.json'
    record = json.loads(path.read_text())
    record['outcome']['status'] = 'changed'
    atomic_json(path, record)
    with pytest.raises(ActionConflictError, match='integrity'):
        store.load('proposal_1', 'inputs-v1')


@pytest.mark.parametrize('unsafe_id', ['..', '../other', '/tmp/task', 'a/b', 'a\\b', '', 'x' * 129])
def test_ids_cannot_escape_store_paths(tmp_path, unsafe_id):
    with pytest.raises(ValueError):
        ActionStore(tmp_path).prepare(action(unsafe_id))
    with pytest.raises(ToolError, match='ID'):
        DraftStore(tmp_path).create(pipeline(), draft_id=unsafe_id)
    with pytest.raises(ValueError):
        with task_lock(tmp_path, unsafe_id):
            pytest.fail('unsafe lock was acquired')


def test_drafts_reload_latest_and_exact_revision_without_session_memory(tmp_path):
    store = DraftStore(tmp_path)
    original = store.create(pipeline(), draft_id='proposal_1', change_reason='first')
    store.edit({'draft_id': original['draft_id'], 'base_revision': 1, 'change_reason': 'adjust',
                'edits': [{'path': '/steps/0/params/sensitivity', 'old': 1, 'new': 2}]})
    restored = DraftStore(tmp_path)
    assert restored.get(original['draft_id'])['revision'] == 2
    assert restored.load(original['draft_id'], 1)['pipeline'] == pipeline()
    with pytest.raises(ToolError) as stale:
        restored.get(original['draft_id'], 1)
    assert stale.value.code == 'revision_conflict'
    assert restored.create(pipeline(), draft_id='proposal_1', change_reason='first') == original
    assert restored.get(original['draft_id'])['revision'] == 2
    with pytest.raises(ToolError) as conflict:
        restored.create({'steps': []}, draft_id='proposal_1')
    assert conflict.value.code == 'draft_conflict'


def test_draft_creation_recovers_after_directory_was_created_without_revision(tmp_path):
    (tmp_path / 'drafts' / 'proposal_1').mkdir(parents=True)
    created = DraftStore(tmp_path).create(pipeline(), draft_id='proposal_1')
    assert DraftStore(tmp_path).load(created['draft_id'])['pipeline'] == pipeline()


def test_modified_draft_is_not_silently_used(tmp_path):
    store = DraftStore(tmp_path)
    created = store.create(pipeline())
    path = store.root / created['draft_id'] / 'revision_1.json'
    record = json.loads(path.read_text())
    record['pipeline']['steps'][0]['params']['sensitivity'] = 10
    atomic_json(path, record)
    with pytest.raises(ToolError) as modified:
        DraftStore(tmp_path).get(created['draft_id'])
    assert modified.value.code == 'draft_modified'


def test_atomic_writes_use_independent_temporary_files_and_leave_complete_json(tmp_path):
    path = tmp_path / 'record.json'
    values = [{'number': number, 'payload': 'x' * 1000} for number in range(30)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda value: atomic_json(path, value), values))
    assert json.loads(path.read_text()) in values
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_write_failure_preserves_previous_document_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / 'record.json'
    atomic_json(path, {'previous': True})
    def fail_replace(*args):
        raise OSError('disk error')
    monkeypatch.setattr('core.experiments.drafts.os.replace', fail_replace)
    with pytest.raises(OSError, match='disk error'):
        atomic_json(path, {'next': True})
    assert json.loads(path.read_text()) == {'previous': True}
    assert list(tmp_path.iterdir()) == [path]


def test_task_lock_is_exclusive_and_released_after_exception(tmp_path):
    def acquire_elsewhere():
        with task_lock(tmp_path, 'task_1'):
            pass
    with pytest.raises(RuntimeError, match='runner failed'):
        with task_lock(tmp_path, 'task_1'):
            with ThreadPoolExecutor(max_workers=1) as pool:
                with pytest.raises(TaskBusyError):
                    pool.submit(acquire_elsewhere).result(timeout=5)
            with task_lock(tmp_path, 'task_2'):
                pass
            raise RuntimeError('runner failed')
    with task_lock(tmp_path, 'task_1'):
        pass


def test_nested_task_locks_reenter_same_thread_but_keep_other_threads_out(tmp_path):
    def acquire_elsewhere():
        with task_lock(tmp_path, 'task_1'):
            return True
    with ThreadPoolExecutor(max_workers=1) as pool:
        with task_lock(tmp_path, 'task_1'):
            with task_lock(tmp_path / '.', 'task_1'):
                with pytest.raises(TaskBusyError):
                    pool.submit(acquire_elsewhere).result(timeout=5)
            with pytest.raises(TaskBusyError):
                pool.submit(acquire_elsewhere).result(timeout=5)
        assert pool.submit(acquire_elsewhere).result(timeout=5)


def test_child_deadline_does_not_cancel_parent_but_parent_cancel_reaches_child():
    from core.request_control import RequestCancelled, RequestControl
    parent_event = threading.Event()
    local = LinkedCancellation(parent_event)
    request = RequestControl(timeout=1, started=-100, cancelled=local)
    with pytest.raises(RequestCancelled):
        request.check()
    assert local.is_set()
    assert not parent_event.is_set()
    next_attempt = LinkedCancellation(parent_event)
    assert not next_attempt.is_set()
    parent_event.set()
    assert next_attempt.is_set()


def test_task_lock_is_released_when_process_exits(tmp_path):
    script = (
        'import os, sys\n'
        'from core.orchestration_runtime import task_lock\n'
        'with task_lock(sys.argv[1], "task_1"):\n'
        '    print("locked", flush=True)\n'
        '    sys.stdin.readline()\n'
        '    os._exit(0)\n'
    )
    process = subprocess.Popen([sys.executable, '-c', script, str(tmp_path)],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == 'locked'
        with pytest.raises(TaskBusyError):
            with task_lock(tmp_path, 'task_1'):
                pytest.fail('parent acquired child lock')
        process.communicate('\n', timeout=20)
        assert process.returncode == 0
        with task_lock(tmp_path, 'task_1'):
            pass
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=20)
