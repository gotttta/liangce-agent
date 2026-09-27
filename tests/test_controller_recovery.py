"""Fault injection at controller checkpoints, action receipts and cancellation boundaries."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import json

from PIL import Image
import pytest

from core.agent_graph import run_agent_graph
from core.memory.checkpoints import get_checkpointer
from core.orchestration import RunController
from core.orchestration_runtime import ActionStore, RunLimits, TaskBusyError, initial_budget, task_lock
from core.request_control import RequestCancelled, RequestControl, control


class SimulatedCrash(BaseException):
    pass


class NeedsInputProvider:
    def __init__(self):
        self.calls = 0

    def propose_action(self, *args, **kwargs):
        self.calls += 1
        return {'kind': 'needs_input', 'reason': 'Specify the expected boundary'}


@pytest.fixture
def target(tmp_path, monkeypatch):
    monkeypatch.setattr('core.sandbox.check_sandbox_available', lambda: {'image_id': 'test-image'})
    path = tmp_path / 'input.png'
    Image.new('L', (16, 16), 128).save(path)
    return path


def pending_state(target):
    budget = initial_budget(RunLimits())
    budget['usage']['model_calls'] = 1
    return {'orchestration_version': 1, 'run_id': 'recovery_run',
            'run_dir': str(target.parent / 'run'), 'graph_thread_id': 'recovery_run',
            'target_image_path': str(target), 'description': 'Find bright regions',
            'output_root': str(target.parent), 'budget': budget, 'phase': 'propose',
            'run_status': 'running', 'action_sequence': 1, 'state_version': 1,
            'failure_counts': {}, 'pending_action': {'id': 'action_0001', 'kind': 'propose',
                                                   'input_hash': 'reserved-inputs'}}


def test_result_saved_before_receipt_is_recovered_without_repeating_model(target, monkeypatch):
    provider = NeedsInputProvider()
    state = pending_state(target)
    pending = state['pending_action']
    def crash_before_receipt(*args):
        raise SimulatedCrash('process exited before receipt write')
    with monkeypatch.context() as patch:
        patch.setattr(ActionStore, 'complete', crash_before_receipt)
        with pytest.raises(SimulatedCrash):
            RunController(provider).perform(state)
    store = ActionStore(state['run_dir'])
    assert store.has_started(pending['id'], pending['input_hash'])
    assert store.load(pending['id'], pending['input_hash']) is None
    result = RunController(provider).perform(state)
    assert provider.calls == 1
    assert result['action_outcome']['data']['kind'] == 'needs_input'
    assert store.load(pending['id'], pending['input_hash']) == result['action_outcome']
    assert result['budget'] == state['budget']


def test_corrupt_saved_result_is_not_promoted_to_completed_receipt(target, monkeypatch):
    from core.experiments.drafts import atomic_json
    state = pending_state(target)
    pending = state['pending_action']
    def crash_before_receipt(*args):
        raise SimulatedCrash('process exited before receipt write')
    with monkeypatch.context() as patch:
        patch.setattr(ActionStore, 'complete', crash_before_receipt)
        with pytest.raises(SimulatedCrash):
            RunController(NeedsInputProvider()).perform(state)
    path = Path(state['run_dir']) / 'actions' / pending['id'] / 'result.json'
    record = json.loads(path.read_text())
    record['outcome']['data']['reason'] = 'corrupted after durable write'
    atomic_json(path, record)
    with pytest.raises(ValueError, match='(?i)(integrity|match|hash)'):
        RunController(None).perform(state)
    assert ActionStore(state['run_dir']).load(pending['id'], pending['input_hash']) is None


def test_receipt_saved_before_effect_checkpoint_replays_unchanged_budget(target):
    provider = NeedsInputProvider()
    state = pending_state(target)
    first = RunController(provider).perform(deepcopy(state))
    # The pre-effect state is the only checkpoint that survived the crash.
    recovered = RunController(provider).perform(deepcopy(state))
    assert recovered['action_outcome'] == first['action_outcome']
    assert recovered['budget'] == state['budget']
    assert provider.calls == 1
    finished = RunController(provider).advance(recovered)
    assert finished['run_status'] == 'awaiting_feedback'
    assert finished['pending_action'] is None


def test_lost_model_response_stops_as_unknown_without_resending(target, monkeypatch):
    from core.experiments.drafts import atomic_json
    provider = NeedsInputProvider()
    state = pending_state(target)
    def lose_result(path, value):
        if Path(path).name == 'result.json':
            raise SimulatedCrash('model returned, process exited before durable save')
        return atomic_json(path, value)
    with monkeypatch.context() as patch:
        patch.setattr('core.orchestration.atomic_json', lose_result)
        with pytest.raises(SimulatedCrash):
            RunController(provider).perform(state)
    recovered = RunController(provider).perform(state)
    assert recovered['action_outcome']['status'] == 'unknown'
    finished = RunController(provider).advance(recovered)
    assert finished['run_status'] == 'interrupted'
    assert finished['stop_reason'] == 'unknown_action_result'
    assert finished['budget'] == state['budget']
    assert provider.calls == 1


def test_cancel_during_effect_produces_durable_terminal_state(target, monkeypatch):
    state = pending_state(target)
    parent = RequestControl(timeout=600)
    token = control.set(parent)
    def cancel(*args):
        parent.cancelled.set()
        raise RequestCancelled('user cancelled during model call')
    monkeypatch.setattr(RunController, '_perform_action', cancel)
    try:
        result = RunController(None).perform(state)
        finished = RunController(None).advance(result)
    finally:
        control.reset(token)
    assert finished['run_status'] == 'cancelled'
    assert finished['pending_action'] is None
    assert finished['budget'] == state['budget']
    pending = state['pending_action']
    assert ActionStore(state['run_dir']).load(pending['id'], pending['input_hash'])['status'] == 'cancelled'


def test_proposal_deadline_preserves_execute_and_review_time(target, monkeypatch):
    state = pending_state(target)
    monkeypatch.setattr(RunController, '_remaining', lambda *args: 60)
    timeouts = []
    def capture_timeout(*args):
        timeouts.append(control.get().timeout)
        return {'kind': 'needs_input', 'reason': 'Need a reference'}
    monkeypatch.setattr(RunController, '_perform_action', capture_timeout)
    RunController(None).perform(state)
    assert timeouts == [10]


def test_recovering_existing_thread_locks_its_original_task(target):
    provider = NeedsInputProvider()
    state = run_agent_graph(target, 'Find bright regions', provider=provider,
                            output_root=target.parent / 'outputs', thread_id='existing_thread')
    def restore():
        return run_agent_graph(target, 'Find bright regions', provider=provider,
                               output_root=target.parent / 'outputs', thread_id='existing_thread')
    with task_lock(state['memory_context']['task_root'], state['task_id']):
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(TaskBusyError):
                pool.submit(restore).result(timeout=10)
    assert provider.calls == 1


def test_proposal_cannot_start_before_budget_checkpoint_is_durable(target, monkeypatch):
    from langgraph.checkpoint.sqlite import SqliteSaver
    provider = NeedsInputProvider()
    original = SqliteSaver.put
    def fail_reserved_checkpoint(self, config, checkpoint, metadata, new_versions):
        state = checkpoint.get('channel_values', {}).get('__root__') or {}
        pending = state.get('pending_action') or {}
        if pending.get('kind') == 'propose' and not state.get('action_outcome'):
            raise SimulatedCrash('reservation checkpoint could not be committed')
        return original(self, config, checkpoint, metadata, new_versions)
    with monkeypatch.context() as patch:
        patch.setattr(SqliteSaver, 'put', fail_reserved_checkpoint)
        with pytest.raises(SimulatedCrash):
            run_agent_graph(target, 'Find bright regions', provider=provider,
                            output_root=target.parent / 'outputs', thread_id='checkpoint_crash')
    assert provider.calls == 0
    saved = get_checkpointer().get_tuple({'configurable': {'thread_id': 'checkpoint_crash'}})
    assert saved is not None
    deadline = saved.checkpoint['channel_values']['__root__']['budget']['deadline_at']
    result = run_agent_graph(target, 'Find bright regions', provider=provider,
                            output_root=target.parent / 'outputs', thread_id='checkpoint_crash')
    assert provider.calls == 1, {key: result.get(key) for key in (
        'phase', 'pending_action', 'run_status', 'stop_reason', 'action_outcome', 'budget')}
    assert result['budget']['usage']['model_calls'] == 1
    assert result['budget']['deadline_at'] == deadline
    assert result['run_status'] == 'awaiting_feedback'
