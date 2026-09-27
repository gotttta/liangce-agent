import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from agent_types import normalize_strategy
from core.agent_graph import run_agent_graph, resume_agent_graph
from core.orchestration import RunController
from core.orchestration_runtime import ActionStore
from core.pipelines.dsl import strategy_to_pipeline


def proposal(sensitivity=1):
    strategy = normalize_strategy({'segmentation': {
        'method': 'bright_threshold', 'sensitivity': sensitivity,
        'min_area_px': 2, 'morphology': 'none'}})
    return {'kind': 'propose', 'understanding': {
        'task_summary': 'Find the bright region', 'recommended_strategy': strategy,
        'target_constraints': {}, 'rendering': {}},
        'pipeline': strategy_to_pipeline(strategy), 'change_reason': 'Threshold bright pixels'}


class Provider:
    def __init__(self, actions=None, reviews=None):
        self.actions = list(actions or [proposal()])
        self.reviews = list(reviews or ['present'])
        self.calls = []

    def propose_action(self, target, description, context=None, **kwargs):
        self.calls.append(('propose', context))
        action = self.actions.pop(0)
        if isinstance(action, BaseException):
            raise action
        return action

    def review_action(self, target, description, candidates, context=None, **kwargs):
        self.calls.append(('review', context))
        decision = self.reviews.pop(0)
        if isinstance(decision, dict):
            return decision
        return {'kind': 'review', 'review': {'decision': decision,
            'selected_candidate': candidates[0]['name'], 'reason': 'Checked the actual result',
            'observed_issues': [] if decision == 'present' else ['Boundary needs correction']}}


@pytest.fixture
def target(tmp_path, monkeypatch):
    monkeypatch.setattr('core.sandbox.check_sandbox_available', lambda: {'image_id': 'test-image'})
    pixels = np.zeros((32, 32), dtype=np.uint8)
    pixels[10:18, 10:18] = 255
    path = tmp_path / 'input.png'
    Image.fromarray(pixels).save(path)
    return path


def run(target, provider, **kwargs):
    return run_agent_graph(target, 'Find bright regions', provider=provider,
                           output_root=target.parent / 'outputs', **kwargs)


def test_controller_delivers_with_two_model_calls_and_one_execution(target):
    provider = Provider()
    state = run(target, provider)
    assert state['run_status'] == 'awaiting_feedback'
    assert state['stop_reason'] == 'review_passed'
    assert state['budget']['usage'] == {'model_calls': 2, 'executions': 1}
    assert [kind for kind, _ in provider.calls] == ['propose', 'review']
    assert state['review']['acceptance']['overall_passed']
    assert Path(state['annotated_image_path']).is_file()
    assert provider.calls[-1][1]['latest_experiment']['experiment_id'] == state['selected_experiment_id']
    task = json.loads((target.parent / 'workspace/tasks' / state['task_id'] / 'task.json').read_text())
    assert task['status'] == 'waiting_for_feedback'
    resumed = resume_agent_graph(state['graph_thread_id'], {'action': 'accept'})
    assert resumed['run_status'] == 'completed'
    assert resumed['budget']['usage'] == state['budget']['usage']


def test_first_proposal_receives_accepted_algorithm_without_extra_model_call(target):
    class Registry:
        def search(self, query, **kwargs):
            assert query['task_summary'] == 'Find bright regions'
            return [{'algorithm_id': 'accepted_1', 'name': 'Prior threshold', 'score': .09,
                     'match_reasons': ['description_similarity'], 'pipeline': proposal()['pipeline']}]
    provider = Provider()
    state = run(target, provider, algorithm_registry=Registry())
    assert provider.calls[0][1]['procedural_memory'][0]['pipeline'] == proposal()['pipeline']
    assert state['budget']['usage']['model_calls'] == 2


def test_revision_uses_one_budget_and_preserves_rejected_experiment(target):
    provider = Provider([proposal(), proposal(.5)], ['revise', 'present'])
    state = run(target, provider)
    assert state['budget']['usage'] == {'model_calls': 4, 'executions': 2}
    assert state['run_status'] == 'awaiting_feedback'
    second = provider.calls[2][1]
    assert second['experiment_summaries'][0]['acceptance_status'] == 'rejected'
    assert second['current_draft']['pipeline']
    assert second['review']['decision'] == 'revise'


def test_budget_exhaustion_preserves_rendered_experiment(target, monkeypatch):
    monkeypatch.setenv('LIANGCE_RUN_MAX_EXECUTIONS', '1')
    provider = Provider(reviews=['revise'])
    state = run(target, provider)
    assert state['run_status'] == 'stopped'
    assert state['stop_reason'] == 'execution_budget_exhausted'
    assert Path(state['annotated_image_path']).is_file()
    assert not state['decision']['automatic_review_passed']
    assert len(provider.calls) == 2


def test_failed_revision_keeps_usable_result_and_its_review(target, monkeypatch):
    from core.experiments.runner import execute_pipeline_sandbox
    monkeypatch.setenv('LIANGCE_RUN_MAX_EXECUTIONS', '2')
    empty = proposal()
    for node in empty['pipeline']['steps']:
        if node['op'] == 'global_threshold':
            node['params']['sensitivity'] = 10
    def fail_revision(image, pipeline, **kwargs):
        if any(node.get('params', {}).get('sensitivity') == 10 for node in pipeline['steps']):
            raise RuntimeError('Revised algorithm failed during execution')
        return execute_pipeline_sandbox(image, pipeline, **kwargs)
    monkeypatch.setattr('core.experiments.runner.execute_pipeline_sandbox', fail_revision)
    provider = Provider([proposal(), empty], ['revise'])
    state = run(target, provider)
    assert state['stop_reason'] == 'execution_budget_exhausted'
    assert state['budget']['usage']['executions'] == 2
    assert len(state['experiment_records']) == 2
    assert state['measurements']['summary']['count'] == 1
    assert state['selected_experiment_id'] == state['experiment_records'][0]['experiment_id']
    assert state['candidate_attempts'][0]['acceptance_status'] == 'rejected'
    assert not state['decision']['automatic_review_passed']
    assert 'iteration_0' in state['annotated_image_path']
    assert Path(state['annotated_image_path']).is_file()


def test_duplicate_failed_algorithm_does_not_execute_again(target):
    provider = Provider([proposal(), proposal()], ['revise'])
    state = run(target, provider)
    assert state['stop_reason'] == 'duplicate_pipeline'
    assert state['budget']['usage']['executions'] == 1
    assert state['experiment_records'][0]['acceptance_status'] == 'rejected'


def test_review_can_read_with_no_remaining_execution_budget(target, monkeypatch):
    monkeypatch.setenv('LIANGCE_RUN_MAX_EXECUTIONS', '1')
    read = {'kind': 'read', 'requests': [{'tool': 'query_operators', 'arguments': {'names': ['normalize']}}]}
    provider = Provider(reviews=[read, 'present'])
    state = run(target, provider)
    assert state['stop_reason'] == 'review_passed'
    assert state['budget']['usage'] == {'model_calls': 3, 'executions': 1}
    assert provider.calls[-1][1]['read_results'][0]['data']['operators']


def test_provider_failure_does_not_generate_fallback(target):
    provider = Provider([RuntimeError('provider offline')])
    state = run(target, provider)
    assert state['run_status'] == 'failed'
    assert state['budget']['usage']['executions'] == 0
    assert not state.get('selected_candidate')


def test_failed_execution_preview_cannot_be_accepted_as_algorithm(target, monkeypatch):
    monkeypatch.setenv('LIANGCE_RUN_MAX_EXECUTIONS', '1')
    def fail(*args, **kwargs):
        raise RuntimeError('Algorithm execution failed')
    monkeypatch.setattr('core.experiments.runner.execute_pipeline_sandbox', fail)
    state = run(target, Provider())
    assert Path(state['annotated_image_path']).is_file()
    with pytest.raises(ValueError, match='未成功执行'):
        resume_agent_graph(state['graph_thread_id'], {'action': 'accept'})


def test_completed_action_receipt_is_replayed_without_provider(target):
    provider = Provider()
    state = run(target, provider)
    action = {'id': 'replay_test', 'kind': 'propose', 'input_hash': 'hash'}
    outcome = {'status': 'ok', 'data': proposal()}
    ActionStore(state['run_dir']).complete(action, outcome)
    restored = RunController(None).perform({**state, 'pending_action': action})
    assert restored['action_outcome'] == outcome
    assert len(provider.calls) == 2


def test_started_unknown_action_is_not_resubmitted(target):
    state = run(target, Provider())
    action = {'id': 'unknown_test', 'kind': 'propose', 'input_hash': 'hash'}
    ActionStore(state['run_dir']).prepare(action)
    restored = RunController(None).perform({**state, 'pending_action': action})
    assert restored['action_outcome']['status'] == 'unknown'


def test_read_request_cannot_mutate_pipeline(target):
    provider = Provider([{'kind': 'read', 'requests': [{'tool': 'execute_pipeline', 'arguments': {}}]},
                         {'kind': 'read', 'requests': [{'tool': 'execute_pipeline', 'arguments': {}}]}])
    state = run(target, provider)
    assert state['budget']['usage']['executions'] == 0
    assert state['run_status'] == 'stopped'


def test_action_narrations_are_persisted_to_trajectory(target):
    provider = Provider()
    state = run(target, provider)
    narrations = {entry['node']: entry['details'].get('narration')
                  for entry in state['trajectory'] if entry.get('node') in {'propose', 'review', 'execute'}}
    assert narrations['propose'] == 'Threshold bright pixels'
    assert narrations['review'] == 'Checked the actual result'
    assert narrations['execute'].startswith('实验完成：检出')


def test_failed_action_marks_trajectory_row_failed(target):
    provider = Provider([RuntimeError('provider offline')])
    state = run(target, provider)
    entry = next(item for item in state['trajectory'] if item.get('node') == 'propose')
    assert entry['status'] == 'failed'
    assert 'narration' not in entry['details']
