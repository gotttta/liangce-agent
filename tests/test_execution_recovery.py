"""Recover real DSL artifacts after execution completed but its action result was lost."""
from copy import deepcopy
import json
from pathlib import Path

from PIL import Image, ImageDraw
import pytest

from agent_types import normalize_strategy
from core.experiments.artifacts import experiment_scope
from core.experiments.drafts import atomic_json, content_hash
from core.agent_workflow import ToolAgentRuntime
from core.orchestration_runtime import ActionStore, RunLimits, initial_budget
from core.pipelines.dsl import execute_pipeline, strategy_to_pipeline


class SimulatedCrash(BaseException):
    pass


@pytest.fixture
def execution_case(tmp_path, monkeypatch):
    image = Image.new('L', (32, 32), 0)
    ImageDraw.Draw(image).rectangle((10, 10, 18, 18), fill=255)
    target = tmp_path / 'input.png'
    image.save(target)
    reference = tmp_path / 'reference.png'
    image.save(reference)
    strategy = normalize_strategy({'segmentation': {
        'method': 'bright_threshold', 'sensitivity': 1,
        'min_area_px': 2, 'morphology': 'none'}})
    calls = []
    def execute(image, pipeline, **kwargs):
        calls.append(content_hash(pipeline))
        return execute_pipeline(image, pipeline, **kwargs)
    runtime = {'python': 'test-runtime', 'dependencies': {}, 'source_sha256': {}}
    monkeypatch.setattr('core.experiments.runner.execute_pipeline_sandbox', execute)
    monkeypatch.setattr('core.sandbox.check_sandbox_available', lambda: {'image_id': 'test-image'})
    monkeypatch.setattr('core.runtime_metadata.runtime_metadata', lambda: deepcopy(runtime))
    monkeypatch.setattr('core.agent_loop.runtime_metadata', lambda: deepcopy(runtime))

    def create(with_reference=False):
        run_dir = tmp_path / ('with_reference' if with_reference else 'without_reference')
        budget = initial_budget(RunLimits())
        budget['usage'] = {'model_calls': 1, 'executions': 1}
        state = {'orchestration_version': 2, 'run_id': run_dir.name,
                 'run_dir': str(run_dir), 'output_root': str(tmp_path), 'graph_thread_id': run_dir.name,
                 'target_image_path': str(target), 'description': 'Find the bright square',
                 'phase': 'execute', 'run_status': 'running', 'budget': budget,
                 'reference_examples': [{'image_path': str(reference), 'description': 'Expected square boundary'}]
                    if with_reference else [],
                 'proposal': {'kind': 'propose', 'understanding': {
                     'task_summary': 'Find the bright square', 'recommended_strategy': strategy,
                     'target_constraints': {}, 'rendering': {}},
                     'pipeline': strategy_to_pipeline(strategy), 'change_reason': 'Threshold bright pixels'},
                 'pending_action': {'id': 'execute_1', 'kind': 'execute', 'input_hash': 'execution-inputs',
                                    'iteration': 0}, 'state_version': 1, 'action_sequence': 3,
                 'failure_counts': {}, 'experiment_records': [], 'experiment_history': []}
        state.update(ToolAgentRuntime(None)._draft(state, {'id': 'draft_1'}))
        draft = state['current_draft']
        state['tool_request'] = {'kind': 'tool', 'tool': 'execute_pipeline', 'arguments': {
            'draft_id': draft['draft_id'], 'revision': draft['revision']}}
        state['input_scope'] = experiment_scope(target, context=state)
        return state

    return {'create': create, 'calls': calls, 'reference': reference, 'runtime': runtime}


def execute_until_result_save(state, monkeypatch):
    def crash_before_result(path, value):
        if Path(path).name == 'result.json':
            raise SimulatedCrash('execution completed before action result write')
        return atomic_json(path, value)
    with monkeypatch.context() as patch:
        patch.setattr('core.agent_workflow.atomic_json', crash_before_result)
        with pytest.raises(SimulatedCrash):
            ToolAgentRuntime(None).perform(state)
    directory = Path(state['run_dir']) / 'iteration_0'
    saved = json.loads((directory / 'graph_state.json').read_text())
    assert saved['candidate_attempts'][0]['status'] == 'selected_for_review'
    pending = state['pending_action']
    store = ActionStore(state['run_dir'])
    assert store.has_started(pending['id'], pending['input_hash'])
    assert store.load(pending['id'], pending['input_hash']) is None
    assert not (store.root / pending['id'] / 'result.json').exists()
    return saved


@pytest.mark.parametrize('with_reference', [False, True])
def test_completed_execution_recovers_real_artifacts_without_running_again(execution_case, monkeypatch, with_reference):
    state = execution_case['create'](with_reference)
    saved = execute_until_result_save(state, monkeypatch)
    result = ToolAgentRuntime(None).perform(state)
    assert result['action_outcome']['status'] == 'ok'
    assert result['action_outcome']['data']['selected_experiment_id'] == saved['selected_experiment_id']
    assert result['budget'] == state['budget']
    assert len(execution_case['calls']) == 1
    resumed = ToolAgentRuntime(None).advance(result)
    # v2 returns the executed result to the Agent; submission is its explicit choice.
    assert resumed['phase'] == 'propose'
    assert resumed['budget']['usage']['executions'] == 1
    assert resumed['pending_action']['kind'] == 'propose'


@pytest.mark.parametrize('refresh_scope', [False, True])
def test_reference_content_change_prevents_execution_recovery(execution_case, monkeypatch, refresh_scope):
    state = execution_case['create'](True)
    execute_until_result_save(state, monkeypatch)
    assert ToolAgentRuntime(None)._recover_execution(state, state['pending_action']) is not None
    Image.new('L', (32, 32), 255).save(execution_case['reference'])
    current_scope = experiment_scope(state['target_image_path'], context=state)
    assert current_scope != state['input_scope']
    if refresh_scope:
        state['input_scope'] = current_scope
    result = ToolAgentRuntime(None).perform(state)
    assert result['action_outcome']['status'] == 'unknown'
    assert len(execution_case['calls']) == 1


@pytest.mark.parametrize('file_name,mutation', [
    ('mask.png', 'remove'), ('mask.png', 'modify'), ('measurements.json', 'remove'),
    ('outputs.json', 'remove'), ('pipeline.json', 'modify'), ('quality_report.json', 'modify'),
])
def test_incomplete_or_modified_execution_evidence_is_not_recovered(execution_case, monkeypatch, file_name, mutation):
    state = execution_case['create']()
    saved = execute_until_result_save(state, monkeypatch)
    candidate = Path(saved['candidate_attempts'][0]['directory'])
    path = candidate / file_name
    assert path.is_file()
    if mutation == 'remove':
        path.unlink()
    elif path.suffix == '.png':
        Image.new('L', (32, 32), 0).save(path)
    else:
        atomic_json(path, {'corrupted': True})
    result = ToolAgentRuntime(None).perform(state)
    assert result['action_outcome']['status'] == 'unknown'
    assert len(execution_case['calls']) == 1
