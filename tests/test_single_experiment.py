import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core.agent_graph import run_agent_graph, resume_agent_graph
from core.agent_loop import plan_candidate_definitions
from core.experiments.lifecycle import compatible_baseline
from core.pipelines.dsl import execute_pipeline
from core.tools.experiments import ExperimentTools
from providers.vision import normalize_task_understanding
from test_tool_agent_graph import Agent, create, execute, submit


def understanding(sensitivity=1, name='trial'):
    return {'task_summary': 'bright region', 'recommended_strategy': {}, 'candidate_pipelines': [{
        'name': name, 'hypothesis': 'bright foreground',
        'change_reason': 'Inspect threshold coverage and adjust sensitivity',
        'expected_change': 'Preserve the visible boundary',
        'pipeline': {'steps': [{'id': 'mask', 'op': 'global_threshold', 'input': 'image',
                                'params': {'polarity': 'bright', 'sensitivity': sensitivity}}]},
    }]}


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setattr('core.sandbox.check_sandbox_available', lambda: {'image_id': 'test-image'})
    monkeypatch.setattr('core.experiments.runner.execute_pipeline_sandbox', execute_pipeline)
    data = np.zeros((32, 32), dtype=np.uint8)
    data[10:18, 10:18] = 255
    path = tmp_path / 'image.png'
    Image.fromarray(data).save(path)
    return path


def run(source, tmp_path, agent, **kwargs):
    return run_agent_graph(source, 'bright region', output_root=tmp_path / 'out', provider=agent, **kwargs)


def needs_input(context):
    return {'kind': 'needs_input', 'reason': 'Waiting for the user'}


def test_default_executes_one_experiment_without_calibration_or_ranking(source, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Default single experiment must not rank or generate extra variants')
    monkeypatch.setattr('core.agent_loop._ground_truth_calibration_candidates', forbidden)
    monkeypatch.setattr('core.agent_loop.select_best_candidate', forbidden)
    state = run(source, tmp_path, Agent(), ground_truth_mask_path=source)
    assert len(state['candidate_attempts']) == 1
    assert state['budget']['usage']['executions'] == 1
    assert state['review']['acceptance']['overall_passed']
    assert state['verified_baseline']['experiment_id'] == state['selected_experiment_id']
    plans = plan_candidate_definitions(understanding(), previous_state=state, max_candidates=3)
    assert len(plans) == 1


def test_revision_receives_rejected_experiment_and_persists_review_lineage(source, tmp_path):
    agent = Agent([create, execute, submit, lambda context: create(context, .5), execute, submit],
                  ['revise', 'present'])
    state = run(source, tmp_path, agent)
    first, second = state['experiment_records']
    assert first['acceptance_status'] == 'rejected'
    assert second['acceptance_status'] == 'passed'
    assert first['algorithm_version'] != second['algorithm_version']
    revising = next(context for kind, context in agent.calls[4:] if kind == 'agent')
    assert revising['review']['reason'] == 'Checked executed image'
    assert revising['execution_feedback']['attempts'][0]['acceptance_status'] == 'rejected'
    for item in (first, second):
        saved = json.loads((Path(item['directory']) / 'experiment.json').read_text())
        assert saved['review'] == item['review']
        assert saved['change_reason']
        assert saved['acceptance_status'] == item['acceptance_status']


@pytest.mark.parametrize('kind', ['exception', 'malformed', 'unknown_selection'])
def test_execution_or_pixel_gate_never_substitutes_for_visual_acceptance(source, tmp_path, kind):
    class Unavailable(Agent):
        def review_action(self, target, description, candidates, context, **kwargs):
            if kind == 'exception':
                raise RuntimeError('review unavailable')
            if kind == 'unknown_selection':
                return {'kind': 'review', 'review': {'decision': 'present', 'selected_candidate': 'nonexistent'}}
            return {'kind': 'review', 'review': None}
    state = run(source, tmp_path, Unavailable(), ground_truth_mask_path=source)
    assert state['evaluation_report']['dice'] == 1
    assert not (state.get('review') or {}).get('acceptance', {}).get('overall_passed')
    assert state.get('verified_baseline') is None
    assert state['experiment_records'][0]['acceptance_status'] != 'passed'


def test_duplicate_rejected_pipeline_is_not_reexecuted(source, tmp_path):
    agent = Agent([create, execute, submit, create, execute, needs_input], ['revise'])
    state = run(source, tmp_path, agent)
    assert len(state['experiment_records']) == 1
    assert state['budget']['usage']['executions'] == 1
    assert state['last_tool_result']['data']['reused'] is True


def test_baseline_cannot_cross_input_contract_or_feedback_changes(source, tmp_path):
    first = run(source, tmp_path, Agent())
    assert compatible_baseline(first)
    changed = deepcopy(first)
    changed['task_contract']['rendering'] = {'contour_color': '#00ff00'}
    assert compatible_baseline(changed) is None
    changed = {**first, 'human_feedback': {'exclude_mask_path': str(source)}}
    assert compatible_baseline(changed) is None
    Image.new('L', (32, 32)).save(source)
    assert compatible_baseline(first) is None


def test_human_acceptance_is_persisted_even_without_automatic_review(source, tmp_path):
    inconclusive = {'kind': 'review', 'review': {'decision': 'uncertain', 'selected_candidate': None,
                                                 'reason': 'Cannot judge the boundary'}}
    state = run(source, tmp_path, Agent(reviews=[inconclusive]))
    assert state['stop_reason'] == 'review_inconclusive'
    assert state['experiment_records'][0]['acceptance_status'] != 'passed'
    accepted = resume_agent_graph(state['graph_thread_id'], {'action': 'accept'})
    baseline = accepted['verified_baseline']
    assert baseline['acceptance_status'] == 'accepted'
    saved = json.loads((Path(baseline['directory']) / 'experiment.json').read_text())
    assert saved['human_review']['action'] == 'accept'
    assert not saved['review']['acceptance']['overall_passed']


@pytest.mark.parametrize('action', ['continue', 'exit'])
def test_human_rejection_invalidates_automatic_baseline(source, tmp_path, action):
    state = run(source, tmp_path, Agent([create, execute, submit, needs_input]))
    assert state['verified_baseline']
    rejected = resume_agent_graph(state['graph_thread_id'], {'action': action})
    assert rejected['verified_baseline'] is None
    record = rejected['experiment_records'][0]
    assert record['acceptance_status'] == 'rejected'
    saved = json.loads((Path(record['directory']) / 'experiment.json').read_text())
    assert saved['human_review']['action'] == action
    assert saved['review']['acceptance']['overall_passed']


def test_tool_comparison_is_optional_and_checks_matching_constraints(source, tmp_path):
    session = ExperimentTools(source, 'bright region', output_root=tmp_path)
    first, _ = session.execute({'pipeline': understanding()['candidate_pipelines'][0]['pipeline'],
                                'change_reason': 'initial evidence', 'expected_change': 'outline'})
    assert session.comparisons == 0
    session.context['human_feedback'] = {'exclude_mask_path': str(source)}
    second, _ = session.execute({'pipeline': understanding()['candidate_pipelines'][0]['pipeline'],
                                 'parent_experiment_id': first['experiment_id']})
    assert not second['reused']
    assert session.attempts[second['experiment_id']]['parent_experiment_id'] == first['experiment_id']
    with pytest.raises(ValueError, match='different task or feedback'):
        session.compare({'experiment_ids': [first['experiment_id'], second['experiment_id']],
                         'reason': 'Check for regression'})


def test_normalization_keeps_one_complete_proposal_and_change_evidence():
    proposal = understanding()
    proposal['candidate_pipelines'].append({'pipeline': {'steps': []}})
    result = normalize_task_understanding(proposal)
    assert len(result['candidate_pipelines']) == 1
    assert result['candidate_pipelines'][0]['change_reason'] == proposal['candidate_pipelines'][0]['change_reason']
