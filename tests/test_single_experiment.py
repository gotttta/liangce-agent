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
    monkeypatch.setattr('core.experiments.runner.execute_pipeline_sandbox', execute_pipeline)
    data = np.zeros((32, 32), dtype=np.uint8)
    data[10:18, 10:18] = 255
    path = tmp_path / 'image.png'
    Image.fromarray(data).save(path)
    return path


class Reviewer:
    def review_candidates(self, target, description, candidates, **kwargs):
        return {'decision': 'present', 'selected_candidate': candidates[0]['name'], 'reason': 'Boundary verified'}


def run(source, tmp_path, **kwargs):
    return run_agent_graph(source, 'bright region', output_root=tmp_path / 'out', **kwargs)


def test_default_executes_one_experiment_without_calibration_or_baseline_replay(source, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Default single experiment must not rank or generate extra variants')
    monkeypatch.setattr('core.agent_loop._ground_truth_calibration_candidates', forbidden)
    monkeypatch.setattr('core.agent_loop.select_best_candidate', forbidden)
    proposal = understanding()
    proposal['candidate_pipelines'].append(understanding(2, 'extra')['candidate_pipelines'][0])
    state = run(source, tmp_path, understanding=proposal, ground_truth_mask_path=source,
                provider=Reviewer())
    assert len(state['candidate_attempts']) == 1
    assert state['review']['acceptance']['overall_passed']
    assert state['verified_baseline']['experiment_id'] == state['selected_experiment_id']
    plans = plan_candidate_definitions(understanding(), previous_state=state, max_candidates=3)
    assert len(plans) == 1


def test_revision_receives_failed_experiment_and_persists_review_lineage(source, tmp_path):
    class Revising(Reviewer):
        reviews = 0
        def understand_task(self, target, description, previous_context=None):
            self.context = previous_context
            return understanding(.5, 'fix')

        def review_candidates(self, target, description, candidates, **kwargs):
            self.reviews += 1
            if self.reviews == 1:
                return {'decision': 'revise', 'selected_candidate': candidates[0]['name'],
                        'reason': 'Boundary expanded', 'observed_issues': ['Boundary expanded'],
                        'revision_plan': ['Inspect the threshold mask']}
            return super().review_candidates(target, description, candidates)
    provider = Revising()
    state = run(source, tmp_path, understanding=understanding(), provider=provider)
    first, second = state['experiment_records']
    assert first['acceptance_status'] == 'rejected'
    assert second['acceptance_status'] == 'passed'
    assert first['algorithm_version'] != second['algorithm_version']
    assert second['parent_experiment_id'] == first['experiment_id']
    assert provider.context['execution_feedback']['attempts'][0]['review']['reason'] == 'Boundary expanded'
    for item in (first, second):
        saved = json.loads((Path(item['directory']) / 'experiment.json').read_text())
        assert saved['review'] == item['review']
        assert saved['change_reason']
        assert saved['expected_change']
        assert saved['acceptance_status'] == item['acceptance_status']


@pytest.mark.parametrize('kind', ['missing', 'exception', 'malformed', 'unknown_selection'])
def test_execution_or_pixel_gate_never_substitutes_for_visual_acceptance(source, tmp_path, kind):
    class Unavailable:
        def review_candidates(self, *args, **kwargs):
            if kind == 'exception':
                raise RuntimeError('review unavailable')
            if kind == 'unknown_selection':
                return {'decision': 'present', 'selected_candidate': 'nonexistent'}
            return None
    state = run(source, tmp_path, understanding=understanding(), ground_truth_mask_path=source,
                provider=None if kind == 'missing' else Unavailable(), max_auto_revisions=0)
    assert state['evaluation_report']['dice'] == 1
    assert not state['review']['acceptance']['overall_passed']
    assert state.get('verified_baseline') is None
    assert state['experiment_records'][0]['acceptance_status'] != 'passed'


def test_duplicate_rejected_pipeline_is_not_reexecuted(source, tmp_path):
    class Repeat:
        def understand_task(self, *args, **kwargs):
            return understanding(name='renamed')
        def review_candidates(self, target, description, candidates, **kwargs):
            return {'decision': 'revise', 'selected_candidate': candidates[0]['name'], 'reason': 'Wrong boundary'}
    state = run(source, tmp_path, understanding=understanding(), provider=Repeat())
    assert len(state['experiment_records']) == 1
    assert state['status'] == 'needs_human_review'


def test_rejected_review_without_a_planner_stops_instead_of_replaying(source, tmp_path):
    class ReviewOnly:
        def review_candidates(self, target, description, candidates, **kwargs):
            return {'decision': 'revise', 'selected_candidate': candidates[0]['name'], 'reason': 'Wrong boundary'}
    state = run(source, tmp_path, understanding=understanding(), provider=ReviewOnly())
    assert len(state['experiment_records']) == 1
    assert state['status'] == 'needs_human_review'


def test_failed_followup_rolls_back_verified_result_without_losing_failure(source, tmp_path):
    first = run(source, tmp_path, understanding=understanding(), provider=Reviewer())
    broken = understanding()
    broken['candidate_pipelines'][0]['pipeline']['steps'][0]['op'] = 'does_not_exist'
    failed = run(source, tmp_path, understanding=broken, previous_state=first, max_auto_revisions=0)
    assert failed['selected_experiment_id'] == first['selected_experiment_id']
    assert failed['retained_experiment_id'] == first['selected_experiment_id']
    assert failed['experiment_records'][-1]['acceptance_status'] == 'rejected'
    assert failed['experiment_records'][-1]['status'] == 'failed'
    assert failed['candidate_attempts'][0]['status'] == 'failed'
    assert failed['pipeline'] == first['pipeline']
    accepted = resume_agent_graph(failed['graph_thread_id'], {'action': 'accept'})
    assert accepted['verified_baseline']['acceptance_status'] == 'accepted'
    assert accepted['experiment_records'][-1]['acceptance_status'] == 'rejected'


def test_baseline_cannot_cross_input_contract_or_feedback_changes(source, tmp_path):
    first = run(source, tmp_path, understanding=understanding(), provider=Reviewer())
    assert compatible_baseline(first)
    changed = deepcopy(first)
    changed['task_contract']['rendering'] = {'contour_color': '#00ff00'}
    assert compatible_baseline(changed) is None
    changed = {**first, 'human_feedback': {'exclude_mask_path': str(source)}}
    assert compatible_baseline(changed) is None
    Image.new('L', (32, 32)).save(source)
    assert compatible_baseline(first) is None


def test_human_acceptance_is_persisted_even_without_automatic_review(source, tmp_path):
    state = run(source, tmp_path, understanding=understanding())
    assert state['experiment_records'][0]['acceptance_status'] == 'pending'
    accepted = resume_agent_graph(state['graph_thread_id'], {'action': 'accept'})
    baseline = accepted['verified_baseline']
    assert baseline['acceptance_status'] == 'accepted'
    saved = json.loads((Path(baseline['directory']) / 'experiment.json').read_text())
    assert saved['human_review']['action'] == 'accept'
    assert not saved['review']['acceptance']['overall_passed']


@pytest.mark.parametrize('action', ['continue', 'exit'])
def test_human_rejection_invalidates_automatic_baseline(source, tmp_path, action):
    state = run(source, tmp_path, understanding=understanding(), provider=Reviewer())
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
