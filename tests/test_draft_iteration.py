"""Regression coverage for syntax repair and ID-based submission, without a model API."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from core.experiments.drafts import content_hash
from core.planning import ModelReply, PlanningSession
from core.tools.experiments import ExperimentTools
from core.tools.contracts import ToolError
from core.sandbox import SandboxExecutionError


def pipeline():
    return {'steps': [{'id': 'mask', 'op': 'global_threshold', 'input': 'image',
                       'params': {'polarity': 'bright', 'sensitivity': 1}}]}


BAD_LINE = '    keep2 = sorted(keep, key=lambda i: -float(stats[i, cv2.CC_STAT_AREA))'
GOOD_LINE = '    keep2 = sorted(keep, key=lambda i: -float(stats[i, cv2.CC_STAT_AREA]))'


def generated_pipeline():
    source = '\n'.join([
        'import cv2', 'import numpy as np', 'def apply(data, params):',
        '    n, labels, stats, centers = cv2.connectedComponentsWithStats((data > 0).astype(np.uint8), 8)',
        '    keep = list(range(1, n))', BAD_LINE, '    return np.isin(labels, keep2)',
    ])
    return {'steps': [{'id': 'mask', 'op': 'extract_regular_block_outlines', 'input': 'image'}],
            'generated_operators': [{'name': 'extract_regular_block_outlines', 'source': source}]}


@pytest.fixture
def session(tmp_path, monkeypatch):
    from core.pipelines.dsl import execute_pipeline
    monkeypatch.setattr('core.experiments.runner.execute_pipeline_sandbox', execute_pipeline)
    monkeypatch.setattr(ExperimentTools, 'preflight', lambda self: {'image_id': 'test-image'})
    pixels = np.zeros((32, 32), dtype=np.uint8)
    pixels[10:20, 10:20] = 255
    path = tmp_path / 'input.png'
    Image.fromarray(pixels).save(path)
    return ExperimentTools(path, '提取亮色方块轮廓', output_root=tmp_path)


def call(session, name, **args):
    return session.dispatch({'tool': name, 'arguments': args}, None)[0]


def save_task(session):
    session.require_task = True
    result = call(session, 'save_task', understanding={
        'task_summary': session.description, 'output_requirements': ['contours'],
        'acceptance_criteria': {'visual_checks': ['所有亮色块的边界正确']},
    })
    assert result['status'] == 'success'


def execute(session):
    result = call(session, 'execute_pipeline', pipeline=pipeline())
    assert result['status'] == 'success'
    return result['data']['experiment_id']


def test_syntax_error_preserves_draft_and_precise_diagnostic_without_execution(session):
    result = call(session, 'execute_pipeline', pipeline=generated_pipeline())
    assert result['error']['code'] == 'source_syntax_error'
    diagnostic = result['error']['details']
    assert diagnostic['operator'] == 'extract_regular_block_outlines'
    assert diagnostic['line'] == 6 and diagnostic['column'] > 0
    assert BAD_LINE in diagnostic['excerpt']
    assert session.executions == 0
    assert not session.attempts
    draft_id = result['data']['draft_id']
    saved = json.loads((session.root / 'drafts' / draft_id / 'revision_1.json').read_text())
    assert saved['pipeline'] == generated_pipeline()
    fixed = call(session, 'edit_draft', draft_id=draft_id, base_revision=1,
                 change_reason='补上缺失的方括号', edits=[{
                     'path': '/generated_operators/0/source', 'old': BAD_LINE, 'new': GOOD_LINE}])
    assert fixed['status'] == 'success' and fixed['data']['validation']['valid']
    assert fixed['data']['revision'] == 2 and session.executions == 0
    assert json.loads((session.root / 'drafts' / draft_id / 'revision_1.json').read_text()) == saved


def test_patch_is_atomic_and_version_checked(session):
    result = call(session, 'create_draft', pipeline=pipeline(), change_reason='first')['data']
    draft_id = result['draft_id']
    args = dict(draft_id=draft_id, base_revision=1, change_reason='change threshold', edits=[{
        'path': '/steps/0/params/sensitivity', 'old': 1, 'new': .5}])
    invalid = call(session, 'edit_draft', **{**args, 'edits': [*args['edits'], {
        'path': '/steps/0/params/polarity', 'old': 'not there', 'new': 'dark'}]})
    assert invalid['error']['code'] == 'patch_conflict'
    assert session.drafts.get(draft_id)['pipeline'] == pipeline()
    assert call(session, 'edit_draft', **args)['status'] == 'success'
    assert call(session, 'edit_draft', **args)['error']['code'] == 'revision_conflict'
    assert call(session, 'execute_pipeline', draft_id=draft_id, revision=1)['error']['code'] == 'revision_conflict'
    assert session.executions == 0


def test_patch_rejects_ambiguous_and_noop_edits(session):
    created = call(session, 'create_draft', pipeline=generated_pipeline(), change_reason='first')['data']
    for old, new, code in [(' ', '  ', 'patch_conflict'), (BAD_LINE, BAD_LINE, 'patch_no_change')]:
        result = call(session, 'edit_draft', draft_id=created['draft_id'], base_revision=1,
                      change_reason='test', edits=[{'path': '/generated_operators/0/source', 'old': old, 'new': new}])
        assert result['error']['code'] == code
    assert session.drafts.get(created['draft_id'])['revision'] == 1


def test_submit_loads_exact_executed_snapshot_and_keeps_visual_review_pending(session):
    save_task(session)
    experiment_id = execute(session)
    attempt = session.attempts[experiment_id]
    draft_id = attempt['source']['draft_id']
    # A later, unexecuted edit cannot change the submitted execution snapshot.
    call(session, 'edit_draft', draft_id=draft_id, base_revision=1, change_reason='unexecuted edit',
         edits=[{'path': '/steps/0/params/sensitivity', 'old': 1, 'new': 2}])
    result = call(session, 'submit_experiment', experiment_id=experiment_id, reason='inspect this output')
    assert result['status'] == 'success'
    assert result['data']['acceptance_status'] == 'pending'
    assert content_hash(session.submitted['candidate_pipelines'][0]['pipeline']) == attempt['algorithm_version']
    assert attempt['acceptance_status'] == 'pending'
    assert (session.root / 'task.json').exists() and (session.root / 'submission.json').exists()


@pytest.mark.parametrize('change,code', [
    ('source', 'experiment_modified'), ('input', 'experiment_scope_changed'),
    ('contract', 'experiment_scope_changed'), ('environment', 'environment_changed'),
    ('failed', 'experiment_not_ready'), ('rejected', 'experiment_rejected'),
])
def test_submit_rejects_invalid_evidence(session, monkeypatch, change, code):
    save_task(session)
    experiment_id = execute(session)
    directory = Path(session.attempts[experiment_id]['directory'])
    if change == 'source':
        value = json.loads((directory / 'pipeline.json').read_text())
        value['steps'][0]['params']['sensitivity'] = 99
        (directory / 'pipeline.json').write_text(json.dumps(value))
    elif change == 'input':
        Image.new('L', (32, 32)).save(session.target)
    elif change == 'contract':
        session.context['task_contract']['rendering']['contour_color'] = '#000000'
    elif change == 'environment':
        monkeypatch.setattr(session, 'preflight', lambda: {'image_id': 'other'})
    else:
        path = directory / 'experiment.json'
        record = json.loads(path.read_text())
        record['execution_status' if change == 'failed' else 'acceptance_status'] = change
        path.write_text(json.dumps(record))
    result = call(session, 'submit_experiment', experiment_id=experiment_id, reason='test')
    assert result['error']['code'] == code
    assert session.submitted is None


def test_unknown_id_and_task_mutation_rejected(session):
    save_task(session)
    assert call(session, 'submit_experiment', experiment_id='../other', reason='test')['error']['code'] == 'unknown_experiment'
    result = call(session, 'save_task', understanding={
        'task_summary': 'changed goal', 'acceptance_criteria': {}, 'output_requirements': ['image']})
    # Existing contract is immutable even if a new summary is supplied.
    assert session.context['task_contract']['task_summary'] == session.description


def test_preflight_stops_before_model_and_never_spends_execution_budget(session, monkeypatch):
    def unavailable():
        raise SandboxExecutionError('Docker unavailable', code='sandbox_unavailable')
    monkeypatch.setattr(session, 'preflight', unavailable)
    planner = PlanningSession(session, None, require_submission=True)
    with pytest.raises(SandboxExecutionError):
        planner.run([], lambda *a: pytest.fail('must not call model'), json.loads, lambda x: x, lambda *a: {})
    assert session.executions == 0
    assert json.loads((session.root / 'failure.json').read_text())['code'] == 'sandbox_unavailable'


def test_daemon_failure_after_preflight_refunds_execution_and_stops(session, monkeypatch):
    save_task(session)
    def unavailable(*args, **kwargs):
        raise SandboxExecutionError('daemon stopped', code='sandbox_unavailable')
    monkeypatch.setattr('core.experiments.runner.execute_pipeline_sandbox', unavailable)
    calls = []
    def complete(*args):
        calls.append(1)
        return ModelReply(calls=[{'id': 'execute', 'name': 'execute_pipeline', 'arguments': {'pipeline': pipeline()}}])
    with pytest.raises(ToolError, match='daemon stopped'):
        PlanningSession(session, None, require_submission=True).run([], complete, json.loads, lambda x: x, lambda *a: {})
    assert len(calls) == 1 and session.executions == 0
    assert len(session.attempts) == 1  # Infrastructure evidence is retained.


@pytest.mark.parametrize('native', [True, False])
def test_finalization_uses_submit_tool_with_no_full_algorithm_json(session, native):
    save_task(session)
    experiment_id = execute(session)
    def complete(messages, specs, final_only):
        assert final_only
        assert [spec['function']['name'] for spec in specs] == ['submit_experiment']
        arguments = {'experiment_id': experiment_id, 'reason': 'ready for independent review'}
        if native:
            return ModelReply(calls=[{'id': 'submit', 'name': 'submit_experiment', 'arguments': json.dumps(arguments)}])
        return json.dumps({'type': 'call_tool', 'tool': 'submit_experiment', 'arguments': arguments})
    result = PlanningSession(session, None, native=native, max_rounds=2, require_submission=True).run(
        [], complete, json.loads, lambda x: x, lambda *a: {})
    assert result['submitted_experiment_id'] == experiment_id
    assert result['candidate_pipelines'][0]['pipeline'] == session.attempts[experiment_id]['pipeline']


def test_no_completed_experiment_does_not_request_fabricated_final_json(session):
    planner = PlanningSession(session, None, max_rounds=2, require_submission=True)
    with pytest.raises(ValueError, match='no_executable_experiment'):
        planner.run([], lambda *a: pytest.fail('no final model call needed'), json.loads, lambda x: x, lambda *a: {})


def test_generated_code_repair_runs_once_in_real_docker(tmp_path, docker_sandbox):
    pixels = np.zeros((32, 32), dtype=np.uint8)
    pixels[10:20, 10:20] = 255
    target = tmp_path / 'source.png'
    Image.fromarray(pixels).save(target)
    session = ExperimentTools(target, 'bright blocks', output_root=tmp_path)
    broken = call(session, 'execute_pipeline', pipeline=generated_pipeline())
    draft_id = broken['data']['draft_id']
    assert broken['error']['code'] == 'source_syntax_error' and session.executions == 0
    fixed = call(session, 'edit_draft', draft_id=draft_id, base_revision=1,
                 change_reason='close bracket', edits=[{
                     'path': '/generated_operators/0/source', 'old': BAD_LINE, 'new': GOOD_LINE}])
    assert fixed['status'] == 'success'
    result = call(session, 'execute_pipeline', draft_id=draft_id, revision=2)
    assert result['status'] == 'success' and session.executions == 1
    assert result['data']['facts']['coverage'] == pytest.approx(100 / 1024, abs=1e-5)


def test_submission_does_not_bypass_formal_visual_review(session, tmp_path):
    from core.agent_graph import run_agent_graph
    save_task(session)
    experiment_id = execute(session)
    assert call(session, 'submit_experiment', experiment_id=experiment_id, reason='review')['status'] == 'success'
    class RejectingReviewer:
        calls = 0
        def review_candidates(self, target, description, candidates, **kwargs):
            self.calls += 1
            return {'decision': 'revise', 'selected_candidate': candidates[0]['name'],
                    'reason': 'Boundary still wrong', 'observed_issues': ['Boundary still wrong']}
    reviewer = RejectingReviewer()
    result = run_agent_graph(session.target, session.description, output_root=tmp_path / 'formal',
                             understanding=session.submitted, provider=reviewer, max_auto_revisions=0)
    assert reviewer.calls == 1
    assert not result['review']['acceptance']['overall_passed']
    assert result.get('verified_baseline') is None
    assert result['status'] == 'needs_human_review'


def test_malformed_tool_json_returns_parse_location_and_can_retry(session):
    seen = []
    def complete(messages, specs, final_only):
        seen.append(1)
        if len(seen) == 1:
            return ModelReply(calls=[{'id': 'broken', 'name': 'create_draft', 'arguments': '{"pipeline":'}],
                              finish_reason='length')
        error = json.loads(messages[-1]['content'])['error']
        assert error['code'] == 'invalid_json'
        assert error['details']['offset'] == 12
        assert error['details']['finish_reason'] == 'length'
        return ModelReply('{"done":true}')
    result = PlanningSession(session, None).run([], complete, json.loads, lambda x: x, lambda *a: {})
    assert result['done'] and session.executions == 0
    assert session.budget.used['editing'] == 1


def test_planning_deadline_restores_outer_control(session, monkeypatch):
    from core.request_control import RequestControl, RequestCancelled, control
    outer = RequestControl(timeout=60)
    token = control.set(outer)
    monkeypatch.setenv('LIANGCE_PLANNING_TIMEOUT_SECONDS', '10')
    def complete(*args):
        current = control.get()
        assert current.timeout == 10
        current.started -= 20
        return ModelReply('{"done":true}')
    try:
        with pytest.raises(RequestCancelled, match='超过总时限'):
            PlanningSession(session, None).run([], complete, json.loads, lambda x: x, lambda *a: {})
        assert control.get() is outer and outer.cancelled.is_set()
    finally:
        control.reset(token)


def test_submission_storage_failure_cannot_commit_in_memory(session, monkeypatch):
    save_task(session)
    experiment_id = execute(session)
    def fail(*args):
        raise OSError('disk full')
    monkeypatch.setattr('core.tools.experiments.atomic_json', fail)
    result = call(session, 'submit_experiment', experiment_id=experiment_id, reason='review')
    assert result['error']['code'] == 'io_error'
    assert session.submitted is None
