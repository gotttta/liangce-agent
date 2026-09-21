import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from core.tools.experiments import ExperimentTools
from providers.vision import AliyunVisionProvider, MockVisionProvider


def pipeline(polarity='bright'):
    return {'name': 'test', 'steps': [{'id': 'mask', 'op': 'global_threshold', 'input': 'image',
                                      'params': {'polarity': polarity, 'sensitivity': 1.0}}]}


@pytest.fixture
def session(tmp_path):
    target = tmp_path / 'target.png'
    data = np.zeros((30, 30), dtype=np.uint8)
    data[10:15, 10:15] = 255
    Image.fromarray(data).save(target)
    return ExperimentTools(target, 'detect bright particles', output_root=tmp_path)


def execute(session, value=None):
    result, images = session.dispatch({'tool': 'execute_pipeline', 'arguments': {
        'pipeline': value or pipeline(), 'hypothesis': 'bright region'}}, None)
    return {**result['data'], 'status': result['status'], 'error': result['error']}, images


def test_execute_reuses_result_and_makes_artifacts_inspectable(session):
    result, images = execute(session)
    assert result['status'] == 'success'
    assert Path(images[0]).exists()
    again, _ = execute(session)
    assert again['reused']
    assert again['experiment_id'] == result['experiment_id']
    assert session.executions == 1
    details, previews = session.dispatch({'tool': 'inspect_artifact', 'arguments': {
        'ids': [result['artifacts'][0]['id']]}}, None)
    assert Path(previews[0]).exists()
    assert details['data']['artifacts']
    assert (session.root / 'session.json').exists()


def test_execute_applies_user_constraints(session, tmp_path):
    excluded = np.zeros((30, 30), dtype=np.uint8)
    excluded[10:15, 10:15] = 255
    path = tmp_path / 'exclude.png'
    Image.fromarray(excluded).save(path)
    session.context['human_feedback'] = {'exclude_mask_path': str(path)}
    result, _ = execute(session)
    assert result['status'] == 'success'  # Empty is diagnostic, not a runtime failure.
    assert result['facts']['coverage'] == 0
    assert result['candidate_status'] == 'selected_for_review'


def test_static_failure_saves_draft_without_spending_execution_budget(session):
    session.max_executions = 1
    bad = {'name': 'bad', 'steps': [{'id': 'mask', 'op': 'not_allowed', 'input': 'image'}]}
    result, images = execute(session, bad)
    assert result['status'] == 'error'
    assert images == []
    record = json.loads(next(session.root.glob('drafts/*/revision_1.json')).read_text())
    assert not record['validation']['valid']
    assert session.executions == 0
    assert execute(session)[0]['status'] == 'success'
    exhausted, _ = execute(session, pipeline('dark'))
    assert exhausted['error']['code'] == 'budget_exhausted'
    assert exhausted['error']['retryable'] is False


def test_compare_returns_aligned_sheet_and_factual_differences(session):
    first, _ = execute(session)
    second, _ = execute(session, pipeline('dark'))
    result, images = session.dispatch({'tool': 'compare_candidates', 'arguments': {
        'experiment_ids': [first['experiment_id'], second['experiment_id']]}}, None)
    with Image.open(images[0]) as sheet:
        assert sheet.size == (60, 62)
    assert result['data']['differences'][0]['changed_pixels'] > 0
    assert 'score' not in result
    assert len(result['data']['candidates']) == 2
    session.max_comparisons = 1
    with pytest.raises(ValueError, match='comparison budget'):
        session.compare({'experiment_ids': [first['experiment_id'], second['experiment_id']]})


def test_compare_rejects_unknown_failed_and_different_source(session):
    first, _ = execute(session)
    second, _ = execute(session, pipeline('dark'))
    with pytest.raises(ValueError, match='outside'):
        session.compare({'experiment_ids': [first['experiment_id'], '/etc/passwd']})
    directory = Path(session.attempts[second['experiment_id']]['directory'])
    manifest = directory / 'experiment.json'
    record = json.loads(manifest.read_text())
    record['input_sha256'] = 'other'
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='different input'):
        session.compare({'experiment_ids': [first['experiment_id'], second['experiment_id']]})
    record['input_sha256'] = session.input_hash
    record['execution_status'] = 'failed'
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='failed experiment'):
        session.compare({'experiment_ids': [first['experiment_id'], second['experiment_id']]})


def test_model_can_execute_compare_and_finish(session, monkeypatch):
    provider = AliyunVisionProvider(api_key='test', base_url='https://example.invalid', tool_mode='text')
    calls = []
    def complete(client, messages, progress_callback=None):
        calls.append(messages)
        index = len(calls)
        if index == 1:
            return json.dumps({'type': 'call_tool', 'tool': 'save_task', 'arguments': {'understanding': {
                'task_summary': 'bright', 'acceptance_criteria': {}}}})
        if index <= 3:
            return json.dumps({'type': 'call_tool', 'tool': 'execute_pipeline', 'arguments': {
                'pipeline': pipeline('bright' if index == 2 else 'dark'), 'hypothesis': 'test'}})
        if index == 4:
            results = [json.loads(message['content'][0]['text']) for message in messages
                       if message['role'] == 'user' and isinstance(message['content'], list)
                       and message['content'][0]['text'].startswith('{')]
            ids = [result['data']['experiment_id'] for result in results if 'experiment_id' in result['data']]
            return json.dumps({'type': 'call_tool', 'tool': 'compare_candidates',
                               'arguments': {'experiment_ids': ids}})
        assert any(part['type'] == 'image_url' for part in messages[-1]['content'])
        results = [json.loads(message['content'][0]['text']) for message in messages
                   if message['role'] == 'user' and isinstance(message['content'], list)
                   and message['content'][0]['text'].startswith('{')]
        experiment_id = next(item['data']['experiment_id'] for item in results if 'experiment_id' in item['data'])
        return json.dumps({'type': 'call_tool', 'tool': 'submit_experiment',
                           'arguments': {'experiment_id': experiment_id, 'reason': 'comparison inspected'}})
    monkeypatch.setattr(provider, '_complete_streaming', complete)
    result = provider.understand_task(session.target, 'bright', previous_context={
        'experiment_output_root': str(session.root.parent)})
    assert len(calls) == 5
    assert result['candidate_pipelines']
    assert result['tool_session']['executions'] == 2
    assert result['tool_session']['comparisons'] == 1


def test_cached_pipeline_does_not_bypass_tool_version_validation(session):
    first, _ = execute(session)
    invalid = pipeline()
    invalid['operator_versions'] = {'global_threshold': '999.0.0'}
    result, _ = execute(session, invalid)
    assert result['status'] == 'error'
    assert 'experiment_id' not in result
    assert result['draft_id']
    assert session.executions == 1
