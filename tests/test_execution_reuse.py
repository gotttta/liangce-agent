"""Submitted-experiment replay: the graph phase must not re-run the sandbox."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from core.input_contract import input_identity
from core.pipelines.dsl import PipelineExecutionResult, execute_pipeline
from core.agent_loop import run_planned_agent
from core.experiments.artifacts import record_experiments
from core.experiments.runner import _serializable_attempt, run_candidate


def source(tmp_path, value=0, shape=(40, 40), name='source.png'):
    path = tmp_path / name
    pixels = np.full(shape, value, dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    return path, pixels


def mask_pipeline():
    return {'schema_version': 3, 'nodes': [
        {'id': 'm', 'operator': 'global_threshold', 'inputs': {'image': '$image'},
         'params': {'sensitivity': 1.0}}], 'outputs': {'mask': 'm'}}


def track_sandbox(monkeypatch):
    calls = []
    def execute(image, pipeline):
        calls.append(pipeline)
        return execute_pipeline(image, pipeline)
    monkeypatch.setattr('core.experiments.runner.execute_pipeline_sandbox', execute)
    return calls


def completed_experiment(tmp_path, monkeypatch):
    """Execute one candidate end-to-end and persist it like the tool session does."""
    calls = track_sandbox(monkeypatch)
    path, pixels = source(tmp_path)
    pixels[10:20, 10:20] = 255
    Image.fromarray(pixels).save(path)
    (tmp_path / 'session').mkdir()
    attempt = run_candidate({'name': 'first', 'pipeline': mask_pipeline()}, path, pixels,
                            tmp_path / 'session' / 'candidate_0')
    assert attempt['status'] == 'selected_for_review', attempt
    record_experiments([attempt], path, {}, 0)
    return path, pixels, attempt, calls


def replay_candidate(first):
    return {'name': 'resubmitted', 'hypothesis': 'same algorithm',
            'pipeline': first['pipeline'],
            'source': {'type': 'agent_tool'},
            'reused_execution': {'experiment_id': first['experiment_id'],
                                 'directory': first['directory']}}


def test_submitted_experiment_is_replayed_without_sandbox(tmp_path, monkeypatch):
    path, pixels, first, first_calls = completed_experiment(tmp_path, monkeypatch)
    calls = track_sandbox(monkeypatch)
    assert len(first_calls) == 1

    (tmp_path / 'graph').mkdir()
    replayed = run_candidate(replay_candidate(first), path, pixels, tmp_path / 'graph' / 'candidate_0')

    assert len(calls) == 0, 'the identical pipeline must not run in the sandbox twice'
    assert replayed['status'] == 'selected_for_review'
    assert replayed['reused_from_experiment'] == first['experiment_id']
    directory = Path(replayed['directory'])
    for file_name in ('mask.png', 'quality_report.json', 'measurements.json',
                      'operator_trace.json', 'contours.json', 'outputs.json',
                      'result_annotation.png', 'pipeline.json'):
        assert (directory / file_name).is_file(), file_name
    assert replayed['quality']['health'] == first['quality']['health']
    assert replayed['measurements']['summary']['count'] == first['measurements']['summary']['count']
    stored_mask = np.asarray(Image.open(directory / 'mask.png').convert('L')) > 0
    original_mask = np.asarray(Image.open(Path(first['directory']) / 'mask.png').convert('L')) > 0
    np.testing.assert_array_equal(stored_mask, original_mask)
    assert replayed['execution'].mask is not None
    assert tuple(replayed['execution'].mask.data.shape) == pixels.shape
    assert json.loads((directory / 'pipeline.json').read_text()) == first['pipeline']


def test_replay_falls_back_to_execution_for_different_input(tmp_path, monkeypatch):
    path, pixels, first, _ = completed_experiment(tmp_path, monkeypatch)
    calls = track_sandbox(monkeypatch)

    other = tmp_path / 'other.png'
    other_pixels = np.full(pixels.shape, 7, dtype=np.uint8)
    Image.fromarray(other_pixels).save(other)
    (tmp_path / 'graph').mkdir()
    result = run_candidate(replay_candidate(first), other, other_pixels,
                           tmp_path / 'graph' / 'candidate_0')

    assert len(calls) == 1, 'a different input image must execute normally'
    assert result['status'] == 'selected_for_review'
    assert 'reused_from_experiment' not in result


def test_replay_rejects_modified_pipeline(tmp_path, monkeypatch):
    path, pixels, first, _ = completed_experiment(tmp_path, monkeypatch)
    calls = track_sandbox(monkeypatch)

    stored = Path(first['directory']) / 'pipeline.json'
    pipeline_record = json.loads(stored.read_text())
    pipeline_record['nodes'][0]['params']['sensitivity'] = 0.5
    stored.write_text(json.dumps(pipeline_record))

    (tmp_path / 'graph').mkdir()
    result = run_candidate(replay_candidate(first), path, pixels, tmp_path / 'graph' / 'candidate_0')

    assert len(calls) == 1, 'a modified stored pipeline must execute normally'
    assert result['status'] == 'selected_for_review'
    assert 'reused_from_experiment' not in result


def test_replay_falls_back_without_persisted_artifacts(tmp_path, monkeypatch):
    path, pixels, first, _ = completed_experiment(tmp_path, monkeypatch)
    calls = track_sandbox(monkeypatch)
    (Path(first['directory']) / 'quality_report.json').unlink()

    (tmp_path / 'graph').mkdir()
    result = run_candidate(replay_candidate(first), path, pixels, tmp_path / 'graph' / 'candidate_0')

    assert len(calls) == 1
    assert result['status'] == 'selected_for_review'
    assert 'reused_from_experiment' not in result


def test_run_planned_agent_replays_submitted_candidate(tmp_path, monkeypatch):
    path, pixels, first, _ = completed_experiment(tmp_path, monkeypatch)
    calls = track_sandbox(monkeypatch)

    state = run_planned_agent(
        target_image_path=path,
        description='检测亮块',
        understanding={'recommended_strategy': {}, 'rendering': {}},
        output_root=str(tmp_path / 'outputs'),
        planned_candidates=[replay_candidate(first)],
        max_candidates=1,
    )

    assert len(calls) == 0, 'the graph execute node must replay, not re-execute'
    assert state['status'] == 'ok'
    assert state['selected_candidate'] == 'resubmitted'
    iteration_dir = Path(state['run_dir']) / 'iteration_0'
    for file_name in ('mask.png', 'result_annotated.png', 'pipeline.json',
                      'quality_report.json', 'measurements.json', 'graph_state.json'):
        assert (iteration_dir / file_name).is_file(), file_name
    summary = state['measurements']['summary']
    assert summary['count'] == first['measurements']['summary']['count']


def test_submit_attaches_replay_reference(tmp_path, monkeypatch):
    from core.agent_loop import plan_candidate_definitions
    from core.tools.experiments import ExperimentTools

    path, pixels, first, _ = completed_experiment(tmp_path, monkeypatch)
    tools = ExperimentTools(path, '检测亮块', output_root=str(tmp_path / 'outputs'))
    tools.understanding = {'task_summary': '检测亮块', 'acceptance_criteria': {}}
    tools.attempts[first['experiment_id']] = _serializable_attempt(first)
    monkeypatch.setattr(tools, 'preflight', lambda: {'image': 'test', 'image_id': 'x'})
    monkeypatch.setattr('core.tools.experiments.runtime_metadata', lambda: {'python': 'test'})
    tools.execution_environments[first['experiment_id']] = {
        'image': 'test', 'image_id': 'x', 'runtime': {'python': 'test'}}

    result = tools.submit({'experiment_id': first['experiment_id'], 'reason': 'replay check'})
    assert result['acceptance_status'] == 'pending'
    assert result['experiment_id'] == first['experiment_id']

    submitted = tools.submitted
    candidate = submitted['candidate_pipelines'][0]
    assert candidate['reused_execution']['experiment_id'] == first['experiment_id']
    assert candidate['reused_execution']['directory'] == first['directory']

    planned = plan_candidate_definitions(submitted)
    assert planned[0]['reused_execution'] == candidate['reused_execution']


def test_submitted_candidate_outranks_retrieved_history(tmp_path, monkeypatch):
    """max_candidates=1 must keep the submitted experiment, not a retrieved replay."""
    from core.agent_loop import plan_candidate_definitions

    submitted = {'candidate_pipelines': [{
        'name': 'experiment_2', 'hypothesis': 'h', 'pipeline': mask_pipeline(),
        'source': {'type': 'agent_tool'},
        'reused_execution': {'experiment_id': 'exp1', 'directory': '/tmp/exp1'},
    }]}
    retrieved = [{'name': 'bright_threshold_and_filter', 'score': 0.9,
                  'pipeline': mask_pipeline(),
                  'source': {'type': 'accepted_algorithm', 'algorithm_id': 'a1'}}]

    planned = plan_candidate_definitions(submitted, retrieved_algorithms=retrieved, max_candidates=1)
    assert planned[0]['reused_execution']['experiment_id'] == 'exp1'

    # 历史算法仍排在全新提议之前
    history_pipeline = {'schema_version': 3, 'nodes': [
        {'id': 'm', 'operator': 'global_threshold', 'inputs': {'image': '$image'},
         'params': {'sensitivity': 0.7}}], 'outputs': {'mask': 'm'}}
    retrieved = [{'name': 'bright_threshold_and_filter', 'score': 0.9,
                  'pipeline': history_pipeline,
                  'source': {'type': 'accepted_algorithm', 'algorithm_id': 'a1'}}]
    fresh = {'candidate_pipelines': [{'name': 'p', 'pipeline': mask_pipeline()}]}
    planned = plan_candidate_definitions(fresh, retrieved_algorithms=retrieved, max_candidates=2)
    assert planned[0]['source']['type'] == 'accepted_algorithm'
