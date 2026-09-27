import json

import numpy as np
from PIL import Image
import pytest

from core.experiments.artifacts import export_artifacts, persist_artifacts
from core.operators import ImageArtifact, MaskArtifact
from core.pipelines.dsl import PipelineExecutionResult
from core.tools.discovery import dispatch_discovery, operator_index
from providers.vision import AliyunVisionProvider, MockVisionProvider, build_task_understanding_messages


def test_disclosure_index_omits_contracts_and_batch_query_restores_them():
    assert all('parameters' not in item for item in operator_index())
    result, images = dispatch_discovery({'tool': 'query_operators', 'arguments': {
        'names': ['normalize', 'global_threshold']}}, None, None)
    assert len(result['operators']) == 2
    assert result['operators'][0]['input_ports'] == {'image': 'ImageArtifact'}
    assert result['operators'][0]['parameters']
    assert images == []
    with pytest.raises(ValueError, match='unknown operators'):
        dispatch_discovery({'tool': 'query_operators', 'arguments': {'names': ['not_real']}}, None, None)


@pytest.mark.parametrize('name, required_guidance', [
    ('line_width_spacing', ('局部法向', '边到边还是中心到中心', 'sampling_direction', 'valid_cross_sections')),
    ('hole_diameter', ('sqrt(4*area/pi)', '不同定义不能混用', 'diameter_definition', 'truncated_holes')),
    ('position_offset', ('缺少基准', 'ΔX、ΔY', '坐标原点', 'target_correspondence')),
    ('area_measurement', ('真实像素数', '孔洞是否扣除', 'roi_denominator', 'component_area_sum')),
    ('contour_deviation', ('缺少参考', '单向或双向距离', '像素差数', 'distance_definition')),
    ('defect_count', ('4/8 连通性', '允许无缺陷结果', 'merged_or_fragmented_targets', 'border_count_policy')),
])
def test_first_skill_load_includes_measurement_rules_and_acceptance(name, required_guidance):
    result, _ = dispatch_discovery({'tool': 'load_skill', 'arguments': {
        'name': name}}, None, None)
    skill = result['skill']
    for guidance in required_guidance:
        assert guidance in skill['content']
    assert skill['resources'] == []
    assert 'operators' not in result
    assert 'pipeline_template' not in skill


def test_intermediate_artifacts_preserve_numeric_data_and_scope(tmp_path):
    data = np.array([[-3, 1], [9, 4]], dtype=np.float32)
    result = PipelineExecutionResult({}, MaskArtifact(data > 2), None, (), {
        '../../escape': ImageArtifact(data), 'mask': MaskArtifact(data > 2)})
    records = persist_artifacts(result, tmp_path)
    original = next(item for item in records if item['node_id'] == '../../escape')
    np.testing.assert_array_equal(np.load(original['raw_path'], allow_pickle=False), data)
    context = {'execution_feedback': {'attempts': [{'artifacts': records}]}}
    reply, images = dispatch_discovery({'tool': 'inspect_artifact', 'arguments': {
        'ids': [original['id']]}}, context, None)
    assert images == [original['preview_path']]
    assert reply['artifacts'][0]['display_range'] == [-3, 9]
    with pytest.raises(ValueError, match='not part'):
        dispatch_discovery({'tool': 'inspect_artifact', 'arguments': {'ids': ['/etc/passwd']}}, context, None)


def test_artifact_export_has_count_and_byte_limits(monkeypatch):
    import core.experiments.artifacts as module
    monkeypatch.setattr(module, 'MAX_BYTES', 32)
    monkeypatch.setattr(module, 'MAX_ARTIFACTS', 2)
    outputs = {str(i): ImageArtifact(np.zeros((2, 2))) for i in range(10)}
    assert len(export_artifacts(outputs)) == 2
    assert sum(item['data'].nbytes for item in export_artifacts(outputs).values()) <= 32


def test_provider_executes_discovery_then_returns_pipeline(tmp_path, monkeypatch):
    target = tmp_path / 'image.png'
    pixels = np.zeros((20, 20), dtype=np.uint8)
    pixels[7:12, 7:12] = 255
    Image.fromarray(pixels).save(target)
    from core.pipelines.dsl import execute_pipeline
    from core.tools.experiments import ExperimentTools
    monkeypatch.setattr('core.experiments.runner.execute_pipeline_sandbox', execute_pipeline)
    monkeypatch.setattr(ExperimentTools, 'preflight', lambda self: {'image_id': 'test'})
    provider = AliyunVisionProvider(api_key='test', base_url='https://example.invalid', tool_mode='text')
    responses = iter([
        json.dumps({'type': 'call_tool', 'tool': 'load_skill', 'arguments': {'name': 'area_measurement'}}),
        json.dumps({'type': 'call_tool', 'tool': 'save_task', 'arguments': {'understanding': {
            'task_summary': '亮颗粒', 'acceptance_criteria': {}}}}),
        json.dumps({'type': 'call_tool', 'tool': 'execute_pipeline', 'arguments': {'pipeline': {'steps': [
            {'id': 'mask', 'op': 'global_threshold', 'input': 'image', 'params': {}}]}}}),
    ])
    prompts = []
    def complete(client, messages, progress_callback=None):
        prompts.append(json.loads(json.dumps(messages)))
        if len(prompts) <= 3:
            return next(responses)
        result = json.loads(messages[-1]['content'][0]['text'])
        return json.dumps({'type': 'call_tool', 'tool': 'submit_experiment', 'arguments': {
            'experiment_id': result['data']['experiment_id'], 'reason': 'ready for review'}})
    monkeypatch.setattr(provider, '_complete_streaming', complete)
    result = provider.understand_task(str(target), '亮颗粒', previous_context={'experiment_output_root': str(tmp_path)})
    assert result['candidate_pipelines']
    assert len(prompts) == 4
    loaded = json.loads(prompts[1][-1]['content'][0]['text'])['data']['skill']
    assert '# 面积量测' in loaded['content']
    assert '真实像素数' in loaded['content']
    assert 'roi_denominator' in loaded['content']
    assert loaded['resources'] == []
    assert 'pipeline_template' not in prompts[1][-1]['content'][0]['text']
    initial = build_task_understanding_messages(target, '亮颗粒')[0]['content'][0]['text']
    assert '"parameters"' not in initial


def test_repeated_discovery_stops_without_unbounded_model_calls(tmp_path, monkeypatch):
    target = tmp_path / 'image.png'
    Image.new('L', (4, 4)).save(target)
    provider = AliyunVisionProvider(api_key='test', base_url='https://example.invalid', tool_mode='text')
    calls = []
    def complete(*args, **kwargs):
        calls.append(1)
        return json.dumps({'type': 'call_tool', 'tool': 'query_operators', 'arguments': {'names': ['normalize']}})
    monkeypatch.setattr(provider, '_complete_streaming', complete)
    with pytest.raises(ValueError, match='budget_exhausted'):
        provider.understand_task(target, '亮颗粒')
    assert len(calls) == 5


def test_candidate_runs_record_failures_artifacts_and_parent(tmp_path):
    from core.agent_loop import run_planned_agent
    target = tmp_path / 'input.png'
    data = np.zeros((30, 30), dtype=np.uint8)
    data[10:15, 10:15] = 255
    Image.fromarray(data).save(target)
    candidates = [
        {'name': 'invalid', 'hypothesis': 'bad operator', 'pipeline': {
            'name': 'invalid', 'steps': [{'id': 'bad', 'op': 'missing', 'input': 'image'}]}},
        {'name': 'valid', 'hypothesis': 'bright object', 'pipeline': {
            'name': 'valid', 'steps': [{'id': 'mask', 'op': 'global_threshold', 'input': 'image',
                                      'params': {'polarity': 'bright', 'sensitivity': 1}}]}},
    ]
    state = run_planned_agent(target, 'bright object', {}, output_root=tmp_path / 'out',
                              planned_candidates=candidates, max_candidates=2,
                              previous_state={'selected_experiment_id': 'parent', 'iteration': 0})
    assert state['selected_experiment_id']
    for attempt in state['candidate_attempts']:
        from pathlib import Path
        record = json.loads((Path(attempt['directory']) / 'experiment.json').read_text())
        assert record['parent_experiment_id'] == 'parent'
        assert record['input_sha256']
        assert record['experiment_id'] == attempt['experiment_id']
    assert state['candidate_attempts'][0]['status'] == 'failed'
    assert state['candidate_attempts'][1]['artifacts']
