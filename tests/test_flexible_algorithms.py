import numpy as np
from PIL import Image
from core.sandbox import execute_pipeline_sandbox, SandboxLimits
from core.pipelines.dsl import normalize_pipeline


def flexible():
    return {'schema_version': 3, 'name': 'rgb_measurement',
            'input_types': {'$rgb': 'ImageArtifact'},
            'generated_operators': [{'name': 'measure', 'input_ports': {
                'gray': 'ImageArtifact', 'color': 'ImageArtifact'},
                'output_artifact': 'MetadataArtifact',
                'source': 'def apply(data, params):\n    return {"red_mean": float(np.mean(data["color"][:,:,0])), "points": [[2, 3]], "boxes": [[1, 1, 4, 4]]}'}],
            'nodes': [{'id': 'stats', 'operator': 'measure', 'inputs': {'gray': '$image', 'color': '$rgb'}}],
            'outputs': {'measurements': 'stats'}}


def test_multi_input_rgb_and_structured_output(docker_sandbox):
    p = normalize_pipeline(flexible())
    out = execute_pipeline_sandbox(np.zeros((8, 8)), p, inputs={'$rgb': np.full((8, 8, 3), 42)})
    assert out.mask is None
    assert out.outputs['measurements'].data['red_mean'] == 42


def test_structured_workflow_can_save_result(tmp_path, docker_sandbox):
    from core.agent_loop import run_planned_agent
    source = tmp_path / 'rgb.png'
    Image.fromarray(np.full((8, 8, 3), 42, dtype=np.uint8)).save(source)
    result = run_planned_agent(source, 'measure red', {}, output_root=tmp_path/'out',
                              planned_candidates=[{'name': 'measure', 'pipeline': flexible()}], max_candidates=1)
    assert result['predicted_mask_path'] is None
    assert result['measurements']['structured_outputs']['measurements']['data']['red_mean'] == 42
    assert '结构化' in result['conversation'][-1]['content']


def test_source_and_operator_count_are_not_truncated():
    p = flexible()
    p['generated_operators'][0]['source'] += '\n#' + 'x'*25000
    for i in range(12):
        p['generated_operators'].append({'name': f'extra{i}', 'source':'def apply(data, params): return data'})
    out = normalize_pipeline(p)
    assert len(out['generated_operators']) == 13
    assert len(out['generated_operators'][0]['source']) > 25000


def test_budget_environment(monkeypatch):
    monkeypatch.setenv('LIANGCE_SANDBOX_TIMEOUT_SECONDS', '90')
    monkeypatch.setenv('LIANGCE_SANDBOX_MEMORY_MB', '2048')
    monkeypatch.setenv('LIANGCE_SANDBOX_MAX_STEPS', '512')
    limits = SandboxLimits.from_env()
    assert (limits.timeout_seconds, limits.memory_mb, limits.max_steps) == (90, 2048, 512)


def test_structured_result_card_escapes_and_displays_measurements():
    from ui.utils.formatters import format_task_card
    card = format_task_card({'measurements': {'structured_outputs': {'value': {'data': {'width': 42, 'note': '<script>'}}}}})
    assert '42' in card and '&lt;script&gt;' in card
    assert '个区域' not in card
