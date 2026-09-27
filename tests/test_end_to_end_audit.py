"""Deterministic acceptance for the numbered 2026-09-20 audit findings."""
import base64
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import threading

import numpy as np
from PIL import Image
import pytest

from core.input_contract import input_identity
from core.pipelines.dsl import execute_pipeline, normalize_pipeline, PipelineExecutionResult
from core.operators import ImageArtifact, MaskArtifact, MetadataArtifact, build_default_registry
from core.experiments.runner import run_candidate
from core.measurement.evaluation import evaluate_prediction, meets_ground_truth_gate
from providers.vision import _extract_explicit_count, normalize_acceptance_criteria


def source(tmp_path, value=0, shape=(40, 40), name='source.png'):
    path = tmp_path / name
    pixels = np.full(shape, value, dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    return path, pixels


def mask_pipeline():
    return {'schema_version': 3, 'nodes': [
        {'id': 'm', 'operator': 'global_threshold', 'inputs': {'image': '$image'},
         'params': {'sensitivity': 1.0}}], 'outputs': {'mask': 'm'}}


@pytest.mark.parametrize('kind', ['ring', 'empty', 'single_pixel', 'nearby', 'full'])
def test_02_binary_ground_truth_is_pixel_exact(tmp_path, kind):
    from core.reference_extraction import extract_ground_truth_mask
    mask = np.zeros((40, 40), dtype=bool)
    if kind == 'ring':
        mask[5:35, 5:35] = True
        mask[10:30, 10:30] = False
    elif kind == 'single_pixel':
        mask[10, 10] = True
    elif kind == 'nearby':
        mask[3:6, 3:6] = True
        mask[3:6, 7:10] = True
    elif kind == 'full':
        mask[:] = True
    path = tmp_path / 'gt.png'
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)
    actual, metadata = extract_ground_truth_mask(path)
    np.testing.assert_array_equal(actual, mask)
    assert metadata['source_kind'] == 'binary'


@pytest.mark.parametrize('text,count', [
    ('检测直径10像素的孔', None), ('检测面积100像素以上的缺陷', None),
    ('不要检测10个孔', None), ('检测最多10个孔', None), ('检测3到5个目标', None),
    ('检测至少5个缺陷', None), ('检测约5个孔', None), ('检测3个直径10像素的孔', 3),
    ('共0个缺陷', 0), ('识别10个目标', 10), ('数量为10', 10), ('检测10像素', None),
    ('检测图中全部2个白色方块，不要标黑色背景', 2), ('当前图预期0个缺陷', 0),
    ('检测3个直径10像素以上的孔', 3), ('检测10个以下目标', None), ('检测10个以上目标', None),
])
def test_04_count_has_explicit_exact_semantics(text, count):
    assert _extract_explicit_count(text) == count


def test_03_revision_keeps_contract_and_normalization_is_idempotent():
    from core.agent_graph import _make_understand_task_node, _make_revise_candidates_node
    understanding = {'task_summary': '所有目标都闭合', 'rendering': {'contour_color': '#39FF14'},
                     'output_requirements': ['mask'], 'candidate_pipelines': [],
                     'acceptance_criteria': {'task_goal': '所有目标都闭合', 'count_policy': 'exact',
                                             'count_source': 'user_explicit', 'expected_count': 3}}
    state = {'description': '检测3个目标，所有目标都闭合，荧光绿', 'understanding': understanding,
             'target_image_path': 'unused.png', 'unit': 'pixel', 'max_candidates': 1}
    state = _make_understand_task_node(None)(state)
    original = deepcopy(state['task_contract'])
    class Provider:
        def understand_task(self, *args, **kwargs):
            return {'task_summary': '只标中央目标', 'rendering': {'contour_color': '#ff0000'},
                    'acceptance_criteria': {'task_goal': '只标中央目标'}, 'candidate_pipelines': []}
    revised = _make_revise_candidates_node(Provider(), 1)(state)
    assert revised['task_contract'] == original
    assert revised['understanding']['rendering']['contour_color'] == '#39FF14'
    assert revised['acceptance_criteria']['expected_count'] == 3
    assert normalize_acceptance_criteria(revised['acceptance_criteria']) == revised['acceptance_criteria']


def test_05_bridge_passes_overlap_but_fails_object_gate():
    ref = np.zeros((30, 40), bool)
    ref[5:15, 5:15] = ref[5:15, 17:27] = True
    pred = ref.copy()
    pred[10, 15:17] = True
    metrics = evaluate_prediction(pred, ref)
    assert metrics['dice'] > .99 and metrics['boundary_f1'] == 1
    assert metrics['count_error'] == -1
    assert not meets_ground_truth_gate(metrics)


def test_05_gt_does_not_bypass_visual_or_delivery_review(tmp_path, monkeypatch):
    from core.graph_nodes import make_review_candidates_node
    monkeypatch.setattr('core.graph_nodes.promote_candidate_result', lambda state, name: state)
    candidate = {'name': 'one', 'status': 'selected_for_review',
                 'quality': {'coverage': .1, 'evaluation': {'status': 'ok', 'dice': 1, 'precision': 1,
                    'recall': 1, 'boundary_f1': 1, 'count_error': 0}},
                 'measurements': {'summary': {'count': 1}}}
    class Provider:
        calls = 0
        def review_candidates(self, *args, **kwargs):
            self.calls += 1
            return {'decision': 'revise', 'selected_candidate': 'one', 'observed_issues': ['wrong_color']}
    provider = Provider()
    state = {'target_image_path': 'x', 'description': '荧光绿', 'ground_truth_mask_path': 'gt',
             'selected_candidate': 'one', 'candidate_attempts': [candidate],
             'task_contract': {'acceptance_criteria': {'requested_output': ['width_measurement']}}}
    result = make_review_candidates_node(provider)(state)
    assert provider.calls == 1
    assert result['review']['decision'] == 'revise'
    assert not result['review']['acceptance']['overall_passed']
    assert 'missing_output:width_measurement' in result['review']['observed_issues']


def test_06_exclusion_rebuilds_all_final_artifacts(tmp_path, monkeypatch):
    path, pixels = source(tmp_path)
    pixels[10:20, 10:20] = 255
    Image.fromarray(pixels).save(path)
    excluded = tmp_path / 'excluded.png'
    Image.fromarray(pixels).save(excluded)
    def execution(image, pipeline):
        result = execute_pipeline(image, pipeline)
        result.outputs['special_measurement'] = MetadataArtifact({'area': 100})
        return result
    monkeypatch.setattr('core.experiments.runner.execute_pipeline_sandbox', execution)
    result = run_candidate({'pipeline': mask_pipeline()}, path, pixels, tmp_path / 'candidate',
        previous_state={**input_identity(path), 'human_feedback': {'exclude_mask_path': str(excluded)}})
    assert result['status'] == 'selected_for_review', result
    directory = Path(result['directory'])
    assert not np.asarray(Image.open(directory / 'mask.png')).any()
    assert json.loads((directory / 'contours.json').read_text())['count'] == 0
    assert result['measurements']['summary']['count'] == 0
    outputs = json.loads((directory / 'outputs.json').read_text())
    assert not np.asarray(outputs['mask']['data']).any()
    assert outputs['contours']['data'] == []
    assert 'special_measurement' not in outputs
    assert result['quality']['invalidated_outputs'] == ['special_measurement']
    assert np.asarray(json.loads((directory / 'raw_outputs.json').read_text())['mask']['data']).sum() == 100
    manifest = json.loads((directory / 'delivery_manifest.json').read_text())
    for name, item in manifest['files'].items():
        assert sha256((directory / item['path']).read_bytes()).hexdigest() == item['sha256']
    assert 'data' not in result['measurements']['structured_outputs']['mask']


def test_07_explicit_image_output_ignores_internal_mask():
    pipeline = mask_pipeline()
    pipeline['nodes'].append({'id': 'final_image', 'operator': 'normalize', 'inputs': {'image': '$image'}, 'params': {}})
    pipeline['outputs'] = {'result_image': 'final_image'}
    result = execute_pipeline(np.ones((5, 5)), normalize_pipeline(pipeline))
    assert result.mask is None and result.contours is None
    assert list(result.outputs) == ['result_image']


@pytest.mark.parametrize('fill', [0, 1])
def test_08_empty_and_full_masks_are_reviewable(fill):
    from core.quality import inspect_mask_health
    assert inspect_mask_health(np.full((5, 5), fill))['usable_for_review']
    assert not inspect_mask_health(np.array([[np.nan]]))['usable_for_review']


def test_09_high_bit_depth_is_preserved_and_preview_mime_matches(tmp_path):
    from core.preprocessing import load_grayscale
    from providers.vision import image_content
    path = tmp_path / 'science.tiff'
    pixels = np.array([[0, 256, 1024, 65535]], np.uint16)
    Image.fromarray(pixels).save(path)
    np.testing.assert_array_equal(load_grayscale(path), pixels)
    url = image_content(path, '')['image_url']['url']
    assert url.startswith('data:image/png;base64,')
    image = Image.open(BytesIO(base64.b64decode(url.split(',')[1])))
    assert image.format == 'PNG' and image.size == (4, 1)


def test_10_final_rgb_is_not_stretched(tmp_path, monkeypatch):
    path, pixels = source(tmp_path)
    artifact = ImageArtifact(np.full((40, 40, 3), 128, np.uint8))
    execution = PipelineExecutionResult({}, None, None, (), {}, {'result_image': artifact})
    monkeypatch.setattr('core.experiments.runner.execute_pipeline_sandbox', lambda *a: execution)
    result = run_candidate({'pipeline': mask_pipeline()}, path, pixels, tmp_path / 'candidate')
    assert result['status'] == 'selected_for_review'
    np.testing.assert_array_equal(np.asarray(Image.open(tmp_path / 'candidate/result_annotation.png')), artifact.data)


def test_11_edge_boundary_is_complete():
    from core.visualization import _mask_boundary
    assert len(_mask_boundary(np.ones((5, 5), bool))) == 16
    assert len(_mask_boundary(np.ones((1, 1), bool))) == 1


def test_13_concurrent_requests_do_not_receive_each_others_events():
    from core.agent_events import register_event_listener, unregister_event_listener, emit_event
    from core.runtime_logging import bind_context
    barrier = threading.Barrier(2)
    def request(name):
        bind_context(run_id=name, task_id=name)
        events = []
        callback = events.append
        register_event_listener(callback)
        try:
            barrier.wait()
            emit_event({'type': 'test', 'message': name})
            barrier.wait()
        finally:
            unregister_event_listener(callback)
        return events
    with ThreadPoolExecutor(2) as pool:
        a, b = list(pool.map(request, ['a', 'b']))
    assert [event['message'] for event in a] == ['a']
    assert [event['message'] for event in b] == ['b']
    assert a[0]['task_id'] == a[0]['run_id'] == 'a'


def test_13_invalid_input_cleans_listener(tmp_path, monkeypatch):
    from ui.annotation_app import run_chat_agent
    from core.agent_events import _event_listeners
    monkeypatch.setattr('ui.annotation_app.TASK_ROOT', tmp_path / 'tasks')
    before = list(_event_listeners)
    for _ in range(2):
        with pytest.raises(Exception):
            run_chat_agent(None, 'detect', [], None)
        assert _event_listeners == before


def test_14_long_source_has_exact_hash_addressed_retrieval():
    from providers.vision import build_revision_context_text
    from core.tools.discovery import query_operators
    code = 'def apply(data, params):\n    return data\n' + '# full source\n' * 1800
    pipeline = {'nodes': [], 'generated_operators': [{'name': 'long_source', 'source': code}]}
    context = {'previous_pipeline': pipeline}
    text = build_revision_context_text(context)
    identifier = 'long_source@' + sha256(code.encode()).hexdigest()
    assert identifier in text
    assert code not in text and 'source_retrieval' in text
    assert query_operators([identifier], context)[0]['source'] == code
    assert pipeline['generated_operators'][0]['source'] == code


def test_design_1_tail_anomalies_appear_in_summary():
    from core.experiments.context import bounded_value
    rows = [{'width': i, 'status': 'ok'} for i in range(100)]
    rows[-1] = {'width': 999, 'status': 'failed'}
    summary = bounded_value(rows, limit=1000)
    assert 99 in summary['anomaly_indices']
    assert summary['statistics']['width']['max'] == 999


def test_design_8_binary_transport_roundtrip_and_malformed_frame():
    from core.sandbox_transport import encode_frame, decode_frame
    pixels = np.zeros((2048, 2048), dtype=np.float32)
    raw = encode_frame({'image': pixels}, 64 * 1024**2)
    assert len(raw) < pixels.nbytes + 1024
    np.testing.assert_array_equal(decode_frame(raw, 64 * 1024**2)['image'], pixels)
    with pytest.raises(ValueError):
        decode_frame(raw[:-1], 64 * 1024**2)


def test_design_8_docker_large_image(docker_sandbox):
    from core.sandbox import execute_pipeline_sandbox
    pixels = np.zeros((2048, 2048), dtype=np.float32)
    pixels[500:510, 500:510] = 100
    result = execute_pipeline_sandbox(pixels, normalize_pipeline(mask_pipeline()))
    assert result.mask.data.shape == (2048, 2048)
    assert result.mask.data.sum() == 100


def test_01_same_size_image_switch_drops_gt_and_old_brush(tmp_path, monkeypatch):
    from core.task_store import TaskStore
    from ui.annotation_app import store_ground_truth_annotation, save_canvas_feedback
    monkeypatch.setattr('ui.annotation_app.TASK_ROOT', tmp_path / 'tasks')
    store = TaskStore(tmp_path / 'tasks')
    task = store.create_task()
    a, _ = source(tmp_path, 0, name='a.png')
    b, _ = source(tmp_path, 1, name='b.png')
    gt, _ = source(tmp_path, 255, name='gt.png')
    task = store_ground_truth_annotation(str(gt), task, str(a))
    assert task['ground_truth']['input_sha256'] == input_identity(a)['input_sha256']
    red = np.zeros((40, 40, 4), np.uint8)
    red[1:3, 1:3] = [255, 0, 0, 255]
    old = save_canvas_feedback({'background': str(a), 'layers': [red]}, task,
                              {**input_identity(a), 'target_image_path': str(a)})
    previous, _ = store.memory_service.prepare(task['id'], '继续', b, old)
    assert previous is None
    # The old upload still present in a Gradio component must not be rebound.
    task = store_ground_truth_annotation(str(gt), task, str(b))
    assert task['ground_truth'] is None
    green = np.zeros_like(red)
    green[7:9, 7:9] = [0, 255, 0, 255]
    new = save_canvas_feedback({'background': str(b), 'layers': [green]}, task,
                              {**input_identity(b), 'target_image_path': str(b)})
    assert new['false_positive_pixel_count'] == 0
    assert new['false_negative_pixel_count'] == 4


def test_12_candidate_switch_refreshes_text_and_removes_stale_files(tmp_path):
    from core.agent_loop import promote_candidate_result
    path, _ = source(tmp_path)
    root = tmp_path / 'run'
    iteration = root / 'iteration_0'
    iteration.mkdir(parents=True)
    (iteration / 'mask.png').write_bytes(b'old mask')
    (iteration / 'evaluation_report.json').write_text('{}')
    directory = tmp_path / 'candidate'
    directory.mkdir()
    Image.new('RGB', (40, 40)).save(directory / 'result_annotation.png')
    (directory / 'outputs.json').write_text('{}')
    candidate = {'name': 'new', 'status': 'selected_for_review', 'directory': str(directory),
                 'pipeline': {}, 'quality': {'output_kind': 'structured'},
                 'measurements': {'summary': {'count': 5, 'total_area': 50, 'unit': 'pixel'}}}
    state = {'run_dir': str(root), 'iteration': 0, 'target_image_path': str(path),
             'conversation': [{'role': 'assistant', 'content': '本轮标出1个区域'}],
             'candidate_attempts': [candidate]}
    result = promote_candidate_result(state, 'new')
    assert result['predicted_mask_path'] is None
    assert not (iteration / 'mask.png').exists()
    assert not (iteration / 'evaluation_report.json').exists()
    assert '1个区域' not in json.dumps(result['conversation'], ensure_ascii=False)
    assert '结构化' in result['conversation'][-1]['content']
    Image.new('L', (40, 40)).save(directory / 'mask.png')
    result = promote_candidate_result(result, 'new')
    assert '5 个区域' in result['conversation'][-1]['content']


def test_design_3_history_compaction_is_recoverable(tmp_path, monkeypatch):
    from core.model_context import compact_optional_history
    monkeypatch.setenv('LIANGCE_LLM_MAX_TEXT_CHARS', '1000')
    monkeypatch.setenv('LIANGCE_CONTEXT_ARCHIVE_DIR', str(tmp_path / 'archive'))
    messages = [{'role': 'user', 'content': 'exact requirements'},
                {'role': 'assistant', 'tool_calls': [{'id': 'old', 'function': {'arguments': 'x' * 2000}}]},
                {'role': 'tool', 'tool_call_id': 'old', 'content': 'old result'},
                {'role': 'assistant', 'tool_calls': [{'id': 'new', 'function': {'arguments': '{}'}}]},
                {'role': 'tool', 'tool_call_id': 'new', 'content': 'latest evidence'}]
    compact = compact_optional_history(messages)
    assert compact[0] == messages[0] and compact[-2:] == messages[-2:]
    archived = json.loads(next((tmp_path / 'archive').glob('*.json')).read_text())
    assert archived == messages[1:3]


def test_design_6_experiment_uses_contract(tmp_path, monkeypatch):
    from core.tools.experiments import ExperimentTools
    path, pixels = source(tmp_path)
    pixels[10:20, 10:20] = 255
    Image.fromarray(pixels).save(path)
    monkeypatch.setattr('core.experiments.runner.execute_pipeline_sandbox', execute_pipeline)
    context = {'task_contract': {'rendering': {'contour_color': '#39FF14'},
        'target_constraints': {'expected_count': 1}, 'unit': 'pixel'}, 'ground_truth_mask_path': str(path)}
    tools = ExperimentTools(path, 'test', context, output_root=tmp_path / 'experiments')
    result, _ = tools.execute({'pipeline': mask_pipeline()})
    attempt = tools.attempts[result['experiment_id']]
    spec = json.loads((Path(attempt['directory']) / 'execution_spec.json').read_text())
    assert spec['rendering'] == context['task_contract']['rendering']
    assert spec['target_constraints'] == context['task_contract']['target_constraints']
    assert attempt['quality']['evaluation']['dice'] == 1


def test_invalid_final_pipeline_is_rejected_before_leaving_planner():
    from providers.vision import normalize_task_understanding
    pipeline = mask_pipeline()
    pipeline['nodes'][0]['params']['radius'] = 3
    with pytest.raises(ValueError, match='unknown params'):
        normalize_task_understanding({'candidate_pipelines': [{'pipeline': pipeline}]})


def test_prose_output_requirements_are_visually_reviewed_not_literal_names():
    from core.task_contract import check_delivery
    candidate = {'quality': {'coverage': .1}, 'measurements': {'summary': {'count': 2}}}
    contract = {'output_requirements': ['mask', 'measurements'], 'acceptance_criteria': {
        'requested_output': ['每个目标的二值mask（可共存）', '荧光绿色闭合轮廓标注图']}}
    result = check_delivery(candidate, contract)
    assert result['passed'] and len(result['semantic_requirements_for_review']) == 2


def test_image_scoped_memory_only_follows_matching_input(tmp_path):
    from core.task_store import TaskStore
    tasks = TaskStore(tmp_path / 'tasks')
    task_id = tasks.create_task()['id']
    a, _ = source(tmp_path, 0, name='a.png')
    b, _ = source(tmp_path, 1, name='b.png')
    service = tasks.memory_service
    _, context = service.prepare(task_id, '只检测左上角', a)
    service.apply_updates(task_id, '只检测左上角', {'memory_updates': [
        {'op': 'set', 'key': 'constraint:region', 'value': '左上角', 'scope': 'image', 'source_quote': '只检测左上角'}]}, context['memory_source_id'])
    assert service.snapshot(task_id, input_identity(a)['input_sha256'])['active_constraints']['region'] == '左上角'
    assert 'region' not in service.snapshot(task_id, input_identity(b)['input_sha256'])['active_constraints']


def test_contract_user_change_has_version_and_source():
    from core.task_contract import establish_contract
    original = establish_contract({'description': '用荧光绿'}, {'rendering': {'contour_color': '#39FF14'}})
    next_contract = establish_contract({'description': '改为红色', 'task_contract': original}, {
        'contract_updates': [{'field': 'rendering', 'value': {'contour_color': '#ff0000'}, 'source_quote': '改为红色'}]})
    assert original['rendering']['contour_color'] == '#39FF14'
    assert next_contract['rendering']['contour_color'] == '#ff0000'
    assert next_contract['version'] == 2 and next_contract['changes'][0]['source_quote'] == '改为红色'


def test_13_stream_deadline_cancels_and_cleans_listener(tmp_path, monkeypatch):
    from ui.annotation_app import run_chat_agent_stream
    from core.agent_events import _event_listeners, register_event_listener, unregister_event_listener, emit_thinking
    from core.request_control import check_cancelled
    import time
    before = list(_event_listeners)
    cleaned = threading.Event()
    def waiting(*args, **kwargs):
        callback = lambda event: None
        register_event_listener(callback)
        try:
            while True:
                check_cancelled()
                time.sleep(.01)
        finally:
            unregister_event_listener(callback)
            cleaned.set()
    monkeypatch.setattr('ui.annotation_app.run_chat_agent', waiting)
    monkeypatch.setenv('LIANGCE_REQUEST_TIMEOUT_SECONDS', '.03')
    updates = list(run_chat_agent_stream('sample.png', 'detect', [], None))
    assert updates and cleaned.wait(1)
    assert _event_listeners == before


def test_10_structured_boxes_use_task_color_after_final_image(tmp_path, monkeypatch):
    path, pixels = source(tmp_path)
    image = ImageArtifact(np.full((40, 40, 3), 128, np.uint8))
    result = PipelineExecutionResult({}, None, None, (), {}, {
        'measurements': MetadataArtifact({'boxes': [[5, 5, 15, 15]]}), 'image': image})
    monkeypatch.setattr('core.experiments.runner.execute_pipeline_sandbox', lambda *a: result)
    attempt = run_candidate({'pipeline': mask_pipeline()}, path, pixels, tmp_path / 'candidate',
                            rendering={'contour_color': '#39FF14'})
    assert attempt['status'] == 'selected_for_review'
    actual = np.asarray(Image.open(tmp_path / 'candidate/result_annotation.png'))
    np.testing.assert_array_equal(actual[5, 5], [57, 255, 20])
    np.testing.assert_array_equal(actual[20, 20], [128, 128, 128])
