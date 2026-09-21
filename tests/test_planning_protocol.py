import json
from types import SimpleNamespace

import pytest
from PIL import Image

from core.planning import ModelReply, PlanningSession
from core.tools.experiments import ExperimentTools
from core.tools.contracts import TOOL_SPECS
from providers.vision import AliyunVisionProvider


@pytest.fixture
def dispatcher(tmp_path):
    target = tmp_path / 'input.png'
    Image.new('L', (8, 8)).save(target)
    return ExperimentTools(target, 'test', output_root=tmp_path)


def run_session(dispatcher, complete, **kwargs):
    return PlanningSession(dispatcher, None, **kwargs).run(
        [], complete, json.loads, lambda raw: raw,
        lambda path, label: {'type': 'image_url', 'image_url': {'url': path}})


def test_native_tool_results_use_ids_and_structured_errors(dispatcher):
    seen = []
    def complete(messages, specs, final_only):
        seen.append(list(messages))
        if len(seen) == 1:
            return ModelReply(calls=[{'id': 'abc', 'name': 'query_operators', 'arguments': '{"names":42}'}])
        result_message = messages[-1]
        assert result_message['role'] == 'tool'
        assert result_message['tool_call_id'] == 'abc'
        error = json.loads(result_message['content'])['error']
        assert error['code'] == 'invalid_arguments'
        assert error['retryable']
        return ModelReply('{"done":true}')
    assert run_session(dispatcher, complete)['done']
    assert dispatcher.budget.used['discovery'] == 1


def test_budget_exhaustion_disables_only_affected_tools(dispatcher):
    dispatcher.budget.limits['discovery'] = 1
    count = 0
    def complete(messages, specs, final_only):
        nonlocal count
        count += 1
        names = {spec['function']['name'] for spec in specs}
        if count == 1:
            return ModelReply(calls=[{'id': 'a', 'name': 'query_operators', 'arguments': '{"names":["normalize"]}'}])
        assert not final_only
        assert 'query_operators' not in names and 'execute_pipeline' in names
        if count == 2:
            return ModelReply(calls=[{'id': 'b', 'name': 'query_operators', 'arguments': '{}'}])
        error = json.loads(messages[-1]['content'])['error']
        assert error['code'] == 'budget_exhausted' and not error['retryable']
        assert '安全校验' not in json.dumps(messages, ensure_ascii=False)
        return ModelReply('{"done":true}')
    assert run_session(dispatcher, complete)['done']


def test_exhaustion_forces_final_without_running_tools(dispatcher):
    for key in dispatcher.budget.limits:
        dispatcher.budget.limits[key] = 0
    count = 0
    def complete(messages, specs, final_only):
        nonlocal count
        count += 1
        assert final_only and not specs
        if count == 1:
            return ModelReply(calls=[{'id': 'blocked', 'name': 'execute_pipeline', 'arguments': '{}'}])
        assert json.loads(messages[-2]['content'])['error']['code'] == 'finalization_required'
        return ModelReply('{"done":true}')
    assert run_session(dispatcher, complete)['done']
    assert not dispatcher.events


def test_dispatch_enforces_budget_without_provider_and_saves_failures(dispatcher):
    for _ in range(3):
        result, _ = dispatcher.dispatch({'tool': 'query_operators', 'arguments': {'names': []}}, None)
        assert result['error']['code'] == 'invalid_arguments'
    result, _ = dispatcher.dispatch({'tool': 'load_skill', 'arguments': {'name': 'area_measurement'}}, None)
    assert result['error']['code'] == 'budget_exhausted'
    assert (dispatcher.root / 'session.json').exists()
    assert 'inspect_artifact' in result['available_tools']


def test_two_measurement_skills_and_batch_operator_query_fit_discovery_budget(dispatcher):
    for name, acceptance in [('area_measurement', 'roi_denominator'),
                             ('defect_count', 'border_count_policy')]:
        result, _ = dispatcher.dispatch({'tool': 'load_skill', 'arguments': {'name': name}}, None)
        assert result['status'] == 'success'
        assert acceptance in result['data']['skill']['content']
    assert result['budget']['discovery'] == 1
    result, _ = dispatcher.dispatch({'tool': 'query_operators', 'arguments': {
        'names': ['normalize', 'global_threshold', 'component_statistics']}}, None)
    assert result['status'] == 'success'
    assert len(result['data']['operators']) == 3
    assert result['budget']['discovery'] == 0
    assert 'query_operators' not in result['available_tools']
    assert 'execute_pipeline' in result['available_tools']


def test_native_stream_accumulates_fragmented_arguments_and_sends_function_schema():
    requests = []
    chunks = [
        {'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'call_1', 'function': {'name': 'query_operators', 'arguments': '{"names":'}}]}}]},
        {'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': '["normalize"]}'}}]}}]},
    ]
    def create(**kwargs):
        requests.append(kwargs)
        return iter(chunks)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    provider = AliyunVisionProvider(api_key='test')
    reply = provider._complete_streaming(client, [], tools=[TOOL_SPECS['query_operators'].function_schema()])
    assert reply.calls == [{'id': 'call_1', 'name': 'query_operators', 'arguments': '{"names":["normalize"]}'}]
    assert requests[0]['tool_choice'] == 'auto'
    assert requests[0]['parallel_tool_calls'] is False
    assert requests[0]['tools'][0]['function']['parameters']['additionalProperties'] is False
    provider._complete_streaming(client, [], tools=[], tool_choice='none')
    assert requests[-1]['tool_choice'] == 'none'


def test_multiple_calls_all_receive_results_before_images(dispatcher, monkeypatch):
    def dispatch(action, root):
        return {'data': {}, 'error': None}, ['/image.png']
    monkeypatch.setattr(dispatcher, 'dispatch', dispatch)
    count = 0
    def complete(messages, specs, final_only):
        nonlocal count
        count += 1
        if count == 1:
            return ModelReply(calls=[{'id': str(i), 'name': 'inspect_artifact', 'arguments': '{"ids":["a"]}'} for i in range(2)])
        assert [message['role'] for message in messages] == ['assistant', 'tool', 'tool', 'user']
        return ModelReply('{"done":true}')
    assert run_session(dispatcher, complete)['done']


@pytest.mark.parametrize('code', ['timeout', 'resource_limit', 'execution_failed'])
def test_runner_failure_codes_survive_dispatch(dispatcher, monkeypatch, code):
    from core.sandbox import SandboxExecutionError
    def fail(*args, **kwargs):
        raise SandboxExecutionError('test failure', code=code)
    monkeypatch.setattr('core.experiments.runner.execute_pipeline_sandbox', fail)
    result, _ = dispatcher.dispatch({'tool': 'execute_pipeline', 'arguments': {'pipeline': {
        'steps': [{'id': 'mask', 'op': 'global_threshold', 'input': 'image', 'params': {}}]}}}, None)
    assert result['error']['code'] == code
    assert result['error']['retryable'] is (code == 'timeout')
    assert dispatcher.executions == 1
    assert result['data']['experiment_id']


def test_native_provider_integration_with_tool_then_final(tmp_path, monkeypatch):
    from core.pipelines.dsl import execute_pipeline
    monkeypatch.setattr('core.experiments.runner.execute_pipeline_sandbox', execute_pipeline)
    monkeypatch.setattr(ExperimentTools, 'preflight', lambda self: {'image_id': 'test'})
    target = tmp_path / 'image.png'
    Image.new('L', (10, 10), 255).save(target)
    requests = []
    def create(**kwargs):
        requests.append(kwargs)
        index = len(requests)
        replies = [json.loads(message['content']) for message in kwargs['messages'] if message['role'] == 'tool']
        if index == 1:
            name, args = 'save_task', {'understanding': {'task_summary': 'bright', 'acceptance_criteria': {}}}
        elif index == 2:
            name, args = 'load_skill', {'name': 'area_measurement'}
        elif index == 3:
            assert replies[-1]['data']['skill']['name'] == 'area_measurement'
            name, args = 'create_draft', {'pipeline': {'steps': [
                {'id': 'mask', 'op': 'global_threshold', 'input': 'image', 'params': {}}]}, 'change_reason': 'bright foreground'}
        elif index == 4:
            name, args = 'execute_pipeline', {key: replies[-1]['data'][key] for key in ('draft_id', 'revision')}
        else:
            assert index == 5
            name, args = 'submit_experiment', {'experiment_id': replies[-1]['data']['experiment_id'], 'reason': 'inspect boundary'}
        return iter([{'choices': [{'finish_reason': 'tool_calls', 'delta': {'tool_calls': [{
            'index': 0, 'id': f'call_{index}', 'function': {'name': name, 'arguments': json.dumps(args)}}]}}]}])
    monkeypatch.setattr('openai.OpenAI', lambda **kwargs: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    provider = AliyunVisionProvider(api_key='test', tool_mode='native')
    result = provider.understand_task(target, 'bright', previous_context={'experiment_output_root': str(tmp_path)})
    assert result['candidate_pipelines'] and result['submitted_experiment_id']
    assert len(requests) == 5
    assert result['tool_session']['budget']['discovery'] == 2
    assert result['tool_session']['executions'] == 1


def test_invalid_json_is_not_reported_as_safety_failure(dispatcher):
    count = 0
    def complete(messages, specs, final_only):
        nonlocal count
        count += 1
        if count == 1:
            return ModelReply('not JSON')
        error = json.loads(messages[-1]['content'])['error']
        assert error['code'] == 'invalid_json'
        return ModelReply('{"done":true}')
    assert run_session(dispatcher, complete)['done']


def test_timeout_retry_executes_again_instead_of_reusing_failure(dispatcher, monkeypatch):
    from core.sandbox import SandboxExecutionError
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise SandboxExecutionError('timeout', code='timeout')
    monkeypatch.setattr('core.experiments.runner.execute_pipeline_sandbox', fail)
    action = {'tool': 'execute_pipeline', 'arguments': {'pipeline': {
        'steps': [{'id': 'mask', 'op': 'global_threshold', 'input': 'image', 'params': {}}]}}}
    first, _ = dispatcher.dispatch(action, None)
    second, _ = dispatcher.dispatch(action, None)
    assert first['error']['retryable'] and second['error']['retryable']
    assert len(calls) == 2
    assert first['data']['experiment_id'] != second['data']['experiment_id']
