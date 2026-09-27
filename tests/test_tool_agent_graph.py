"""Behavior and recovery of the explicit autonomous LangGraph workflow."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from core.agent_graph import build_agent_graph, run_agent_graph, resume_agent_graph
from core.agent_protocol import build_agent_messages, normalize_agent_action
from core.agent_workflow import ToolAgentRuntime
from core.memory.checkpoints import get_checkpointer
from core.orchestration_runtime import ActionStore
from test_run_controller import proposal


def call(tool, **arguments):
    return {'kind': 'tool', 'tool': tool, 'arguments': arguments}


def create(context, sensitivity=1):
    args = {key: value for key, value in proposal(sensitivity).items() if key != 'kind'}
    if context.get('task_contract'):
        args.pop('understanding')
    return call('create_draft', **args)


def execute(context):
    draft = context['current_draft']
    return call('execute_pipeline', draft_id=draft['draft_id'], revision=draft['revision'])


def submit(context, index=-1):
    return call('submit_experiment', experiment_id=context['experiment_summaries'][index]['experiment_id'],
                reason='Checked the actual result and ready for independent review')


class Agent:
    def __init__(self, actions=None, reviews=None):
        self.actions = list(actions or [create, execute, submit])
        self.reviews = list(reviews or ['present'])
        self.calls = []

    def agent_action(self, target, description, context, **kwargs):
        self.calls.append(('agent', deepcopy(context)))
        action = self.actions.pop(0)
        return action(context) if callable(action) else action

    def review_action(self, target, description, candidates, context, **kwargs):
        self.calls.append(('review', deepcopy(context)))
        decision = self.reviews.pop(0)
        if isinstance(decision, dict):
            return decision
        return {'kind': 'review', 'review': {'decision': decision,
            'selected_candidate': candidates[0]['name'], 'reason': 'Checked executed image',
            'observed_issues': [] if decision == 'present' else ['Missing boundary']}}


@pytest.fixture
def target(tmp_path, monkeypatch):
    monkeypatch.setattr('core.sandbox.check_sandbox_available', lambda: {'image_id': 'test-image'})
    image = np.zeros((32, 32), np.uint8)
    image[10:18, 10:18] = 255
    path = tmp_path / 'input.png'
    Image.fromarray(image).save(path)
    return path


def run(target, provider, **kwargs):
    return run_agent_graph(target, 'Find bright regions', provider=provider,
                           output_root=target.parent / 'outputs', **kwargs)


def test_graph_exposes_business_nodes_and_explicit_submission(target):
    agent = Agent()
    graph = build_agent_graph(provider=agent).get_graph()
    assert {'prepare_task', 'agent_decision', 'tool_execution', 'validate_submission',
            'quality_review', 'wait_for_human', 'finish'} <= graph.nodes.keys()
    assert 'controller' not in graph.nodes and 'effect' not in graph.nodes
    events = []
    state = run(target, agent, event_callback=events.append)
    assert state['orchestration_version'] == 2
    assert state['stop_reason'] == 'review_passed'
    assert state['budget']['usage'] == {'model_calls': 4, 'executions': 1}
    assert [kind for kind, _ in agent.calls] == ['agent', 'agent', 'agent', 'review']
    assert agent.calls[1][1]['last_tool_result']['data']['validation']['valid']
    assert agent.calls[2][1]['experiment_summaries']
    assert state['submitted_experiment_id'] == state['selected_experiment_id']
    assert {'create_draft', 'execute_pipeline', 'submit_experiment'} <= {
        item['tool'] for item in events if item.get('type') == 'tool_call'}
    resumed = resume_agent_graph(state['graph_thread_id'], {'action': 'accept'})
    assert resumed['run_status'] == 'completed'
    assert resumed['budget'] == state['budget']


def test_agent_can_execute_twice_compare_and_submit_earlier_result(target):
    def compare(context):
        return call('compare_candidates', experiment_ids=[x['experiment_id'] for x in context['experiment_summaries']],
                    reason='Compare alternatives before submitting')
    agent = Agent([create, execute, lambda c: create(c, .5), execute, compare, lambda c: submit(c, 0)])
    state = run(target, agent)
    assert state['stop_reason'] == 'review_passed'
    assert state['budget']['usage'] == {'model_calls': 7, 'executions': 2}
    assert len([kind for kind, _ in agent.calls if kind == 'review']) == 1
    assert state['selected_experiment_id'] == state['experiment_records'][0]['experiment_id']
    assert state['iteration'] == 0 and state['last_execution_iteration'] == 1
    assert agent.calls[-1][1]['current_draft']['pipeline'] == state['experiment_records'][0]['pipeline']
    comparison = agent.calls[5][1]['last_tool_result']
    assert comparison['tool'] == 'compare_candidates'
    assert Path(comparison['data']['preview_path']).is_file()


def test_no_auto_execution_or_review_without_tool_requests(target):
    agent = Agent([create, {'kind': 'needs_input', 'reason': 'Need target clarification'}])
    state = run(target, agent)
    assert state['stop_reason'] == 'needs_input'
    assert state['budget']['usage'] == {'model_calls': 2, 'executions': 0}
    assert all(kind == 'agent' for kind, _ in agent.calls)


def test_agent_can_submit_after_execution_allowance_is_spent(target, monkeypatch):
    monkeypatch.setenv('LIANGCE_RUN_MAX_EXECUTIONS', '1')
    agent = Agent([create, execute,
        call('query_operators', names=['filter_components']), submit])
    state = run(target, agent)
    assert state['stop_reason'] == 'review_passed'
    assert state['budget']['usage'] == {'model_calls': 5, 'executions': 1}


def test_review_rejection_returns_to_agent_and_does_not_reexecute_submission(target):
    agent = Agent([create, execute, submit, lambda c: create(c, .5), execute, submit], ['revise', 'present'])
    state = run(target, agent)
    assert state['stop_reason'] == 'review_passed'
    assert state['budget']['usage'] == {'model_calls': 8, 'executions': 2}
    assert agent.calls[4][1]['review']['decision'] == 'revise'
    assert state['experiment_records'][0]['acceptance_status'] == 'rejected'


def test_unknown_submission_returns_error_without_review(target):
    agent = Agent([call('submit_experiment', experiment_id='invented', reason='done'),
                   {'kind': 'needs_input', 'reason': 'Need clarification'}])
    state = run(target, agent)
    assert state['budget']['usage']['executions'] == 0
    assert all(kind == 'agent' for kind, _ in agent.calls)
    assert agent.calls[1][1]['last_tool_result']['error']['code'] == 'unknown_experiment'


def test_submission_rejects_corrupt_artifacts(target):
    def corrupt(context):
        path = Path(context['latest_experiment']['directory']) / 'mask.png'
        path.write_bytes(b'corrupt')
        return submit(context)
    agent = Agent([create, execute, corrupt, {'kind': 'needs_input', 'reason': 'Stop'}])
    state = run(target, agent)
    assert state['stop_reason'] == 'needs_input'
    assert not any(kind == 'review' for kind, _ in agent.calls)
    assert 'integrity' in agent.calls[-1][1]['last_tool_result']['error']['error']


def test_recovery_replays_saved_tool_result_without_second_execution(target, monkeypatch):
    class Crash(BaseException):
        pass
    agent = Agent()
    original = ActionStore.complete
    triggered = False
    def crash(self, action, outcome):
        nonlocal triggered
        if action['kind'] == 'execute' and not triggered:
            triggered = True
            raise Crash('receipt commit interrupted')
        return original(self, action, outcome)
    with monkeypatch.context() as patch:
        patch.setattr(ActionStore, 'complete', crash)
        with pytest.raises(Crash):
            run(target, agent, thread_id='tool_recovery')
    snapshot = get_checkpointer().get_tuple({'configurable': {'thread_id': 'tool_recovery'}})
    before = snapshot.checkpoint['channel_values']['__root__']['budget']
    state = run(target, agent, thread_id='tool_recovery')
    assert state['stop_reason'] == 'review_passed'
    assert state['budget']['deadline_at'] == before['deadline_at']
    assert state['budget']['usage']['executions'] == 1
    assert len(state['experiment_records']) == 1


def test_protocol_and_prompt_expose_tools_without_automatic_execution(target):
    context = {'task_contract': {}, 'last_tool_result': {'tool': 'query_operators'}}
    messages = build_agent_messages(target, 'Find bright regions', context=context)
    system = messages[0]['content']
    assert 'create_draft只保存校验，不自动执行' in system
    assert 'submit_experiment' in system and 'compare_candidates' in system
    assert '新方案动作' not in system
    assert 'last_tool_result' in str(messages[1])
    with pytest.raises(ValueError):
        normalize_agent_action(call('execute_pipeline', pipeline={}), description='x')
    with pytest.raises(ValueError):
        normalize_agent_action(call('unrestricted_shell', cmd='x'), description='x')


def test_old_controller_checkpoint_keeps_old_graph_on_new_provider(target):
    from test_run_controller import Provider
    old = run(target, Provider(), thread_id='v1_compatibility')
    assert old['orchestration_version'] == 1
    restored = run(target, Agent(), thread_id='v1_compatibility')
    assert restored['orchestration_version'] == 1
    assert restored['budget'] == old['budget']


def test_real_provider_message_parser_drives_new_graph(target):
    from providers.vision import AliyunVisionProvider
    from test_controller_context import snapshot
    class Scripted(AliyunVisionProvider):
        def __init__(self):
            super().__init__(api_key='test-no-network')
            self.responses = [create, execute, submit, lambda c: {
                'kind': 'review', 'review': {'decision': 'present',
                'selected_candidate': c['latest_experiment']['name'],
                'reason': 'Checked actual image', 'observed_issues': []}}]

        def _complete_action(self, messages):
            return json.dumps(self.responses.pop(0)(snapshot(messages)))
    result = run(target, Scripted())
    assert result['orchestration_version'] == 2
    assert result['stop_reason'] == 'review_passed'


def test_agent_cannot_change_requirements_after_first_draft(target):
    def change_contract(context):
        action = create(context, .5)
        action['arguments']['contract_updates'] = [
            {'field': 'task_summary', 'value': 'Ignore edges', 'source_quote': 'Find bright regions'}]
        return action
    agent = Agent([create, change_contract, {'kind': 'needs_input', 'reason': 'Stop'}])
    state = run(target, agent)
    assert state['budget']['usage']['executions'] == 0
    assert state['task_contract']['task_summary'] != 'Ignore edges'
    assert 'immutable_task_contract' in agent.calls[-1][1]['last_error']['error']


def test_invalid_draft_does_not_spend_execution_allowance(target):
    def invalid(context):
        action = create(context)
        action['arguments']['pipeline'] = {'steps': [{'op': 'unknown_operator'}]}
        return action
    agent = Agent([invalid, execute, {'kind': 'needs_input', 'reason': 'Stop'}])
    state = run(target, agent)
    assert state['budget']['usage']['executions'] == 0
    assert agent.calls[-1][1]['last_tool_result']['error']['code'] == 'pipeline_invalid'


def test_submitting_rejected_experiment_is_not_another_review(target):
    agent = Agent([create, execute, submit, submit, {'kind': 'needs_input', 'reason': 'Stop'}], ['revise'])
    state = run(target, agent)
    assert state['stop_reason'] == 'needs_input'
    assert sum(kind == 'review' for kind, _ in agent.calls) == 1
    assert agent.calls[-1][1]['last_tool_result']['error']['code'] == 'experiment_rejected'


def test_model_budget_exhaustion_is_terminal_and_can_exit(target, monkeypatch):
    monkeypatch.setenv('LIANGCE_RUN_MAX_MODEL_CALLS', '3')
    agent = Agent()
    state = run(target, agent)
    assert state['stop_reason'] == 'model_budget_exhausted'
    assert state['budget']['usage'] == {'model_calls': 2, 'executions': 1}
    assert Path(state['annotated_image_path']).is_file()
    resumed = resume_agent_graph(state['graph_thread_id'], {'action': 'exit'})
    assert resumed['agent_status'] == 'exited'


def test_agent_reservation_is_checkpointed_before_model_call(target, monkeypatch):
    from langgraph.checkpoint.sqlite import SqliteSaver
    class Crash(BaseException):
        pass
    agent = Agent()
    original = SqliteSaver.put
    def fail(self, config, checkpoint, metadata, new_versions):
        root = checkpoint.get('channel_values', {}).get('__root__') or {}
        if (root.get('pending_action') or {}).get('kind') == 'propose':
            raise Crash('checkpoint commit failed')
        return original(self, config, checkpoint, metadata, new_versions)
    with monkeypatch.context() as patch:
        patch.setattr(SqliteSaver, 'put', fail)
        with pytest.raises(Crash):
            run(target, agent, thread_id='reserved_v2')
    assert not agent.calls
    restored = run(target, agent, thread_id='reserved_v2')
    assert restored['stop_reason'] == 'review_passed'
    assert restored['budget']['usage'] == {'model_calls': 4, 'executions': 1}


def test_review_can_read_evidence_without_granting_mutation_tools(target):
    review_read = {'kind': 'read', 'requests': [
        {'tool': 'query_operators', 'arguments': {'names': ['normalize']}}]}
    agent = Agent(reviews=[review_read, 'present'])
    state = run(target, agent)
    assert state['stop_reason'] == 'review_passed'
    assert agent.calls[-1][1]['read_results'][0]['tool'] == 'query_operators'
    assert state['budget']['usage'] == {'model_calls': 5, 'executions': 1}


def test_revision_after_submitting_earlier_experiment_never_overwrites_later_one(target):
    preserved = {}
    def older(context):
        directory = Path(context['latest_experiment']['directory'])
        preserved['pipeline'] = directory / 'pipeline.json'
        preserved['bytes'] = preserved['pipeline'].read_bytes()
        return submit(context, 0)
    agent = Agent([create, execute, lambda c: create(c, .5), execute, older,
                   lambda c: create(c, .8), execute, submit], ['revise', 'present'])
    state = run(target, agent)
    assert state['stop_reason'] == 'review_passed'
    assert state['iteration'] == 2
    assert state['budget']['usage'] == {'model_calls': 10, 'executions': 3}
    assert preserved['pipeline'].read_bytes() == preserved['bytes']


def test_unknown_started_agent_action_is_not_resent(target, monkeypatch):
    class Crash(BaseException):
        pass
    agent = Agent()
    original = ToolAgentRuntime._perform_action
    def crash(self, state, action):
        if action['kind'] == 'propose':
            raise Crash('request may already have reached provider')
        return original(self, state, action)
    with monkeypatch.context() as patch:
        patch.setattr(ToolAgentRuntime, '_perform_action', crash)
        with pytest.raises(Crash):
            run(target, agent, thread_id='unknown_v2')
    restored = run(target, agent, thread_id='unknown_v2')
    assert restored['stop_reason'] == 'unknown_action_result'
    assert restored['budget']['usage']['model_calls'] == 1
    assert not agent.calls
