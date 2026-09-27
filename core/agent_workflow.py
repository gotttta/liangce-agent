"""Explicit LangGraph business nodes around one autonomous tool-using Agent.

Every node settles one durable action and reserves the next before checkpointing.
The shared runtime owns receipts/deadlines, never a nested model planning loop.
Version-one checkpoints continue to use core.orchestration's original graph.
"""
from copy import deepcopy
import json
from pathlib import Path
from typing import Annotated

from langgraph.graph import END, StateGraph

from core.agent_protocol import normalize_agent_action
from core.experiments.drafts import DraftStore, content_hash
from core.experiments.artifacts import experiment_scope
from core.experiments.delivery import verify_manifest
from core.orchestration import RunController, READ_TOOLS
from core.orchestration_runtime import ActionStore
from core.tools.contracts import ToolError


def build_tool_agent_graph(provider, checkpointer, algorithm_registry=None):
    from core.agent_graph import _wait_for_human
    runtime = ToolAgentRuntime(provider, algorithm_registry)
    # Routing is exclusive. Explicit replacement also lets LangGraph's graph
    # renderer simulate converging conditional branches without LastValue errors.
    graph = StateGraph(Annotated[dict, _replace_state])
    graph.add_node('initialize_run', runtime.advance)
    nodes = {
        'prepare_task': {'prepare'},
        'agent_decision': {'propose'},
        'tool_execution': {'read', 'draft', 'execute', 'compare'},
        'validate_submission': {'submit'},
        'quality_review': {'review'},
        'review_evidence': {'read'},
    }
    for name, phases in nodes.items():
        graph.add_node(name, runtime.node(phases))
    graph.add_node('wait_for_human', _wait_for_human)
    graph.add_node('finish', lambda state: state)
    graph.set_entry_point('initialize_run')
    routes = {
        'initialize_run': ['prepare_task', 'finish'],
        'prepare_task': ['agent_decision', 'finish'],
        'agent_decision': ['agent_decision', 'tool_execution', 'validate_submission', 'wait_for_human', 'finish'],
        'tool_execution': ['agent_decision', 'finish'],
        'validate_submission': ['quality_review', 'agent_decision', 'finish'],
        'quality_review': ['quality_review', 'review_evidence', 'agent_decision', 'wait_for_human', 'finish'],
        'review_evidence': ['quality_review', 'wait_for_human', 'finish'],
    }
    for name, destinations in routes.items():
        graph.add_conditional_edges(name, runtime.route, destinations)
    graph.add_edge('wait_for_human', 'finish')
    graph.add_edge('finish', END)
    return graph.compile(checkpointer=checkpointer)


def _replace_state(previous, current):
    return current


class ToolAgentRuntime(RunController):
    orchestration_version = 2
    autonomous_tools = True

    @staticmethod
    def route(state):
        action = state.get('pending_action')
        if not action:
            return 'wait_for_human' if state.get('run_status') == 'awaiting_feedback' else 'finish'
        phase = action['kind']
        if phase == 'read' and state.get('read_return_phase') == 'review':
            return 'review_evidence'
        return {'prepare': 'prepare_task', 'propose': 'agent_decision',
                'submit': 'validate_submission', 'review': 'quality_review'}.get(phase, 'tool_execution')

    def node(self, phases):
        def run(state):
            action = state.get('pending_action') or {}
            if action.get('kind') not in phases:
                raise ValueError('Checkpoint action does not belong to this graph node')
            request = state.get('tool_request') if action['kind'] in {'draft', 'execute', 'compare', 'submit'} or (
                action['kind'] == 'read' and state.get('read_return_phase') == 'propose') else None
            if request:
                from core.agent_events import emit_tool_call
                # Source and task text live in the draft/action store, not progress logs.
                arguments = {key: value for key, value in request['arguments'].items()
                             if key in {'draft_id', 'revision', 'base_revision', 'experiment_id',
                                        'experiment_ids', 'names', 'name', 'path', 'report', 'region', 'ids'}}
                emit_tool_call(request['tool'], arguments)
            result = self.perform(state)
            if request:
                from core.agent_events import emit_tool_result
                from core.request_control import RequestCancelled
                outcome = result['action_outcome']
                try:
                    emit_tool_result(request['tool'], {'action_id': action['id'], 'status': outcome['status']},
                                     success=outcome['status'] == 'ok')
                except RequestCancelled:
                    pass  # Result is already durable; advance records cancellation.
            return self.advance(result)
        return run

    def advance(self, value):
        state = super().advance(value)
        action = state.get('pending_action') or {}
        if action.get('kind') == 'execute':
            # Selecting an earlier result must not overwrite a later iteration.
            action['iteration'] = max(action['iteration'], int(state.get('last_execution_iteration', -1)) + 1)
        return state

    def _context(self, state):
        context = super()._context(state)
        context = {**context, 'last_tool_result': state.get('last_tool_result'),
                'draft_catalog': state.get('draft_catalog', []),
                'submitted_experiment_id': state.get('submitted_experiment_id')}
        reviewing = state.get('phase') == 'review' or (
            state.get('phase') == 'read' and state.get('read_return_phase') == 'review')
        submitted = state.get('submitted_draft')
        if reviewing and submitted:
            draft = DraftStore(state['run_dir']).load(submitted['draft_id'], submitted['revision'])
            context.update(current_draft=draft, previous_pipeline=draft['pipeline'], draft_catalog=[submitted])
        return context

    def _perform_action(self, state, action):
        if action['kind'] == 'propose':
            self._validate_scope(state)
            context = self._context(state)
            raw = self.provider.agent_action(state['target_image_path'], state['description'],
                context=self._context(state), reference_examples=state.get('reference_examples', []))
            return normalize_agent_action(raw, description=state['description'], context=context)
        if action['kind'] == 'submit':
            return self._submission(state)
        if action['kind'] == 'compare':
            return self._compare(state, action)
        if action['kind'] == 'execute':
            args = state['tool_request']['arguments']
            draft = DraftStore(state['run_dir']).get(args['draft_id'], args['revision'])
            if not draft['validation']['valid']:
                raise ToolError('pipeline_invalid', 'Repair the draft before execution')
            state = {**state, 'current_draft': DraftStore.summary(draft),
                     'iteration': action['iteration'] - 1}
        return super()._perform_action(state, action)

    def _apply(self, state, action, outcome):
        phase = action['kind']
        if outcome['status'] != 'ok':
            # Return tool diagnostics to the Agent; infrastructure/unknown actions
            # and provider errors retain the common conservative stop policy.
            if phase in {'submit', 'compare', 'draft', 'read', 'execute'} and outcome['status'] == 'error':
                if outcome.get('error_type') in {'ValueError', 'ToolError', 'GeneratedSourceError'}:
                    signature = content_hash([phase, state.get('tool_request'), outcome.get('error')])
                    if signature in state.get('tool_failures', []):
                        return self._finish(state, 'stopped', 'repeated_tool_failure', outcome['error'])
                    state.setdefault('tool_failures', []).append(signature)
                    self._tool_result(state, error=outcome)
                    state.update(phase='review' if phase == 'read' and state.get('read_return_phase') == 'review'
                                 else 'propose', last_error=outcome)
                    return state
            return super()._apply(state, action, outcome)
        handlers = {'propose': self._decision, 'draft': self._draft_result,
                    'execute': self._execution_result, 'submit': self._submission_result,
                    'compare': self._comparison_result}
        if phase in handlers:
            return handlers[phase](state, action, outcome['data'])
        state = super()._apply(state, action, outcome)
        if phase == 'read':
            self._tool_result(state, data=outcome['data'])
        return state

    @staticmethod
    def _tool_result(state, *, data=None, error=None):
        request = state.get('tool_request') or {}
        state['last_tool_result'] = {'tool': request.get('tool'),
                                    'status': 'error' if error else 'success', 'data': data, 'error': error}

    def _decision(self, state, action, data):
        # Validate here as well as at the provider boundary, so custom providers
        # cannot skip contract immutability or tool argument restrictions.
        if data['kind'] == 'needs_input':
            return self._finish(state, 'awaiting_feedback', 'needs_input', data['reason'])
        state['tool_request'] = data
        tool, args = data['tool'], data['arguments']
        state['last_error'] = None
        if tool in READ_TOOLS:
            return super()._apply(state, action, {'status': 'ok', 'data': {
                'kind': 'read', 'requests': [{'tool': tool, 'arguments': args}]}})
        if tool in {'create_draft', 'edit_draft'}:
            state.update(phase='draft', proposal={
                'kind': 'propose' if tool == 'create_draft' else 'edit', **args})
        elif tool == 'execute_pipeline':
            try:
                draft = DraftStore(state['run_dir']).get(args['draft_id'], args['revision'])
                if not draft['validation']['valid']:
                    raise ToolError('pipeline_invalid', 'Draft is invalid; repair before execution')
                from core.agent_loop import pipeline_fingerprint
                duplicate = next((item for item in state.get('experiment_records', [])
                    if item.get('scope') == state['input_scope'] and
                    pipeline_fingerprint(item.get('pipeline')) == pipeline_fingerprint(draft['pipeline'])), None)
                if duplicate:
                    self._tool_result(state, data={'experiment_id': duplicate['experiment_id'],
                        'reused': True, 'status': duplicate.get('status'),
                        'note': 'Already executed; inspect, compare or submit this experiment, or change the algorithm.'})
                    state['phase'] = 'propose'
                    return state
                if state['budget']['usage']['executions'] >= state['budget']['limits']['max_executions']:
                    raise ToolError('execution_budget_exhausted', 'No executions remain; inspect or submit existing experiments')
                state.update(current_draft=DraftStore.summary(draft), phase='execute')
            except (ValueError, ToolError) as exc:
                self._tool_result(state, error={'code': getattr(exc, 'code', 'invalid_draft'), 'message': str(exc)})
                state['phase'] = 'propose'
        elif tool == 'compare_candidates':
            state['phase'] = 'compare'
        elif tool == 'submit_experiment':
            state['phase'] = 'submit'
        return state

    def _draft_result(self, state, action, data):
        state.update(data)
        state['input_scope'] = experiment_scope(state['target_image_path'], context={**(state.get('previous_state') or {}), **state})
        summary = state['current_draft']
        catalog = {item['draft_id']: item for item in state.get('draft_catalog', [])}
        catalog[summary['draft_id']] = summary
        state.update(draft_catalog=list(catalog.values()), phase='propose')
        self._tool_result(state, data=summary)
        return state

    def _execution_result(self, state, action, data):
        state = super()._apply(state, action, {'status': 'ok', 'data': data})
        state['last_execution_iteration'] = action['iteration']
        receipts = state.setdefault('execution_receipts', {})
        for item in data.get('candidate_attempts', []):
            receipts[item['experiment_id']] = {'action': deepcopy(action),
                                             'draft': deepcopy(state['current_draft'])}
        from core.experiments.context import candidate_for_model
        self._tool_result(state, data=[candidate_for_model(item) for item in data.get('candidate_attempts', [])])
        if state.get('run_status') == 'running':
            state['phase'] = 'propose'
        return state

    def _validated_experiment(self, state, experiment_id, *, submission=False):
        self._validate_scope(state)
        item = next((item for item in state.get('experiment_records', [])
                     if item.get('experiment_id') == experiment_id), None)
        receipt = state.get('execution_receipts', {}).get(experiment_id)
        if item is None or receipt is None:
            raise ToolError('unknown_experiment', 'Experiment must have been executed in this run')
        if submission and item.get('acceptance_status') == 'rejected':
            raise ToolError('experiment_rejected', 'Independent review rejected this experiment; revise before submitting')
        if item.get('status') not in {'completed', 'selected_for_review'}:
            raise ToolError('experiment_not_ready', 'Failed experiments cannot be submitted or compared')
        if item.get('scope') != state['input_scope']:
            raise ToolError('experiment_scope_changed', 'Input or task contract has changed')
        outcome = ActionStore(state['run_dir']).load(receipt['action']['id'], receipt['action']['input_hash'])
        if not outcome or outcome.get('status') != 'ok':
            raise ToolError('missing_execution_receipt', 'No verified execution receipt')
        directory = Path(item['directory'])
        expected = Path(state['run_dir']) / f"iteration_{receipt['action']['iteration']}"
        if not directory.resolve().is_relative_to(expected.resolve()):
            raise ValueError('Experiment artifacts are outside their execution directory')
        from core.runtime_metadata import runtime_metadata
        if json.loads((expected / 'runtime_environment.json').read_text()) != runtime_metadata():
            raise ToolError('environment_changed', 'Runtime changed since execution')
        from core.sandbox import check_sandbox_available
        if check_sandbox_available() != {key: value for key, value in state['environment'].items() if key != 'runtime'}:
            raise ToolError('environment_changed', 'Sandbox changed since execution')
        verify_manifest(directory, state['input_scope']['input_sha256'])
        record = json.loads((directory / 'experiment.json').read_text())
        pipeline = json.loads((directory / 'pipeline.json').read_text())
        if (record.get('experiment_id') != experiment_id or record.get('execution_status') != 'completed'
                or record.get('scope') != state['input_scope'] or pipeline != item['pipeline']
                or record.get('algorithm_version') != content_hash(pipeline)):
            raise ToolError('experiment_modified', 'Executed experiment identity, scope or algorithm differs')
        saved = next((candidate for candidate in outcome['data'].get('candidate_attempts', [])
                      if candidate.get('experiment_id') == experiment_id), None)
        if saved is None or saved.get('pipeline') != pipeline or saved.get('quality') != item.get('quality'):
            raise ToolError('experiment_modified', 'Experiment differs from its durable execution receipt')
        return item, outcome['data'], receipt

    def _submission(self, state):
        experiment_id = state['tool_request']['arguments']['experiment_id']
        item, result, receipt = self._validated_experiment(state, experiment_id, submission=True)
        # Restore the submitted result, not the most recently edited draft. No CV rerun.
        keys = ('iteration', 'pipeline', 'annotated_image_path', 'predicted_mask_path', 'measurements',
                'quality_report', 'selected_candidate', 'selected_experiment_id', 'strategy',
                'contours_path', 'rendering', 'evaluation_report')
        return {**{key: result[key] for key in keys if key in result},
                'candidate_attempts': [deepcopy(item)], 'retained_action': receipt['action'],
                'submitted_draft': receipt['draft'],
                'submitted_experiment_id': experiment_id,
                'submission_reason': state['tool_request']['arguments']['reason']}

    def _submission_result(self, state, action, data):
        state.update(data, phase='review', review={})
        # Independent reviewer starts from submitted artifacts and static definitions.
        state['read_results'] = self._static_evidence(state)
        self._tool_result(state, data={'experiment_id': data['submitted_experiment_id'], 'status': 'submitted'})
        return state

    def _compare(self, state, action):
        from core.tools.experiments import ExperimentTools
        args = state['tool_request']['arguments']
        for experiment_id in args['experiment_ids']:
            self._validated_experiment(state, experiment_id)
        data, images = ExperimentTools.compare_results(
            state['target_image_path'], state['input_scope']['input_sha256'],
            {item['experiment_id']: item for item in state['experiment_records']},
            Path(state['run_dir']) / 'actions' / action['id'] / 'comparison.png', args)
        return {'tool': 'compare_candidates', 'arguments': args, 'data': data, 'images': images}

    def _comparison_result(self, state, action, data):
        state['read_results'] = [*state.get('read_results', []), data]
        state['phase'] = 'propose'
        self._tool_result(state, data=data['data'])
        return state
