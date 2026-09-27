"""One durable controller for proposal, execution, evidence and review."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from types import SimpleNamespace

from langgraph.graph import END, StateGraph

from agent_types import normalize_strategy
from core.agent_events import emit_node_start, emit_node_complete
from core.experiments.context import candidate_for_model
from core.experiments.drafts import DraftStore, atomic_json, content_hash, locate
from core.experiments.artifacts import experiment_scope
from core.orchestration_runtime import ActionStore, RunLimits, RunDeadline, initial_budget
from core.request_control import RequestControl, RequestCancelled, control
from core.runtime_logging import logger
from core.task_contract import establish_contract, apply_contract
from core.tools.contracts import TOOL_SPECS


READ_TOOLS = {'query_operators', 'load_skill', 'inspect_experiment', 'inspect_artifact', 'read_draft'}
MODEL_PHASES = {'propose', 'review'}


class InputScopeChangedError(ValueError):
    pass


def build_controller_graph(provider, checkpointer, algorithm_registry=None):
    from core.graph_nodes import wait_for_human
    runner = RunController(provider, algorithm_registry)
    graph = StateGraph(dict)
    graph.add_node('controller', runner.advance)
    graph.add_node('effect', runner.perform)
    graph.add_node('wait_for_human', wait_for_human)
    graph.set_entry_point('controller')
    graph.add_conditional_edges('controller', lambda state: (
        'effect' if state.get('pending_action') else
        'wait_for_human' if state.get('run_status') == 'awaiting_feedback' else END
    ))
    graph.add_edge('effect', 'controller')
    graph.add_edge('wait_for_human', END)
    return graph.compile(checkpointer=checkpointer)


def _trajectory(state, kind, outcome, duration=None):
    """Append one auditable timeline entry carrying the model's own narration."""
    narration = RunController._narration(kind, outcome)
    entry = {'node': kind, 'status': 'completed' if outcome.get('status') == 'ok' else 'failed',
             'duration_seconds': round(float(duration), 3) if duration is not None else 0.0,
             'details': {**({'narration': narration} if narration else {})}}
    return [*state.get('trajectory', []), entry]


class RunController:
    orchestration_version = 1
    autonomous_tools = False
    def __init__(self, provider, algorithm_registry=None):
        self.provider = provider
        self.algorithm_registry = algorithm_registry
        self.deadlines = {}

    def _remaining(self, state):
        key = (state['run_id'], state['budget']['deadline_at'])
        if key not in self.deadlines:
            self.deadlines[key] = RunDeadline(key[1])
        return self.deadlines[key].remaining()

    def advance(self, value):
        state = deepcopy(value)
        if not state.get('orchestration_version'):
            limits = RunLimits.from_env()
            budget = initial_budget(limits)
            budget['limits']['max_executions'] = min(
                limits.max_executions, max(1, int(state.get('max_auto_revisions', 2)) + 1))
            state.update(orchestration_version=self.orchestration_version, phase='prepare', run_status='running',
                         budget=budget, action_sequence=0, pending_action=None,
                         run_id=state.get('run_id') or state['graph_thread_id'],
                         run_started_at=state.get('run_started_at') or datetime.now(timezone.utc).isoformat(),
                         run_dir=str(Path(state['output_root']) / state['graph_thread_id']),
                         read_results=[], read_keys=[], failure_counts={}, state_version=0,
                         allow_contract_updates=True)
        action = state.get('pending_action')
        outcome = state.pop('action_outcome', None)
        if outcome is not None:
            state['pending_action'] = None
            state = self._apply(state, action, outcome)
        if state.get('run_status') != 'running':
            return self._publish(state)
        parent = control.get()
        if parent is not None and parent.cancelled.is_set():
            return self._publish(self._finish(state, 'cancelled', 'cancelled', '任务已取消，已保留完成的实验。'))
        if self._remaining(state) <= 0:
            return self._publish(self._finish(state, 'stopped', 'deadline_exceeded', '本次运行已达到总时限。'))
        # A checkpoint can resume immediately before the effect node.
        if state.get('pending_action'):
            return self._publish(state)
        phase = state['phase']
        usage, limits = state['budget']['usage'], state['budget']['limits']
        if phase in MODEL_PHASES:
            needed = 2 if phase == 'propose' else 1
            if usage['model_calls'] + needed > limits['max_model_calls']:
                return self._publish(self._finish(state, 'stopped', 'model_budget_exhausted', '模型调用额度已用完，保留已有结果。'))
            if phase == 'propose':
                if not self.autonomous_tools and usage['executions'] >= limits['max_executions']:
                    return self._publish(self._finish(state, 'stopped', 'execution_budget_exhausted', '本次实验次数已用完。'))
                reserve = min(30, limits['model_call_timeout_seconds']) + 20
                if self._remaining(state) <= reserve:
                    return self._publish(self._finish(state, 'stopped', 'finalization_reserved', '剩余时间不足以完成新实验及复查。'))
            usage['model_calls'] += 1
        elif phase == 'execute':
            if usage['executions'] >= limits['max_executions']:
                return self._publish(self._finish(state, 'stopped', 'execution_budget_exhausted', '本次实验次数已用完。'))
            usage['executions'] += 1
        state['action_sequence'] += 1
        identity = {'phase': phase, 'scope': state.get('input_scope'),
                    'draft': state.get('current_draft'), 'experiment': state.get('selected_experiment_id'),
                    'request': state.get('read_requests'), 'sequence': state['action_sequence'],
                    'tool_request': state.get('tool_request')}
        state['pending_action'] = {'id': f"action_{state['action_sequence']:04d}", 'kind': phase,
                                   'input_hash': content_hash(identity)}
        if phase == 'execute':
            from core.sandbox import action_container_name
            previous_iteration = state.get('iteration', (state.get('previous_state') or {}).get('iteration'))
            state['pending_action']['iteration'] = 0 if previous_iteration is None else int(previous_iteration) + 1
            state['pending_action']['container_name'] = action_container_name(state['run_id'], state['pending_action']['id'])
        return self._publish(state)

    def _publish(self, state):
        state['state_version'] = int(state.get('state_version', 0)) + 1
        return state

    def _project(self, state):
        # The preceding synchronous checkpoint is already committed at effect entry.
        context = state.get('memory_context') or {}
        if context.get('task_root') and state.get('task_id'):
            from core.task_store import TaskStore
            TaskStore(context['task_root']).save_run_state(state['task_id'], state)

    def perform(self, value):
        state = dict(value)
        self._project(state)
        action = state['pending_action']
        store = ActionStore(Path(state['run_dir']))
        outcome = store.load(action['id'], action['input_hash'])
        if outcome is not None:
            self._validate_scope(state)
            return {**state, 'action_outcome': outcome, 'trajectory': _trajectory(state, action['kind'], outcome)}
        if store.has_started(action['id'], action['input_hash']):
            saved = Path(state['run_dir']) / 'actions' / action['id'] / 'result.json'
            if saved.is_file():
                record = json.loads(saved.read_text(encoding='utf-8'))
                if record.get('input_hash') != action['input_hash']:
                    raise ValueError('Recovered action input does not match')
                if 'outcome' not in record or record.get('outcome_hash') != content_hash(record['outcome']):
                    raise ValueError('Recovered action outcome integrity check failed')
                outcome = record['outcome']
                self._validate_scope(state)
                store.complete(action, outcome)
                return {**state, 'action_outcome': outcome, 'trajectory': _trajectory(state, action['kind'], outcome)}
            if action['kind'] == 'execute':
                if action.get('container_name'):
                    from core.sandbox import cleanup_action_container
                    cleanup_action_container(action['container_name'])
                restored = self._recover_execution(state, action)
                if restored is not None:
                    outcome = {'status': 'ok', 'data': restored}
                    store.complete(action, outcome)
                    return {**state, 'action_outcome': outcome, 'trajectory': _trajectory(state, action['kind'], outcome)}
            return {**state, 'action_outcome': {'status': 'unknown', 'error': 'Action started without a durable result'}}
        store.prepare(action)
        parent = control.get()
        remaining = self._remaining(state)
        timeout = min(remaining, state['budget']['limits']['model_call_timeout_seconds']) if action['kind'] in MODEL_PHASES else remaining
        if action['kind'] == 'propose':
            timeout = min(timeout, remaining - min(30, state['budget']['limits']['model_call_timeout_seconds']) - 20)
        if parent is not None:
            try:
                timeout = min(timeout, parent.remaining())
            except RequestCancelled:
                timeout = 0
        duration = None
        if timeout <= 0:
            outcome = {'status': 'cancelled', 'error': 'Run deadline or cancellation reached'}
        else:
            from core.orchestration_runtime import LinkedCancellation
            request = RequestControl(timeout=timeout, **({'cancelled': LinkedCancellation(parent.cancelled)} if parent else {}))
            token = control.set(request)
            from core.sandbox import container_name
            container_token = container_name.set(action.get('container_name'))
            started = time.monotonic()
            try:
                emit_node_start(action['kind'], {'propose': 'Agent 决定下一步工具调用' if self.autonomous_tools else '生成或修改一个算法版本', 'execute': '执行并保存实验',
                                                'review': '独立复查实验', 'read': '读取所需证据',
                                                'prepare': '校验输入和执行环境', 'draft': '保存并校验草稿',
                                                'submit': '校验已提交的实验', 'compare': '比较已有实验'}[action['kind']])
                outcome = {'status': 'ok', 'data': self._perform_action(state, action)}
                self._validate_scope(state)
            except RequestCancelled as exc:
                outcome = {'status': 'cancelled' if parent and parent.cancelled.is_set() else 'timeout', 'error': str(exc)}
            except Exception as exc:
                logger.warning('Controller action failed action=%s kind=%s', action['id'], action['kind'], exc_info=True)
                outcome = {'status': 'error', 'error': str(exc), 'error_type': type(exc).__name__,
                           'code': getattr(exc, 'code', None)}
            finally:
                container_name.reset(container_token)
                control.reset(token)
            try:
                duration = time.monotonic() - started
                narration = self._narration(action['kind'], outcome)
                emit_node_complete(action['kind'], duration,
                                   {'action_id': action['id'], 'status': outcome['status'],
                                    **({'narration': narration} if narration else {}),
                                    **({'error': outcome['error']} if outcome.get('error') else {})})
            except RequestCancelled:
                logger.info('Controller action settled during cancellation action=%s status=%s', action['id'], outcome['status'])
        atomic_json(Path(state['run_dir']) / 'actions' / action['id'] / 'result.json',
                    {'input_hash': action['input_hash'], 'outcome': outcome,
                     'outcome_hash': content_hash(outcome)})
        store.complete(action, outcome)
        return {**state, 'action_outcome': outcome, 'trajectory': _trajectory(state, action['kind'], outcome, duration)}

    @staticmethod
    def _narration(kind, outcome):
        """One line of user-facing intent in the model's own words."""
        if outcome.get('status') != 'ok':
            return None
        data = outcome.get('data')
        if kind == 'propose' and isinstance(data, dict):
            if data.get('kind') == 'tool':
                args = data.get('arguments') or {}
                reason = args.get('change_reason') or args.get('reason') or ''
                return f"调用 {data['tool']}。{reason}"
            if data.get('kind') == 'read':
                tools = '、'.join(str(item.get('tool')) for item in data.get('requests') or []
                                 if isinstance(item, dict))
                return f"我先读取证据：{tools}。" if tools else '我先读取证据。'
            if data.get('kind') == 'needs_input':
                return str(data.get('reason') or '') or None
            return str(data.get('change_reason') or '') or None
        if kind == 'review' and isinstance(data, dict) and isinstance(data.get('review'), dict):
            return str(data['review'].get('reason') or '') or None
        if kind == 'execute' and isinstance(data, dict):
            attempts = data.get('candidate_attempts') or []
            selected = next((item for item in attempts if item.get('status') == 'selected_for_review'), None)
            if selected is None:
                return '实验执行完成，但没有产生可提交复查的结果。' if attempts else None
            quality = selected.get('quality') or {}
            if quality:
                return (f"实验完成：检出 {quality.get('component_count', '?')} 个连通域，"
                        f"覆盖率 {round(float(quality.get('coverage') or 0), 4)}。")
        return None

    def _perform_action(self, state, action):
        self._validate_scope(state)
        phase = action['kind']
        if phase == 'prepare':
            from core.graph_nodes import prepare_inputs
            from core.sandbox import check_sandbox_available
            prepared = prepare_inputs(state)
            from core.runtime_metadata import runtime_metadata
            environment = {**check_sandbox_available(), 'runtime': runtime_metadata()}
            memory = dict(state.get('memory_context') or {})
            matches = state.get('retrieved_algorithms')
            if not isinstance(matches, list):
                matches = []
                registry = self.algorithm_registry
                if registry is None and memory.get('task_root'):
                    from core.task_store import TaskStore
                    registry = TaskStore(memory['task_root']).algorithm_registry
                if registry is not None and not (state.get('previous_state') or {}).get('pipeline'):
                    # Before visual understanding, only the registry's text signal exists.
                    matches = [item for item in registry.search({'task_summary': state['description']}, limit=2, min_score=0.045)
                               if 'description_similarity' in item.get('match_reasons', [])]
            memory['procedural_memory'] = [{key: item.get(key) for key in (
                'algorithm_id', 'name', 'pipeline', 'strategy', 'source_task_id', 'match_reasons')}
                for item in matches[:2]]
            return {'reference_masks': prepared['reference_masks'], 'environment': environment,
                    'trajectory': prepared['trajectory'], 'retrieved_algorithms': matches[:2],
                    'memory_context': memory}
        if phase == 'propose':
            return self.provider.propose_action(state['target_image_path'], state['description'],
                context=self._context(state), reference_examples=state.get('reference_examples', []))
        if phase == 'review':
            return self.provider.review_action(state['target_image_path'], state['description'],
                state.get('candidate_attempts', []), acceptance_criteria=state.get('acceptance_criteria'),
                context=self._context(state), reference_examples=state.get('reference_examples', []))
        if phase == 'read':
            return self._read(state)
        if phase == 'draft':
            return self._draft(state, action)
        if phase == 'execute':
            from core.agent_loop import run_planned_agent
            draft = DraftStore(Path(state['run_dir'])).get(state['current_draft']['draft_id'], state['current_draft']['revision'])
            previous = {**(state.get('previous_state') or {}), **{key: state[key] for key in (
                'iteration', 'pipeline', 'selected_experiment_id', 'annotated_image_path',
                'human_feedback', 'include_mask_path', 'exclude_mask_path',
                'false_positive_mask_path', 'false_negative_mask_path', 'reference_examples') if key in state}}
            return run_planned_agent(state['target_image_path'], state['description'], state['understanding'],
                output_root=state['output_root'], run_dir=state['run_dir'], unit=state.get('unit', 'pixel'),
                retrieved_algorithms=state.get('retrieved_algorithms'),
                max_candidates=1, previous_state=previous, return_failure_state=True,
                ground_truth_mask_path=state.get('ground_truth_mask_path'), task_contract=state['task_contract'],
                planned_candidates=[{'name': f"experiment_{state['budget']['usage']['executions']}",
                    'pipeline': draft['pipeline'], 'hypothesis': draft.get('change_reason', ''),
                    'change_reason': draft.get('change_reason', ''), 'expected_change': draft.get('expected_change', '')}])
        raise ValueError(f'Unknown controller phase: {phase}')

    def _validate_scope(self, state):
        if state.get('input_scope') and state['input_scope'] != experiment_scope(
                state['target_image_path'], context={**(state.get('previous_state') or {}), **state}):
            raise InputScopeChangedError('输入图片、参考图或反馈在运行期间发生变化，请重新发起运行。')

    def _recover_execution(self, state, action):
        from core.experiments.runner import pipeline_fingerprint
        from core.experiments.delivery import verify_manifest
        from core.runtime_metadata import runtime_metadata
        directory = Path(state['run_dir']) / f"iteration_{action['iteration']}"
        try:
            scope = experiment_scope(state['target_image_path'], context={**(state.get('previous_state') or {}), **state})
            if scope != state['input_scope']:
                return None
            restored = json.loads((directory / 'graph_state.json').read_text(encoding='utf-8'))
            runtime = json.loads((directory / 'runtime_environment.json').read_text(encoding='utf-8'))
            if runtime != runtime_metadata():
                return None
            draft = DraftStore(Path(state['run_dir'])).get(state['current_draft']['draft_id'], state['current_draft']['revision'])
            attempts = restored.get('candidate_attempts') or []
            if not attempts:
                return None
            for attempt in attempts:
                candidate_dir = Path(attempt['directory'])
                if not candidate_dir.resolve().is_relative_to(directory.resolve()):
                    return None
                record = json.loads((candidate_dir / 'experiment.json').read_text(encoding='utf-8'))
                if record.get('scope') != state['input_scope'] or pipeline_fingerprint(record.get('pipeline')) != pipeline_fingerprint(draft['pipeline']):
                    return None
                if record.get('execution_status') != 'completed':
                    return None
                verify_manifest(candidate_dir, scope['input_sha256'])
                if pipeline_fingerprint(json.loads((candidate_dir / 'pipeline.json').read_text(encoding='utf-8'))) != pipeline_fingerprint(draft['pipeline']):
                    return None
                if json.loads((candidate_dir / 'quality_report.json').read_text(encoding='utf-8')) != attempt.get('quality'):
                    return None
            manifest = verify_manifest(directory, scope['input_sha256'])
            if restored.get('predicted_mask_path') and 'mask.png' not in manifest['files']:
                return None
            return restored
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _context(self, state):
        records = state.get('experiment_records') or []
        latest = next((item for item in records if item.get('experiment_id') == state.get('selected_experiment_id')), None)
        draft = None
        if state.get('current_draft'):
            draft = DraftStore(Path(state['run_dir'])).get(state['current_draft']['draft_id'])
        return {**(state.get('memory_context') or {}),
                'task_contract': state.get('task_contract') or {}, 'current_draft': draft,
                'latest_experiment': {**candidate_for_model(latest), 'directory': latest.get('directory')} if latest else None,
                'experiment_summaries': [candidate_for_model(item) for item in records],
                'previous_pipeline': (draft or {}).get('pipeline') or (state.get('previous_state') or {}).get('pipeline'),
                'review': state.get('review') or {}, 'read_results': state.get('read_results') or [],
                'last_error': state.get('last_error'), 'budget': state['budget'],
                'remaining_seconds': round(self._remaining(state), 2),
                'allow_contract_updates': state.get('allow_contract_updates', False),
                'reference_masks': state.get('reference_masks', []),
                'execution_feedback': {'attempts': records},
                'human_feedback': state.get('human_feedback') or (state.get('previous_state') or {}).get('human_feedback', {})}

    def _draft(self, state, action):
        from providers.vision import normalize_task_understanding
        from core.pipelines.dsl import pin_pipeline_operator_versions
        proposal = state['proposal']
        understanding = state.get('understanding') or proposal.get('understanding') or {}
        contract = state.get('task_contract') or {}
        if proposal.get('contract_updates'):
            if not state.get('allow_contract_updates'):
                raise ValueError('Requirements are fixed during automatic revision')
            contract = establish_contract(state, {'contract_updates': proposal['contract_updates']})
        if state.get('task_contract'):
            understanding = apply_contract(understanding, contract)
        drafts = DraftStore(Path(state['run_dir']))
        if proposal['kind'] == 'edit':
            summary = drafts.edit({key: value for key, value in proposal.items()
                                   if key not in {'kind', 'contract_updates', 'memory_updates'}})
        else:
            pipeline = proposal.get('pipeline')
            if not isinstance(pipeline, dict):
                raise ValueError('Proposal must contain one pipeline object')
            summary = drafts.create(pipeline, draft_id=action['id'], change_reason=proposal.get('change_reason', ''),
                                    expected_change=proposal.get('expected_change', ''))
        draft = drafts.get(summary['draft_id'])
        # Preserve invalid drafts for local repair before normalizing executable code.
        if summary['validation']['valid']:
            normalized = normalize_task_understanding({**understanding, 'candidate_pipelines': [{'pipeline': draft['pipeline']}]},
                                                      task_description=state['description'])
            contract = contract or establish_contract(state, normalized)
            normalized = apply_contract(normalized, contract)
            effective = pin_pipeline_operator_versions(normalized['candidate_pipelines'][0]['pipeline'])
            normalized['candidate_pipelines'][0]['pipeline'] = effective
            if effective != draft['pipeline']:
                summary = drafts.create(effective, draft_id=action['id'] + '_normalized',
                    change_reason=proposal.get('change_reason', ''), expected_change=proposal.get('expected_change', ''))
            understanding = normalized
        else:
            contract = contract or establish_contract(state, understanding)
        memory = dict(state.get('memory_context') or {})
        if state.get('allow_contract_updates') and memory.get('task_root') and memory.get('memory_source_id'):
            from core.task_store import TaskStore
            service = TaskStore(memory['task_root']).memory_service
            memory['task_memory'] = service.apply_updates(state['task_id'], state['description'],
                {**understanding, 'memory_updates': proposal.get('memory_updates') or understanding.get('memory_updates', [])},
                memory['memory_source_id'])
        return {'current_draft': summary, 'understanding': apply_contract(understanding, contract),
                'task_contract': contract, 'acceptance_criteria': contract['acceptance_criteria'],
                'memory_context': memory, 'allow_contract_updates': False}

    def _read(self, state):
        from core.tools.discovery import dispatch_discovery
        from core.tools.evidence import inspect_experiment
        results = []
        context = self._context(state)
        attempts = {item['experiment_id']: item for item in state.get('experiment_records', [])}
        requests = state['read_requests']
        if not isinstance(requests, list) or not 1 <= len(requests) <= 5:
            raise ValueError('Read action requires one to five requests')
        for request in requests:
            tool, args = request.get('tool'), request.get('arguments', {})
            if tool not in READ_TOOLS:
                raise ValueError('Only read-only evidence requests are allowed')
            TOOL_SPECS[tool].validate(args)
            if tool == 'inspect_experiment':
                data, images = inspect_experiment(args, attempts, state['target_image_path'], state['run_dir'])
            elif tool == 'read_draft':
                draft = DraftStore(Path(state['run_dir'])).get(args['draft_id'])
                data = draft['pipeline']
                if args.get('path'):
                    container, key = locate(data, args['path'])
                    data = container[key]
                data, images = {'draft_id': draft['draft_id'], 'revision': draft['revision'], 'value': data}, []
            else:
                data, images = dispatch_discovery(request, context, Path(__file__).resolve().parents[1] / 'workspace' / 'skills')
            results.append({**request, 'data': data, 'images': images})
        return results

    def _apply(self, state, action, outcome):
        phase = action['kind']
        if outcome['status'] == 'cancelled':
            return self._finish(state, 'cancelled', 'cancelled_or_timeout', outcome['error'])
        if outcome['status'] == 'timeout':
            if phase in MODEL_PHASES and self._remaining(state) > 0 and state.get('network_retries', 0) < 1:
                state.update(network_retries=1, phase=phase, last_error=outcome)
                return state
            reason = 'deadline_exceeded' if self._remaining(state) <= 0 else 'model_timeout'
            return self._finish(state, 'stopped', reason, outcome['error'])
        if outcome['status'] == 'unknown':
            return self._finish(state, 'interrupted', 'unknown_action_result',
                                '上次动作已经启动但结果未完整保存，已停止自动重试并保留产物。')
        if outcome['status'] == 'error':
            state['last_error'] = outcome
            signature = content_hash([phase, outcome.get('error_type'), outcome.get('error')])
            count = state['failure_counts'].get(signature, 0) + 1
            state['failure_counts'][signature] = count
            network_errors = {'APIConnectionError', 'APITimeoutError', 'RateLimitError', 'InternalServerError', 'TimeoutError'}
            if phase in MODEL_PHASES and outcome.get('error_type') in network_errors:
                retries = state.get('network_retries', 0)
                if retries < 1:
                    state.update(network_retries=retries + 1, phase=phase)
                    return state
            if phase == 'review':
                return self._finish(state, 'awaiting_feedback', 'review_unavailable', '复查服务未能完成，已保留实验，仍需人工验收。')
            if count < 2 and phase in {'propose', 'draft', 'read'} and outcome.get('error_type') in {'ValueError', 'ToolError', 'GeneratedSourceError'}:
                state['phase'] = state.get('read_return_phase', 'propose') if phase == 'read' else 'propose'
                return state
            return self._finish(state, 'failed', outcome.get('code') or 'action_failed', outcome['error'])
        data = outcome['data']
        if phase == 'prepare':
            state.update(data)
            state['input_scope'] = experiment_scope(state['target_image_path'], context={**(state.get('previous_state') or {}), **state})
            state['phase'] = 'propose'
        elif phase in MODEL_PHASES:
            if not isinstance(data, dict):
                return self._finish(state, 'failed', 'invalid_action', '模型必须返回结构化动作。')
            kind = data.get('kind')
            if kind == 'read':
                requests = data.get('requests')
                signature = content_hash([phase, state.get('selected_experiment_id'), state.get('current_draft'), requests])
                if signature in state['read_keys']:
                    return self._finish(state, 'stopped', 'repeated_evidence_request', '模型重复请求相同证据，已停止无进展的循环。')
                state['read_keys'].append(signature)
                state.update(read_requests=requests, read_return_phase=phase, phase='read')
            elif kind == 'needs_input':
                return self._finish(state, 'awaiting_feedback', 'needs_input', data.get('reason') or '需要补充任务信息。')
            elif phase == 'propose' and kind in {'propose', 'edit'}:
                state.update(proposal=data, phase='draft')
            elif phase == 'review' and kind == 'review':
                from core.graph_nodes import make_review_candidates_node
                review = data.get('review')
                if not isinstance(review, dict):
                    return self._finish(state, 'failed', 'invalid_review', '模型未返回有效复查结论。')
                adapter = SimpleNamespace(review_candidates=lambda *args, **kwargs: deepcopy(review))
                state = make_review_candidates_node(adapter)(state)
                state['read_results'] = self._static_evidence(state)
                if (state['review'].get('acceptance') or {}).get('overall_passed'):
                    return self._finish(state, 'awaiting_feedback', 'review_passed', state['review'].get('reason', '复查通过，等待人工确认。'))
                if state['review'].get('decision') != 'revise':
                    return self._finish(state, 'awaiting_feedback', 'review_inconclusive', state['review'].get('reason', '结果仍需人工验收。'))
                state['revision_count'] = int(state.get('revision_count', 0)) + 1
                state['phase'] = 'propose'
            else:
                return self._finish(state, 'failed', 'invalid_action', f'当前阶段不允许动作 {kind}')
        elif phase == 'read':
            reads = {content_hash([item['tool'], item.get('arguments')]): item for item in state.get('read_results', [])}
            reads.update({content_hash([item['tool'], item.get('arguments')]): item for item in data})
            state.update(read_results=list(reads.values()), phase=state['read_return_phase'])
        elif phase == 'draft':
            state.update(data)
            state['input_scope'] = experiment_scope(state['target_image_path'], context={**(state.get('previous_state') or {}), **state})
            validation = state['current_draft']['validation']
            if not validation['valid']:
                state.update(last_error=validation['error'], phase='propose')
            else:
                from core.experiments.runner import pipeline_fingerprint
                draft = DraftStore(Path(state['run_dir'])).get(state['current_draft']['draft_id'])
                repeated = any(item.get('scope') == state['input_scope'] and
                               pipeline_fingerprint(item.get('pipeline')) == pipeline_fingerprint(draft['pipeline'])
                               for item in state.get('experiment_records', []))
                if repeated:
                    return self._finish(state, 'stopped', 'duplicate_pipeline', '算法未发生有效变化，已保留原实验及其复查结论。')
                state.update(phase='execute', last_error=None, read_results=self._static_evidence(state))
        elif phase == 'execute':
            state.update(data)
            records = {item['experiment_id']: item for item in state.get('experiment_records', [])}
            records.update({item['experiment_id']: item for item in data.get('candidate_attempts', [])})
            state['experiment_records'] = list(records.values())
            state['experiment_history'] = [*state.get('experiment_history', []), {
                'iteration': state.get('iteration'), 'selected_candidate': state.get('selected_candidate'),
                'candidate_attempts': state.get('candidate_attempts', []), 'pipeline': state.get('pipeline')}]
            if any(item.get('status') == 'selected_for_review' for item in data.get('candidate_attempts', [])):
                state.update(phase='review', retained_action=action)
            else:
                failures = data.get('candidate_attempts', [])
                error = next((item for item in failures if item.get('failure_type') in {'sandbox_unavailable', 'cleanup_failed', 'io_error'}), None)
                if error:
                    return self._finish(state, 'failed', error['failure_type'], str(error.get('quality', {}).get('error', '实验基础设施不可用。')))
                state.update(phase='propose', last_error={'code': 'execution_failed', 'attempts': [candidate_for_model(item) for item in failures]})
        return state

    @staticmethod
    def _static_evidence(state):
        return [item for item in state.get('read_results', []) if item.get('tool') in {'query_operators', 'load_skill'}]

    def _finish(self, state, status, reason, message):
        retained = state.get('retained_action')
        usable = any(item.get('status') == 'selected_for_review' for item in state.get('candidate_attempts', []))
        if retained and not usable:
            outcome = ActionStore(Path(state['run_dir'])).load(retained['id'], retained['input_hash'])
            if outcome and outcome.get('status') == 'ok':
                for key in ('annotated_image_path', 'predicted_mask_path', 'measurements', 'quality_report',
                            'pipeline', 'selected_candidate', 'selected_experiment_id', 'strategy', 'contours_path',
                            'rendering', 'evaluation_report'):
                    if key in outcome['data']:
                        state[key] = outcome['data'][key]
                retained_ids = {item['experiment_id'] for item in outcome['data'].get('candidate_attempts', [])}
                state['candidate_attempts'] = [item for item in state.get('experiment_records', [])
                                              if item['experiment_id'] in retained_ids]
        state.setdefault('strategy', normalize_strategy({}))
        state.setdefault('measurements', {'summary': {'count': 0, 'unit': state.get('unit', 'pixel'), 'total_area': 0}, 'results': []})
        state.setdefault('quality_report', {})
        state.setdefault('understanding', {'task_summary': state['description'], 'recommended_strategy': state['strategy']})
        state.update(run_status=status, stop_reason=reason, phase='finished', pending_action=None,
                     agent_status='waiting_for_acceptance' if reason == 'review_passed' else
                     'waiting_for_feedback' if status == 'awaiting_feedback' else status,
                     status='ok' if reason == 'review_passed' else 'needs_human_review' if state.get('annotated_image_path') else status,
                     decision={'next_action': 'wait_for_acceptance' if status == 'awaiting_feedback' else 'stop',
                               'reason': message, 'automatic_review_passed': reason == 'review_passed'})
        state['conversation'] = [*state.get('conversation', []), {'role': 'assistant', 'content': message}]
        from core.graph_nodes import write_trajectory
        write_trajectory(state)
        return state
