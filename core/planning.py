"""Bounded planning state machine, independent of provider and image encoding."""
from dataclasses import dataclass, field
import json
import os
from copy import deepcopy
from core.tools.contracts import TOOL_SPECS
from core.agent_events import emit_thinking


# Model preamble before tool calls, surfaced as a thinking row; clipped so
# the unified event log does not carry full model replies.
INTENT_DISPLAY_LIMIT = 1000


def _clip_text(text, limit):
    text = str(text)
    return text if len(text) <= limit else text[:limit] + "…（截断）"


@dataclass
class ModelReply:
    text: str = ""
    calls: list = field(default_factory=list)
    finish_reason: str | None = None


class PlanningSession:
    def __init__(self, dispatcher, skill_root, *, native=True, max_rounds=10, final_attempts=2, require_submission=False):
        self.dispatcher = dispatcher
        self.skill_root = skill_root
        self.native = native
        self.max_rounds = max_rounds
        self.final_attempts = final_attempts
        self.require_submission = require_submission
        self.final_instruction = '探索已结束。现在只允许输出包含一个最终方案的完整 JSON，不能调用工具。'
        if require_submission:
            dispatcher.require_task = True
            self.final_instruction = '探索已结束。只调用submit_experiment提交本会话已执行且可复查的实验ID及reason。不得重写算法或声称视觉验收已通过。'

    @classmethod
    def for_task(cls, target_image_path, description, context, output_root, skill_root, *, native=True, require_submission=False):
        from core.tools.experiments import ExperimentTools
        dispatcher = ExperimentTools(target_image_path, description, context, output_root=output_root)
        return cls(dispatcher, skill_root, native=native, require_submission=require_submission,
                   max_rounds=24 if require_submission else 10)

    def run(self, messages, complete, parse, normalize, image_content):
        from core.request_control import RequestControl, RequestCancelled, control
        parent = control.get()
        timeout = float(os.getenv('LIANGCE_PLANNING_TIMEOUT_SECONDS', '900'))
        if timeout <= 0:
            raise ValueError('LIANGCE_PLANNING_TIMEOUT_SECONDS must be positive')
        request = RequestControl(timeout=min(timeout, parent.remaining()) if parent else timeout,
                                 **({'cancelled': parent.cancelled} if parent else {}))
        token = control.set(request)
        try:
            if self.require_submission:
                self.dispatcher.preflight()
            return self._run(messages, complete, parse, normalize, image_content)
        except (Exception, RequestCancelled) as exc:
            if self.require_submission:
                from core.experiments.drafts import atomic_json
                try:
                    atomic_json(self.dispatcher.root / 'failure.json', {
                        'code': getattr(exc, 'code', 'cancelled' if isinstance(exc, RequestCancelled) else 'planning_failed'),
                        'message': str(exc), 'session': self.dispatcher.summary(),
                    })
                except OSError as storage_error:
                    exc.add_note(f'Could not persist planning failure: {storage_error}')
            raise
        finally:
            control.reset(token)

    def _run(self, messages, complete, parse, normalize, image_content):
        messages = list(messages)
        final_only = False
        final_used = 0
        invalid_final = 0
        unavailable_requests = 0
        last_error = None
        for turn in range(self.max_rounds):
            from core.request_control import check_cancelled
            check_cancelled()
            available = self.dispatcher.budget.available()
            final_only = final_only or not available or turn >= self.max_rounds - self.final_attempts
            if final_only:
                if self.require_submission and not any(
                    attempt.get('status') in {'completed', 'selected_for_review'}
                    for attempt in self.dispatcher.attempts.values()
                ):
                    last_error = last_error or {'code': 'no_executable_experiment', 'message': 'No completed experiment is available'}
                    break
                if final_used >= self.final_attempts:
                    break
                final_used += 1
                messages.append({'role': 'user', 'content': self.final_instruction})
            allowed = (['submit_experiment'] if self.require_submission and 'submit_experiment' in available else []) if final_only else available
            specs = [TOOL_SPECS[name].function_schema() for name in allowed]
            response = complete(messages, specs, final_only)
            check_cancelled()
            reply = response if isinstance(response, ModelReply) else ModelReply(text=response)
            calls = reply.calls
            intent = str(reply.text or "").strip()
            if calls and intent and not intent.startswith("{"):
                emit_thinking(_clip_text(intent, INTENT_DISPLAY_LIMIT), "tool_intent")
            if not calls:
                phase = 'invalid_json'
                try:
                    raw = parse(reply.text)
                    if not isinstance(raw, dict):
                        raise ValueError('Final output must be a JSON object')
                    phase = 'invalid_final_output'
                    if raw.get('type') == 'call_tool':
                        if self.native:
                            raise ValueError('Use native function calls, not call_tool text')
                        calls = [{'id': f'call_{turn}', 'name': raw.get('tool'), 'arguments': raw.get('arguments', {})}]
                    else:
                        if self.require_submission:
                            raise ValueError('Use save_task, create_draft/edit_draft, execute_pipeline, then submit_experiment. Final algorithm JSON is not accepted.')
                        result = normalize(raw)
                        if self.dispatcher.events:
                            result['tool_session'] = self.dispatcher.summary()
                        return result
                except (ValueError, TypeError) as exc:
                    invalid_final += 1
                    last_error = {'code': phase, 'message': str(exc), 'retryable': True}
                    from core.runtime_logging import logger
                    logger.warning('Planning protocol error code=%s chars=%s offset=%s finish_reason=%s',
                                   phase, len(reply.text or ''), getattr(exc, 'pos', None), reply.finish_reason)
                    messages.extend([{'role': 'assistant', 'content': reply.text},
                                     {'role': 'user', 'content': json.dumps({'error': last_error}, ensure_ascii=False)}])
                    if invalid_final >= self.final_attempts:
                        break
                    continue
            if self.native:
                messages.append({'role': 'assistant', 'content': reply.text or None, 'tool_calls': [
                    {'id': call['id'], 'type': 'function', 'function': {
                        'name': call['name'], 'arguments': call['arguments'] if isinstance(call['arguments'], str)
                        else json.dumps(call['arguments'])}} for call in calls]})
            else:
                messages.append({'role': 'assistant', 'content': reply.text})
            image_parts = []
            for call in calls:
                error = None
                if self.require_submission and self.dispatcher.submitted is not None:
                    error = {'code': 'session_submitted', 'message': 'Session already submitted', 'retryable': False}
                elif final_only and not (self.require_submission and call['name'] == 'submit_experiment'):
                    error = {'code': 'finalization_required', 'message': 'Only final output is allowed', 'retryable': False}
                elif call['name'] not in self.dispatcher.budget.available() and call['name'] in TOOL_SPECS:
                    unavailable_requests += 1
                    error = {'code': 'budget_exhausted', 'message': f"{call['name']} budget exhausted", 'retryable': False}
                if error:
                    result, images = {'call_id': call['id'], 'status': 'error', 'data': {}, 'error': error}, []
                else:
                    argument_error = None
                    try:
                        args = json.loads(call['arguments']) if isinstance(call['arguments'], str) else call['arguments']
                    except (ValueError, TypeError) as exc:
                        # Invalid JSON still reaches dispatch to consume the call budget.
                        args = None
                        argument_error = {'message': str(exc), 'line': getattr(exc, 'lineno', None),
                                          'column': getattr(exc, 'colno', None), 'offset': getattr(exc, 'pos', None),
                                          'finish_reason': reply.finish_reason}
                        from core.runtime_logging import logger
                        logger.warning('Tool argument JSON invalid tool=%s chars=%s offset=%s finish_reason=%s',
                                       call['name'], len(str(call['arguments'])), argument_error['offset'], reply.finish_reason)
                    result, images = self.dispatcher.dispatch({
                        'call_id': call['id'], 'tool': call['name'], 'arguments': args,
                        'argument_error': argument_error}, self.skill_root)
                last_error = result.get('error')
                if self.require_submission and last_error and last_error.get('code') in {'sandbox_unavailable', 'cleanup_failed'}:
                    from core.tools.contracts import ToolError
                    raise ToolError(last_error['code'], last_error['message'])
                if self.native:
                    messages.append({'role': 'tool', 'tool_call_id': call['id'],
                                     'content': json.dumps(result, ensure_ascii=False)})
                    image_parts.extend(image_content(path, f"工具 {call['id']} 返回的实验图像") for path in images)
                else:
                    messages.append({'role': 'user', 'content': [
                        {'type': 'text', 'text': json.dumps(result, ensure_ascii=False)},
                        *[image_content(path, '工具返回的实验图像') for path in images]]})
            if image_parts:
                # Compatible multimodal message; every tool call is resolved before images.
                messages.append({'role': 'user', 'content': image_parts})
            if self.require_submission and self.dispatcher.submitted is not None:
                submitted = self.dispatcher.submitted
                result = normalize(submitted)
                # The delivered source is the saved execution snapshot, never a model rewrite.
                result['candidate_pipelines'] = deepcopy(submitted['candidate_pipelines'])
                result['submitted_experiment_id'] = submitted['submitted_experiment_id']
                result['tool_session'] = self.dispatcher.summary()
                return result
            if unavailable_requests >= 2:
                final_only = True
        label = 'submit an executable experiment' if self.require_submission else 'produce final output'
        raise ValueError(f'Planning could not {label}: {json.dumps(last_error, ensure_ascii=False)}')
