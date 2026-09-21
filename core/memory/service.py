"""Memory lifecycle shared by UI and graph callers.

Model observations remain hypotheses. Explicit, source-backed memory operations
are applied to task facts; unmentioned constraints survive each iteration.
"""
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from .store import MemoryStore
from .context import build_context


def image_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest() if path and Path(path).is_file() else None


class MemoryService:
    def __init__(self, task_store):
        self.tasks = task_store
        self.store = MemoryStore(task_store.root.parent / 'memory.sqlite3')

    def episode(self, task_id, event, data, source_id=None):
        source_id = source_id or uuid4().hex
        return self.store.put(task_id, 'episodic', source_id,
                              {'event': event, **data}, source_id=source_id)

    def set_fact(self, task_id, key, value, *, source_id, source_quote='', status='active', input_sha256=None):
        return self.store.put(task_id, 'semantic', key,
                              {'value': value, 'source_quote': source_quote, **({'input_sha256': input_sha256, 'coordinate_version': 'stored-pixels-v1'} if input_sha256 else {})},
                              source_id=source_id, status=status)

    def migrate(self, task_id):
        if self.store.list(task_id, 'semantic', history=True):
            return
        legacy = self.tasks.load_memory(task_id)
        if legacy.get('task_goal'):
            self.set_fact(task_id, 'current_goal', legacy['task_goal'],
                          source_id='legacy:memory.json', status='hypothesis')
        for key, value in (legacy.get('active_constraints') or {}).items():
            self.set_fact(task_id, 'constraint:' + key, value,
                          source_id='legacy:memory.json', status='hypothesis')

    def snapshot(self, task_id, input_sha256=None):
        self.migrate(task_id)
        records = self.store.list(task_id, 'semantic')
        facts = [r for r in records if r['status'] == 'active' and
                 (not r['data'].get('input_sha256') or r['data']['input_sha256'] == input_sha256)]
        goals = [r for r in records if r['key'] == 'current_goal']
        constraints = [r for r in facts if r['key'].startswith('constraint:')]
        legacy = self.tasks.load_memory(task_id)
        return {
            **legacy,
            'current_goal': goals[-1]['data']['value'] if goals else legacy.get('task_goal'),
            'active_constraints': {r['key'][11:]: r['data']['value'] for r in constraints},
            'constraint_records': constraints,
            'semantic_hypotheses': [r for r in records if r['status'] == 'hypothesis'],
            'recent_episodes': [r for r in self.store.list(task_id, 'episodic')
                                if r['data'].get('event') in {'user_request', 'iteration_result', 'result_accepted', 'task_exited'}][-6:],
        }

    def prepare(self, task_id, message, image_path, previous_state=None, history=None):
        self.migrate(task_id)
        previous = dict(previous_state or {})
        current_hash = image_hash(image_path)
        prior_hash = previous.get('input_sha256') or image_hash(previous.get('target_image_path'))
        if prior_hash and prior_hash != current_hash:
            # Pixel constraints, old result images and comparison baselines are image-local.
            previous = {}
        message_id = uuid4().hex
        self.episode(task_id, 'user_request', {'content': message, 'input_sha256': current_hash}, message_id)
        facts = self.store.list(task_id, 'semantic')
        if not any(r['key'] == 'current_goal' for r in facts):
            self.set_fact(task_id, 'current_goal', message, source_id=message_id, source_quote=message)
        context = {
            'task_id': task_id, 'task_root': str(self.tasks.root), 'memory_source_id': message_id,
            'input_sha256': current_hash,
            'original_task_goal': previous.get('original_task_goal') or self.snapshot(task_id).get('current_goal') or message,
            'task_memory': self.snapshot(task_id, current_hash),
            'conversation': self.tasks.load_messages(task_id) or history or [],
            'previous_pipeline': previous.get('pipeline'),
            'previous_quality': previous.get('quality_report'),
            'previous_result_image_path': previous.get('annotated_image_path'),
            'human_feedback': previous.get('human_feedback', {}),
        }
        from core.input_contract import evidence_matches
        task = self.tasks.load_task(task_id)
        if task.get('ground_truth') and not evidence_matches(task['ground_truth'], image_path):
            task['inactive_ground_truth'] = task['ground_truth']
            task['ground_truth'] = None
            self.tasks._write_json(self.tasks.task_dir(task_id) / 'task.json', task)
        if prior_hash and prior_hash != current_hash:
            context['task_memory'] = {key: val for key, val in context['task_memory'].items()
                if key not in {'quality_report', 'measurement_summary', 'structured_outputs',
                               'latest_result_image_path', 'latest_mask_path', 'recent_episodes', 'semantic_hypotheses'}}
            context['input_changed'] = True
        return previous or None, context

    def conversation_context(self, task_id, messages, summarize=None):
        from .conversation import roll_conversation
        rows = [r for r in self.store.list(task_id, 'episodic')
                if r['key'] == 'conversation_summary']
        cached = rows[-1]['data'] if rows else {}
        try:
            result = roll_conversation(messages, cached, summarize)
        except Exception:
            from core.runtime_logging import logger
            logger.warning('Conversation summary unavailable; retaining unsummarized history', exc_info=True)
            result = roll_conversation(messages, cached)
        if result['summary'] and (result['prefix_hash'] != cached.get('prefix_hash')
                                  or result['summary'] != cached.get('summary')):
            self.store.put(task_id, 'episodic', 'conversation_summary', {
                key: result[key] for key in ('summary', 'covered_messages', 'prefix_hash', 'recent_turns')
            }, source_id='conversation:' + result['prefix_hash'])
        return result

    def apply_updates(self, task_id, message, understanding, source_id):
        """Only source-backed operations from this user's message can change facts.

        The model maps natural language to keys. It cannot change scope, invent a
        source, or promote observations via target_constraints.
        """
        requests = [r for r in self.store.list(task_id, 'episodic') if r['source_id'] == source_id]
        request_hash = (requests[-1]['data'].get('input_sha256') if requests else None)
        for update in understanding.get('memory_updates') or []:
            if not isinstance(update, dict):
                continue
            op, key, quote = update.get('op'), update.get('key'), update.get('source_quote')
            if (op not in {'set', 'revoke'} or not isinstance(key, str)
                    or not (key == 'current_goal' or key.startswith('constraint:'))
                    or not isinstance(quote, str) or not quote.strip() or quote not in message):
                continue
            if key == 'current_goal' and op == 'revoke':
                continue
            value = update.get('value')
            if op == 'set' and (value is None or value == ''):
                continue
            pixel_local = update.get('scope') == 'image' or any(token in key.lower() for token in ('roi', 'bbox', 'coordinate', 'brush', 'pixel_region'))
            self.set_fact(task_id, key, value, source_id=source_id, source_quote=quote,
                          input_sha256=request_hash if pixel_local else None, status='revoked' if op == 'revoke' else 'active')
        for key, value in (understanding.get('target_constraints') or {}).items():
            self.set_fact(task_id, 'observation:' + key, value,
                          source_id=source_id, status='hypothesis')
        return self.snapshot(task_id, request_hash)

    def record_result(self, task_id, message, understanding, state, previous_state=None):
        source = state.get('graph_thread_id') or uuid4().hex
        from core.experiments.context import quality_for_model
        quality_summary = quality_for_model(state.get('quality_report') or {})
        if state.get('run_dir'):
            quality_summary['report_path'] = str(Path(state['run_dir']) / f"iteration_{state.get('iteration', 0)}" / 'quality_report.json')
        self.episode(task_id, 'iteration_result', {
            'iteration': state.get('iteration'), 'graph_thread_id': state.get('graph_thread_id'),
            'selected_candidate': state.get('selected_candidate'),
            'review': state.get('review'), 'quality_report': quality_summary,
            'result_image': state.get('annotated_image_path'),
            'experiments': [{'id': item.get('experiment_id'), 'directory': item.get('directory'),
                             'status': item.get('status'), 'failure_type': item.get('failure_type')}
                            for item in state.get('candidate_attempts', [])],
        }, 'result:' + source)
        existing = self.snapshot(task_id, state.get('input_sha256'))
        corrections = list(existing.get('corrections') or [])
        if previous_state and (previous_state.get('feedback_pixel_count') or
                               previous_state.get('agent_status') == 'waiting_for_feedback'):
            corrections.append({'iteration': state.get('iteration'), 'request': message})
        payload = {**existing, 'task_goal': existing.get('task_goal') or existing.get('current_goal') or message,
                   'latest_user_request': message, 'latest_iteration': state.get('iteration'),
                   'selected_pipeline': state.get('pipeline', {}), 'strategy': state.get('strategy', {}),
                   'pipeline_diff': state.get('pipeline_diff', {}), 'quality_report': quality_summary,
                   'measurement_summary': state.get('measurements', {}).get('summary', {}),
                   'structured_outputs': state.get('measurements', {}).get('structured_outputs', {}),
                   'latest_result_image_path': state.get('annotated_image_path'),
                   'latest_mask_path': state.get('predicted_mask_path'), 'corrections': corrections[-20:]}
        return self.tasks.save_memory(task_id, payload)

    def context_manifest(self, task_id, context):
        _, manifest = build_context(context)
        self.episode(task_id, 'context_built', manifest)
        return manifest

    def publish(self, task_id, algorithm):
        return self.store.put('algorithms', 'procedural', algorithm['id'], {
            'algorithm_id': algorithm['id'], 'path': algorithm['path'],
            'source_task_id': task_id, 'validation': 'user_accepted_on_source_input',
            'applicability': {k: algorithm.get(k) for k in
                              ('defect_type', 'measurement_type', 'background_type', 'polarity')},
        }, source_id='acceptance:' + task_id)
