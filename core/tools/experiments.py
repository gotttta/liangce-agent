"""Bounded experiment tools scoped to one model planning/revision request."""
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw

from core.agent_events import emit_tool_call, emit_tool_result
from core.tools.discovery import dispatch_discovery
from core.tools.contracts import TOOL_SPECS, ToolError, ToolResult
from core.tools.budget import ToolBudget
from core.skills import SkillRegistry
from core.experiments.runner import run_candidate, pipeline_fingerprint, _serializable_attempt
from core.experiments.artifacts import record_experiments, experiment_scope
from core.experiments.context import candidate_for_model
from core.tools.evidence import inspect_experiment
from core.preprocessing import load_grayscale
from core.runtime_metadata import runtime_metadata
from core.experiments.drafts import DraftStore, atomic_json, content_hash, locate


class ExperimentTools:
    def __init__(self, target_image_path, description, context=None, output_root='outputs',
                 max_executions=2, max_comparisons=2):
        self.target = Path(target_image_path).resolve()
        self.description = description
        self.context = deepcopy(context or {})
        self.root = Path(output_root).resolve() / 'agent_experiments' / uuid4().hex
        self.budget = ToolBudget()
        edit_limit = int(os.getenv('LIANGCE_MAX_DRAFT_EDITS', '8'))
        if edit_limit < 1:
            raise ValueError('LIANGCE_MAX_DRAFT_EDITS must be positive')
        self.budget.limits['editing'] = edit_limit
        self.max_executions = max_executions
        self.max_comparisons = max_comparisons
        self.skills = None
        self.attempts = {}
        self.cache = {}
        self.events = []
        self.drafts = DraftStore(self.root)
        self.understanding = None
        self.submitted = None
        self.require_task = False
        self.execution_environments = {}
        self.execution_sequence = 0
        self.draft_experiments = {}
        self.input_hash = sha256(self.target.read_bytes()).hexdigest()
        unresolved = len((self.context.get('review') or {}).get('observed_issues') or [])
        self.budget.limits['inspection'] = min(24, max(8, 2 * unresolved))
        for attempt in (self.context.get('execution_feedback') or {}).get('attempts', []):
            if attempt.get('experiment_id'):
                self.attempts[attempt['experiment_id']] = attempt

    @property
    def max_executions(self):
        return self.budget.limits['execution']

    @max_executions.setter
    def max_executions(self, value):
        self.budget.limits['execution'] = value

    @property
    def max_comparisons(self):
        return self.budget.limits['comparison']

    @max_comparisons.setter
    def max_comparisons(self, value):
        self.budget.limits['comparison'] = value

    @property
    def executions(self):
        return self.budget.used.get('execution', 0)

    @property
    def comparisons(self):
        return self.budget.used.get('comparison', 0)

    def dispatch(self, action, skill_root):
        tool = action.get('tool')
        args = action.get('arguments', {})
        reply = ToolResult(str(action.get('call_id') or uuid4().hex))
        emit_tool_call(str(tool), args)
        try:
            spec = TOOL_SPECS.get(tool)
            if spec is None:
                raise ToolError('unknown_tool', f'unknown tool: {tool}', retryable=True)
            # Even malformed known calls consume interaction budget.
            category = 'navigation' if tool == 'inspect_experiment' and isinstance(args, dict) and not args.get('selector') and 'region' not in args else spec.budget
            self.budget.consume(category)
            if action.get('argument_error'):
                diagnostic = action['argument_error']
                raise ToolError('invalid_json', diagnostic['message'], retryable=True, details=diagnostic)
            spec.validate(args)
            if tool == 'execute_pipeline':
                reply.data, reply.images = self.execute(args)
            elif tool == 'save_task':
                reply.data = self.save_task(args['understanding'])
            elif tool == 'create_draft':
                self._check_task()
                self._check_parent(args.get('parent_experiment_id'))
                reply.data = self.drafts.create(**args)
            elif tool == 'edit_draft':
                reply.data = self.drafts.edit(args)
            elif tool == 'read_draft':
                draft = self.drafts.get(args['draft_id'])
                value = draft['pipeline']
                if args.get('path'):
                    container, key = locate(value, args['path'])
                    value = container[key]
                reply.data = {**self.drafts.summary(draft), 'value': value}
            elif tool == 'submit_experiment':
                reply.data = self.submit(args)
            elif tool == 'compare_candidates':
                reply.data, reply.images = self.compare(args)
            elif tool == 'inspect_experiment':
                reply.data, reply.images = inspect_experiment(args, self.attempts, self.target, self.root)
            else:
                if self.skills is None:
                    self.skills = SkillRegistry(skill_root)
                reply.data, reply.images = dispatch_discovery(action, self.context, skill_root, registry=self.skills)
            validation = reply.data.get('validation')
            if validation and not validation['valid'] and tool != 'read_draft':
                error = validation['error']
                reply.error = ToolError(error['code'], error['message'], retryable=True,
                                        details=error.get('details'))
            elif reply.data.get('status') == 'error':
                reply.error = ToolError(reply.data.get('error') or 'execution_failed',
                                        str(reply.data.get('facts', {}).get('error') or 'Experiment failed'),
                                        retryable=reply.data.get('error') in {'pipeline_invalid', 'timeout'})
        except ToolError as exc:
            reply.error = exc
        except (ValueError, TypeError) as exc:
            reply.error = ToolError('invalid_arguments', str(exc), retryable=True)
        except FileNotFoundError as exc:
            reply.error = ToolError('artifact_missing', str(exc))
        except TimeoutError as exc:
            reply.error = ToolError('timeout', str(exc), retryable=True)
        except MemoryError as exc:
            reply.error = ToolError('resource_limit', str(exc))
        except OSError as exc:
            reply.error = ToolError('io_error', str(exc), retryable=True)
        except Exception as exc:
            from core.sandbox import SandboxExecutionError
            if not isinstance(exc, SandboxExecutionError):
                raise
            reply.error = ToolError(exc.code, str(exc))
        result = reply.as_dict()
        result['budget'] = self.budget.snapshot()
        result['available_tools'] = self.budget.available()
        emit_tool_result(str(tool), result, success=reply.error is None)
        self.events.append({'tool': tool, 'status': result['status'], 'error': result['error']})
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / 'session.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(self.summary(), ensure_ascii=False, indent=2), encoding='utf-8')
        temp.replace(path)
        return result, reply.images

    def summary(self):
        return {'directory': str(self.root), 'executions': self.executions,
                'comparisons': self.comparisons, 'events': self.events,
                'experiment_ids': list(self.attempts), 'budget': self.budget.snapshot(),
                'drafts': [self.drafts.summary(draft) for draft in self.drafts.current.values()]}

    def preflight(self):
        from core.sandbox import check_sandbox_available
        return check_sandbox_available()

    def _check_task(self):
        if self.require_task and self.understanding is None:
            raise ToolError('task_required', 'Call save_task with understanding and acceptance criteria first', retryable=True)

    def _check_parent(self, parent):
        if parent is not None and (not isinstance(parent, str) or parent not in self.attempts):
            raise ToolError('invalid_arguments', 'parent experiment is outside this session', retryable=True)

    def save_task(self, raw):
        from providers.vision import normalize_task_understanding
        from core.task_contract import establish_contract, apply_contract
        if 'candidate_pipelines' in raw or 'candidate_plans' in raw:
            raise ToolError('invalid_arguments', 'save_task must not contain algorithms; use create_draft', retryable=True)
        if not isinstance(raw.get('task_summary'), str) or not raw['task_summary'].strip() or not isinstance(raw.get('acceptance_criteria'), dict):
            raise ToolError('invalid_arguments', 'task_summary and acceptance_criteria are required', retryable=True)
        normalized = normalize_task_understanding(raw, task_description=self.description)
        contract = establish_contract({**self.context, 'description': self.description,
                                       'input_sha256': self.input_hash}, normalized)
        normalized = apply_contract(normalized, contract)
        if self.understanding is not None and normalized != self.understanding:
            raise ToolError('task_locked', 'Task criteria are fixed for this session; edit the algorithm instead')
        atomic_json(self.root / 'task.json', {'understanding': normalized, 'task_contract': contract})
        self.understanding = normalized
        self.context['task_contract'] = contract
        return {'task_summary': normalized['task_summary'], 'task_contract': contract}

    def submit(self, args):
        self._check_task()
        experiment_id = args['experiment_id']
        attempt = self.attempts.get(experiment_id)
        if attempt is None:
            raise ToolError('unknown_experiment', 'experiment is outside this session', retryable=True)
        directory = Path(attempt['directory'])
        record = json.loads((directory / 'experiment.json').read_text(encoding='utf-8'))
        pipeline = json.loads((directory / 'pipeline.json').read_text(encoding='utf-8'))
        if (record.get('experiment_id') != experiment_id or record.get('execution_status') != 'completed'
                or record.get('status') not in {'selected_for_review', 'completed'}):
            raise ToolError('experiment_not_ready', 'Only an executable, reviewable experiment can be submitted', retryable=True)
        if record.get('acceptance_status') == 'rejected':
            raise ToolError('experiment_rejected', 'This experiment was rejected; revise and execute again', retryable=True)
        if record.get('input_sha256') != sha256(self.target.read_bytes()).hexdigest() or record.get('scope') != experiment_scope(self.target, context=self.context):
            raise ToolError('experiment_scope_changed', 'Input or task/feedback requirements changed; execute again', retryable=True)
        if (content_hash(pipeline) != record.get('algorithm_version') or pipeline != attempt.get('pipeline')):
            raise ToolError('experiment_modified', 'Saved pipeline differs from the executed version; execute again')
        environment = self.execution_environments.get(experiment_id)
        if environment is None or environment != {**self.preflight(), 'runtime': runtime_metadata()}:
            raise ToolError('environment_changed', 'Execution environment changed or is not recorded in this session; execute again', retryable=True)
        candidate = {key: deepcopy(record.get(key, '')) for key in
                     ('name', 'hypothesis', 'change_reason', 'expected_change')}
        candidate['pipeline'] = pipeline
        # The workflow's execute node replays this persisted execution instead
        # of running the identical pipeline in the sandbox a second time.
        candidate['reused_execution'] = {
            'experiment_id': experiment_id,
            'directory': str(directory),
        }
        submitted = {**deepcopy(self.understanding or {}), 'candidate_pipelines': [candidate],
                     'submitted_experiment_id': experiment_id, 'submission_reason': args['reason']}
        atomic_json(self.root / 'submission.json', {
            'experiment_id': experiment_id, 'algorithm_version': record['algorithm_version'],
            'reason': args['reason'], 'acceptance_status': 'pending',
        })
        self.submitted = submitted
        return {'experiment_id': experiment_id, 'algorithm_version': record['algorithm_version'],
                'acceptance_status': 'pending', 'note': 'Formal execution and independent visual review are still required.'}

    def execute(self, args):
        from core.pipelines.dsl import normalize_pipeline, pin_pipeline_operator_versions

        self._check_task()
        if set(args) - {'pipeline', 'draft_id', 'revision', 'hypothesis', 'parent_experiment_id', 'change_reason', 'expected_change'}:
            raise ValueError('execute_pipeline received unknown experiment fields')
        if ('pipeline' in args) == ('draft_id' in args):
            raise ValueError('provide either draft_id with revision, or pipeline')
        if 'draft_id' in args:
            if 'revision' not in args:
                raise ValueError('revision is required with draft_id')
            draft = self.drafts.get(args['draft_id'], args['revision'])
        else:
            if not isinstance(args['pipeline'], dict) or 'revision' in args:
                raise ValueError('pipeline must be an object without a revision argument')
            self.budget.consume('editing')
            created = self.drafts.create(args['pipeline'], change_reason=args.get('change_reason') or args.get('hypothesis', ''),
                                         expected_change=args.get('expected_change', ''),
                                         parent_experiment_id=args.get('parent_experiment_id'))
            draft = self.drafts.get(created['draft_id'])
        if not draft['validation']['valid']:
            return self.drafts.summary(draft), []
        pipeline = draft['pipeline']
        hypothesis = args.get('hypothesis', '')
        if not isinstance(hypothesis, str):
            raise ValueError('hypothesis must be a string')
        default_parent = self.context.get('selected_experiment_id')
        parent = (args.get('parent_experiment_id') or self.draft_experiments.get(draft['draft_id'])
                  or draft.get('parent_experiment_id') or (default_parent if default_parent in self.attempts else None))
        self._check_parent(parent)
        # Cache reuse requires identical source, constraints, operator versions and environment.
        if self.understanding is not None:
            from providers.vision import normalize_task_understanding
            pipeline = normalize_task_understanding({**self.understanding, 'candidate_pipelines': [{'pipeline': pipeline}]},
                                                   task_description=self.description)['candidate_pipelines'][0]['pipeline']
        pinned = pin_pipeline_operator_versions(normalize_pipeline(pipeline))
        environment = {**self.preflight(), 'runtime': runtime_metadata()}
        key = pipeline_fingerprint(pinned) + json.dumps(pinned.get('operator_versions', {}), sort_keys=True)
        key += content_hash(environment)
        scope = experiment_scope(self.target, context=self.context)
        key += json.dumps(scope, sort_keys=True)
        if key in self.cache:
            result, images = self.cache[key]
            return {**result, 'reused': True}, images
        self.budget.consume('execution')  # Only validated, environment-ready executions count.
        self.execution_sequence += 1
        previous = deepcopy(self.context)
        previous['selected_experiment_id'] = parent
        candidate = {'name': f'experiment_{self.executions}', 'pipeline': pinned,
                     'hypothesis': hypothesis or draft.get('change_reason', ''), 'source': {
                         'type': 'agent_tool', 'draft_id': draft['draft_id'], 'revision': draft['revision'],
                         'source_hash': draft['source_hash']}}
        iteration = int(previous.get('iteration', -1)) + 1
        iteration_dir = self.root / f'execution_{self.execution_sequence}' / f'iteration_{iteration}'
        iteration_dir.mkdir(parents=True)
        atomic_json(iteration_dir / 'runtime_environment.json', environment)
        from core.agent_loop import _load_ground_truth_mask
        contract = self.context.get('task_contract') or self.context
        image = load_grayscale(self.target)
        attempt = run_candidate(candidate, self.target, image,
                                iteration_dir / 'candidate_0', previous_state=previous,
                                rendering=contract.get('rendering'), target_constraints=contract.get('target_constraints'),
                                ground_truth_mask=_load_ground_truth_mask(self.context.get('ground_truth_mask_path'), image.shape),
                                unit=contract.get('unit', 'pixel'))
        attempt.update(change_reason=args.get('change_reason') or draft.get('change_reason') or hypothesis,
                       expected_change=args.get('expected_change') or draft.get('expected_change', ''))
        if attempt.get('failure_type') == 'sandbox_unavailable':
            self.budget.used['execution'] -= 1
        record_experiments([attempt], self.target, previous, iteration, scope=scope)
        attempt = _serializable_attempt(attempt)
        experiment_id = attempt['experiment_id']
        self.attempts[experiment_id] = attempt
        self.execution_environments[experiment_id] = environment
        self.draft_experiments[draft['draft_id']] = experiment_id
        feedback = self.context.setdefault('execution_feedback', {})
        feedback.setdefault('attempts', []).append(attempt)
        directory = Path(attempt['directory'])
        overlay = directory / 'result_annotation.png'
        result = {**candidate_for_model(attempt),
                  'status': 'error' if attempt['status'] == 'failed' else 'success',
                  'experiment_id': experiment_id, 'candidate_status': attempt['status'],
                  'hypothesis': hypothesis,
                  'artifacts': attempt.get('artifacts', []), 'reused': False,
                  'remaining_executions': self.max_executions - self.executions,
                  'error': attempt.get('failure_type'),
                  'draft_id': draft['draft_id'], 'revision': draft['revision'],
                  'note': 'Execution success is not visual acceptance. Submit experiment_id for independent workflow review.'}
        images = [str(overlay)] if overlay.exists() else []
        if attempt.get('failure_type') not in {'timeout', 'worker_terminated', 'io_error', 'sandbox_unavailable', 'cleanup_failed'}:
            self.cache[key] = result, images
        return result, images

    def compare(self, args):
        if 'experiment_ids' not in args or set(args) - {'experiment_ids', 'reason'}:
            raise ValueError('compare_candidates requires experiment_ids and optional reason')
        ids = args.get('experiment_ids')
        if (not isinstance(ids, list) or not 2 <= len(ids) <= 3
                or not all(isinstance(item, str) for item in ids) or len(set(ids)) != len(ids)):
            raise ValueError('provide 2 to 3 distinct experiment IDs')
        if any(item not in self.attempts for item in ids):
            raise ValueError('experiment is outside this session')
        self.budget.consume('comparison')
        return self.compare_results(self.target, self.input_hash, self.attempts,
                                    self.root / f'comparison_{self.comparisons}.png', args)

    @staticmethod
    def compare_results(target, input_hash, attempts, output_path, args):
        """Deterministic comparison, with budgeting owned by the caller's run."""
        ids = args.get('experiment_ids')
        if (not isinstance(ids, list) or not 2 <= len(ids) <= 3
                or not all(isinstance(item, str) for item in ids) or len(set(ids)) != len(ids)):
            raise ValueError('provide 2 to 3 distinct experiment IDs')
        if any(item not in attempts for item in ids):
            raise ValueError('experiment is outside this session')
        masks, overlays, summaries = [], [], []
        comparison_scope = None
        with Image.open(target) as source:
            shape = (source.height, source.width)
        for experiment_id in ids:
            attempt = attempts[experiment_id]
            directory = Path(attempt['directory'])
            record = json.loads((directory / 'experiment.json').read_text(encoding='utf-8'))
            if record.get('input_sha256') != input_hash:
                raise ValueError('cannot compare experiments from different input images')
            if comparison_scope is not None and record.get('scope') != comparison_scope:
                raise ValueError('cannot compare experiments with different task or feedback constraints')
            comparison_scope = record.get('scope')
            if record.get('execution_status') != 'completed':
                raise ValueError('cannot compare a failed experiment')
            mask = None
            if (directory / 'mask.png').exists():
                with Image.open(directory / 'mask.png') as image:
                    mask = np.asarray(image.convert('L')) > 0
            if mask is not None and mask.shape != shape:
                raise ValueError('candidate mask does not match source coordinates')
            masks.append(mask)
            with Image.open(directory / 'result_annotation.png') as image:
                if image.size != (shape[1], shape[0]):
                    raise ValueError('candidate overlay does not match source coordinates')
                preview = image.convert('RGB')
                preview.thumbnail((640, 640))
                overlays.append(preview)
            summaries.append({**candidate_for_model(attempt),
                              'feedback_constraints': record.get('feedback_constraints', {})})
        differences = []
        for index in range(1, len(masks)):
            if masks[0] is None or masks[index] is None:
                differences.append({"baseline": ids[0], "candidate": ids[index], "pixel_comparison": "not_applicable"})
                continue
            differences.append({'baseline': ids[0], 'candidate': ids[index],
                                'added_pixels': int(np.count_nonzero(masks[index] & ~masks[0])),
                                'removed_pixels': int(np.count_nonzero(masks[0] & ~masks[index])),
                                'changed_pixels': int(np.count_nonzero(masks[index] ^ masks[0]))})
        width, height = overlays[0].size
        sheet = Image.new('RGB', (width * len(ids), height + 32), 'white')
        draw = ImageDraw.Draw(sheet)
        for index, (preview, experiment_id) in enumerate(zip(overlays, ids)):
            sheet.paste(preview, (index * width, 32))
            draw.text((index * width + 5, 8), f'{index + 1}: {experiment_id}', fill='black')
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(path)
        result = {'status': 'success', 'candidates': summaries, 'differences': differences,
                  'reason': args.get('reason') or 'Explicit comparison of selected experiment IDs',
                  'preview_path': str(path), 'note': 'Pixel differences are not accuracy scores. Check constraints and visually review all candidates.'}
        path.with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        return result, [str(path)]
