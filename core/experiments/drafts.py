"""Versioned algorithm drafts. Static validation never executes generated code."""
from copy import deepcopy
from hashlib import sha256
import json
from uuid import uuid4

from core.operators.generated import GeneratedSourceError
from core.pipelines.dsl import normalize_pipeline, pin_pipeline_operator_versions
from core.tools.contracts import ToolError


def content_hash(value):
    return sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def locate(value, pointer):
    if not pointer.startswith('/'):
        raise ToolError('invalid_arguments', 'path must be a non-root JSON Pointer', retryable=True)
    parts = [part.replace('~1', '/').replace('~0', '~') for part in pointer[1:].split('/')]
    try:
        for part in parts[:-1]:
            value = value[int(part)] if isinstance(value, list) and part.isdecimal() else value[part]
        key = int(parts[-1]) if isinstance(value, list) and parts[-1].isdecimal() else parts[-1]
        value[key]  # Only existing fields can be edited.
        return value, key
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ToolError('patch_conflict', f'path does not exist: {pointer}; read_draft first', retryable=True) from exc


class DraftStore:
    def __init__(self, root):
        self.root = root / 'drafts'
        self.current = {}

    def get(self, draft_id, revision=None):
        if draft_id not in self.current:
            raise ToolError('unknown_draft', 'draft is outside this session', retryable=True)
        draft = self.current[draft_id]
        if revision is not None and revision != draft['revision']:
            raise ToolError('revision_conflict', 'draft version changed; read_draft before editing/executing',
                            retryable=True, details={'current_revision': draft['revision']})
        return deepcopy(draft)

    def create(self, pipeline, **metadata):
        return self._save({'draft_id': uuid4().hex, 'revision': 1,
                           'pipeline': deepcopy(pipeline), **metadata})

    def edit(self, args):
        draft = self.get(args['draft_id'], args['base_revision'])
        before = deepcopy(draft['pipeline'])
        for edit in args['edits']:
            container, key = locate(draft['pipeline'], edit['path'])
            current, old, new = container[key], edit['old'], edit['new']
            if isinstance(current, str):
                if not isinstance(old, str) or not isinstance(new, str) or not old or current.count(old) != 1:
                    raise ToolError('patch_conflict', 'old text must match exactly once; read_draft first', retryable=True)
                container[key] = current.replace(old, new, 1)
            elif content_hash(current) == content_hash(old):
                container[key] = deepcopy(new)
            else:
                raise ToolError('patch_conflict', 'old value does not match; read_draft first', retryable=True)
        if draft['pipeline'] == before:
            raise ToolError('patch_no_change', 'patch made no change; inspect the diagnosis before retrying', retryable=True)
        draft.update(revision=draft['revision'] + 1, change_reason=args['change_reason'])
        if 'expected_change' in args:
            draft['expected_change'] = args['expected_change']
        return self._save(draft)

    def _save(self, draft):
        try:
            pin_pipeline_operator_versions(normalize_pipeline(draft['pipeline']))
            validation = {'valid': True, 'error': None}
        except (ValueError, TypeError, KeyError) as exc:
            validation = {'valid': False, 'error': {
                'code': 'source_syntax_error' if isinstance(exc, GeneratedSourceError) else 'pipeline_invalid',
                'message': str(exc), 'retryable': True,
                'details': getattr(exc, 'diagnostic', {}),
            }}
        draft.update(validation=validation, source_hash=content_hash(draft['pipeline']))
        path = self.root / draft['draft_id'] / f"revision_{draft['revision']}.json"
        atomic_json(path, draft)
        self.current[draft['draft_id']] = deepcopy(draft)
        return self.summary(draft)

    @staticmethod
    def summary(draft):
        return {key: deepcopy(draft[key]) for key in (
            'draft_id', 'revision', 'source_hash', 'validation', 'change_reason', 'expected_change') if key in draft}
