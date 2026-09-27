"""Versioned algorithm drafts. Static validation never executes generated code."""
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile
from uuid import uuid4

from core.operators.generated import GeneratedSourceError
from core.pipelines.dsl import normalize_pipeline, pin_pipeline_operator_versions
from core.tools.contracts import ToolError


def content_hash(value):
    return sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=f'.{path.name}.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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
        self.root = Path(root) / 'drafts'
        self.current = {}

    def get(self, draft_id, revision=None):
        draft = self.load(draft_id)
        if revision is not None and revision != draft['revision']:
            raise ToolError('revision_conflict', 'draft version changed; read_draft before editing/executing',
                            retryable=True, details={'current_revision': draft['revision']})
        return deepcopy(draft)

    def _directory(self, draft_id):
        if not isinstance(draft_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', draft_id):
            raise ToolError('unknown_draft', 'invalid draft ID', retryable=True)
        directory = self.root / draft_id
        if not directory.resolve().is_relative_to(self.root.resolve()):
            raise ToolError('unknown_draft', 'draft is outside this store', retryable=True)
        return directory

    def load(self, draft_id, revision=None):
        """Read a persisted snapshot; get() additionally guards edits against stale revisions."""
        directory = self._directory(draft_id)
        if revision is None:
            revisions = [int(match.group(1)) for path in directory.glob('revision_*.json')
                         if (match := re.fullmatch(r'revision_([1-9][0-9]*)\.json', path.name))]
            if not revisions:
                raise ToolError('unknown_draft', 'draft is outside this store', retryable=True)
            revision = max(revisions)
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
            raise ToolError('revision_conflict', 'invalid draft revision', retryable=True)
        path = directory / f'revision_{revision}.json'
        try:
            draft = json.loads(path.read_text(encoding='utf-8'))
        except FileNotFoundError as exc:
            raise ToolError('unknown_draft', 'draft revision does not exist in this store', retryable=True) from exc
        if (draft.get('draft_id') != draft_id or draft.get('revision') != revision
                or draft.get('source_hash') != content_hash(draft.get('pipeline'))):
            raise ToolError('draft_modified', 'persisted draft identity or source hash changed')
        self.current[draft_id] = deepcopy(draft)
        return deepcopy(draft)

    def create(self, pipeline, draft_id=None, **metadata):
        draft_id = uuid4().hex if draft_id is None else draft_id
        directory = self._directory(draft_id)
        if (directory / 'revision_1.json').exists():
            existing = self.load(draft_id, 1)
            if existing['pipeline'] != pipeline or any(existing.get(key) != value for key, value in metadata.items()):
                raise ToolError('draft_conflict', 'draft ID already belongs to another creation')
            return self.summary(existing)
        return self._save({**metadata, 'draft_id': draft_id, 'revision': 1,
                           'pipeline': deepcopy(pipeline)})

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
