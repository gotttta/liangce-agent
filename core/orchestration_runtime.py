"""Durable action records and process-local safeguards for the run controller."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import re
import threading
import time

from core.experiments.drafts import atomic_json, content_hash


@dataclass(frozen=True)
class RunLimits:
    timeout_seconds: float = 600
    max_model_calls: int = 10
    max_executions: int = 3
    model_call_timeout_seconds: float = 120

    def __post_init__(self):
        for name in ('timeout_seconds', 'model_call_timeout_seconds'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('max_model_calls', 'max_executions'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')

    @classmethod
    def from_env(cls):
        values = {}
        for field, variable, convert in (
            ('timeout_seconds', 'LIANGCE_RUN_TIMEOUT_SECONDS', float),
            ('max_model_calls', 'LIANGCE_RUN_MAX_MODEL_CALLS', int),
            ('max_executions', 'LIANGCE_RUN_MAX_EXECUTIONS', int),
            ('model_call_timeout_seconds', 'LIANGCE_MODEL_CALL_TIMEOUT_SECONDS', float),
        ):
            raw = os.getenv(variable, '').strip()
            if raw:
                try:
                    values[field] = convert(raw)
                except ValueError as exc:
                    raise ValueError(f'{variable} has an invalid value') from exc
        return cls(**values)

    def as_dict(self):
        return asdict(self)


def initial_budget(limits, *, now=None):
    started_at = time.time() if now is None else now
    if not math.isfinite(started_at):
        raise ValueError('budget start time must be finite')
    return {
        'limits': limits.as_dict(),
        'usage': {'model_calls': 0, 'executions': 0},
        'started_at': started_at,
        'deadline_at': started_at + limits.timeout_seconds,
    }


class RunDeadline:
    """Restore an absolute deadline once, then resist wall-clock changes during a run."""
    def __init__(self, deadline_at, *, wall_clock=None, monotonic=None):
        if not math.isfinite(deadline_at):
            raise ValueError('run deadline must be finite')
        self._monotonic = monotonic or time.monotonic
        self._expires = self._monotonic() + max(0.0, deadline_at - (wall_clock or time.time)())

    def remaining(self):
        return max(0.0, self._expires - self._monotonic())


class ActionConflictError(ValueError):
    """An action ID was reused with different inputs or results."""


class TaskBusyError(RuntimeError):
    """Another runner already holds the task lock."""


class LinkedCancellation:
    """Propagate parent cancellation inward without propagating child timeouts outward."""
    def __init__(self, parent_event=None):
        self._parent = parent_event
        self._local = threading.Event()

    def is_set(self):
        return self._local.is_set() or (self._parent is not None and self._parent.is_set())

    def set(self):
        self._local.set()


_held_task_locks = threading.local()


def _safe_id(value, kind):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', value):
        raise ValueError(f'invalid {kind} ID')
    return value


class ActionStore:
    def __init__(self, root):
        self.root = Path(root) / 'actions'

    def _path(self, action_id, name):
        path = self.root / _safe_id(action_id, 'action') / name
        if not path.resolve().is_relative_to(self.root.resolve()):
            raise ValueError('action path is outside this store')
        return path

    def _read(self, action_id, input_hash, name):
        if not isinstance(input_hash, str) or not input_hash:
            raise ValueError('action input_hash must be a non-empty string')
        path = self._path(action_id, name)
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
        except FileNotFoundError:
            return None
        if record.get('action_id') != action_id or record.get('input_hash') != input_hash:
            raise ActionConflictError(f'action {action_id} identity does not match persisted record')
        return record

    def load(self, action_id, input_hash):
        record = self._read(action_id, input_hash, 'outcome.json')
        if record is None:
            return None
        if 'outcome' not in record or record.get('outcome_hash') != content_hash(record['outcome']):
            raise ActionConflictError(f'action {action_id} outcome integrity check failed')
        return deepcopy(record['outcome'])

    def has_started(self, action_id, input_hash):
        return self._read(action_id, input_hash, 'started.json') is not None

    def prepare(self, action):
        action_id, input_hash = action['id'], action['input_hash']
        record = self._read(action_id, input_hash, 'started.json')
        if record is not None:
            if record.get('action') != action:
                raise ActionConflictError(f'action {action_id} changed after starting')
            return
        self.load(action_id, input_hash)
        atomic_json(self._path(action_id, 'started.json'), {
            'action_id': action_id, 'input_hash': input_hash,
            'action': deepcopy(action), 'started_at': time.time(),
        })

    def complete(self, action, outcome):
        if outcome is None:
            raise ValueError('action outcome cannot be None')
        action_id, input_hash = action['id'], action['input_hash']
        existing = self.load(action_id, input_hash)
        if existing is not None:
            if content_hash(existing) != content_hash(outcome):
                raise ActionConflictError(f'action {action_id} already completed with another outcome')
            return
        self.prepare(action)
        atomic_json(self._path(action_id, 'outcome.json'), {
            'action_id': action_id, 'input_hash': input_hash,
            'outcome': deepcopy(outcome), 'outcome_hash': content_hash(outcome),
            'completed_at': time.time(),
        })


@contextmanager
def task_lock(task_root, task_id):
    """Hold the same lock inode until exit; never unlink advisory lock files."""
    directory = Path(task_root) / '.locks'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{_safe_id(task_id, "task")}.lock'
    key = (os.getpid(), str(path.resolve()))
    held = getattr(_held_task_locks, 'paths', None)
    if held is None:
        held = _held_task_locks.paths = {}
    if key in held:
        held[key] += 1
        try:
            yield
        finally:
            held[key] -= 1
        return
    with path.open('a+', encoding='utf-8') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise TaskBusyError(f'task {task_id} already has an active runner') from exc
        held[key] = 1
        try:
            yield
        finally:
            del held[key]
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
