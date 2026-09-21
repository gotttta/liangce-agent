"""Versioned SQLite records; large artifacts stay in their existing files."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .models import MemoryKind, STATUSES


class MemoryStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY, scope TEXT NOT NULL, kind TEXT NOT NULL,
                    key TEXT NOT NULL, version INTEGER NOT NULL, status TEXT NOT NULL,
                    source_id TEXT NOT NULL, data TEXT NOT NULL, created_at TEXT NOT NULL,
                    UNIQUE(scope, kind, key, version));
                CREATE INDEX IF NOT EXISTS memory_scope ON memories(scope, kind, status);
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def put(self, scope, kind, key, data, *, source_id, status='active'):
        kind = MemoryKind(kind).value
        if not scope or not key or not source_id or status not in STATUSES:
            raise ValueError('Memory requires scope, key, source and valid status')
        encoded = json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT * FROM memories WHERE scope=? AND kind=? AND key=? '
                             'ORDER BY version DESC LIMIT 1', (scope, kind, key)).fetchone()
            if old and old['data'] == encoded and old['status'] == status and old['source_id'] == source_id:
                return self.decode(old)
            version = old['version'] + 1 if old else 1
            if old and old['status'] in {'active', 'hypothesis'}:
                db.execute("UPDATE memories SET status='superseded' WHERE id=?", (old['id'],))
            record_id = uuid4().hex
            db.execute('INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?)', (
                record_id, scope, kind, key, version, status, source_id, encoded,
                datetime.now(timezone.utc).isoformat()))
            return self.decode(db.execute('SELECT * FROM memories WHERE id=?', (record_id,)).fetchone())

    def list(self, scope, kind=None, *, history=False):
        query, args = 'SELECT * FROM memories WHERE scope=?', [scope]
        if kind:
            query += ' AND kind=?'
            args.append(MemoryKind(kind).value)
        if not history:
            query += " AND status IN ('active', 'hypothesis')"
        query += ' ORDER BY created_at, version'
        with self.connect() as db:
            return [self.decode(row) for row in db.execute(query, args)]

    @staticmethod
    def decode(row):
        result = dict(row)
        result['data'] = json.loads(result['data'])
        return result
