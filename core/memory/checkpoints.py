"""Process-independent LangGraph checkpoints with lazy connection creation."""
import atexit
import os
from pathlib import Path
import sqlite3
from threading import Lock

from langgraph.checkpoint.sqlite import SqliteSaver

_lock = Lock()
_savers = {}


def get_checkpointer():
    default = Path(__file__).resolve().parents[2] / 'workspace' / 'checkpoints.sqlite3'
    path = str(Path(os.environ.get('LIANGCE_CHECKPOINT_PATH', default)).resolve())
    with _lock:
        key = (os.getpid(), path)
        if key not in _savers:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(path, check_same_thread=False, timeout=30)
            connection.execute('PRAGMA journal_mode=WAL')
            _savers[key] = SqliteSaver(connection)
        return _savers[key]


@atexit.register
def close_checkpointers():
    with _lock:
        for saver in _savers.values():
            saver.conn.close()
        _savers.clear()
