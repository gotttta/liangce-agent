"""Application logging with request-local context and redacted rotating output."""
import contextvars
import functools
import inspect
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import re
import threading
import time
from uuid import uuid4

_context = contextvars.ContextVar("agent_log_context", default=None)
_lock = threading.RLock()
logger = logging.getLogger("liangce")
logger.addHandler(logging.NullHandler())
logger.propagate = False


def current_context():
    return dict(_context.get() or {})


def bind_context(**fields):
    _context.set({**(_context.get() or {}), **fields})


def redact(value):
    text = str(value)
    for key, secret in os.environ.items():
        if any(word in key.upper() for word in ("KEY", "TOKEN", "PASSWORD", "SECRET")) and len(secret) >= 8:
            text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"data:image/[^\s\"']+", "[IMAGE REDACTED]", text, flags=re.I)
    text = re.sub(r"(?i)(bearer\s+)[\w.+=/\-]+", r"\1[REDACTED]", text)
    text = re.sub(r'''(?i)((?:api[_-]?key|authorization|password|secret|access[_-]?token)["']?\s*[:=]\s*["']?)[^\s,"'}]+''', r"\1[REDACTED]", text)
    text = re.sub(r"(?<![\w/])[A-Za-z0-9+/]{256,}={0,2}", "[BINARY REDACTED]", text)
    return text


class SafeFormatter(logging.Formatter):
    def format(self, record):
        # Context is resolved synchronously in the emitting thread, including traceback.
        context = _context.get() or {}
        record.task_id = context.get("task_id", "-")
        record.run_id = context.get("run_id", "-")
        record.stage = context.get("stage", "-")
        return redact(super().format(record))


def configure_logging(log_dir=None, level=None, max_bytes=None, backup_count=None):
    """Idempotent setup; only this application's handlers are configured."""
    directory = Path(log_dir or os.getenv("LIANGCE_LOG_DIR") or Path(__file__).resolve().parents[1] / "workspace/logs")
    level = str(level or os.getenv("LIANGCE_LOG_LEVEL", "INFO")).upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError("LIANGCE_LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR or CRITICAL")
    size = int(max_bytes if max_bytes is not None else os.getenv("LIANGCE_LOG_MAX_BYTES", "10485760"))
    backups = int(backup_count if backup_count is not None else os.getenv("LIANGCE_LOG_BACKUP_COUNT", "5"))
    if size <= 0 or backups < 1:
        raise ValueError("Log size and backup count must be positive")
    config = (str(directory.resolve()), level, size, backups)
    with _lock:
        if getattr(logger, "_liangce_config", None) == config:
            return directory / "agent.log"
        directory.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(directory / "agent.log", maxBytes=size, backupCount=backups, encoding="utf-8")
        formatter = SafeFormatter("%(asctime)s %(levelname)s task=%(task_id)s run=%(run_id)s stage=%(stage)s %(message)s")
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        for handler in (logging.StreamHandler(), file_handler):
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.propagate = False
        logger.setLevel(level)
        logger._liangce_config = config
    return directory / "agent.log"


def logged_operation(stage):
    """Wrap synchronous entry points; restore context even when they fail."""
    def decorate(function):
        signature = inspect.signature(function)
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            arguments = signature.bind(*args, **kwargs).arguments
            parent = _context.get() or {}
            task = arguments.get("task") or {}
            context = {**parent, "stage": stage}
            context.setdefault("run_id", uuid4().hex)
            if isinstance(task, dict) and task.get("id"):
                context["task_id"] = task["id"]
            token = _context.set(context)
            started = time.monotonic()
            metadata = {}
            if "self" in arguments and hasattr(arguments["self"], "model"):
                metadata["model"] = arguments["self"].model
            try:
                logger.info("%s started %s", stage, metadata)
                result = function(*args, **kwargs)
                details = {key: result[key] for key in ("run_dir", "agent_status", "selected_candidate", "graph_thread_id") if isinstance(result, dict) and key in result}
                logger.info("%s completed duration_seconds=%.3f %s", stage, time.monotonic() - started, details)
                return result
            except Exception:
                logger.exception("%s failed duration_seconds=%.3f", stage, time.monotonic() - started)
                raise
            finally:
                _context.reset(token)
        return wrapped
    return decorate


def log_event(event):
    event_type = event.get("type", "event")
    if event_type == "llm_chunk":
        return
    if event.get("node"):
        bind_context(stage=event["node"])
    # Full model replies and source code belong in artifacts, not runtime logs.
    payload = {key: value for key, value in event.items() if key not in {"content", "content_preview", "timestamp", "args"}}
    level = logging.ERROR if event_type == "error" or event.get("success") is False else logging.INFO
    logger.log(level, "%s", json.dumps(payload, ensure_ascii=False, default=str))
    if event_type == "tool_call" and logger.isEnabledFor(logging.DEBUG):
        logger.debug("tool_args %s", redact(json.dumps(event.get("args", {}), ensure_ascii=False, default=str))[:4000])
