"""Cooperative cancellation and an optional shared request deadline."""
from contextvars import ContextVar
from dataclasses import dataclass, field
import math
import os
import threading
import time


class RequestCancelled(BaseException):
    pass


@dataclass
class RequestControl:
    timeout: float | None = None
    cancelled: threading.Event = field(default_factory=threading.Event)
    started: float = field(default_factory=time.monotonic)

    def __post_init__(self):
        if self.timeout is not None:
            if not math.isfinite(self.timeout) or self.timeout < 0:
                raise ValueError('Request timeout must be finite and non-negative')
            if self.timeout == 0:
                self.timeout = None

    @classmethod
    def from_env(cls):
        # Unset, empty, or zero disables only the overall request deadline.
        raw = os.getenv('LIANGCE_REQUEST_TIMEOUT_SECONDS', '').strip()
        return cls(timeout=float(raw) if raw else None)

    def check(self):
        if self.cancelled.is_set():
            raise RequestCancelled('任务已取消。')
        if self.timeout is not None and time.monotonic() - self.started >= self.timeout:
            self.cancelled.set()
            raise RequestCancelled('任务超过总时限，已请求取消。')

    def remaining(self):
        self.check()
        if self.timeout is None:
            return math.inf
        return max(0.1, self.timeout - (time.monotonic() - self.started))


control = ContextVar('request_control', default=None)


def check_cancelled():
    current = control.get()
    if current is not None:
        current.check()
