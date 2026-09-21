import math
from types import SimpleNamespace

import pytest

from core.request_control import RequestCancelled, RequestControl, check_cancelled, control


@pytest.mark.parametrize('configured', [None, '', '0'])
def test_unlimited_request_survives_elapsed_time_and_can_be_cancelled(monkeypatch, configured):
    if configured is None:
        monkeypatch.delenv('LIANGCE_REQUEST_TIMEOUT_SECONDS', raising=False)
    else:
        monkeypatch.setenv('LIANGCE_REQUEST_TIMEOUT_SECONDS', configured)
    request = RequestControl.from_env()
    monkeypatch.setattr('core.request_control.time', SimpleNamespace(monotonic=lambda: request.started + 3600))
    token = control.set(request)
    try:
        check_cancelled()
        assert math.isinf(request.remaining())
        request.cancelled.set()
        with pytest.raises(RequestCancelled, match='任务已取消'):
            check_cancelled()
    finally:
        control.reset(token)


def test_explicit_deadline_remains_available(monkeypatch):
    monkeypatch.setenv('LIANGCE_REQUEST_TIMEOUT_SECONDS', '600')
    request = RequestControl.from_env()
    monkeypatch.setattr('core.request_control.time', SimpleNamespace(monotonic=lambda: request.started + 599))
    assert request.remaining() == 1
    monkeypatch.setattr('core.request_control.time', SimpleNamespace(monotonic=lambda: request.started + 600))
    with pytest.raises(RequestCancelled, match='超过总时限'):
        request.check()
    assert request.cancelled.is_set()


@pytest.mark.parametrize('configured', ['-1', 'nan', 'inf', 'invalid'])
def test_invalid_deadline_is_rejected(monkeypatch, configured):
    monkeypatch.setenv('LIANGCE_REQUEST_TIMEOUT_SECONDS', configured)
    with pytest.raises(ValueError):
        RequestControl.from_env()
