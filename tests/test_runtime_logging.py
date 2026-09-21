import io
import concurrent.futures
import threading

import pytest

from core.runtime_logging import (
    _context, configure_logging, logged_operation, logger,
)


@pytest.fixture
def log_path(tmp_path):
    token = _context.set(None)
    old_handlers = logger.handlers[:]
    old_level = logger.level
    old_config = getattr(logger, '_liangce_config', None)
    logger.handlers = []
    logger._liangce_config = None
    path = configure_logging(tmp_path)
    yield path
    for handler in logger.handlers:
        handler.close()
    logger.handlers = old_handlers
    logger.setLevel(old_level)
    logger._liangce_config = old_config
    _context.reset(token)


def test_exception_is_redacted_in_file_and_console(log_path, monkeypatch):
    console = io.StringIO()
    logger.handlers[0].setStream(console)
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'private-key-value-123')

    @logged_operation('test_failure')
    def fail(task):
        raise RuntimeError('private-key-value-123 data:image/png;base64,AAAA api_key=anothersecret')

    with pytest.raises(RuntimeError):
        fail({'id': 'task_failure'})
    for text in (log_path.read_text(), console.getvalue()):
        assert 'Traceback' in text
        assert 'RuntimeError' in text
        assert 'task=task_failure' in text
        assert 'private-key-value-123' not in text
        assert 'anothersecret' not in text
        assert 'base64,AAAA' not in text
    assert _context.get() is None


def test_concurrent_task_context_and_nested_calls(log_path):
    barrier = threading.Barrier(2)

    @logged_operation('model')
    def nested():
        logger.info('nested marker')

    @logged_operation('web')
    def request(task):
        barrier.wait(timeout=5)
        nested()
        logger.info('finished %s', task['id'])
        return _context.get()['run_id']

    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        runs = list(pool.map(request, [{'id': 'task_a'}, {'id': 'task_b'}]))
    assert runs[0] != runs[1]
    text = log_path.read_text()
    for task, run in zip(['task_a', 'task_b'], runs):
        rows = [line for line in text.splitlines() if f'task={task} ' in line]
        assert rows
        assert all(f'run={run} ' in line for line in rows)
        assert any('stage=model' in line for line in rows)
        assert any('stage=web' in line and 'finished' in line for line in rows)


def test_rotation_and_idempotent_configuration(log_path):
    configure_logging(log_path.parent, max_bytes=400, backup_count=2)
    handlers = logger.handlers[:]
    configure_logging(log_path.parent, max_bytes=400, backup_count=2)
    assert logger.handlers == handlers
    for i in range(30):
        logger.info('rotation %s %s', i, 'x' * 70)
    assert (log_path.parent / 'agent.log.1').exists()
    assert (log_path.parent / 'agent.log.2').exists()
    assert not (log_path.parent / 'agent.log.3').exists()
    assert 'rotation 29' in log_path.read_text()


def test_events_and_debug_arguments(log_path):
    from core.agent_events import emit_event
    emit_event({'type': 'tool_call', 'tool': 'example', 'args': {'api_key': 'dontwrite'}})
    emit_event({'type': 'llm_chunk', 'content': 'chunk should not appear'})
    assert 'tool_call' in log_path.read_text()
    assert 'tool_args' not in log_path.read_text()
    configure_logging(log_path.parent, level='DEBUG')
    emit_event({'type': 'tool_call', 'tool': 'example', 'args': {'api_key': 'dontwrite', 'size': 7}})
    text = log_path.read_text()
    assert 'tool_args' in text and '"size": 7' in text
    assert 'dontwrite' not in text
    assert 'chunk should not appear' not in text


def test_request_size_log_contains_only_counts(log_path):
    from core.model_context import check_request_budget
    check_request_budget([{"role": "user", "content": [
        {"type": "text", "text": "private-business-prompt"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}])
    text = log_path.read_text()
    assert "llm_input_size" in text and "total_text_chars" in text
    assert "private-business-prompt" not in text and "base64" not in text


def test_tool_diagnostics_precede_long_hypotheses_and_source_is_not_logged(log_path):
    from core.agent_events import emit_tool_result
    emit_tool_result('execute_pipeline', {
        'status': 'error', 'data': {'hypothesis': 'long hypothesis' * 1000, 'draft_id': 'draft'},
        'error': {'code': 'source_syntax_error', 'message': 'Missing bracket at line 103'},
    }, success=False)
    emit_tool_result('read_draft', {
        'status': 'success', 'error': None, 'data': {'value': 'private-full-source-marker'},
    })
    text = log_path.read_text()
    assert 'source_syntax_error' in text and 'line 103' in text
    assert 'private-full-source-marker' not in text
    assert 'long hypothesis' not in text
