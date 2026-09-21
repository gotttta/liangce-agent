"""
Agent 事件系统 - 捕获和发送 Agent 执行过程中的详细信息到前端
"""
from core.runtime_logging import logger, log_event, current_context

import threading
import time
from typing import Any, Callable, Dict, List, Optional

# 全局事件监听器列表
_event_listeners: List[Callable[[Dict[str, Any]], None]] = []
_listeners_lock = threading.Lock()
_listener_scopes = {}


def register_event_listener(listener: Callable[[Dict[str, Any]], None], *, run_id=None, task_id=None):
    """注册一个事件监听器"""
    with _listeners_lock:
        if listener not in _event_listeners:
            _event_listeners.append(listener)
            context = current_context()
            _listener_scopes[listener] = (run_id or context.get('run_id'), task_id, threading.get_ident())


def unregister_event_listener(listener: Callable[[Dict[str, Any]], None]):
    """取消注册事件监听器"""
    with _listeners_lock:
        if listener in _event_listeners:
            _event_listeners.remove(listener)
            _listener_scopes.pop(listener, None)


def clear_event_listeners():
    """清空所有事件监听器"""
    with _listeners_lock:
        _event_listeners.clear()
        _listener_scopes.clear()


def emit_event(event: Dict[str, Any]):
    """发送事件到所有监听器"""
    from core.request_control import check_cancelled
    check_cancelled()
    context = current_context()
    event = {**event, 'task_id': context.get('task_id'), 'run_id': context.get('run_id')}
    log_event(event)
    with _listeners_lock:
        listeners = []
        for listener in _event_listeners:
            run, task, thread = _listener_scopes[listener]
            if ((run is not None and run != event.get('run_id'))
                    or (run is None and thread != threading.get_ident())
                    or (task is not None and task != event.get('task_id'))):
                continue
            listeners.append(listener)

    for listener in listeners:
        try:
            listener(event)
        except Exception as exc:
            logger.exception("Event listener failed")


# 便捷的事件发送函数

def emit_node_start(node_name: str, description: str):
    """节点开始执行"""
    emit_event({
        "type": "node_start",
        "node": node_name,
        "description": description,
        "timestamp": time.time(),
    })


def emit_node_complete(node_name: str, duration: float, metadata: Optional[Dict] = None):
    """节点执行完成"""
    emit_event({
        "type": "node_complete",
        "node": node_name,
        "duration": duration,
        "metadata": metadata or {},
        "timestamp": time.time(),
    })


def emit_thinking(message: str, context: Optional[str] = None):
    """Agent 思考过程"""
    emit_event({
        "type": "thinking",
        "message": message,
        "context": context,
        "timestamp": time.time(),
    })


def emit_tool_call(tool_name: str, args: Dict[str, Any]):
    """工具调用"""
    emit_event({
        "type": "tool_call",
        "tool": tool_name,
        "args": args,
        "timestamp": time.time(),
    })


def emit_tool_result(tool_name: str, result: Any, success: bool = True):
    """工具执行结果"""
    if isinstance(result, dict):
        # Diagnostics must precede long hypotheses/artifact lists in the UI.
        import json
        data = result.get('data') or {}
        summary = {'status': result.get('status'), 'error': result.get('error'),
                   **{key: data[key] for key in ('draft_id', 'revision', 'experiment_id') if key in data}}
        if not result.get('error'):
            # Source/large report contents stay in tool messages and artifacts.
            summary['data'] = {key: data[key] for key in (
                'remaining_executions', 'candidate_status', 'reused', 'acceptance_status', 'note',
            ) if key in data}
        preview = json.dumps(summary, ensure_ascii=False)[:2000]
    else:
        preview = str(result)[:2000] if result else None
    emit_event({
        "type": "tool_result",
        "tool": tool_name,
        "result": preview,
        "success": success,
        "timestamp": time.time(),
    })


def emit_llm_request(provider: str, model: str, message_count: int, has_images: bool = False):
    """LLM 请求"""
    emit_event({
        "type": "llm_request",
        "provider": provider,
        "model": model,
        "message_count": message_count,
        "has_images": has_images,
        "timestamp": time.time(),
    })


def emit_llm_response(provider: str, content_preview: str, usage: Optional[Dict] = None,
                      model: str = "", context_window: Optional[int] = None):
    """LLM 响应；usage 为 token 统计，供前端上下文用量指示器使用"""
    emit_event({
        "type": "llm_response",
        "provider": provider,
        "model": model,
        "content_preview": content_preview,
        "usage": usage,
        "context_window": context_window,
        "timestamp": time.time(),
    })


def emit_llm_chunk(content: str, provider: str = "", model: str = ""):
    """Emit one visible chunk from a streaming model response."""
    emit_event({
        "type": "llm_chunk",
        "provider": provider,
        "model": model,
        "content": content,
        "timestamp": time.time(),
    })


def emit_error(error_type: str, message: str, node: Optional[str] = None):
    """错误事件"""
    emit_event({
        "type": "error",
        "error_type": error_type,
        "message": message,
        "node": node,
        "timestamp": time.time(),
    })
