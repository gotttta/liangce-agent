"""
Agent 事件系统 - 捕获和发送 Agent 执行过程中的详细信息到前端
"""
import threading
import time
from typing import Any, Callable, Dict, List, Optional

# 全局事件监听器列表
_event_listeners: List[Callable[[Dict[str, Any]], None]] = []
_listeners_lock = threading.Lock()


def register_event_listener(listener: Callable[[Dict[str, Any]], None]):
    """注册一个事件监听器"""
    with _listeners_lock:
        if listener not in _event_listeners:
            _event_listeners.append(listener)


def unregister_event_listener(listener: Callable[[Dict[str, Any]], None]):
    """取消注册事件监听器"""
    with _listeners_lock:
        if listener in _event_listeners:
            _event_listeners.remove(listener)


def clear_event_listeners():
    """清空所有事件监听器"""
    with _listeners_lock:
        _event_listeners.clear()


def emit_event(event: Dict[str, Any]):
    """发送事件到所有监听器"""
    with _listeners_lock:
        listeners = _event_listeners.copy()

    for listener in listeners:
        try:
            listener(event)
        except Exception as exc:
            print(f"Event listener error: {exc}")


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
    emit_event({
        "type": "tool_result",
        "tool": tool_name,
        "result": str(result)[:500] if result else None,
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


def emit_llm_response(provider: str, content_preview: str):
    """LLM 响应"""
    emit_event({
        "type": "llm_response",
        "provider": provider,
        "content_preview": content_preview,
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
