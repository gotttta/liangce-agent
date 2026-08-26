"""流式事件系统，用于实时传递 Agent 的执行状态到前端。

这个模块提供了一个全局事件队列和发送函数，用于在 Agent 执行过程中
实时推送思考过程、工具调用、节点执行等事件到前端。
"""

import queue
import time
from typing import Any, Dict, Optional

# 全局事件队列，用于线程间通信
_event_queue: Optional[queue.Queue] = None


def set_event_queue(q: queue.Queue):
    """设置全局事件队列，通常在 Gradio 应用启动时调用"""
    global _event_queue
    _event_queue = q


def get_event_queue() -> Optional[queue.Queue]:
    """获取当前的全局事件队列"""
    return _event_queue


def _emit_event(event_type: str, data: Dict[str, Any]):
    """内部函数：发送事件到队列"""
    if _event_queue is None:
        return

    event = {
        "type": event_type,
        "timestamp": time.time(),
        "data": data
    }

    try:
        _event_queue.put_nowait(event)
    except queue.Full:
        # 队列满时忽略，避免阻塞
        pass


def emit_thinking(message: str, stage: str = "general"):
    """发送思考过程事件

    Args:
        message: 思考内容
        stage: 当前阶段标识
    """
    _emit_event("thinking", {
        "message": message,
        "stage": stage
    })


def emit_tool_call(tool_name: str, parameters: Dict[str, Any]):
    """发送工具调用事件

    Args:
        tool_name: 工具名称
        parameters: 工具参数
    """
    _emit_event("tool_call", {
        "tool": tool_name,
        "parameters": parameters
    })


def emit_tool_result(tool_name: str, result: Any):
    """发送工具执行结果事件

    Args:
        tool_name: 工具名称
        result: 执行结果
    """
    _emit_event("tool_result", {
        "tool": tool_name,
        "result": result
    })


def emit_node_progress(node_name: str, progress: float, message: str = ""):
    """发送节点执行进度事件

    Args:
        node_name: 节点名称
        progress: 进度百分比 (0-100)
        message: 进度消息
    """
    _emit_event("node_progress", {
        "node": node_name,
        "progress": progress,
        "message": message
    })


def emit_llm_chunk(content: str, model: str = ""):
    """发送 LLM 流式输出片段

    Args:
        content: 文本片段
        model: 模型名称
    """
    _emit_event("llm_chunk", {
        "content": content,
        "model": model
    })


def emit_error(message: str, error_type: str = "error"):
    """发送错误事件

    Args:
        message: 错误信息
        error_type: 错误类型
    """
    _emit_event("error", {
        "message": message,
        "error_type": error_type
    })


def emit_step_complete(step_name: str, duration: float, result: Any = None):
    """发送步骤完成事件

    Args:
        step_name: 步骤名称
        duration: 执行时长（秒）
        result: 执行结果（可选）
    """
    _emit_event("step_complete", {
        "step": step_name,
        "duration": duration,
        "result": result
    })
