"""UI 事件协议（计划 §8）：把 core 事件翻译成稳定的前端事件。

纯转换，不分配 seq；run_id/ts 在这里填好，seq 由 RunManager 分配。
配对规则：step 按节点出现次序编号，model call 与最近一个未完成的请求配对，
tool 按 FIFO 配对（core 事件没有调用 id，同名并发只能按到达顺序配）。
"""
import time

from core.agent_workflow import NODE_LABELS
from core.runtime_logging import logger

from api.files import FileRoots, file_url


class EventTranslator:
    def __init__(self, run_id: str, roots: FileRoots):
        self.run_id = run_id
        self.roots = roots
        self._step_seq = {}        # node -> 出现次数
        self._open_steps = []      # [(node, step_id)]，后进先出关闭
        self._model_call_seq = 0
        self._open_model_calls = []  # [(call_id, ts)]
        self._last_call_id = None
        self._tool_seq = 0
        self._open_tools = []      # FIFO: [(tool_id, ts)]

    def translate(self, core_event: dict) -> list[dict]:
        handlers = {
            "node_start": self._on_node_start,
            "node_complete": self._on_node_complete,
            "thinking": self._on_thinking,
            "thinking_delta": self._on_thinking_delta,
            "llm_request": self._on_llm_request,
            "llm_response": self._on_llm_response,
            "llm_chunk": self._on_llm_chunk,
            "tool_call": self._on_tool_call,
            "tool_result": self._on_tool_result,
            "reference_candidate": self._on_reference_candidate,
            "iteration_scored": self._on_passthrough,
            "error": self._on_error,
        }
        handler = handlers.get(core_event.get("type"))
        if handler is None:
            logger.debug("Dropping unknown core event type: %s", core_event.get("type"))
            return []
        return handler(core_event)

    # --- 内部工具 ---

    def _ui(self, event_type: str, ts, **fields) -> dict:
        return {"type": event_type, "run_id": self.run_id,
                "ts": float(ts if ts is not None else time.time()), **fields}

    def _current_step(self):
        return self._open_steps[-1][1] if self._open_steps else None

    # --- 事件处理 ---

    def _on_node_start(self, event) -> list[dict]:
        node = str(event.get("node") or "")
        count = self._step_seq.get(node, 0) + 1
        self._step_seq[node] = count
        step_id = f"node:{node}:{count}"
        self._open_steps.append((node, step_id))
        label = event.get("description") or NODE_LABELS.get(node, node)
        return [self._ui("step_started", event.get("timestamp"),
                         step_id=step_id, node=node, label=str(label))]

    def _on_node_complete(self, event) -> list[dict]:
        node = str(event.get("node") or "")
        step_id = None
        for index in range(len(self._open_steps) - 1, -1, -1):
            if self._open_steps[index][0] == node:
                _, step_id = self._open_steps.pop(index)
                break
        if step_id is None:
            logger.debug("node_complete without open step: %s", node)
            return []
        return [self._ui("step_finished", event.get("timestamp"),
                         step_id=step_id, duration=event.get("duration"),
                         metadata=event.get("metadata") or {})]

    def _on_thinking(self, event) -> list[dict]:
        return [self._ui("thinking", event.get("timestamp"),
                         step_id=self._current_step(), context=event.get("context"),
                         text=str(event.get("message") or ""), delta=False)]

    def _on_thinking_delta(self, event) -> list[dict]:
        return [self._ui("thinking", event.get("timestamp"),
                         step_id=self._current_step(), context=event.get("context"),
                         text=str(event.get("content") or ""), delta=True)]

    def _on_llm_request(self, event) -> list[dict]:
        self._model_call_seq += 1
        call_id = f"call:{self._model_call_seq}"
        self._open_model_calls.append((call_id, event.get("timestamp")))
        self._last_call_id = call_id
        return [self._ui("model_call_started", event.get("timestamp"),
                         call_id=call_id, model=str(event.get("model") or ""),
                         message_count=event.get("message_count") or 0,
                         has_images=bool(event.get("has_images")))]

    def _on_llm_response(self, event) -> list[dict]:
        duration = None
        call_id = None
        if self._open_model_calls:
            call_id, started_ts = self._open_model_calls.pop()
            if started_ts is not None and event.get("timestamp") is not None:
                duration = max(0.0, float(event["timestamp"]) - float(started_ts))
        if call_id is None:
            call_id = self._last_call_id
        return [self._ui("model_call_finished", event.get("timestamp"),
                         call_id=str(call_id or ""), usage=event.get("usage"),
                         context_window=event.get("context_window"), duration=duration)]

    def _on_llm_chunk(self, event) -> list[dict]:
        active = self._open_model_calls[-1][0] if self._open_model_calls else self._last_call_id
        return [self._ui("model_output", event.get("timestamp"),
                         call_id=str(active or ""), text=str(event.get("content") or ""))]

    def _on_tool_call(self, event) -> list[dict]:
        self._tool_seq += 1
        tool_id = f"tool:{self._tool_seq}"
        self._open_tools.append((tool_id, event.get("timestamp")))
        return [self._ui("tool_started", event.get("timestamp"),
                         tool_id=tool_id, tool=str(event.get("tool") or ""),
                         args=event.get("args") or {})]

    def _on_tool_result(self, event) -> list[dict]:
        duration = None
        tool_id = None
        if self._open_tools:
            tool_id, started_ts = self._open_tools.pop(0)
            if started_ts is not None and event.get("timestamp") is not None:
                duration = max(0.0, float(event["timestamp"]) - float(started_ts))
        return [self._ui("tool_finished", event.get("timestamp"),
                         tool_id=str(tool_id or ""), tool=str(event.get("tool") or ""),
                         result=event.get("result"), success=bool(event.get("success", True)),
                         duration=duration)]

    def _on_reference_candidate(self, event) -> list[dict]:
        overlay = event.get("overlay_path") or ""
        return [self._ui("reference_candidate", event.get("timestamp"),
                         image_id=str(event.get("image_id") or ""),
                         overlay_url=file_url(overlay, self.roots) if overlay else None,
                         sam_score=float(event.get("sam_score") or 0.0),
                         low_quality=bool(event.get("low_quality")))]

    def _on_passthrough(self, event) -> list[dict]:
        payload = {key: value for key, value in event.items()
                   if key not in {"type", "timestamp", "task_id", "run_id"}}
        return [self._ui(str(event.get("type")), event.get("timestamp"), **payload)]

    def _on_error(self, event) -> list[dict]:
        return [self._ui("error", event.get("timestamp"),
                         message=str(event.get("message") or ""),
                         node=event.get("node"))]
