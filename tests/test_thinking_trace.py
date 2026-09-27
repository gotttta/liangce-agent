"""思考过程真实内容上屏：reasoning_content、任务理解摘要与工具轮次说明。"""
import json
from types import SimpleNamespace

from core.agent_events import register_event_listener, unregister_event_listener
from core.planning import ModelReply, PlanningSession
from providers.vision import AliyunVisionProvider, understanding_summary_lines


def _streaming_client(chunks):
    def create(**kwargs):
        return iter(chunks)

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _listening():
    events = []
    register_event_listener(events.append)
    return events


def test_reasoning_content_is_surfaced_as_thinking_event():
    chunks = [
        {"choices": [{"delta": {"reasoning_content": "REASONING_MARKER 优先尝试亮阈值。"}}]},
        {"choices": [{"delta": {"content": "{\"task_summary\": \"提取亮目标\"}"}}]},
    ]
    provider = AliyunVisionProvider(api_key="test")
    events = _listening()
    try:
        content = provider._complete_streaming(_streaming_client(chunks), [])
    finally:
        unregister_event_listener(events.append)

    assert content == '{"task_summary": "提取亮目标"}'
    thinking = [e for e in events if e.get("type") == "thinking"]
    assert thinking, "reasoning_content 应产生思考事件"
    assert any("REASONING_MARKER" in str(e.get("message")) for e in thinking)


def test_stream_without_reasoning_emits_no_synthetic_thinking():
    chunks = [{"choices": [{"delta": {"content": "ok"}}]}]
    provider = AliyunVisionProvider(api_key="test")
    events = _listening()
    try:
        provider._complete_streaming(_streaming_client(chunks), [])
    finally:
        unregister_event_listener(events.append)

    assert not [e for e in events if e.get("type") == "thinking"]
    assert not [e for e in events if e.get("type") == "thinking_delta"]


def test_reasoning_chunks_stream_as_live_thinking_deltas():
    chunks = [
        {"choices": [{"delta": {"reasoning_content": "先观察背景亮度。"}}]},
        {"choices": [{"delta": {"reasoning_content": "再选择亮极性阈值。"}}]},
        {"choices": [{"delta": {"content": "{\"task_summary\": \"提取亮目标\"}"}}]},
    ]
    provider = AliyunVisionProvider(api_key="test")
    events = _listening()
    try:
        content = provider._complete_streaming(_streaming_client(chunks), [])
    finally:
        unregister_event_listener(events.append)

    assert content == '{"task_summary": "提取亮目标"}'
    deltas = [e for e in events if e.get("type") == "thinking_delta"]
    assert [e["content"] for e in deltas] == ["先观察背景亮度。", "再选择亮极性阈值。"]
    # 完整思考事件仍按原样发出，且在所有增量之后，供时间线收敛与回放。
    final = [e for e in events if e.get("type") == "thinking"]
    assert final and events.index(deltas[-1]) < events.index(final[0])
    assert any("先观察背景亮度。" in str(e.get("message")) for e in final)


def test_reasoning_delta_stream_is_capped_at_display_limit():
    from providers.vision import REASONING_DISPLAY_LIMIT
    piece = "思" * 1000
    chunks = [{"choices": [{"delta": {"reasoning_content": piece}}]} for _ in range(6)]
    provider = AliyunVisionProvider(api_key="test")
    events = _listening()
    try:
        provider._complete_streaming(_streaming_client(chunks), [])
    finally:
        unregister_event_listener(events.append)

    deltas = [e for e in events if e.get("type") == "thinking_delta"]
    assert sum(len(e["content"]) for e in deltas) == REASONING_DISPLAY_LIMIT
    final = next(e for e in events if e.get("type") == "thinking")
    assert len(final["message"]) == REASONING_DISPLAY_LIMIT + len("…（截断）")


def test_understanding_summary_lines_render_task_and_candidates():
    lines = understanding_summary_lines({
        "task_summary": "TASK_MARKER 提取暗场中的亮颗粒",
        "target_defect": "DEFECT_MARKER 异常亮点",
        "candidate_pipelines": [
            {"name": "bright_threshold", "hypothesis": "HYPOTHESIS_MARKER 颗粒比背景亮"},
            {"name": "tophat_baseline"},
        ],
    })

    assert any("TASK_MARKER" in line for line in lines)
    assert any("DEFECT_MARKER" in line for line in lines)
    assert any(
        "bright_threshold" in line and "HYPOTHESIS_MARKER" in line for line in lines
    )
    # 没有假设的候选仍要出现，且不会被当成有假设处理。
    fallback = next(line for line in lines if "tophat_baseline" in line)
    assert "HYPOTHESIS_MARKER" not in fallback


class _StubDispatcher:
    def __init__(self):
        self.budget = SimpleNamespace(available=lambda: ["query_operators"])
        self.events = True

    def dispatch(self, request, skill_root):
        return {"call_id": request["call_id"], "status": "success", "data": {}}, []

    def summary(self):
        return {}


def test_tool_round_intent_text_is_emitted_as_thinking():
    session = PlanningSession(_StubDispatcher(), None, native=True)
    rounds = [
        ModelReply(
            text="INTENT_MARKER 先查 threshold 算子定义，确认参数后再组装 Pipeline。",
            calls=[{"id": "call_1", "name": "query_operators",
                    "arguments": json.dumps({"names": ["normalize"]})}],
        ),
        '{"candidate_pipelines": []}',
    ]

    def complete(messages, specs, final_only):
        return rounds.pop(0)

    events = _listening()
    try:
        result = session.run(
            [], complete, json.loads, lambda raw: {"final": raw}, lambda path, label: {}
        )
    finally:
        unregister_event_listener(events.append)

    assert result["final"]["candidate_pipelines"] == []
    thinking = [str(e.get("message")) for e in events if e.get("type") == "thinking"]
    assert any("INTENT_MARKER" in message for message in thinking)


def test_final_json_round_does_not_emit_intent_thinking():
    session = PlanningSession(_StubDispatcher(), None, native=True)

    def complete(messages, specs, final_only):
        return '{"candidate_pipelines": []}'

    events = _listening()
    try:
        result = session.run(
            [], complete, json.loads, lambda raw: {"final": raw}, lambda path, label: {}
        )
    finally:
        unregister_event_listener(events.append)

    assert result["final"]["candidate_pipelines"] == []
    assert not [e for e in events if e.get("type") == "thinking"]
