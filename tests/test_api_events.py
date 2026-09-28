"""EventTranslator 单元测试（计划 §8 末尾清单）。"""
from pathlib import Path

from api.events import EventTranslator
from api.files import file_roots


def translator(tmp_path) -> EventTranslator:
    return EventTranslator("agent_trans", file_roots(tmp_path))


def test_step_pairing_and_repeated_nodes(tmp_path):
    events = translator(tmp_path)
    started = events.translate({"type": "node_start", "node": "prepare",
                                "description": "校验", "timestamp": 1.0})
    assert started[0]["type"] == "step_started"
    assert started[0]["step_id"] == "node:prepare:1"
    assert started[0]["label"] == "校验"

    finished = events.translate({"type": "node_complete", "node": "prepare",
                                 "duration": 0.3, "metadata": {"next_node": "iterate"},
                                 "timestamp": 1.3})
    assert finished[0]["type"] == "step_finished"
    assert finished[0]["step_id"] == "node:prepare:1"
    assert finished[0]["duration"] == 0.3

    again = events.translate({"type": "node_start", "node": "prepare",
                              "description": "校验", "timestamp": 2.0})
    assert again[0]["step_id"] == "node:prepare:2"  # 同名节点再次出现编号 +1


def test_thinking_attributed_to_current_step(tmp_path):
    events = translator(tmp_path)
    orphan = events.translate({"type": "thinking", "message": "开场", "timestamp": 1.0})
    assert orphan[0]["step_id"] is None
    assert orphan[0]["text"] == "开场"
    assert orphan[0]["delta"] is False

    events.translate({"type": "node_start", "node": "iterate", "description": "迭代",
                      "timestamp": 2.0})
    inside = events.translate({"type": "thinking_delta", "content": "流式",
                               "context": "model_reasoning", "timestamp": 2.1})
    assert inside[0]["step_id"] == "node:iterate:1"
    assert inside[0]["delta"] is True
    assert inside[0]["text"] == "流式"

    events.translate({"type": "node_complete", "node": "iterate", "duration": 1.0,
                      "timestamp": 3.0})
    after = events.translate({"type": "thinking", "message": "收尾", "timestamp": 3.1})
    assert after[0]["step_id"] is None


def test_model_call_pairing_and_duration(tmp_path):
    events = translator(tmp_path)
    first = events.translate({"type": "llm_request", "provider": "aliyun",
                              "model": "deepseek", "message_count": 3,
                              "has_images": True, "timestamp": 10.0})
    assert first[0] == {"type": "model_call_started", "run_id": "agent_trans",
                        "ts": 10.0, "call_id": "call:1", "model": "deepseek",
                        "message_count": 3, "has_images": True}

    chunk = events.translate({"type": "llm_chunk", "provider": "aliyun",
                              "content": "部分输出", "timestamp": 10.5})
    assert chunk[0]["type"] == "model_output"
    assert chunk[0]["call_id"] == "call:1"

    done = events.translate({"type": "llm_response", "provider": "aliyun",
                             "content_preview": "x", "usage": {"total_tokens": 12},
                             "context_window": 131072, "timestamp": 11.0})
    assert done[0]["type"] == "model_call_finished"
    assert done[0]["call_id"] == "call:1"
    assert done[0]["duration"] == 1.0
    assert done[0]["usage"] == {"total_tokens": 12}

    # 第二次调用拿到不同的 call_id；chunk 在无未完成调用时归到最近一次
    events.translate({"type": "llm_request", "provider": "aliyun", "model": "deepseek",
                      "message_count": 1, "timestamp": 20.0})
    trailing = events.translate({"type": "llm_chunk", "content": "tail", "timestamp": 20.2})
    assert trailing[0]["call_id"] == "call:2"
    second = events.translate({"type": "llm_response", "provider": "aliyun",
                               "timestamp": 21.0})
    assert second[0]["call_id"] == "call:2"


def test_tool_fifo_pairing_with_same_name(tmp_path):
    events = translator(tmp_path)
    events.translate({"type": "tool_call", "tool": "same_tool",
                      "args": {"a": 1}, "timestamp": 1.0})
    events.translate({"type": "tool_call", "tool": "same_tool",
                      "args": {"b": 2}, "timestamp": 2.0})

    first = events.translate({"type": "tool_result", "tool": "same_tool",
                              "result": {"status": 1}, "success": True, "timestamp": 3.0})
    second = events.translate({"type": "tool_result", "tool": "same_tool",
                               "result": {"status": 2}, "success": False, "timestamp": 4.0})
    # 同名并发按 FIFO 配对：先完成配第一个 started
    assert first[0]["tool_id"] == "tool:1"
    assert first[0]["duration"] == 2.0
    assert first[0]["success"] is True
    assert second[0]["tool_id"] == "tool:2"
    assert second[0]["duration"] == 2.0


def test_unknown_event_dropped(tmp_path):
    events = translator(tmp_path)
    assert events.translate({"type": "mystery_event", "timestamp": 1.0}) == []


def test_paths_converted_to_urls(tmp_path):
    overlay = tmp_path / "outputs" / "run1" / "reference_candidate" / "img_a.png"
    overlay.parent.mkdir(parents=True, exist_ok=True)
    overlay.write_bytes(b"png")
    events = translator(tmp_path)
    translated = events.translate({"type": "reference_candidate", "image_id": "img_a",
                                   "overlay_path": str(overlay), "sam_score": 0.55,
                                   "low_quality": True, "timestamp": 1.0})
    assert translated[0]["type"] == "reference_candidate"
    assert translated[0]["overlay_url"] == "/api/files?path=outputs/run1/reference_candidate/img_a.png"
    assert translated[0]["sam_score"] == 0.55
    assert translated[0]["low_quality"] is True


def test_iteration_scored_passthrough(tmp_path):
    events = translator(tmp_path)
    translated = events.translate({"type": "iteration_scored", "iteration": 3,
                                   "composite_mean": 0.7, "best_score_before": 0.6,
                                   "improved": True, "pipeline": [{"op": "normalize"}],
                                   "notes": "v3",
                                   "image_scores": [{"image_id": "img_a", "iou_mean": 0.7}],
                                   "task_id": "t", "run_id": "r", "timestamp": 5.0})
    assert translated[0]["type"] == "iteration_scored"
    assert translated[0]["iteration"] == 3
    assert translated[0]["pipeline"] == [{"op": "normalize"}]
    assert translated[0]["image_scores"][0]["image_id"] == "img_a"
    assert "task_id" not in translated[0]  # core 元数据不透传


def test_error_event(tmp_path):
    events = translator(tmp_path)
    translated = events.translate({"type": "error", "error_type": "ValueError",
                                   "message": "模型输出无效", "node": "iterate",
                                   "timestamp": 1.0})
    assert translated[0] == {"type": "error", "run_id": "agent_trans", "ts": 1.0,
                             "message": "模型输出无效", "node": "iterate"}
