"""RunManager + SSE 集成测试（计划 §阶段3 清单，fake runner 注入）。

fake runner 在 RunManager 的 worker 线程里发真实 core 事件——
顺带验证 listener 按 run_id 过滤确实拿到事件（bind_context 先于 run_agent_graph）。
"""
import io
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from api.app import create_app
from core.agent_events import (
    emit_llm_chunk,
    emit_llm_request,
    emit_llm_response,
    emit_node_complete,
    emit_node_start,
    emit_thinking_delta,
    emit_tool_call,
    emit_tool_result,
)


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("L", (16, 16), color=90).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeFlows:
    """start/resume 两个 fake 工作流；可编程延迟、失败与取消。"""

    def __init__(self, root: Path):
        self.root = root
        self.start_calls = []
        self.resume_calls = []
        self.gate = threading.Event()      # set 后 start 才继续（busy/cancel 测试用）
        self.resume_gate = None            # set 后 resume 才开始发事件（订阅时机测试用）
        self.start_raises = None
        self.overlay = root / "outputs" / "run_overlay.png"
        self.overlay.parent.mkdir(parents=True, exist_ok=True)
        self.overlay.write_bytes(_png_bytes())

    def start(self, **kwargs):
        self.start_calls.append(kwargs)
        if self.start_raises is not None:
            raise self.start_raises
        self.gate.wait(timeout=30)
        emit_node_start("prepare", "校验输入并检查参考掩膜")
        emit_thinking_delta("分析任务", "model_reasoning")
        emit_thinking_delta("，规划算子", "model_reasoning")
        emit_llm_request("aliyun", "deepseek-v4.1-flash", 3, has_images=True)
        emit_llm_chunk("拟采用 normalize", provider="aliyun")
        emit_llm_response("aliyun", "拟采用 normalize",
                          usage={"total_tokens": 128}, context_window=131072)
        emit_tool_call("pipeline_preview", {"api_key": "SECRET123456", "steps": 2})
        emit_tool_result("pipeline_preview", {"status": "ok"}, True)
        emit_event_raw({"type": "reference_candidate", "image_id": "img_a",
                        "overlay_path": str(self.overlay), "sam_score": 0.92,
                        "low_quality": False})
        emit_node_complete("prepare", 0.2, {"next_node": "iterate"})
        return {"run_status": "awaiting_feedback", "best_score": 0.5,
                "interrupt": [{"id": "int1", "value": {
                    "kind": "human_review", "stage": "reference",
                    "message": "请确认 img_a 的参考掩膜",
                    "overlay_path": str(self.overlay),
                    "best_score": 0.5, "stop_reason": None}}]}

    def resume(self, **kwargs):
        self.resume_calls.append(kwargs)
        if self.resume_gate is not None:
            self.resume_gate.wait(timeout=30)
        emit_node_start("iterate", "提出算子序列调整")
        emit_node_complete("iterate", 0.4, {"pipeline_length": 3})
        return {"run_status": "completed", "best_score": 0.9}


def emit_event_raw(event):
    from core.agent_events import emit_event

    payload = {"timestamp": time.time(), **event}
    emit_event(payload)


@pytest.fixture
def env(tmp_path):
    flows = FakeFlows(tmp_path)
    app = create_app(root=tmp_path, run_fn=flows.start, resume_fn=flows.resume,
                     provider_factory=lambda: SimpleNamespace(model="fake-model"))
    with TestClient(app) as client:
        task = client.post("/api/tasks", json={"title": "运行测试"}).json()
        client.post(f"/api/tasks/{task['id']}/samples",
                    files=[("files", ("a.png", _png_bytes(), "image/png"))])
        yield SimpleNamespace(client=client, task=task, flows=flows,
                              root=tmp_path)
        flows.gate.set()


def _collect_events(client, task_id, run_id, after=None):
    """读完整条 SSE 流，返回 UI 事件列表；遇到 event: end 结束。"""
    params = {"after": after} if after is not None else None
    events = []
    with client.stream("GET",
                       f"/api/tasks/{task_id}/runs/{run_id}/events",
                       params=params) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        buffer = []
        for line in response.iter_lines():
            if line == "":
                continue
            buffer.append(line)
            if line.startswith("data: "):
                parsed = json.loads(line[len("data: "):])
                if parsed.get("type"):          # end 帧的 data: {} 不计入事件
                    events.append(parsed)
            if line == "event: end":
                break
    return events


def _wait_finished(client, task_id, run_id, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = client.get(f"/api/tasks/{task_id}/runs/{run_id}").json()
        if snapshot["status"] != "running":
            return snapshot
        time.sleep(0.05)
    pytest.fail("run did not finish in time")


def test_sse_full_stream_order_and_review(tmp_path, env):
    env.flows.gate.set()
    run_id = env.client.post(f"/api/tasks/{env.task['id']}/runs",
                             json={"message": "找出亮块", "target_type": "defect"}).json()["run_id"]
    events = _collect_events(env.client, env.task["id"], run_id)

    seqs = [event["seq"] for event in events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)  # 严格递增无重复
    assert seqs[0] == 1
    assert events[0]["type"] == "run_started"
    assert events[0]["message"] == "找出亮块"
    assert events[0]["image_count"] == 1 and events[0]["resumed"] is False

    types = [event["type"] for event in events]
    assert types.count("run_started") == 1
    assert "step_started" in types and "step_finished" in types
    assert "model_call_started" in types and "model_call_finished" in types
    assert "tool_started" in types and "tool_finished" in types
    assert "reference_candidate" in types

    # 两条 thinking_delta 同 context 落在同一合并窗口 → 合成一条
    thinking = [event for event in events if event["type"] == "thinking"]
    assert len(thinking) == 1
    assert thinking[0]["text"] == "分析任务，规划算子"
    assert thinking[0]["delta"] is True

    assert types[-2] == "review_requested"
    assert types[-1] == "run_finished"
    review = events[-2]
    assert review["stage"] == "reference"
    assert review["overlay_url"].startswith("/api/files?path=")
    assert env.client.get(review["overlay_url"]).status_code == 200
    assert events[-1]["status"] == "awaiting_review"

    # 传给 runner 的参数：RunManager 预生成的 run_id 作为 thread_id
    call = env.flows.start_calls[0]
    assert call["thread_id"] == run_id
    assert call["task_id"] == env.task["id"]
    assert len(call["target_image_paths"]) == 1
    assert call["target_type"] == "defect"


def test_reconnect_with_after_returns_only_newer_events(tmp_path, env):
    env.flows.gate.set()
    run_id = env.client.post(f"/api/tasks/{env.task['id']}/runs",
                             json={"message": "找出亮块"}).json()["run_id"]
    _wait_finished(env.client, env.task["id"], run_id)
    events = _collect_events(env.client, env.task["id"], run_id, after=3)
    assert all(event["seq"] > 3 for event in events)
    assert events[0]["seq"] == 4


def test_second_start_while_running_returns_409(tmp_path, env):
    run_id = env.client.post(f"/api/tasks/{env.task['id']}/runs",
                             json={"message": "第一次"}).json()["run_id"]
    try:
        # 运行中：任务详情暴露活动 run（latest_run_id 落盘前的刷新恢复窗口）
        detail = env.client.get(f"/api/tasks/{env.task['id']}").json()
        assert detail["running"] is True
        assert detail["active_run_id"] == run_id
        response = env.client.post(f"/api/tasks/{env.task['id']}/runs",
                                   json={"message": "第二次"})
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "task_busy"
        # 运行中的任务也不能删除
        response = env.client.delete(f"/api/tasks/{env.task['id']}")
        assert response.status_code == 409
    finally:
        env.flows.gate.set()
        _wait_finished(env.client, env.task["id"], run_id)
    detail = env.client.get(f"/api/tasks/{env.task['id']}").json()
    assert detail["running"] is False
    assert detail["active_run_id"] is None


def test_cancel_produces_cancelled_finish(tmp_path, env):
    run_id = env.client.post(f"/api/tasks/{env.task['id']}/runs",
                             json={"message": "找出亮块"}).json()["run_id"]
    # start 卡在 gate 上：取消要通过 RequestCancelled 从事件发射点抛出
    def canceller():
        time.sleep(0.3)
        env.client.post(f"/api/tasks/{env.task['id']}/runs/{run_id}/cancel")

    threading.Thread(target=canceller, daemon=True).start()

    # 让 start 在 gate 等待期间收到取消：gate 释放后第一个 emit 检查点抛 RequestCancelled
    def releaser():
        time.sleep(0.6)
        env.flows.gate.set()

    threading.Thread(target=releaser, daemon=True).start()

    events = _collect_events(env.client, env.task["id"], run_id)
    finishes = [event for event in events if event["type"] == "run_finished"]
    assert finishes[-1]["status"] == "cancelled"


def test_disconnect_does_not_cancel_and_stream_jsonl_complete(tmp_path, env):
    run_id = env.client.post(f"/api/tasks/{env.task['id']}/runs",
                             json={"message": "找出亮块"}).json()["run_id"]
    # 订阅后立刻断开（只读一个事件就关闭连接）
    with env.client.stream("GET",
                           f"/api/tasks/{env.task['id']}/runs/{run_id}/events") as response:
        for line in response.iter_lines():
            if line.startswith("data: "):
                break
    env.flows.gate.set()
    snapshot = _wait_finished(env.client, env.task["id"], run_id)
    assert snapshot["status"] == "awaiting_review"  # 运行没有被断连打断

    stream_path = (env.root / "workspace" / "tasks" / env.task["id"]
                   / "runs" / run_id / "stream.jsonl")
    lines = [json.loads(line) for line in
             stream_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [event["seq"] for event in lines] == [event["seq"] for event in snapshot["events"]]
    # 持久化内容经过 redact：工具参数里的假密钥不能落盘
    dumped = stream_path.read_text(encoding="utf-8")
    assert "SECRET123456" not in dumped
    assert "[REDACTED]" in dumped


def test_resume_continues_seq_on_same_stream(tmp_path, env):
    env.flows.gate.set()
    run_id = env.client.post(f"/api/tasks/{env.task['id']}/runs",
                             json={"message": "找出亮块"}).json()["run_id"]
    first = _wait_finished(env.client, env.task["id"], run_id)
    last_seq = first["events"][-1]["seq"]

    response = env.client.post(f"/api/tasks/{env.task['id']}/runs/{run_id}/review",
                               json={"action": "continue", "feedback": "边界偏小"})
    assert response.status_code == 200
    second = _wait_finished(env.client, env.task["id"], run_id)
    assert second["status"] == "completed"
    resumed_started = [event for event in second["events"]
                       if event["type"] == "run_started" and event.get("resumed")]
    assert resumed_started[0]["seq"] == last_seq + 1
    assert resumed_started[0]["action"] == "continue"
    assert resumed_started[0]["feedback"] == "边界偏小"
    seqs = [event["seq"] for event in second["events"]]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)

    resume_call = env.flows.resume_calls[0]
    assert resume_call["thread_id"] == run_id
    assert resume_call["response"] == {"action": "continue", "feedback": "边界偏小"}


def test_subscribe_during_resumed_segment_keeps_streaming(tmp_path, env):
    """决策后立刻重开 SSE（前端行为）：resume 段进行中不能提前收到 end。

    回归：resume 曾忘记把 handle.finished 复位，重开的订阅会立刻收到
    event: end，时间线在 resume 段中途断流。"""
    env.flows.gate.set()
    run_id = env.client.post(f"/api/tasks/{env.task['id']}/runs",
                             json={"message": "找出亮块"}).json()["run_id"]
    first = _wait_finished(env.client, env.task["id"], run_id)
    last_seq = first["events"][-1]["seq"]

    # resume 先卡在 gate 上：POST /review 返回后、事件尚未发出时完成订阅
    env.flows.resume_gate = threading.Event()
    assert env.client.post(f"/api/tasks/{env.task['id']}/runs/{run_id}/review",
                           json={"action": "continue"}).status_code == 200

    lines = []
    reader = threading.Thread(
        target=lambda: _read_stream_into(env.client, env.task["id"], run_id,
                                         last_seq, lines),
        daemon=True)
    reader.start()
    time.sleep(0.3)          # 确保订阅已经打开并处于等待
    env.flows.resume_gate.set()
    reader.join(timeout=10)

    events = [json.loads(line[len("data: "):]) for line in lines
              if line.startswith("data: ") and line != "data: {}"]
    assert events, "resume 段没有收到任何事件（订阅被提前 end 掐断）"
    assert events[0]["type"] == "run_started" and events[0]["seq"] == last_seq + 1
    assert events[-1]["type"] == "run_finished" and events[-1]["status"] == "completed"
    assert lines[-1] == "event: end"


def _read_stream_into(client, task_id, run_id, after, lines):
    with client.stream("GET",
                       f"/api/tasks/{task_id}/runs/{run_id}/events",
                       params={"after": after}) as response:
        for line in response.iter_lines():
            lines.append(line)
            if line == "event: end":
                break


def test_runner_exception_maps_to_error_and_failed(tmp_path, env):
    env.flows.start_raises = ValueError("生成失败 api_key=SECRET123456 泄漏")
    env.flows.gate.set()
    run_id = env.client.post(f"/api/tasks/{env.task['id']}/runs",
                             json={"message": "找出亮块"}).json()["run_id"]
    events = _collect_events(env.client, env.task["id"], run_id)
    error = next(event for event in events if event["type"] == "error")
    assert "[REDACTED]" in error["message"]
    assert "SECRET123456" not in error["message"]
    finishes = [event for event in events if event["type"] == "run_finished"]
    assert finishes[-1]["status"] == "failed"


def test_start_without_samples_is_rejected(tmp_path, env):
    empty_task = env.client.post("/api/tasks", json={"title": "无图"}).json()
    response = env.client.post(f"/api/tasks/{empty_task['id']}/runs",
                               json={"message": "找出亮块"})
    assert response.status_code == 400
    assert "请先上传样本图" in response.json()["error"]["message"]


def test_unknown_run_returns_404(tmp_path, env):
    response = env.client.get(f"/api/tasks/{env.task['id']}/runs/agent_none")
    assert response.status_code == 404
    response = env.client.get(
        f"/api/tasks/{env.task['id']}/runs/agent_none/events")
    assert response.status_code == 404
