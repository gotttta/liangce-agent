"""导出前端时间线测试 fixture（计划 §阶段5）。

用与 tests/test_api_runs.py 相同的 fake 工作流注入机制，走真实的
RunManager + EventTranslator + stream.jsonl 全链路，跑一遍完整流程：

    上传 2 张图 → 启动 → 参考确认 img_a（正常）→ 修正反馈 →
    参考确认 img_b（低质量）→ 确认 → 3 轮迭代打分 → final 确认 → 接受

然后把 stream.jsonl 原样写到 web/tests/fixtures/run_events.json，
保证前端 fixture 与后端事件协议一致。

用法（项目根目录）：
    .venv/bin/python tests/fixtures/export_run_events.py
"""
import io
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from api.app import create_app  # noqa: E402
from core.agent_events import (  # noqa: E402
    emit_event,
    emit_llm_chunk,
    emit_llm_request,
    emit_llm_response,
    emit_node_complete,
    emit_node_start,
    emit_thinking,
    emit_thinking_delta,
    emit_tool_call,
    emit_tool_result,
)

FIXTURE_PATH = ROOT / "web" / "tests" / "fixtures" / "run_events.json"
SAMPLE_NAMES = ("img_a.png", "img_b.png")


def _png_bytes(color: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("L", (32, 32), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


class FixtureFlows:
    """阶段 3 fake 工作流的扩展版：按 resume 次数推进固定剧本。"""

    def __init__(self, root: Path):
        self.root = root
        self.resume_count = 0
        self.overlays: dict[str, Path] = {}
        output = root / "outputs"
        output.mkdir(parents=True, exist_ok=True)
        for name in SAMPLE_NAMES:
            overlay = output / f"overlay_{name}"
            overlay.write_bytes(_png_bytes(160))
            self.overlays[name] = overlay

    # --- 第 1 段：启动运行，停在 img_a 的参考确认 ---

    def start(self, **kwargs):
        emit_node_start("prepare", "校验输入并检查参考掩膜")
        emit_thinking_delta("样本共 2 张", "model_reasoning")
        emit_thinking_delta("，先为第一张生成参考", "model_reasoning")
        emit_node_complete("prepare", 0.31,
                           {"next_node": "gen_reference", "scoreable_references": 0})
        emit_node_start("gen_reference", "生成参考掩膜")
        emit_llm_request("aliyun", "qwen-vl-max", 4, has_images=True)
        emit_llm_chunk("正在定位亮块区域")
        emit_llm_response("aliyun", "已定位亮块区域",
                          usage={"prompt_tokens": 512, "completion_tokens": 96,
                                 "total_tokens": 608},
                          context_window=131072)
        emit_tool_call("generate_reference_mask", {"image": SAMPLE_NAMES[0]})
        emit_tool_result("generate_reference_mask",
                         {"status": "ok", "data": {"sam_points": 3}}, True)
        emit_event({
            "type": "reference_candidate",
            "image_id": "img_a",
            "overlay_path": str(self.overlays[SAMPLE_NAMES[0]]),
            "sam_score": 0.91,
            "low_quality": False,
            "timestamp": time.time(),
        })
        emit_node_complete("gen_reference", 41.6, {"image_id": "img_a"})
        emit_node_start("human_gate", "等待用户确认")
        return {"run_status": "awaiting_feedback", "best_score": 0.0,
                "interrupt": [{"id": "int1", "value": {
                    "kind": "human_review", "stage": "reference",
                    "message": "请确认 img_a 的参考掩膜：叠加图见附件。",
                    "overlay_path": str(self.overlays[SAMPLE_NAMES[0]]),
                    "best_score": 0.0, "stop_reason": None}}]}

    # --- resume：第 1 次 = 修正反馈；第 2 次 = 确认 img_b；第 3 次 = 接受结果 ---

    def resume(self, **kwargs):
        self.resume_count += 1
        if self.resume_count == 1:
            self._resume_reference_img_b()
            return self._interrupt_reference("img_b", 0.52)
        if self.resume_count == 2:
            self._resume_iterations()
            return self._interrupt_final()
        emit_node_start("finish", "写出最优算法和分数")
        emit_node_complete("finish", 0.42, {"best_score": 0.71})
        return {"run_status": "completed", "best_score": 0.71}

    def _resume_reference_img_b(self):
        emit_node_start("gen_reference", "生成参考掩膜")
        emit_thinking("按反馈调整 SAM 提示后重试第二张图", "reference_retry")
        emit_tool_call("generate_reference_mask", {"image": SAMPLE_NAMES[1]})
        emit_tool_result("generate_reference_mask",
                         {"status": "ok", "data": {"sam_points": 2}}, True)
        emit_event({
            "type": "reference_candidate",
            "image_id": "img_b",
            "overlay_path": str(self.overlays[SAMPLE_NAMES[1]]),
            "sam_score": 0.52,
            "low_quality": True,
            "timestamp": time.time(),
        })
        emit_node_complete("gen_reference", 37.9, {"image_id": "img_b"})
        emit_node_start("human_gate", "等待用户确认")

    def _resume_iterations(self):
        rounds = [
            (1, 0.62, True, "normalize → clahe → adaptive_threshold"),
            (2, 0.71, True, "normalize → clahe → otsu_threshold"),
            (3, 0.58, False, "normalize → gamma → otsu_threshold"),
        ]
        best = 0.0
        for iteration, score, improved, ops in rounds:
            emit_node_start("iterate", "提出算子序列调整")
            if iteration == 1:
                emit_thinking_delta(f"第 {iteration} 轮：", "model_reasoning")
                emit_thinking_delta(f"尝试 {ops}", "model_reasoning")
            else:
                emit_thinking(f"第 {iteration} 轮：沿用方向微调参数", "model_reasoning")
            emit_node_complete("iterate", 8.13, {"pipeline_length": 3})
            emit_node_start("score", "在参考图上打分")
            emit_tool_call("score_pipeline", {"iteration": iteration})
            emit_tool_result("score_pipeline",
                             {"status": "ok", "data": {"remaining_executions": 27}}, True)
            pipeline = []
            for op in ops.split(" → "):
                pipeline.append({"op": op} if op != "clahe" else {"op": op, "clip": 2.0})
            emit_event({
                "type": "iteration_scored",
                "iteration": iteration,
                "composite_mean": score,
                "best_score_before": best,
                "improved": improved,
                "pipeline": pipeline,
                "notes": f"第 {iteration} 轮：{ops}，composite_mean={score:.2f}",
                "image_scores": [
                    {"image_id": "img_a", "iou_mean": score + 0.05,
                     "false_positive_count": 1, "false_negative_count": 0,
                     "ref_count": 4, "composite": score + 0.05},
                    {"image_id": "img_b", "iou_mean": score - 0.05,
                     "false_positive_count": 2, "false_negative_count": 1,
                     "ref_count": 3, "composite": score - 0.05},
                ],
                "timestamp": time.time(),
            })
            emit_node_complete("score", 5.47, {"composite_mean": score, "improved": improved})
            if improved:
                emit_node_start("promote", "更新最优算法")
                emit_node_complete("promote", 0.08, {"best_score": score})
                best = score

    def _interrupt_reference(self, image_id: str, sam_score: float):
        quality = ("（SAM 分数 0.52 低于 0.7，分割质量存疑，确认后将不参与打分）"
                   if sam_score < 0.7 else "")
        emit_node_start("human_gate", "等待用户确认")
        return {"run_status": "awaiting_feedback", "best_score": 0.0,
                "interrupt": [{"id": "int2", "value": {
                    "kind": "human_review", "stage": "reference",
                    "message": f"请确认 {image_id} 的参考掩膜：叠加图见附件。{quality}",
                    "overlay_path": str(self.overlays[f"{image_id}.png"]),
                    "best_score": 0.0, "stop_reason": None}}]}

    def _interrupt_final(self):
        emit_node_start("human_gate", "等待用户确认")
        return {"run_status": "awaiting_feedback", "best_score": 0.71,
                "interrupt": [{"id": "int3", "value": {
                    "kind": "human_review", "stage": "final",
                    "message": "达到停止条件 target_reached：当前最优 composite_mean=0.710。",
                    "overlay_path": "",
                    "best_score": 0.71, "stop_reason": "target_reached"}}]}


def _wait_finished(client, task_id, run_id, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = client.get(f"/api/tasks/{task_id}/runs/{run_id}").json()
        if snapshot["status"] != "running":
            return snapshot
        time.sleep(0.02)
    raise TimeoutError(f"run {run_id} did not finish")


def main():
    with tempfile.TemporaryDirectory(prefix="liangce-fixture-") as temp:
        root = Path(temp)
        flows = FixtureFlows(root)
        app = create_app(root=root, run_fn=flows.start, resume_fn=flows.resume)
        with TestClient(app) as client:
            task = client.post("/api/tasks", json={"title": "亮块检测"}).json()
            task_id = task["id"]
            files = [("files", (name, _png_bytes(90), "image/png"))
                     for name in SAMPLE_NAMES]
            assert client.post(f"/api/tasks/{task_id}/samples", files=files).status_code == 200

            run_id = client.post(f"/api/tasks/{task_id}/runs",
                                 json={"message": "找出图中的亮块", "target_type": "defect"}
                                 ).json()["run_id"]
            _wait_finished(client, task_id, run_id)

            steps = [
                {"action": "continue", "feedback": "边缘偏小，请包含完整的亮块"},
                {"action": "continue"},
                {"action": "accept"},
            ]
            for step in steps:
                response = client.post(f"/api/tasks/{task_id}/runs/{run_id}/review", json=step)
                assert response.status_code == 200, response.text
                _wait_finished(client, task_id, run_id)

            stream_path = root / "workspace" / "tasks" / task_id / "runs" / run_id / "stream.jsonl"
            events = [json.loads(line) for line in
                      stream_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(events, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    types = [event["type"] for event in events]
    print(f"wrote {len(events)} events to {FIXTURE_PATH}")
    print("types:", json.dumps({name: types.count(name) for name in sorted(set(types))}))


if __name__ == "__main__":
    main()
