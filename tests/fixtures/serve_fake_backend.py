"""阶段 5 手动验收用 fake 后端：在 127.0.0.1:8765 提供与真实 API 相同的接口，
但运行逻辑换成动态 fake 工作流（复用阶段 3 的注入机制，走真实 RunManager /
EventTranslator / SSE / stream.jsonl 全链路）。

剧本：prepare → 按样本逐张 gen_reference + 参考确认（第 2 张低质量）→
3 轮迭代打分 → final 确认（target_reached）→ 接受后 finish。
任务描述包含“慢速”时进入长时间分割循环（每 2 秒一条思考增量），
用于验证刷新恢复与停止按钮。

用法（项目根目录）：
    .venv/bin/python tests/fixtures/serve_fake_backend.py [--port 8765]
"""
import argparse
import io
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from api.app import create_app  # noqa: E402
from core.agent_events import (  # noqa: E402
    emit_event,
    emit_llm_chunk,
    emit_llm_request,
    emit_llm_response,
    emit_node_complete,
    emit_node_start,
    emit_thinking_delta,
    emit_tool_call,
    emit_tool_result,
)
from core.agent_workflow import NODE_LABELS  # noqa: E402

WORKSPACE = Path("/tmp/liangce-fake-backend")


def _overlay_bytes(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    image = np.full((240, 320, 3), 40, dtype=np.uint8)
    for _ in range(4):
        y, x = rng.integers(30, 200), rng.integers(30, 280)
        h, w = rng.integers(18, 45, size=2)
        image[y : y + h, x : x + w] = 235
    mask = image[:, :, 0] > 120
    overlay = image.copy()
    overlay[mask] = [0, 255, 0]
    buffer = io.BytesIO()
    Image.fromarray(overlay).save(buffer, format="PNG")
    return buffer.getvalue()


class DemoFlows:
    def __init__(self, root: Path):
        self.root = root
        self.paths: list[str] = []
        self.slow = False
        self.image_index = 0
        self.iterated = False
        self.best = 0.0
        self.task_id = ""
        self.run_id = ""

    def _save_snapshot(self, status: str, message: str = "", stop_reason: str | None = None,
                       pending_image: str | None = None, overlay: str | None = None):
        """模仿 run_agent_graph 的 save_run_state：任务详情的 runs 列表读 latest.json，
        刷新页面后的时间线回放依赖它。"""
        run_dir = self.root / "workspace" / "tasks" / self.task_id / "runs" / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "run_id": self.run_id,
            "run_status": status,
            "run_started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "stop_reason": stop_reason,
            "best_score": self.best,
            "conversation": [{"role": "assistant", "content": message}] if message else [],
        }
        if pending_image:
            snapshot["pending_reference_image_id"] = pending_image
            snapshot["pending_reference_overlay_path"] = overlay or ""
        (run_dir / "latest.json").write_text(json.dumps(snapshot, ensure_ascii=False),
                                             encoding="utf-8")

    # --- 第 1 段：prepare → 第一张图的参考确认 ---

    def start(self, *, target_image_paths, description, task_id, thread_id, **kwargs):
        self.paths = list(target_image_paths)
        self.slow = "慢速" in str(description or "")
        self.task_id = task_id
        self.run_id = thread_id
        emit_node_start("prepare", NODE_LABELS["prepare"])
        emit_thinking_delta(f"收到任务，共 {len(self.paths)} 张样本图。", "model_reasoning")
        emit_thinking_delta("先检查输入，再逐张生成参考掩膜。", "model_reasoning")
        emit_node_complete("prepare", 0.42,
                           {"next_node": "gen_reference", "scoreable_references": 0})
        if self.slow:
            emit_node_start("gen_reference", NODE_LABELS["gen_reference"])
            for tick in range(600):
                time.sleep(2)
                emit_thinking_delta(f"分割进行中 {tick + 1}… ", "model_reasoning")
            return {"run_status": "completed", "best_score": 0.0}
        return self._gen_current()

    # --- resume：确认/修正当前图 → 下一张图；全部确认后迭代 → final；接受后 finish ---

    def resume(self, *, response, **kwargs):
        action = str(response.get("action"))
        if action == "exit":
            emit_node_start("finish", NODE_LABELS["finish"])
            emit_node_complete("finish", 0.3, {"best_score": self.best})
            self._save_snapshot("completed", "任务已结束")
            return {"run_status": "completed", "best_score": self.best}

        self.image_index += 1
        if self.image_index < len(self.paths):
            return self._gen_current()
        if not self.iterated:
            self.iterated = True
            self._iterate()
            return self._final_interrupt()
        emit_node_start("finish", NODE_LABELS["finish"])
        emit_node_complete("finish", 0.35, {"best_score": self.best})
        self._save_snapshot("completed", "已写出最优算法与分数")
        return {"run_status": "completed", "best_score": self.best}

    # --- 内部：各阶段事件 ---

    def _gen_current(self):
        index = self.image_index
        image_id = Path(self.paths[index]).stem
        overlay_dir = self.root / "outputs"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = overlay_dir / f"overlay_{image_id}.png"
        overlay_path.write_bytes(_overlay_bytes(index + 1))

        emit_node_start("gen_reference", NODE_LABELS["gen_reference"])
        for delta in (f"正在分析 {image_id} 的亮块分布", "，使用 SAM 生成候选掩膜…"):
            emit_thinking_delta(delta, "model_reasoning")
            time.sleep(1.2)
        emit_llm_request("aliyun", "qwen-vl-max", 4, has_images=True)
        for chunk in ("定位到若干高亮区域", "，正在输出多边形…"):
            emit_llm_chunk(chunk)
            time.sleep(0.8)
        emit_llm_response("aliyun", "已生成候选掩膜",
                          usage={"prompt_tokens": 640, "completion_tokens": 88,
                                 "total_tokens": 728},
                          context_window=131072)
        emit_tool_call("generate_reference_mask", {"image": f"{image_id}.png"})
        time.sleep(1.5)
        emit_tool_result("generate_reference_mask",
                         {"status": "ok", "data": {"sam_points": 3}}, True)
        sam_score = 0.52 if index == 1 else 0.88 + 0.04 * (index % 3)
        emit_event({
            "type": "reference_candidate",
            "image_id": image_id,
            "overlay_path": str(overlay_path),
            "sam_score": round(sam_score, 3),
            "low_quality": sam_score < 0.7,
            "timestamp": time.time(),
        })
        emit_node_complete("gen_reference", 12.0 + index, {"image_id": image_id})
        emit_node_start("human_gate", NODE_LABELS["human_gate"])
        quality = (f"（SAM 分数 {sam_score:.2f} 低于 0.7，分割质量存疑，确认后将不参与打分）"
                   if sam_score < 0.7 else "")
        message = f"请确认 {image_id} 的参考掩膜：叠加图见附件。{quality}"
        self._save_snapshot("awaiting_feedback", message,
                            pending_image=image_id, overlay=str(overlay_path))
        return {"run_status": "awaiting_feedback", "best_score": self.best,
                "interrupt": [{"id": f"int_{index}", "value": {
                    "kind": "human_review", "stage": "reference",
                    "message": message,
                    "overlay_path": str(overlay_path),
                    "best_score": self.best, "stop_reason": None}}]}

    def _iterate(self):
        rounds = [(1, 0.62, True, ["normalize", "clahe", "adaptive_threshold"]),
                  (2, 0.71, True, ["normalize", "clahe", "otsu_threshold"]),
                  (3, 0.58, False, ["normalize", "gamma", "otsu_threshold"])]
        for iteration, score, improved, ops in rounds:
            emit_node_start("iterate", NODE_LABELS["iterate"])
            emit_thinking_delta(f"第 {iteration} 轮：尝试 ", "model_reasoning")
            emit_thinking_delta(" → ".join(ops), "model_reasoning")
            time.sleep(1.0)
            emit_node_complete("iterate", 6.0 + iteration, {"pipeline_length": len(ops)})
            emit_node_start("score", NODE_LABELS["score"])
            emit_tool_call("score_pipeline", {"iteration": iteration})
            time.sleep(1.2)
            emit_tool_result("score_pipeline",
                             {"status": "ok", "data": {"remaining_executions": 25}}, True)
            emit_event({
                "type": "iteration_scored",
                "iteration": iteration,
                "composite_mean": score,
                "best_score_before": self.best,
                "improved": improved,
                "pipeline": [{"op": op} for op in ops],
                "notes": f"第 {iteration} 轮组合，composite_mean={score:.2f}",
                "image_scores": [
                    {"image_id": Path(path).stem, "iou_mean": score + 0.03,
                     "false_positive_count": 1, "false_negative_count": 0,
                     "ref_count": 4, "composite": score + 0.03}
                    for path in self.paths[:2]
                ],
                "timestamp": time.time(),
            })
            emit_node_complete("score", 4.5 + iteration,
                               {"composite_mean": score, "improved": improved})
            if improved:
                emit_node_start("promote", NODE_LABELS["promote"])
                emit_node_complete("promote", 0.1, {"best_score": score})
                self.best = score
            time.sleep(0.8)

    def _final_interrupt(self):
        emit_node_start("human_gate", NODE_LABELS["human_gate"])
        message = (f"达到停止条件 target_reached：当前最优 composite_mean={self.best:.3f}。")
        self._save_snapshot("awaiting_feedback", message, stop_reason="target_reached")
        return {"run_status": "awaiting_feedback", "best_score": self.best,
                "interrupt": [{"id": "int_final", "value": {
                    "kind": "human_review", "stage": "final",
                    "message": message,
                    "overlay_path": "",
                    "best_score": self.best, "stop_reason": "target_reached"}}]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--keep", action="store_true", help="保留已有演示工作区")
    args = parser.parse_args()

    if WORKSPACE.exists() and not args.keep:
        shutil.rmtree(WORKSPACE)
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    flows = DemoFlows(WORKSPACE)
    app = create_app(root=WORKSPACE, run_fn=flows.start, resume_fn=flows.resume)

    import uvicorn

    print(f"fake backend on http://127.0.0.1:{args.port} (workspace: {WORKSPACE})")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
