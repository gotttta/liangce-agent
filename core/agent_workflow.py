"""IoU-driven reference-scoring workflow replacing the v2 tool-agent graph.

Reference masks confirmed once by the user anchor all scoring. The model
iterates operator pipelines against them; promotion and stopping are decided
by IterationTracker, never by model self-review.
"""
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import json
from pathlib import Path
from types import SimpleNamespace
import time
from typing import Annotated, Any

import cv2
import numpy as np
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt
from skimage.measure import label as label_instances

from core.agent_events import emit_node_complete, emit_node_start
from core.iteration_tracker import IterationTracker
from core.pipelines.dsl import execute_pipeline, pipeline_operator_catalog
from core.reference_store import ReferenceStore
from core.runtime_logging import logger
from core.sam_session import (
    MIN_MEAN_IOU,
    SamSession,
    _select_array_masks,
    _select_defect_masks,
    _tint,
)
from core.scoring import RunScore, score_image, score_run
from core.workflow_state import (
    AlgorithmSpec,
    WorkflowState,
    WorkflowStateChannel,
    restore_state,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCES_ROOT = PROJECT_ROOT / "workspace" / "references"
DEFAULT_SAM_MODEL = "facebook/sam-vit-base"
NODE_LABELS = {
    "prepare": "校验输入并检查参考掩膜",
    "gen_reference": "生成参考掩膜",
    "human_gate": "等待用户确认",
    "iterate": "提出算子序列调整",
    "score": "在参考图上打分",
    "promote": "更新最优算法",
    "finish": "写出最优算法和分数",
}


@dataclass(frozen=True)
class IterationPolicy:
    patience: int = 5
    target: float = 0.85
    max_iterations: int = 100


def route(state) -> str:
    if isinstance(state, dict):
        return str(state.get("next_node", ""))
    return state.next_node


def _utc_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@lru_cache(maxsize=1)
def _operator_catalog() -> dict[str, list[str]]:
    return {
        entry["name"]: [parameter["name"] for parameter in entry["parameters"]]
        for entry in pipeline_operator_catalog()
    }


def _to_dsl_pipeline(steps: list[dict]) -> dict:
    dsl_steps = []
    previous_id = "image"
    for index, step in enumerate(steps):
        step_id = f"step_{index}"
        dsl_steps.append({
            "id": step_id,
            "op": step["op"],
            "input": previous_id,
            "params": {key: value for key, value in step.items() if key != "op"},
        })
        previous_id = step_id
    return {"name": "agent_pipeline", "steps": dsl_steps}


class _ProviderChatClient:
    """把 provider 的单次 JSON 请求适配成 sam_session 需要的 OpenAI 风格客户端。"""

    def __init__(self, provider):
        self._provider = provider
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model=None, messages=None, **kwargs):
        content = self._provider._complete_action(messages)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class WorkflowRuntime:
    def __init__(self, provider, references_root=None, policy=None, sam_model=DEFAULT_SAM_MODEL):
        self.provider = provider
        self.references_root = Path(references_root or DEFAULT_REFERENCES_ROOT)
        self.policy = policy or IterationPolicy()
        self.sam_model = sam_model

    # 模型调用（测试通过 mock 这两个入口避免真实推理）

    def _vision_json(self, prompt: str, image_paths=()) -> dict:
        from providers.vision import extract_json_object, image_content

        content = [image_content(str(path), "输入图片") for path in image_paths]
        content.append({"type": "text", "text": prompt})
        return extract_json_object(self.provider._complete_action([{"role": "user", "content": content}]))

    def _model_name(self) -> str:
        return str(getattr(self.provider, "model", "vision") or "vision")

    # 节点

    def prepare(self, state: WorkflowState) -> WorkflowState:
        emit_node_start("prepare", NODE_LABELS["prepare"])
        started = time.monotonic()
        if not state.task:
            raise ValueError("缺少任务描述")
        if not state.image_paths:
            raise ValueError("缺少样本图")
        for path in state.image_paths:
            if not Path(path).is_file():
                raise FileNotFoundError(f"样本图不存在: {path}")
        run_dir = Path(state.run_dir or Path(state.output_root or "outputs") / (state.run_id or "run"))
        run_dir.mkdir(parents=True, exist_ok=True)
        reference_ids = ReferenceStore(self.references_root).list_scoreable()
        next_node = "iterate" if reference_ids else "gen_reference"
        emit_node_complete("prepare", time.monotonic() - started, {
            "next_node": next_node, "scoreable_references": len(reference_ids)})
        return replace(state, run_dir=str(run_dir), next_node=next_node)

    def gen_reference(self, state: WorkflowState) -> WorkflowState:
        emit_node_start("gen_reference", NODE_LABELS["gen_reference"])
        started = time.monotonic()
        store = ReferenceStore(self.references_root)
        image_path = self._first_missing_reference(state, store)
        if image_path is None:
            if not store.list_scoreable():
                raise ValueError("所有样本图都已有参考记录但没有可打分的参考（全部被跳过）")
            return replace(state, pending_reference_image_id="", next_node="iterate", human_message="")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取样本图: {image_path}")
        task_text = state.task
        if state.user_feedback:
            task_text += f"\n用户对上一次标注的反馈，请据此修正：{state.user_feedback}"
        try:
            client = _ProviderChatClient(self.provider)
            session = SamSession(self.sam_model)
            if state.target_type == "array":
                parts = _select_array_masks(client, self._model_name(), session, image, task_text)
            else:
                parts = _select_defect_masks(client, self._model_name(), session, image, task_text)
        except ImportError as exc:
            raise ValueError(
                f"生成参考掩膜需要 SAM（transformers/torch），当前环境不可用：{exc}"
                "请先在开发机用 tools/gen_reference.py 生成参考掩膜") from exc
        if not parts:
            message = f"未能为 {image_path} 生成参考掩膜。回复 continue 重试，或回复 exit 结束。"
            emit_node_complete("gen_reference", time.monotonic() - started, {"generated": 0})
            return replace(state, pending_reference_image_id="", pending_reference_mask_path="",
                           human_message=message, user_feedback="", next_node="human_gate")
        merged = np.zeros(image.shape[:2], dtype=bool)
        for _, mask in parts:
            merged |= mask
        mean_score = float(np.mean([score for score, _ in parts]))
        image_id = Path(image_path).stem
        candidate_dir = Path(state.run_dir) / "reference_candidate"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        mask_path = candidate_dir / f"{image_id}.npy"
        overlay_path = candidate_dir / f"{image_id}.png"
        np.save(mask_path, merged)
        cv2.imwrite(str(overlay_path), _tint(image, merged, (0, 255, 0)))
        quality_note = "" if mean_score >= MIN_MEAN_IOU else \
            f"（SAM 分数 {mean_score:.2f} 低于 {MIN_MEAN_IOU}，分割质量存疑，将标记 skip_scoring）"
        message = (f"请确认 {image_id} 的参考掩膜：叠加图 {overlay_path}{quality_note}。"
                   "确认请回复 action=continue 且不带 feedback；"
                   "需要修正请回复 action=continue 并在 feedback 写明哪里不对/漏了；放弃请回复 action=exit。")
        emit_node_complete("gen_reference", time.monotonic() - started,
                           {"generated": len(parts), "sam_iou": round(mean_score, 3)})
        return replace(state, pending_reference_image_id=image_id,
                       pending_reference_mask_path=str(mask_path),
                       pending_reference_overlay_path=str(overlay_path),
                       pending_reference_score=mean_score,
                       human_message=message, user_feedback="", next_node="human_gate")

    def human_gate(self, state: WorkflowState) -> WorkflowState:
        emit_node_start("human_gate", NODE_LABELS["human_gate"])
        stage = "reference" if state.pending_reference_image_id else "final" if state.stop_reason else "retry"
        request = {
            "kind": "human_review",
            "thread_id": state.run_id,
            "stage": stage,
            "message": state.human_message,
            "overlay_path": state.pending_reference_overlay_path or None,
            "best_score": state.best_score,
            "stop_reason": state.stop_reason or None,
        }
        response = interrupt(request) or {}
        action = str(response.get("action", "continue"))
        if action not in {"accept", "continue", "exit"}:
            raise ValueError(f"未知的人类决策：{action}")
        feedback = str(response.get("feedback") or response.get("incremental_description") or "")
        if action == "exit":
            return replace(state, user_feedback=feedback,
                           stop_reason=state.stop_reason or "user_exited",
                           human_message="", next_node="finish")
        if state.pending_reference_image_id:
            if feedback:
                return replace(state, user_feedback=feedback, next_node="gen_reference")
            state = self._save_confirmed_reference(state)
            store = ReferenceStore(self.references_root)
            needs_reference = bool(self._first_missing_reference(state, store)
                                   or not store.list_scoreable())
            next_node = "gen_reference" if needs_reference else "iterate"
            return replace(state, reference_confirmed=True, user_feedback="",
                           human_message="", next_node=next_node)
        if state.stop_reason:
            if action == "accept":
                return replace(state, next_node="finish", human_message="")
            # 用户要求继续：给一轮耐心窗口，避免 should_stop 立刻再次触发
            return replace(state, stop_reason="", grace_iterations=self.policy.patience,
                           user_feedback=feedback, human_message="", next_node="iterate")
        return replace(state, user_feedback=feedback, next_node="gen_reference")

    def iterate(self, state: WorkflowState) -> WorkflowState:
        emit_node_start("iterate", NODE_LABELS["iterate"])
        started = time.monotonic()
        prompt = self._iteration_prompt(state)
        spec = None
        error = ""
        for _ in range(2):
            try:
                raw = self._vision_json(prompt)
                spec, error = self._parse_spec(raw)
            except ValueError as exc:
                spec, error = None, str(exc)
            if spec is not None:
                break
            prompt += f"\n\n上一次输出无效：{error}。请修正后重新输出完整 JSON。"
        if spec is None:
            raise ValueError(f"模型未能给出有效算子序列: {error}")
        emit_node_complete("iterate", time.monotonic() - started,
                           {"pipeline_length": len(spec.pipeline)})
        return replace(state, current_spec=spec, iteration=state.iteration + 1, next_node="score")

    def score(self, state: WorkflowState) -> WorkflowState:
        emit_node_start("score", NODE_LABELS["score"])
        started = time.monotonic()
        store = ReferenceStore(self.references_root)
        image_ids = store.list_scoreable()
        if not image_ids:
            return replace(state, stop_reason="", human_message="没有可打分的参考掩膜，需要先生成参考。",
                           next_node="gen_reference")
        image_scores = []
        for image_id in image_ids:
            ref_mask, meta = store.load(image_id)
            image = self._read_image(meta["image_path"])
            pred_masks = self._predicted_instances(image, state.current_spec.pipeline)
            image_scores.append(score_image(image_id, pred_masks, ref_mask))
        run_score = score_run(image_scores)
        tracker = IterationTracker(Path(state.run_dir))
        tracker.record(run_score, {"pipeline": state.current_spec.pipeline,
                                   "notes": state.current_spec.notes})
        state = replace(state, last_run_score=run_score)
        if run_score.composite_mean > state.best_score:
            emit_node_complete("score", time.monotonic() - started,
                               {"composite_mean": round(run_score.composite_mean, 4), "improved": True})
            return replace(state, next_node="promote")
        emit_node_complete("score", time.monotonic() - started,
                           {"composite_mean": round(run_score.composite_mean, 4), "improved": False})
        return self._stop_or_continue(state, tracker)

    def promote(self, state: WorkflowState) -> WorkflowState:
        emit_node_start("promote", NODE_LABELS["promote"])
        started = time.monotonic()
        run_score = state.last_run_score
        state = replace(state, best_spec=state.current_spec,
                        best_score=run_score.composite_mean, best_run_score=run_score)
        emit_node_complete("promote", time.monotonic() - started,
                           {"best_score": round(state.best_score, 4)})
        return self._stop_or_continue(state, IterationTracker(Path(state.run_dir)))

    def finish(self, state: WorkflowState) -> WorkflowState:
        emit_node_start("finish", NODE_LABELS["finish"])
        started = time.monotonic()
        run_dir = Path(state.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        spec = state.best_spec or state.current_spec
        if spec is not None:
            (run_dir / "algorithm.json").write_text(
                json.dumps({"pipeline": spec.pipeline, "notes": spec.notes},
                           ensure_ascii=False, indent=2), encoding="utf-8")
        run_score = state.best_run_score or state.last_run_score
        if run_score is not None:
            (run_dir / "score.json").write_text(
                json.dumps(_run_score_json(run_score), ensure_ascii=False, indent=2),
                encoding="utf-8")
        emit_node_complete("finish", time.monotonic() - started,
                           {"best_score": round(state.best_score, 4),
                            "stop_reason": state.stop_reason or None})
        return replace(state, next_node="")

    # 内部工具

    def _first_missing_reference(self, state: WorkflowState, store: ReferenceStore) -> str | None:
        for path in state.image_paths:
            try:
                store.load(Path(path).stem)
            except FileNotFoundError:
                return str(path)
        return None

    def _save_confirmed_reference(self, state: WorkflowState) -> WorkflowState:
        store = ReferenceStore(self.references_root)
        mask = np.load(state.pending_reference_mask_path)
        image_path = next((path for path in state.image_paths
                           if Path(path).stem == state.pending_reference_image_id),
                          state.pending_reference_image_id)
        score = float(state.pending_reference_score)
        skip = score < MIN_MEAN_IOU
        meta = {
            "image_id": state.pending_reference_image_id,
            "image_path": str(image_path),
            "task": state.task,
            "target_type": state.target_type,
            "sam_iou_score": score,
            "skip_scoring": skip,
            "skip_reason": "" if not skip else f"SAM IoU 分数均值 {score:.3f} 低于 {MIN_MEAN_IOU}",
            "confirmed_at": _utc_now_iso(),
            "confirmed_by": "user",
        }
        store.save(state.pending_reference_image_id, mask, meta)
        return replace(state, pending_reference_image_id="", pending_reference_mask_path="",
                       pending_reference_overlay_path="", pending_reference_score=0.0)

    def _iteration_prompt(self, state: WorkflowState) -> str:
        catalog = [
            {"name": name, "parameters": parameters}
            for name, parameters in _operator_catalog().items()
        ]
        prompt = (
            f"任务：{state.task}\n目标类型：{state.target_type}\n"
            "你要设计一条按顺序执行的 CV 算子流水线，最终输出二值掩膜。\n"
            f"可用算子及参数（参数名只允许列出的名字）：{json.dumps(catalog, ensure_ascii=False)}\n"
            '只输出JSON：{"pipeline":[{"op":"算子名", ...参数...}, ...],"notes":"一句话说明"}\n'
            "第一台通常是 normalize；最后一步必须是输出掩膜的算子"
            "（global_threshold / adaptive_threshold / statistical_threshold / residual_threshold 等）。")
        if state.current_spec is None:
            prompt += "\n这是第一轮，请提出初始算子序列。"
        else:
            prompt += f"\n上一轮算子序列：{json.dumps(state.current_spec.pipeline, ensure_ascii=False)}"
            if state.last_run_score is not None:
                lines = [f"  {item.image_id}: iou_mean={item.iou_mean:.3f} 误检={item.false_positive_count}"
                         f" 漏检={item.false_negative_count} 参考目标数={item.ref_count}"
                         for item in state.last_run_score.image_scores]
                prompt += (f"\n上一轮分数：composite_mean={state.last_run_score.composite_mean:.3f}\n"
                           + "\n".join(lines)
                           + "\n请根据误检/漏检/边界偏差调整算子结构或参数。")
        if state.user_feedback:
            prompt += f"\n用户反馈：{state.user_feedback}"
        return prompt

    def _parse_spec(self, raw) -> tuple[AlgorithmSpec | None, str]:
        if not isinstance(raw, dict):
            return None, "输出必须是 JSON 对象"
        pipeline = raw.get("pipeline")
        if not isinstance(pipeline, list) or not pipeline:
            return None, "pipeline 必须是非空列表"
        catalog = _operator_catalog()
        steps = []
        for index, step in enumerate(pipeline):
            if not isinstance(step, dict) or not isinstance(step.get("op"), str):
                return None, f"第 {index} 步缺少 op"
            op = step["op"]
            if op not in catalog:
                return None, f"第 {index} 步的算子 {op} 不在可用列表里"
            allowed = set(catalog[op])
            params = {key: value for key, value in step.items() if key != "op" and key in allowed}
            steps.append({"op": op, **params})
        return AlgorithmSpec(pipeline=steps, notes=str(raw.get("notes", ""))), ""

    def _read_image(self, image_path: str) -> np.ndarray:
        path = Path(image_path)
        if not path.is_absolute():
            resolved = PROJECT_ROOT / path
            path = resolved if resolved.is_file() else path
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"无法读取样本图: {image_path}")
        return image

    def _predicted_instances(self, image: np.ndarray, pipeline: list[dict]) -> list[np.ndarray]:
        result = execute_pipeline(image, _to_dsl_pipeline(pipeline))
        if result.mask is None:
            return []
        labels, count = label_instances(result.mask.data, return_num=True)
        return [labels == k for k in range(1, count + 1)]

    def _stop_or_continue(self, state: WorkflowState, tracker: IterationTracker) -> WorkflowState:
        if state.grace_iterations > 0:
            return replace(state, grace_iterations=state.grace_iterations - 1, next_node="iterate")
        stop, reason = tracker.should_stop(self.policy.patience, self.policy.target,
                                           self.policy.max_iterations)
        if stop:
            message = (f"达到停止条件 {reason}：当前最优 composite_mean={state.best_score:.3f}。"
                       "回复 action=accept 接受最优结果，action=continue 继续迭代，action=exit 放弃。")
            return replace(state, stop_reason=reason, human_message=message, next_node="human_gate")
        return replace(state, next_node="iterate")


def _run_score_json(run_score: RunScore) -> dict:
    return {
        "composite_mean": run_score.composite_mean,
        "image_scores": [asdict(item) for item in run_score.image_scores],
    }


def build_workflow_graph(provider=None, checkpointer=None, algorithm_registry=None, *,
                         references_root=None, policy=None, sam_model=DEFAULT_SAM_MODEL):
    runtime = WorkflowRuntime(provider, references_root=references_root,
                              policy=policy, sam_model=sam_model)

    def with_restore(handler):
        def node(state):
            return handler(restore_state(state))
        return node

    graph = StateGraph(WorkflowStateChannel)
    for name in ("prepare", "gen_reference", "human_gate", "iterate", "score", "promote", "finish"):
        graph.add_node(name, with_restore(getattr(runtime, name)))
    graph.set_entry_point("prepare")
    graph.add_conditional_edges("prepare", route, ["gen_reference", "iterate"])
    graph.add_conditional_edges("gen_reference", route, ["human_gate"])
    graph.add_conditional_edges("human_gate", route, ["gen_reference", "iterate", "finish"])
    graph.add_conditional_edges("iterate", route, ["score"])
    graph.add_conditional_edges("score", route, ["promote", "iterate", "gen_reference", "human_gate"])
    graph.add_conditional_edges("promote", route, ["iterate", "human_gate"])
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer)
