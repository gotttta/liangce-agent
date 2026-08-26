"""Canonical Agent orchestration for the visual pipeline developer.

The graph deliberately keeps pixel processing in the deterministic pipeline
executor. LangGraph owns task state, planning boundaries and the human decision
boundary; it does not replace the CV execution layer.
"""

import json
import time
from pathlib import Path
from typing import Optional, TypedDict
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from core.agent_events import (
    emit_node_complete,
    emit_node_start,
    register_event_listener,
    unregister_event_listener,
)
from core.agent_loop import (
    pipeline_fingerprint,
    plan_candidate_definitions,
    promote_candidate_result,
    run_planned_agent,
)
from core.reference_extraction import extract_annotation_from_reference, reference_mask_stats
from core.task_store import save_rejection_record
from agent_types import normalize_strategy
from providers.vision import normalize_acceptance_criteria, normalize_reference_examples


_CHECKPOINTER = MemorySaver()

GROUND_TRUTH_GATE = {
    "dice": 0.85,
    "recall": 0.90,
    "precision": 0.85,
    "boundary_f1": 0.80,
}


class AgentGraphState(TypedDict, total=False):
    target_image_path: str
    description: str
    original_task_goal: str
    reference_examples: list[dict]
    ground_truth_mask_path: Optional[str]
    ground_truth_annotation_path: Optional[str]
    output_root: str
    unit: str
    max_candidates: int
    max_auto_revisions: int
    revision_count: int
    experiment_history: list[dict]
    previous_state: Optional[dict]
    understanding: dict
    acceptance_criteria: dict
    retrieved_algorithms: list[dict]
    planned_candidates: list[dict]
    trajectory: list[dict]
    decision: dict
    review: dict
    agent_status: str
    status: str
    run_dir: str
    iteration: int
    parent_iteration: Optional[int]
    parent_result_image_path: Optional[str]
    strategy: dict
    selected_candidate: str
    rendering: dict
    quality_report: dict
    evaluation_report: Optional[dict]
    candidate_attempts: list[dict]
    pipeline: dict
    pipeline_diff: dict
    feedback: dict
    measurements: dict
    annotated_image_path: str
    predicted_mask_path: str
    conversation: list[dict]
    segmentation: dict
    errors: list[str]
    graph_thread_id: str
    human_response: dict
    human_feedback: dict
    reference_masks: list[dict]
    interrupt: list[dict]


def build_agent_graph(
    provider=None,
    algorithm_registry=None,
    max_candidates=3,
    checkpointer=None,
):
    workflow = StateGraph(AgentGraphState)
    workflow.add_node("prepare_inputs", _prepare_inputs)
    workflow.add_node("understand_task", _make_understand_task_node(provider))
    workflow.add_node("retrieve_algorithms", _make_retrieve_algorithms_node(algorithm_registry))
    workflow.add_node("plan_candidates", _make_plan_candidates_node(max_candidates))
    workflow.add_node("execute_candidates", _execute_candidates)
    workflow.add_node("review_candidates", _make_review_candidates_node(provider))
    workflow.add_node("revise_candidates", _make_revise_candidates_node(provider, max_candidates))
    workflow.add_node("report_failure", _report_failure)
    workflow.add_node("decide_next_action", _decide_next_action)
    workflow.add_node("wait_for_human", _wait_for_human)

    workflow.set_entry_point("prepare_inputs")
    workflow.add_edge("prepare_inputs", "understand_task")
    workflow.add_edge("understand_task", "retrieve_algorithms")
    workflow.add_edge("retrieve_algorithms", "plan_candidates")
    workflow.add_edge("plan_candidates", "execute_candidates")
    workflow.add_edge("execute_candidates", "review_candidates")
    workflow.add_conditional_edges(
        "review_candidates",
        _route_after_review,
        {"revise": "revise_candidates", "present": "decide_next_action", "fail": "report_failure"},
    )
    workflow.add_conditional_edges(
        "revise_candidates",
        _route_after_revision,
        {"execute": "execute_candidates", "fail": "report_failure"},
    )
    workflow.add_edge("report_failure", END)
    workflow.add_edge("decide_next_action", "wait_for_human")
    workflow.add_edge("wait_for_human", END)
    if checkpointer is False:
        return workflow.compile()
    return workflow.compile(checkpointer=checkpointer or _CHECKPOINTER)


def run_agent_graph(
    target_image_path,
    description,
    output_root="outputs",
    reference_annotation_path=None,
    reference_examples=None,
    ground_truth_mask_path=None,
    ground_truth_annotation_path=None,
    unit="pixel",
    max_candidates=3,
    max_auto_revisions=2,
    provider=None,
    algorithm_registry=None,
    understanding=None,
    retrieved_algorithms=None,
    previous_state=None,
    thread_id=None,
    event_callback=None,
):
    graph_thread_id = thread_id or f"agent_{uuid4().hex}"
    initial_state: AgentGraphState = {
        "target_image_path": str(target_image_path),
        "description": description,
        "reference_examples": normalize_reference_examples(
            [*(reference_examples or []), *([reference_annotation_path] if reference_annotation_path else [])]
        ),
        "ground_truth_mask_path": str(ground_truth_mask_path) if ground_truth_mask_path else None,
        "ground_truth_annotation_path": str(ground_truth_annotation_path) if ground_truth_annotation_path else None,
        "output_root": str(output_root),
        "unit": unit,
        "max_candidates": max_candidates,
        "max_auto_revisions": max(0, int(max_auto_revisions)),
        "revision_count": 0,
        "experiment_history": [],
        "previous_state": previous_state,
        "original_task_goal": (previous_state or {}).get("original_task_goal") or description,
        "understanding": understanding,
        "retrieved_algorithms": retrieved_algorithms,
        "trajectory": [],
        "status": "pending",
        "graph_thread_id": graph_thread_id,
    }
    graph = build_agent_graph(
        provider=provider,
        algorithm_registry=algorithm_registry,
        max_candidates=max_candidates,
    )
    if event_callback:
        register_event_listener(event_callback)
    try:
        result = graph.invoke(
            initial_state,
            config={"configurable": {"thread_id": graph_thread_id}},
        )
    finally:
        if event_callback:
            unregister_event_listener(event_callback)
    if result.get("__interrupt__"):
        result["interrupt"] = _serialize_interrupts(result["__interrupt__"])
        _write_trajectory(result)
    return result


def resume_agent_graph(thread_id, response, event_callback=None):
    """Resume a paused human-review node without re-running earlier nodes."""
    if not thread_id:
        raise ValueError("缺少 Agent thread ID，无法恢复工作流")
    graph = build_agent_graph()
    if event_callback:
        register_event_listener(event_callback)
    try:
        result = graph.invoke(
            Command(resume=response),
            config={"configurable": {"thread_id": thread_id}},
        )
    finally:
        if event_callback:
            unregister_event_listener(event_callback)
    _write_trajectory(result)
    return result


def _prepare_inputs(state):
    emit_node_start("prepare_inputs", "校验图片、描述和参考标注")
    started = time.monotonic()
    target = Path(state["target_image_path"])
    if not target.exists():
        raise FileNotFoundError(f"Image not found: {target}")
    if not state.get("description"):
        raise ValueError("缺少缺陷描述")
    Path(state["output_root"]).mkdir(parents=True, exist_ok=True)
    reference_masks = []
    for example in state.get("reference_examples", []):
        path = None
        if isinstance(example, dict):
            path = example.get("image_path") or example.get("path")
        if not path:
            continue
        mask, confidence = extract_annotation_from_reference(path)
        if mask is not None and confidence > 0.3:
            reference_masks.append({
                "image_path": str(path),
                "confidence": confidence,
                "stats": reference_mask_stats(mask),
            })
    result = {**state, "reference_masks": reference_masks}
    emit_node_complete("prepare_inputs", time.monotonic() - started, {
        "reference_template_count": len(reference_masks),
    })
    return _with_event(result, "prepare_inputs", started, {
        "target_image_path": str(target),
        "reference_template_count": len(reference_masks),
    })


def _make_understand_task_node(provider):
    def understand_task(state):
        emit_node_start("understand_task", "分析图像和任务描述，理解检测目标")
        started = time.monotonic()
        understanding = state.get("understanding")
        if not isinstance(understanding, dict):
            if provider is None:
                raise ValueError("缺少任务理解结果或 Vision Provider")
            try:
                understanding = _call_understand_task(
                    provider,
                    state["target_image_path"],
                    state["description"],
                    previous_context={
                        "original_task_goal": (state.get("previous_state") or {}).get("original_task_goal")
                        or state.get("original_task_goal")
                        or state.get("description"),
                        "previous_pipeline": (state.get("previous_state") or {}).get("pipeline"),
                        "previous_quality": (state.get("previous_state") or {}).get("quality_report"),
                        "reference_examples": state.get("reference_examples", []),
                        "reference_masks": state.get("reference_masks", []),
                        "ground_truth_mask_path": state.get("ground_truth_mask_path"),
                        "ground_truth_annotation_path": state.get("ground_truth_annotation_path"),
                        "human_feedback": (state.get("previous_state") or {}).get("human_feedback", {}),
                    },
                    reference_examples=state.get("reference_examples", []),
                )
            except Exception as exc:
                # Provider/schema failures are recoverable. Preserve the error
                # as context while letting the deterministic planner continue.
                description = state["description"]
                periodic = any(token in description.lower() for token in ("periodic", "周期", "条纹", "重复"))
                understanding = {
                    "task_summary": description,
                    "target_defect": "用户描述的视觉异常",
                    "normal_context": "周期背景" if periodic else "未知背景",
                    "ambiguities": [f"provider_invalid_json_or_call: {type(exc).__name__}: {exc}"],
                    "questions": [],
                    "candidate_pipelines": [],
                    "recommended_strategy": normalize_strategy({
                        "visual_observation": {
                            "background_pattern": "periodic_lines" if periodic else "unknown",
                        },
                    }),
                    "target_constraints": {},
                    "rendering": {},
                }
        criteria = normalize_acceptance_criteria(
            understanding.get("acceptance_criteria"),
            task_summary=understanding.get("task_summary") or state.get("description"),
            output_requirements=understanding.get("output_requirements"),
        )
        understanding = {**understanding, "acceptance_criteria": criteria}
        duration = time.monotonic() - started
        emit_node_complete(
            "understand_task",
            duration,
            {
                "provider": type(provider).__name__ if provider else "precomputed",
                "strategy": understanding.get("recommended_strategy", {}).get("name"),
            }
        )
        return _with_event(
            {**state, "understanding": understanding, "acceptance_criteria": criteria},
            "understand_task",
            started,
            {"provider": type(provider).__name__ if provider else "precomputed"},
        )

    return understand_task


def _make_retrieve_algorithms_node(algorithm_registry):
    def retrieve_algorithms(state):
        emit_node_start("retrieve_algorithms", "检索历史成功的算法")
        started = time.monotonic()
        matches = state.get("retrieved_algorithms")
        if not isinstance(matches, list):
            matches = []
            if algorithm_registry is not None and not (state.get("previous_state") or {}).get("pipeline"):
                matches = algorithm_registry.search(state["understanding"], limit=2, min_score=0.2)
        summary = [
            {
                "algorithm_id": item.get("algorithm_id"),
                "name": item.get("name"),
                "score": item.get("score"),
                "match_reasons": item.get("match_reasons", []),
            }
            for item in matches
            if isinstance(item, dict)
        ]
        duration = time.monotonic() - started
        emit_node_complete("retrieve_algorithms", duration, {"match_count": len(summary)})
        return _with_event(
            {**state, "retrieved_algorithms": matches},
            "retrieve_algorithms",
            started,
            {"match_count": len(summary), "matches": summary},
        )

    return retrieve_algorithms


def _make_plan_candidates_node(max_candidates):
    def plan_candidates(state):
        emit_node_start("plan_candidates", "生成候选算法方案")
        started = time.monotonic()
        candidates = plan_candidate_definitions(
            state["understanding"],
            previous_state=state.get("previous_state"),
            retrieved_algorithms=state.get("retrieved_algorithms"),
            max_candidates=state.get("max_candidates", max_candidates),
        )
        summary = [
            {
                "name": item.get("name"),
                "source": item.get("source", {"type": "qwen"}),
            }
            for item in candidates
        ]
        duration = time.monotonic() - started
        emit_node_complete("plan_candidates", duration, {"candidate_count": len(candidates)})
        return _with_event(
            {**state, "planned_candidates": candidates},
            "plan_candidates",
            started,
            {"generated_pipeline": summary[0] if summary else None},
        )

    return plan_candidates


def _execute_candidates(state):
    emit_node_start("execute_candidates", "执行候选算法并生成结果")
    started = time.monotonic()
    result = run_planned_agent(
        target_image_path=state["target_image_path"],
        description=state["description"],
        understanding=state["understanding"],
        output_root=state["output_root"],
        unit=state.get("unit", "pixel"),
        max_candidates=state.get("max_candidates", 3),
        previous_state=state.get("previous_state"),
        retrieved_algorithms=state.get("retrieved_algorithms"),
        planned_candidates=state.get("planned_candidates"),
        run_dir=state.get("run_dir"),
        return_failure_state=True,
        ground_truth_mask_path=state.get("ground_truth_mask_path"),
    )
    experiment = {
        "iteration": result.get("iteration"),
        "selected_candidate": result.get("selected_candidate"),
        "quality_report": result.get("quality_report", {}),
        "pipeline": result.get("pipeline", {}),
        "candidate_attempts": result.get("candidate_attempts", []),
        "pipeline_diff": result.get("pipeline_diff", ),
    }
    merged = {
        **state,
        **result,
        "experiment_history": [*(state.get("experiment_history") or []), experiment],
    }
    duration = time.monotonic() - started
    emit_node_complete("execute_candidates", duration, {
        "attempt_count": len(result.get("candidate_attempts", [])),
        "selected_candidate": result.get("selected_candidate"),
    })
    return _with_event(
        merged,
        "execute_candidates",
        started,
        {
            "attempt_count": len(result.get("candidate_attempts", [])),
            "selected_candidate": result.get("selected_candidate"),
            "quality": result.get("quality_report", {}),
        },
    )


def _decide_next_action(state):
    emit_node_start("decide_next_action", "准备人工验收")
    started = time.monotonic()
    review = state.get("review") if isinstance(state.get("review"), dict) else {}
    decision = {
        "next_action": "wait_for_acceptance",
        "reason": review.get("reason") or "标注结果已生成，等待用户确认标注是否准确。",
    }
    result = {
        **state,
        "decision": decision,
        "agent_status": "waiting_for_acceptance",
    }
    result = _with_event(result, "decide_next_action", started, decision)
    emit_node_complete("decide_next_action", time.monotonic() - started, decision)
    _write_trajectory(result)
    return result


def _make_review_candidates_node(provider):
    def review_candidates(state):
        emit_node_start("review_candidates", "评估候选结果质量")
        started = time.monotonic()
        attempts = state.get("candidate_attempts", [])
        completed = [item for item in attempts if item.get("status") == "selected_for_review"]
        if not completed:
            can_revise = provider is not None and hasattr(provider, "understand_task")
            review = {
                "decision": "revise" if can_revise else "failed",
                "selected_candidate": None,
                "reason": (
                    "这次没有识别到目标，正在自动换一种方法重试。"
                    if can_revise
                    else "这次没有识别到目标，而且当前无法自动更换识别方法。"
                ),
                "observed_issues": ["没有生成可用标注"],
            }
        else:
            review = None
            objective_review = _ground_truth_review(state, completed)
            if objective_review is not None:
                review = objective_review
            elif provider is not None and hasattr(provider, "review_candidates") and completed:
                try:
                    review = _call_review_candidates(
                        provider,
                        state["target_image_path"],
                        state["description"],
                        completed,
                        reference_examples=state.get("reference_examples", []),
                        acceptance_criteria=state.get("acceptance_criteria")
                        or (state.get("understanding") or {}).get("acceptance_criteria"),
                    )
                except Exception as exc:
                    review = {
                        "decision": "revise",
                        "selected_candidate": None,
                        "reason": f"视觉复查调用失败：{type(exc).__name__}: {exc}",
                    }
            if not isinstance(review, dict):
                review = {
                    "decision": "present",
                    "selected_candidate": state.get("selected_candidate"),
                    "reason": "当前没有配置自动视觉复查，结果已生成，等待用户直观确认。",
                    "observed_issues": [],
                }
        selected_name = review.get("selected_candidate")
        if selected_name not in {item.get("name") for item in completed}:
            selected_name = state.get("selected_candidate")
        merged = promote_candidate_result(
            {**state, "review": review},
            selected_name,
        )
        merged["review"] = {
            **review,
            "selected_candidate": selected_name,
        }
        duration = time.monotonic() - started
        emit_node_complete("review_candidates", duration, {
            "decision": review.get("decision"),
            "selected": selected_name,
        })
        return _with_event(
            merged,
            "review_candidates",
            started,
            merged["review"],
        )

    return review_candidates


def _route_after_review(state):
    review = state.get("review") if isinstance(state.get("review"), dict) else {}
    revision_count = int(state.get("revision_count", 0))
    max_revisions = int(state.get("max_auto_revisions", 1))
    if review.get("decision") == "revise":
        return "revise" if revision_count < max_revisions else "fail"
    if review.get("decision") == "failed":
        return "fail"
    return "present"


def _ground_truth_review(state, completed):
    """Make the ReAct decision from same-image Ground Truth metrics.

    This path intentionally bypasses visual-model judgement: when a reference
    mask is pixel-aligned, objective overlap is more reliable than another
    interpretation of the same image.
    """
    if not state.get("ground_truth_mask_path"):
        return None
    selected_name = state.get("selected_candidate")
    selected = next((item for item in completed if item.get("name") == selected_name), None)
    if selected is None:
        selected = completed[0]
        selected_name = selected.get("name")
    evaluation = (selected.get("quality") or {}).get("evaluation") or {}
    if evaluation.get("status") != "ok":
        return {
            "decision": "revise",
            "selected_candidate": selected_name,
            "reason": "Ground Truth 评估无效，无法比较候选标注。",
            "observed_issues": [str(evaluation.get("reason") or "evaluation_invalid")],
            "revision_plan": ["检查 Ground Truth Mask 与原图尺寸和标注范围。"],
        }
    failed = [
        key for key, threshold in GROUND_TRUTH_GATE.items()
        if float(evaluation.get(key, 0.0)) < threshold
    ]
    metric_text = "，".join(
        f"{key}={float(evaluation.get(key, 0.0)):.3f}"
        for key in ("dice", "recall", "precision", "boundary_f1")
    )
    if not failed:
        return {
            "decision": "present",
            "selected_candidate": selected_name,
            "reason": f"Ground Truth 指标达到当前开发门槛：{metric_text}。",
            "observed_issues": [],
            "revision_plan": [],
        }
    revisions = []
    if "recall" in failed:
        revisions.append("漏检偏多：扩大目标响应范围，并检查最小面积与阈值。")
    if "precision" in failed:
        revisions.append("误检偏多：加强背景抑制和候选区域过滤。")
    if "boundary_f1" in failed:
        revisions.append("边界偏差较大：调整去噪、形态学和轮廓处理。")
    if "dice" in failed:
        revisions.append("整体重叠不足：根据误检和漏检同时重规划 Pipeline。")
    return {
        "decision": "revise",
        "selected_candidate": selected_name,
        "reason": f"Ground Truth 指标未达到当前开发门槛：{metric_text}。",
        "observed_issues": [f"{key}_below_gate" for key in failed],
        "revision_plan": revisions,
    }


def _route_after_revision(state):
    if state.get("planned_candidates"):
        return "execute"
    return "fail"


def _make_revise_candidates_node(provider, max_candidates):
    def revise_candidates(state):
        emit_node_start("revise_candidates", "根据复查结果调整候选算法")
        started = time.monotonic()
        if provider is None or not hasattr(provider, "understand_task"):
            emit_node_complete("revise_candidates", time.monotonic() - started, {
                "skipped": True,
                "reason": "缺少 Vision Provider",
            })
            return state
        previous_state = {
            "iteration": state.get("iteration", 0),
            "original_task_goal": state.get("original_task_goal")
            or (state.get("previous_state") or {}).get("original_task_goal")
            or state.get("description"),
            "annotated_image_path": state.get("annotated_image_path"),
            "predicted_mask_path": state.get("predicted_mask_path"),
            "pipeline": state.get("pipeline", {}),
            "quality_report": state.get("quality_report", {}),
            "evaluation_report": state.get("evaluation_report"),
            "ground_truth_mask_path": state.get("ground_truth_mask_path"),
            "ground_truth_annotation_path": state.get("ground_truth_annotation_path"),
            "feedback_image_path": state.get("feedback", {}).get("feedback_image_path"),
            "false_positive_mask_path": state.get("feedback", {}).get("false_positive_mask_path"),
            "false_negative_mask_path": state.get("feedback", {}).get("false_negative_mask_path"),
            "human_feedback": dict(state.get("human_feedback") or {}),
            "include_mask_path": state.get("include_mask_path")
            or (state.get("human_feedback") or {}).get("include_mask_path"),
            "exclude_mask_path": state.get("exclude_mask_path")
            or (state.get("human_feedback") or {}).get("exclude_mask_path"),
        }
        has_completed_result = any(
            item.get("status") == "selected_for_review"
            for item in state.get("candidate_attempts", [])
        )
        execution_feedback = {
            "status": "needs_visual_revision" if has_completed_result else "no_usable_annotation",
            "instruction": (
                "上一种方法已有标注，但视觉复查认为需要调整。请结合复查意见生成改进方法。"
                if has_completed_result
                else "上一种方法没有得到可用标注。分析每一步记录，生成不同的方法，不要原样重复。"
            ),
            "attempts": state.get("candidate_attempts", []),
            "review": state.get("review", {}),
        }
        try:
            understanding = _call_understand_task(
                provider,
                state["target_image_path"],
                state["description"],
                previous_context={
                    "previous_pipeline": state.get("pipeline", {}),
                    "previous_quality": state.get("quality_report", {}),
                    "previous_evaluation": state.get("evaluation_report"),
                    "previous_result_image_path": state.get("annotated_image_path"),
                    "review": state.get("review", {}),
                    "execution_feedback": execution_feedback,
                    "original_task_goal": previous_state.get("original_task_goal"),
                    "reference_examples": state.get("reference_examples", []),
                    "reference_masks": state.get("reference_masks", []),
                    "ground_truth_mask_path": state.get("ground_truth_mask_path"),
                    "ground_truth_annotation_path": state.get("ground_truth_annotation_path"),
                },
                reference_examples=state.get("reference_examples", []),
            )
        except Exception as exc:
            understanding = _fallback_understanding(state["description"], exc)
        criteria = normalize_acceptance_criteria(
            understanding.get("acceptance_criteria"),
            task_summary=understanding.get("task_summary") or state.get("description"),
            output_requirements=understanding.get("output_requirements"),
        )
        understanding = {**understanding, "acceptance_criteria": criteria}
        candidates = plan_candidate_definitions(
            understanding,
            previous_state=previous_state,
            retrieved_algorithms=[],
            max_candidates=state.get("max_candidates", max_candidates),
        )
        failed_fingerprints = {
            pipeline_fingerprint(attempt.get("pipeline"))
            for experiment in state.get("experiment_history", [])
            for attempt in experiment.get("candidate_attempts", [])
            if (
                isinstance(attempt, dict)
                and attempt.get("pipeline")
                and attempt.get("status") in {
                    "failed", "no_annotation", "health_failed", "duplicate_pipeline",
                }
            )
        }
        candidates = [
            candidate for candidate in candidates
            if pipeline_fingerprint(candidate.get("pipeline")) not in failed_fingerprints
        ]
        result = {
            **state,
            "understanding": understanding,
            "acceptance_criteria": criteria,
            "planned_candidates": candidates,
            "previous_state": previous_state,
            "retrieved_algorithms": [],
            "revision_count": int(state.get("revision_count", 0)) + 1,
        }
        emit_node_complete("revise_candidates", time.monotonic() - started, {
            "revision_count": result["revision_count"],
            "candidate_count": len(candidates),
        })
        return _with_event(
            result,
            "revise_candidates",
            started,
            {
                "revision_count": result["revision_count"],
                "reason": (state.get("review") or {}).get("reason"),
                "candidate_count": len(candidates),
            },
        )

    return revise_candidates


def _report_failure(state):
    """Hand the last rendered result to the user after automatic retries.

    A visual review can reject an otherwise renderable candidate. That is not
    an execution error: the user must still be able to inspect and decide on
    the final candidate instead of losing its image in an exception.
    """
    emit_node_start("report_failure", "自动复查未通过，交由用户判断")
    started = time.monotonic()
    attempts = int(state.get("revision_count", 0)) + 1
    state = _promote_best_ground_truth_attempt(state)
    review = state.get("review") if isinstance(state.get("review"), dict) else {}
    had_result = any(
        item.get("status") == "selected_for_review"
        for item in state.get("candidate_attempts", [])
        if isinstance(item, dict)
    )
    reason = "结果没有满足当前任务的验收条件" if had_result else "没有生成可用结果"
    message = (
        f"自动复查没有通过：{reason}。系统已经尝试了 {attempts} 种方法。"
        "我仍会展示最后一版结果，请由你判断是否可用；可用则确认，不可用则继续修改。"
    )
    decision = {
        "next_action": "wait_for_acceptance",
        "reason": message,
        "automatic_review_passed": False,
    }
    result = _with_event(
        {
            **state,
            "status": "needs_human_review",
            "agent_status": "waiting_for_feedback",
            "decision": decision,
            "review": {**review, "reason": review.get("reason") or message},
        },
        "report_failure",
        started,
        {"attempt_count": attempts, "message": message, "had_result": had_result},
    )
    emit_node_complete("report_failure", time.monotonic() - started, {
        "attempt_count": attempts,
        "message": message,
    })
    _write_trajectory(result)
    return result


def _promote_best_ground_truth_attempt(state):
    """Keep the strongest evaluated candidate across all ReAct iterations."""
    if not state.get("ground_truth_mask_path"):
        return state
    evaluated = []
    for experiment in state.get("experiment_history", []):
        for attempt in experiment.get("candidate_attempts", []):
            evaluation = (attempt.get("quality") or {}).get("evaluation") or {}
            if attempt.get("status") == "selected_for_review" and evaluation.get("status") == "ok":
                evaluated.append(attempt)
    if not evaluated:
        return state

    def rank(attempt):
        evaluation = (attempt.get("quality") or {}).get("evaluation") or {}
        return tuple(float(evaluation.get(key, -1.0)) for key in (
            "dice", "boundary_f1", "recall", "precision", "iou",
        ))

    selected = max(evaluated, key=rank)
    current_evaluation = state.get("evaluation_report") or {}
    selected_evaluation = (selected.get("quality") or {}).get("evaluation") or {}
    if rank(selected) < tuple(float(current_evaluation.get(key, -1.0)) for key in (
        "dice", "boundary_f1", "recall", "precision", "iou",
    )):
        return state
    promoted = promote_candidate_result(
        {**state, "candidate_attempts": [selected]},
        selected.get("name"),
    )
    promoted["candidate_attempts"] = state.get("candidate_attempts", [])
    promoted["conversation"] = [
        {
            "role": "assistant",
            "content": f"自动重试完成，已保留 Ground Truth 指标最好的候选 {selected.get('name')}。",
        },
        {
            "role": "assistant",
            "content": (
                f"本轮标出 {((selected.get('measurements') or {}).get('summary') or {}).get('count', 0)} 个区域，"
                f"总面积 {((selected.get('measurements') or {}).get('summary') or {}).get('total_area', 0)} pixel。"
            ),
        },
    ]
    return promoted


def _call_understand_task(provider, target_image_path, description, previous_context, reference_examples):
    try:
        return provider.understand_task(
            target_image_path,
            description,
            previous_context=previous_context,
            reference_examples=reference_examples,
        )
    except TypeError as exc:
        if "reference_examples" not in str(exc):
            raise
        return provider.understand_task(
            target_image_path,
            description,
            previous_context=previous_context,
        )


def _fallback_understanding(description, error):
    periodic = any(token in str(description).lower() for token in ("periodic", "周期", "条纹", "重复"))
    return {
        "task_summary": str(description),
        "target_defect": "用户描述的视觉异常",
        "normal_context": "周期背景" if periodic else "未知背景",
        "ambiguities": [f"provider_invalid_json_or_call: {type(error).__name__}: {error}"],
        "questions": [],
        "candidate_pipelines": [],
        "recommended_strategy": normalize_strategy({
            "visual_observation": {
                "background_pattern": "periodic_lines" if periodic else "unknown",
            },
        }),
        "target_constraints": {},
        "rendering": {},
    }


def _call_review_candidates(
    provider,
    target_image_path,
    description,
    candidates,
    reference_examples,
    acceptance_criteria=None,
):
    try:
        return provider.review_candidates(
            target_image_path,
            description,
            candidates,
            reference_examples=reference_examples,
            acceptance_criteria=acceptance_criteria,
        )
    except TypeError as exc:
        if "reference_examples" not in str(exc) and "acceptance_criteria" not in str(exc):
            raise
        try:
            return provider.review_candidates(
                target_image_path,
                description,
                candidates,
                reference_examples=reference_examples,
            )
        except TypeError as fallback_exc:
            if "reference_examples" not in str(fallback_exc):
                raise
            return provider.review_candidates(target_image_path, description, candidates)

def _wait_for_human(state):
    emit_node_start("wait_for_human", "等待用户确认标注结果")
    request = {
        "kind": "human_review",
        "thread_id": state.get("graph_thread_id"),
        "agent_status": state.get("agent_status"),
        "decision": state.get("decision", {}),
        "selected_candidate": state.get("selected_candidate"),
        "quality_report": state.get("quality_report", {}),
        "annotated_image_path": state.get("annotated_image_path"),
        "predicted_mask_path": state.get("predicted_mask_path"),
    }
    response = interrupt(request)
    action = str((response or {}).get("action", "continue"))
    if action not in {"accept", "continue", "exit"}:
        raise ValueError(f"未知的人类决策：{action}")
    started = time.monotonic()
    human_feedback = {}
    if action == "continue":
        for key in ("incremental_description", "include_mask_path", "exclude_mask_path"):
            value = (response or {}).get(key)
            if value:
                human_feedback[key] = value
    elif action == "exit" and (response or {}).get("rejection_reason"):
        human_feedback["rejection_reason"] = response["rejection_reason"]
    if action == "exit":
        save_rejection_record(
            task_id=state.get("graph_thread_id"),
            pipeline=state.get("pipeline"),
            rejection_reason=human_feedback.get("rejection_reason"),
            quality_report=state.get("quality_report"),
            task_root=state.get("output_root"),
        )
    status_by_action = {
        "accept": "accepted",
        "continue": "waiting_for_feedback",
        "exit": "exited",
    }
    result = _with_event(
        {
            **state,
            "human_response": dict(response or {}),
            "human_feedback": human_feedback,
            "agent_status": status_by_action[action],
        },
        "resume_after_human",
        started,
        {"action": action, "has_feedback": bool(human_feedback)},
    )
    emit_node_complete("wait_for_human", time.monotonic() - started, {
        "action": action,
        "has_feedback": bool(human_feedback),
    })
    _write_trajectory(result)
    return result


def _with_event(state, node, started, details):
    event = {
        "node": node,
        "status": "completed",
        "duration_seconds": round(time.monotonic() - started, 6),
        "details": details,
    }
    return {**state, "trajectory": [*(state.get("trajectory") or []), event]}


def _write_trajectory(state):
    run_dir = state.get("run_dir")
    iteration = state.get("iteration", 0)
    if not run_dir:
        return
    iteration_dir = Path(run_dir) / f"iteration_{iteration}"
    iteration_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = iteration_dir / "agent_trajectory.json"
    trajectory_path.write_text(
        json.dumps(state.get("trajectory", []), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    persisted_state = {
        key: value
        for key, value in state.items()
        if key not in {"__interrupt__", "previous_state", "planned_candidates", "output_root"}
    }
    (iteration_dir / "graph_state.json").write_text(
        json.dumps(persisted_state, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _serialize_interrupts(interrupts):
    return [
        {
            "id": item.id,
            "value": item.value,
        }
        for item in interrupts
    ]
