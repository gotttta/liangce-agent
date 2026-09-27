"""LangGraph nodes shared by the legacy, controller and tool-agent graphs.

These nodes own input preparation, independent candidate review, the human
decision interrupt and trajectory persistence. Graph builders import them from
here instead of reaching into each other's private helpers.
"""

import json
import time
from pathlib import Path

from langgraph.types import interrupt

from core.agent_events import emit_node_complete, emit_node_start
from core.agent_loop import promote_candidate_result
from core.experiments.lifecycle import record_human_review, record_review
from core.measurement.evaluation import GROUND_TRUTH_GATE, meets_ground_truth_gate
from core.reference_extraction import extract_annotation_from_reference, reference_mask_stats
from core.runtime_logging import logger
from core.task_contract import check_delivery
from core.task_store import save_rejection_record

def with_event(state, node, started, details):
    event = {
        "node": node,
        "status": "completed",
        "duration_seconds": round(time.monotonic() - started, 6),
        "details": details,
    }
    return {**state, "trajectory": [*(state.get("trajectory") or []), event]}


def write_trajectory(state):
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


def prepare_inputs(state):
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
    return with_event(result, "prepare_inputs", started, {
        "target_image_path": str(target),
        "reference_template_count": len(reference_masks),
    })


def wait_for_human(state):
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
    return apply_human_response(state, response)


def apply_human_response(state, response):
    action = str((response or {}).get("action", "continue"))
    if action not in {"accept", "continue", "exit"}:
        raise ValueError(f"未知的人类决策：{action}")
    if action == 'accept' and state.get('orchestration_version') and not Path(state.get('annotated_image_path') or '').is_file():
        raise ValueError('没有可供人工验收的已执行结果')
    if action == 'accept' and state.get('orchestration_version') and not any(
            item.get('experiment_id') == state.get('selected_experiment_id') and
            item.get('status') in {'selected_for_review', 'completed'} for item in state.get('candidate_attempts', [])):
        raise ValueError('当前实验未成功执行，诊断用原图不能作为算法结果验收')
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
    result = with_event(
        {
            **state,
            "human_response": dict(response or {}),
            "human_feedback": human_feedback,
            "agent_status": status_by_action[action],
            "run_status": {'accept': 'completed', 'continue': 'awaiting_feedback', 'exit': 'stopped'}[action],
            "state_version": int(state.get('state_version', 0)) + 1,
        },
        "resume_after_human",
        started,
        {"action": action, "has_feedback": bool(human_feedback)},
    )
    result = record_human_review(result, response or {})
    emit_node_complete("wait_for_human", time.monotonic() - started, {
        "action": action,
        "has_feedback": bool(human_feedback),
    })
    write_trajectory(result)
    return result


def make_review_candidates_node(provider):
    def review_candidates(state):
        emit_node_start("review_candidates", "检查实验结果与任务验收条件")
        started = time.monotonic()
        attempts = state.get("candidate_attempts", [])
        completed = [item for item in attempts if item.get("status") == "selected_for_review"]
        if not completed:
            can_revise = provider is not None and hasattr(provider, "understand_task")
            review = {
                "decision": "revise" if can_revise else "failed",
                "selected_candidate": None,
                "reason": (
                    "本次实验没有生成可用结果，正在根据执行记录定位并修正问题。"
                    if can_revise
                    else "这次没有识别到目标，而且当前无法自动更换识别方法。"
                ),
                "observed_issues": ["没有生成可用标注"],
            }
        else:
            review = None
            visual_available = False
            objective_review = ground_truth_review(state, completed)
            if provider is not None and hasattr(provider, "review_candidates") and completed:
                try:
                    review = call_review_candidates(
                        provider,
                        state["target_image_path"],
                        state["description"],
                        completed,
                        reference_examples=state.get("reference_examples", []),
                        acceptance_criteria={
                            **(state.get("acceptance_criteria") or (state.get("understanding") or {}).get("acceptance_criteria") or {}),
                            "rendering": (state.get("task_contract") or {}).get("rendering") or state.get("rendering"),
                            "unit": (state.get("task_contract") or {}).get("unit"),
                            "target_constraints": (state.get("task_contract") or {}).get("target_constraints"),
                            "memory_contract": {key: ((state.get("memory_context") or {}).get("task_memory") or {}).get(key)
                                                for key in ("current_goal", "active_constraints")},
                        },
                    )
                    visual_available = isinstance(review, dict) and review.get('decision') in {'present', 'revise'}
                except Exception as exc:
                    logger.warning("Graph operation recovered from exception", exc_info=True)
                    review = {
                        "decision": "review_unavailable",
                        "selected_candidate": None,
                        "reason": f"视觉复查调用失败：{type(exc).__name__}: {exc}",
                        "observed_issues": ["review_service_unavailable"],
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
            if completed and visual_available and review.get('decision') == 'present':
                review = {**review, 'decision': 'revise', 'reason': '视觉复查未指出通过验收的有效实验。'}
            selected_name = state.get("selected_candidate")
        selected = next((item for item in completed if item.get("name") == selected_name), None)
        if selected is not None:
            delivery = check_delivery(selected, state.get('task_contract') or {
                'acceptance_criteria': state.get('acceptance_criteria') or {},
                'target_constraints': (state.get('understanding') or {}).get('target_constraints') or {},
            })
            objective_review = ground_truth_review({**state, 'selected_candidate': selected_name}, completed)
            review['acceptance'] = {
                'pixel_metrics': None if objective_review is None else objective_review['decision'] == 'present',
                'delivery': delivery,
                'visual_task_conditions': review.get('decision') == 'present' if visual_available else None,
                'overall_passed': bool(visual_available and review.get('decision') == 'present'
                    and delivery['passed'] and (objective_review is None or objective_review['decision'] == 'present')),
            }
            if not delivery['passed'] or (objective_review and objective_review['decision'] != 'present'):
                review = {**review, 'decision': 'revise',
                    'observed_issues': [*(review.get('observed_issues') or []), *delivery['issues'],
                        *((objective_review or {}).get('observed_issues') or [])],
                    'reason': (objective_review or {}).get('reason') or '交付结果不满足任务契约。'}
            elif objective_review and not visual_available and review.get('decision') != 'review_unavailable':
                review['reason'] = objective_review['reason'] + '任务及显示条件仍待人工验收。'
        merged = promote_candidate_result(
            {**state, "review": review},
            selected_name,
        )
        merged["review"] = {
            **review,
            "selected_candidate": selected_name,
        }
        merged = record_review(merged)
        duration = time.monotonic() - started
        emit_node_complete("review_candidates", duration, {
            "decision": review.get("decision"),
            "selected": selected_name,
        })
        return with_event(
            merged,
            "review_candidates",
            started,
            merged["review"],
        )

    return review_candidates


def ground_truth_review(state, completed):
    """Pixel/instance checks only; task and delivery checks are combined by review."""
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
    if meets_ground_truth_gate(evaluation):
        return {
            "decision": "present",
            "selected_candidate": selected_name,
            "reason": f"Ground Truth 指标达到当前开发门槛：{metric_text}。",
            "observed_issues": [],
            "revision_plan": [],
        }
    if evaluation.get("count_error", 0) != 0:
        failed.append('component_count')
    revisions = []
    if 'component_count' in failed:
        revisions.append('检查目标粘连与分离：候选连通域数量与 Ground Truth 不一致。')
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


def call_review_candidates(
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
