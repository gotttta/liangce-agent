from datetime import datetime
from copy import deepcopy
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from core.stream_events import emit_thinking, emit_tool_call, emit_tool_result
from core.measurement.area import measure_components
from core.measurement.evaluation import evaluate_prediction
from core.pipelines.dsl import normalize_pipeline, strategy_to_pipeline
from core.preprocessing import load_grayscale
from core.quality import evaluate_mask_quality, inspect_mask_health
from core.sandbox import execute_pipeline_sandbox
from core.visualization import save_annotated_image, save_mask_image


def run_planned_agent(
    target_image_path,
    description,
    understanding,
    output_root="outputs",
    unit="pixel",
    max_candidates=3,
    previous_state=None,
    retrieved_algorithms=None,
    planned_candidates=None,
    run_dir=None,
    return_failure_state=False,
    ground_truth_mask_path=None,
):
    image_path = Path(target_image_path)
    image = load_grayscale(image_path)
    run_dir = Path(run_dir) if run_dir else _make_run_dir(output_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    parent_iteration = (
        int(previous_state.get("iteration", 0))
        if isinstance(previous_state, dict) and previous_state.get("iteration") is not None
        else None
    )
    iteration = 0 if parent_iteration is None else parent_iteration + 1
    iteration_dir = run_dir / f"iteration_{iteration}"
    iteration_dir.mkdir(parents=True)
    strategy = understanding.get("recommended_strategy", {})
    previous_pipeline = (previous_state or {}).get("pipeline") if isinstance(previous_state, dict) else None
    candidates = (
        planned_candidates
        if planned_candidates is not None
        else plan_candidate_definitions(
            understanding,
            previous_state=previous_state,
            retrieved_algorithms=retrieved_algorithms,
            max_candidates=max_candidates,
        )
    )
    if not candidates:
        candidates = _fallback_candidate_definitions(understanding)
    rendering = understanding.get("rendering") or {}
    target_constraints = understanding.get("target_constraints") or {}
    ground_truth_mask = _load_ground_truth_mask(ground_truth_mask_path, image.shape)
    if ground_truth_mask is not None:
        candidates = [
            *candidates,
            *_ground_truth_calibration_candidates(candidates, ground_truth_mask),
        ]
    attempts = []
    executed_fingerprints = set()

    for index, candidate in enumerate(candidates):
        candidate_name = candidate.get("name") or f"candidate_{index + 1}"
        emit_thinking(f"正在执行候选算法: {candidate_name}", "execute_candidate")

        candidate_dir = iteration_dir / f"candidate_{index}"
        candidate_dir.mkdir()
        raw_pipeline = candidate.get("pipeline")
        try:
            pipeline = normalize_pipeline(raw_pipeline, name=candidate_name)
        except Exception as exc:
            failure = {"issues": ["pipeline_invalid"], "error": f"{type(exc).__name__}: {exc}"}
            _write_json(candidate_dir / "pipeline.json", raw_pipeline or {})
            _write_json(candidate_dir / "quality_report.json", failure)
            attempts.append({
                "index": index,
                "name": candidate_name,
                "hypothesis": candidate.get("hypothesis", ""),
                "source": candidate.get("source", {"type": "qwen"}),
                "status": "failed",
                "failure_type": "pipeline_invalid",
                "pipeline": raw_pipeline or {},
                "quality": failure,
                "directory": str(candidate_dir),
            })
            continue
        _write_json(candidate_dir / "pipeline.json", pipeline)

        fingerprint = pipeline_fingerprint(pipeline)
        if fingerprint in executed_fingerprints:
            failure = {"issues": ["duplicate_pipeline"], "message": "同一任务中不重复执行相同 Pipeline。"}
            _write_json(candidate_dir / "quality_report.json", failure)
            attempts.append({
                "index": index, "name": candidate_name, "hypothesis": candidate.get("hypothesis", ""),
                "source": candidate.get("source", {"type": "qwen"}), "status": "duplicate_pipeline",
                "failure_type": "duplicate_pipeline",
                "pipeline": pipeline, "quality": failure, "directory": str(candidate_dir),
            })
            continue
        executed_fingerprints.add(fingerprint)

        emit_tool_call("execute_pipeline_sandbox", {"pipeline": candidate_name, "steps": len(pipeline.get("steps", []))})

        try:
            execution = execute_pipeline_sandbox(image, pipeline)
            emit_tool_result("execute_pipeline_sandbox", {"success": True, "result": "流水线执行成功"})

            emit_thinking("应用用户约束", "apply_constraints")
            constrained_mask, constraint_report = apply_user_constraints(
                execution.mask.data, previous_state,
            )
            execution.mask.data = constrained_mask

            emit_thinking("评估质量", "evaluate_quality")
            quality = evaluate_mask_quality(constrained_mask)
            quality["user_constraints"] = constraint_report
            quality = _apply_feedback_quality(quality, constrained_mask, previous_state)
            health = inspect_mask_health(constrained_mask, target_constraints)
            quality["health"] = health
            if ground_truth_mask is not None:
                quality["evaluation"] = evaluate_prediction(
                    constrained_mask,
                    ground_truth_mask,
                )

            emit_thinking("测量组件", "measure_components")
            measurements = measure_components(execution.mask.data, min_area=1, unit=unit)

            emit_thinking("保存可视化结果", "save_visualization")
            save_mask_image(execution.mask.data, candidate_dir / "mask.png")
            save_annotated_image(
                image_path,
                measurements["results"],
                candidate_dir / "result_annotation.png",
                mask=execution.mask.data,
                contour_color=rendering.get("contour_color", "#ff4030"),
                contour_thickness=int(rendering.get("contour_thickness", 1)),
                annotation_mode=rendering.get("annotation_mode", "contour"),
                mask_alpha=int(rendering.get("mask_alpha", 72)),
            )
            _write_json(candidate_dir / "operator_trace.json", list(execution.trace))
            _write_json(candidate_dir / "quality_report.json", quality)
            _write_json(candidate_dir / "measurements.json", measurements)
            _write_json(candidate_dir / "contours.json", _serialize_contours(execution.contours))
            # A same-image Ground Truth is the objective selection signal. Keep
            # every executable candidate, including an empty result, so recall
            # and false positives can guide the next ReAct revision.
            if ground_truth_mask is not None:
                status = "selected_for_review"
            else:
                status = "selected_for_review" if health["usable_for_review"] else (
                    "no_annotation" if "empty_mask" in health["issues"] else "health_failed"
                )
            attempts.append({
                "index": index,
                "name": candidate.get("name") or pipeline["name"],
                "hypothesis": candidate.get("hypothesis", ""),
                "source": candidate.get("source", {"type": "qwen"}),
                "status": status,
                "pipeline": pipeline,
                "execution": execution,
                "quality": quality,
                "measurements": measurements,
                "directory": str(candidate_dir),
            })
        except Exception as exc:
            emit_tool_result("execute_pipeline_sandbox", {"success": False, "error": str(exc)})
            issue = "pipeline_invalid" if isinstance(exc, ValueError) else "pipeline_execution_failed"
            failure = {
                "issues": [issue],
                "error": f"{type(exc).__name__}: {exc}",
            }
            _write_json(candidate_dir / "quality_report.json", failure)
            attempts.append({
                "index": index,
                "name": candidate.get("name") or pipeline["name"],
                "hypothesis": candidate.get("hypothesis", ""),
                "source": candidate.get("source", {"type": "qwen"}),
                "status": "failed",
                "failure_type": issue,
                "pipeline": pipeline,
                "quality": failure,
                "directory": str(candidate_dir),
            })

    completed = [attempt for attempt in attempts if attempt["status"] == "selected_for_review"]
    if not completed:
        emit_thinking("所有候选算法都失败了", "all_failed")
        if return_failure_state:
            return _build_failure_state(
                image_path=image_path,
                description=description,
                run_dir=run_dir,
                iteration=iteration,
                parent_iteration=parent_iteration,
                previous_state=previous_state,
                strategy=strategy,
                rendering=rendering,
                attempts=attempts,
                retrieved_algorithms=retrieved_algorithms,
                iteration_dir=iteration_dir,
                unit=unit,
            )
        if any(attempt["status"] == "no_annotation" for attempt in attempts):
            if int(max_candidates) == 1:
                raise RuntimeError("agent-generated pipeline produced empty annotations")
            raise RuntimeError("all candidate pipelines produced empty annotations")
        errors = [attempt["quality"].get("error", ", ".join(attempt["quality"].get("issues", [])) or "unknown error") for attempt in attempts]
        prefix = "agent-generated pipeline failed: " if int(max_candidates) == 1 else "all candidate pipelines failed: "
        raise RuntimeError(prefix + "; ".join(errors))
    # The visual review node may replace this provisional selection after it
    # compares the rendered candidate images. This provisional result keeps the
    # execution node independently useful and guarantees a renderable fallback.
    selected = select_best_candidate(completed, prefer_evaluation=ground_truth_mask is not None)
    emit_thinking(f"选择最佳候选: {selected['name']}", "select_best")
    execution = selected["execution"]
    measurements = selected["measurements"]

    emit_thinking("保存最终结果", "save_final_results")
    mask_path = iteration_dir / "mask.png"
    result_path = iteration_dir / "result_annotated.png"
    save_mask_image(execution.mask.data, mask_path)
    save_annotated_image(
        image_path,
        measurements["results"],
        result_path,
        mask=execution.mask.data,
        contour_color=rendering.get("contour_color", "#ff4030"),
        contour_thickness=int(rendering.get("contour_thickness", 1)),
        annotation_mode=rendering.get("annotation_mode", "contour"),
        mask_alpha=int(rendering.get("mask_alpha", 72)),
    )
    _write_json(iteration_dir / "pipeline.json", selected["pipeline"])
    _write_json(iteration_dir / "operator_trace.json", list(execution.trace))
    _write_json(iteration_dir / "quality_report.json", selected["quality"])
    if selected["quality"].get("evaluation") is not None:
        _write_json(iteration_dir / "evaluation_report.json", selected["quality"]["evaluation"])
    _write_json(iteration_dir / "measurements.json", measurements)
    contours_path = iteration_dir / "contours.json"
    _write_json(contours_path, _serialize_contours(execution.contours))
    _write_json(iteration_dir / "candidate_summary.json", [_serializable_attempt(item) for item in attempts])

    summary = measurements["summary"]
    state = {
        "target_image_path": str(image_path),
        "description": description,
        "original_task_goal": (previous_state or {}).get("original_task_goal") or description,
        "run_dir": str(run_dir),
        "iteration": iteration,
        "parent_iteration": parent_iteration,
        "parent_result_image_path": (previous_state or {}).get("annotated_image_path"),
        "strategy": strategy,
        "pipeline": selected["pipeline"],
        "selected_candidate": selected["name"],
        "candidate_attempts": [_serializable_attempt(item) for item in attempts],
        "retrieved_algorithms": [
            _serializable_retrieved_algorithm(item)
            for item in (retrieved_algorithms or [])
        ],
        "quality_report": selected["quality"],
        "evaluation_report": selected["quality"].get("evaluation"),
        "ground_truth_mask_path": str(ground_truth_mask_path) if ground_truth_mask_path else None,
        "rendering": rendering,
        "measurements": measurements,
        "annotated_image_path": str(result_path),
        "predicted_mask_path": str(mask_path),
        "contours_path": str(contours_path),
        "pipeline_diff": _pipeline_diff(previous_pipeline, selected["pipeline"]),
        "feedback": {
            "feedback_image_path": (previous_state or {}).get("feedback_image_path"),
            "false_positive_mask_path": (previous_state or {}).get("false_positive_mask_path"),
            "false_negative_mask_path": (previous_state or {}).get("false_negative_mask_path"),
            "false_positive_pixel_count": (previous_state or {}).get("false_positive_pixel_count", 0),
            "false_negative_pixel_count": (previous_state or {}).get("false_negative_pixel_count", 0),
        },
        "human_feedback": dict((previous_state or {}).get("human_feedback") or {}),
        "include_mask_path": (previous_state or {}).get("include_mask_path")
        or ((previous_state or {}).get("human_feedback") or {}).get("include_mask_path"),
        "exclude_mask_path": (previous_state or {}).get("exclude_mask_path")
        or ((previous_state or {}).get("human_feedback") or {}).get("exclude_mask_path"),
        "false_positive_mask_path": (previous_state or {}).get("false_positive_mask_path"),
        "false_negative_mask_path": (previous_state or {}).get("false_negative_mask_path"),
        "status": "ok",
        "agent_status": "waiting_for_acceptance",
        "conversation": [
            {
                "role": "assistant",
                "content": (
                    f"Agent 已生成并执行算法 {selected['name']}。"
                    if int(max_candidates) == 1
                    else f"已执行 {len(attempts)} 个候选并选择 {selected['name']}。"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    f"本轮标出 {summary['count']} 个区域，总面积 "
                    f"{summary['total_area']} {summary['unit']}。"
                ),
            },
        ],
    }
    _write_json(iteration_dir / "graph_state.json", state)
    return state


def _build_failure_state(
    *,
    image_path,
    description,
    run_dir,
    iteration,
    parent_iteration,
    previous_state,
    strategy,
    rendering,
    attempts,
    retrieved_algorithms,
    iteration_dir,
    unit,
):
    serialized_attempts = [_serializable_attempt(item) for item in attempts]
    last_attempt = serialized_attempts[-1] if serialized_attempts else {}
    statuses = {item.get("status") for item in serialized_attempts}
    had_empty_mask = "no_annotation" in statuses
    had_health_failure = "health_failed" in statuses
    issue = "empty_annotation" if had_empty_mask else (
        "mask_health_failed" if had_health_failure else "pipeline_execution_failed"
    )
    message = (
        "这次方法没有标出任何目标，系统会自动换一种方法再试。"
        if had_empty_mask
        else (
            "这次方法产生了不可靠的标注范围，系统会自动换一种方法再试。"
            if had_health_failure else "这次方法没有运行成功，系统会自动换一种方法再试。"
        )
    )
    quality = {
        "issues": [issue],
        "message": message,
        "attempt_count": len(serialized_attempts),
        "recommended_action": "请提供 ROI、include/exclude 约束或新的参考图，以便继续缩小目标范围。",
    }
    # A failed detection is still a meaningful visual result. Persist a blank
    # mask and the unmodified source image so the UI can always show the final
    # attempt and let a person decide what to do next.
    empty_mask = np.zeros_like(load_grayscale(image_path), dtype=bool)
    mask_path = iteration_dir / "mask.png"
    result_path = iteration_dir / "result_annotated.png"
    save_mask_image(empty_mask, mask_path)
    save_annotated_image(
        image_path,
        [],
        result_path,
        mask=empty_mask,
        contour_color=rendering.get("contour_color", "#ff4030"),
        contour_thickness=int(rendering.get("contour_thickness", 1)),
        annotation_mode=rendering.get("annotation_mode", "contour"),
        mask_alpha=int(rendering.get("mask_alpha", 72)),
    )
    state = {
        "target_image_path": str(image_path),
        "description": description,
        "run_dir": str(run_dir),
        "iteration": iteration,
        "parent_iteration": parent_iteration,
        "parent_result_image_path": (previous_state or {}).get("annotated_image_path"),
        "strategy": strategy,
        "pipeline": last_attempt.get("pipeline", {}),
        "selected_candidate": None,
        "candidate_attempts": serialized_attempts,
        "retrieved_algorithms": [
            _serializable_retrieved_algorithm(item)
            for item in (retrieved_algorithms or [])
        ],
        "quality_report": quality,
        "rendering": rendering,
        "measurements": {"results": [], "summary": {"count": 0, "total_area": 0, "unit": unit}},
        "annotated_image_path": str(result_path),
        "predicted_mask_path": str(mask_path),
        "pipeline_diff": _pipeline_diff(
            (previous_state or {}).get("pipeline"),
            last_attempt.get("pipeline", {}),
        ),
        "feedback": {
            "feedback_image_path": (previous_state or {}).get("feedback_image_path"),
            "false_positive_mask_path": (previous_state or {}).get("false_positive_mask_path"),
            "false_negative_mask_path": (previous_state or {}).get("false_negative_mask_path"),
        },
        "human_feedback": dict((previous_state or {}).get("human_feedback") or {}),
        "include_mask_path": (previous_state or {}).get("include_mask_path")
        or ((previous_state or {}).get("human_feedback") or {}).get("include_mask_path"),
        "exclude_mask_path": (previous_state or {}).get("exclude_mask_path")
        or ((previous_state or {}).get("human_feedback") or {}).get("exclude_mask_path"),
        "false_positive_mask_path": (previous_state or {}).get("false_positive_mask_path"),
        "false_negative_mask_path": (previous_state or {}).get("false_negative_mask_path"),
        # The graph may still choose another automatic revision. Once its
        # retry budget is exhausted, report_failure promotes this to the
        # human-review state while retaining these artifacts.
        "status": "retry_needed",
        "agent_status": "retrying",
        "conversation": [{
            "role": "assistant",
            "content": "最后一种方法没有标出目标。我会显示原图，请你判断是否应继续修改。",
        }],
    }
    _write_json(iteration_dir / "candidate_summary.json", serialized_attempts)
    _write_json(iteration_dir / "graph_state.json", state)
    return state


def _load_ground_truth_mask(path, shape):
    if not path:
        return None
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Ground Truth Mask 不存在：{source}")
    mask = np.asarray(Image.open(source).convert("L")) > 0
    if tuple(mask.shape) != tuple(shape):
        raise ValueError(
            "Ground Truth Mask 尺寸与待测图不一致："
            f"{list(mask.shape)} != {list(shape)}"
        )
    return mask


def select_best_candidate(attempts, prefer_evaluation=False):
    """Select a reproducible candidate, using Ground Truth metrics when present."""
    if not attempts:
        raise ValueError("cannot select from an empty candidate list")
    if not prefer_evaluation:
        return attempts[0]

    def rank(attempt):
        evaluation = (attempt.get("quality") or {}).get("evaluation") or {}
        # Dice measures region overlap; boundary F1 then rewards faithful
        # contours. Recall and precision break ties without a synthetic score.
        return tuple(float(evaluation.get(key, -1.0)) for key in (
            "dice", "boundary_f1", "recall", "precision", "iou",
        ))

    return max(attempts, key=rank)


def _ground_truth_calibration_candidates(candidates, ground_truth_mask, limit=6):
    """Create a bounded local parameter search around a residual pipeline.

    Ground Truth is used only to set aggregate component/area constraints and
    rank the executed variants. Pixel coordinates are never copied into the
    generated algorithm, so the resulting Pipeline remains replayable.
    """
    reference_summary = measure_components(ground_truth_mask, min_area=1)["summary"]
    reference_count = int(reference_summary.get("count", 0))
    reference_area = int(reference_summary.get("total_area", 0))
    if reference_count <= 0 or reference_area <= 0:
        return []

    residual_candidates = []
    for candidate in candidates:
        pipeline = candidate.get("pipeline") if isinstance(candidate, dict) else None
        steps = pipeline.get("steps", []) if isinstance(pipeline, dict) else []
        if any(step.get("op") in {"local_background_residual", "morphological_residual"} for step in steps):
            residual_candidates.append(candidate)
    if not residual_candidates:
        return []

    base = residual_candidates[-1]
    base_pipeline = base.get("pipeline") or {}
    average_area = max(1, reference_area // reference_count)
    min_area = max(4, int(average_area * 0.15))
    max_area = max(min_area, int(average_area * 3.0))
    variants = []
    seen = {pipeline_fingerprint(item.get("pipeline")) for item in candidates}
    residual_step = next(
        (step for step in base_pipeline.get("steps", []) if step.get("op") == "local_background_residual"),
        None,
    )
    base_sigma = float((residual_step or {}).get("params", {}).get("sigma", 10.0))
    sigma_values = [base_sigma * 0.67, base_sigma, base_sigma * 1.25] if residual_step else [None]

    for sigma in sigma_values:
        for radius in (1, 2):
            pipeline = deepcopy(base_pipeline)
            pipeline["name"] = f"{base_pipeline.get('name', 'residual')}_gt_calibrated_s{sigma or 0:.1f}_r{radius}"
            steps = _remove_pipeline_steps(pipeline.get("steps", []), {"fill_holes"})
            filter_index = next(
                (index for index, step in enumerate(steps) if step.get("op") == "filter_components"),
                None,
            )
            if filter_index is None:
                continue
            filter_step = steps[filter_index]
            dilate_id = "gt_calibration_dilate"
            dilate_step = {
                "id": dilate_id,
                "op": "morphology",
                "input": filter_step.get("input"),
                "params": {"method": "dilate", "radius": radius},
            }
            filter_step["input"] = dilate_id
            filter_step["params"] = {
                **(filter_step.get("params") or {}),
                "min_area": min_area,
                "max_area": max_area,
                "max_components": reference_count,
            }
            steps.insert(filter_index, dilate_step)
            if sigma is not None:
                for step in steps:
                    if step.get("op") == "local_background_residual":
                        step["params"] = {**(step.get("params") or {}), "sigma": round(sigma, 3)}
            pipeline["steps"] = steps
            fingerprint = pipeline_fingerprint(pipeline)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            variants.append({
                "name": pipeline["name"],
                "hypothesis": "Ground Truth 指标驱动的残差、膨胀和连通域参数校准。",
                "source": {"type": "ground_truth_calibration"},
                "pipeline": pipeline,
            })
            if len(variants) >= max(1, int(limit)):
                return variants
    return variants


def _remove_pipeline_steps(steps, removed_operators):
    replacements = {}
    retained = []
    for raw_step in steps:
        step = deepcopy(raw_step)
        if step.get("op") in removed_operators:
            replacements[step.get("id")] = step.get("input")
            continue
        retained.append(step)

    def resolve(input_id):
        visited = set()
        while input_id in replacements and input_id not in visited:
            visited.add(input_id)
            input_id = replacements[input_id]
        return input_id

    for step in retained:
        step["input"] = resolve(step.get("input"))
    return retained


def promote_candidate_result(state, selected_candidate):
    """Promote a reviewed candidate's already-rendered artifacts to iteration root."""
    attempts = [item for item in state.get("candidate_attempts", []) if item.get("status") == "selected_for_review"]
    selected = next(
        (item for item in attempts if item.get("name") == selected_candidate),
        None,
    )
    if selected is None:
        return state
    iteration_dir = Path(state["run_dir"]) / f"iteration_{state.get('iteration', 0)}"
    candidate_dir = Path(selected["directory"])
    for source_name, target_name in (
        ("mask.png", "mask.png"),
        ("result_annotation.png", "result_annotated.png"),
        ("pipeline.json", "pipeline.json"),
        ("operator_trace.json", "operator_trace.json"),
        ("quality_report.json", "quality_report.json"),
        ("evaluation_report.json", "evaluation_report.json"),
        ("measurements.json", "measurements.json"),
        ("contours.json", "contours.json"),
    ):
        source = candidate_dir / source_name
        if source.exists():
            shutil.copy2(source, iteration_dir / target_name)
    updated = {
        **state,
        "selected_candidate": selected["name"],
        "pipeline": selected["pipeline"],
        "quality_report": selected.get("quality", {}),
        "evaluation_report": (selected.get("quality") or {}).get("evaluation"),
        "measurements": selected.get("measurements", state.get("measurements", {})),
        "annotated_image_path": str(iteration_dir / "result_annotated.png"),
        "predicted_mask_path": str(iteration_dir / "mask.png"),
        "pipeline_diff": _pipeline_diff(
            (state.get("previous_state") or {}).get("pipeline"),
            selected["pipeline"],
        ),
    }
    return updated


def _load_feedback_mask(path, shape):
    if not path or not Path(path).exists():
        return None
    mask = np.asarray(Image.open(path).convert("L")) > 0
    return mask if mask.shape == shape else None


def apply_user_constraints(predicted_mask, previous_state):
    """Apply explicit user include/exclude masks to a predicted mask.

    The masks are persisted feedback artifacts, so this remains deterministic
    even when a later pipeline revision does not reproduce the same regions.
    """
    mask = np.asarray(predicted_mask, dtype=bool).copy()
    if not isinstance(previous_state, dict):
        return mask, {}

    feedback = previous_state.get("human_feedback") or {}
    legacy_feedback = previous_state.get("feedback") or {}
    # Legacy false-negative/positive artifacts are the same user intent.
    include_path = (
        feedback.get("include_mask_path")
        or previous_state.get("false_negative_mask_path")
        or legacy_feedback.get("include_mask_path")
        or legacy_feedback.get("false_negative_mask_path")
    )
    exclude_path = (
        feedback.get("exclude_mask_path")
        or previous_state.get("false_positive_mask_path")
        or legacy_feedback.get("exclude_mask_path")
        or legacy_feedback.get("false_positive_mask_path")
    )
    report = {}

    include = _load_feedback_mask(include_path, mask.shape)
    if include is not None:
        added = include & ~mask
        mask |= include
        report["included_pixels"] = int(np.count_nonzero(added))
    exclude = _load_feedback_mask(exclude_path, mask.shape)
    if exclude is not None:
        removed = exclude & mask
        mask &= ~exclude
        report["excluded_pixels"] = int(np.count_nonzero(removed))
    return mask, report


def _apply_feedback_quality(quality, predicted_mask, previous_state):
    if not isinstance(previous_state, dict):
        return quality
    predicted = np.asarray(predicted_mask, dtype=bool)
    false_positive = _load_feedback_mask(
        previous_state.get("false_positive_mask_path"),
        predicted.shape,
    )
    false_negative = _load_feedback_mask(
        previous_state.get("false_negative_mask_path"),
        predicted.shape,
    )
    result = dict(quality)
    if false_positive is not None and np.any(false_positive):
        remaining = float(np.mean(predicted[false_positive]))
        result["false_positive_remaining"] = round(remaining, 6)
    if false_negative is not None and np.any(false_negative):
        recovered = float(np.mean(predicted[false_negative]))
        result["false_negative_recovered"] = round(recovered, 6)
    return result


def plan_candidate_definitions(
    understanding,
    previous_state=None,
    retrieved_algorithms=None,
    max_candidates=3,
):
    strategy = understanding.get("recommended_strategy") or {}
    candidates = _candidate_definitions(
        understanding,
        strategy,
        retrieved_algorithms=retrieved_algorithms,
    )
    previous_pipeline = (
        previous_state.get("pipeline")
        if isinstance(previous_state, dict)
        else None
    )
    if isinstance(previous_pipeline, dict) and previous_pipeline.get("steps"):
        candidates = [*candidates, {
            "name": "previous_pipeline_baseline",
            "hypothesis": "Replay the previous accepted working pipeline as an explicit baseline.",
            "pipeline": previous_pipeline,
            "source": {"type": "previous_iteration"},
        }]
    candidates = _deduplicate_candidates(candidates)
    # An accepted historical algorithm is itself a confirmed baseline; do not
    # dilute a one-semantic-candidate replay with speculative fallbacks.
    has_confirmed_history = any(
        isinstance(item, dict) and item.get("pipeline")
        for item in (retrieved_algorithms or [])
    )
    if int(max_candidates) >= 2 and len(candidates) < 2 and not has_confirmed_history:
        candidates = _deduplicate_candidates([*candidates, *_fallback_candidate_definitions(understanding)])
    return candidates[: max(1, int(max_candidates))]


def _pipeline_diff(previous, current):
    if not isinstance(previous, dict):
        return {"status": "initial", "changes": []}
    previous_steps = {step.get("id"): step for step in previous.get("steps", []) if isinstance(step, dict)}
    current_steps = {step.get("id"): step for step in current.get("steps", []) if isinstance(step, dict)}
    changes = []
    for step_id in sorted(set(previous_steps) | set(current_steps)):
        before = previous_steps.get(step_id)
        after = current_steps.get(step_id)
        if before is None:
            changes.append({"step": step_id, "change": "added", "after": after})
            continue
        if after is None:
            changes.append({"step": step_id, "change": "removed", "before": before})
            continue
        if before.get("op") != after.get("op"):
            changes.append({
                "step": step_id,
                "change": "operator_changed",
                "before": before.get("op"),
                "after": after.get("op"),
            })
        before_params = before.get("params") or {}
        after_params = after.get("params") or {}
        for key in sorted(set(before_params) | set(after_params)):
            if before_params.get(key) != after_params.get(key):
                changes.append({
                    "step": step_id,
                    "parameter": key,
                    "before": before_params.get(key),
                    "after": after_params.get(key),
                })
    return {
        "status": "unchanged" if not changes else "changed",
        "previous_pipeline": previous.get("name"),
        "current_pipeline": current.get("name"),
        "changes": changes,
    }


def _candidate_definitions(understanding, strategy, retrieved_algorithms=None):
    historical = []
    for item in retrieved_algorithms or []:
        if not isinstance(item, dict) or not isinstance(item.get("pipeline"), dict):
            continue
        historical.append({
            "name": f"accepted::{item.get('name') or item.get('algorithm_id') or 'historical'}",
            "hypothesis": (
                "Replay a user-accepted pipeline from a similar historical task "
                f"(match score {float(item.get('score', 0.0)):.3f})."
            ),
            "pipeline": item["pipeline"],
            "source": {
                "type": "accepted_algorithm",
                "algorithm_id": item.get("algorithm_id"),
                "source_task_id": item.get("source_task_id"),
                "match_score": item.get("score"),
                "match_reasons": item.get("match_reasons", []),
                "path": item.get("path"),
            },
        })
    candidates = understanding.get("candidate_pipelines")
    if isinstance(candidates, list) and candidates:
        generated = [item for item in candidates if isinstance(item, dict)]
        for item in generated:
            item.setdefault("source", {"type": "qwen"})
        return [*generated, *historical]
    return [*historical, *_fallback_candidate_definitions(understanding)]


def _fallback_candidate_definitions(understanding):
    """Supply bounded local baselines when the provider gives no executable plan."""
    strategy = understanding.get("recommended_strategy") or {}
    observation = strategy.get("visual_observation") or {}
    text = " ".join(str(value).lower() for value in (
        understanding.get("task_summary", ""), understanding.get("target_defect", ""),
        observation.get("background_pattern", ""),
    ))
    candidates = []
    if any(token in text for token in ("periodic", "周期", "repeat", "stripe", "line pattern")):
        candidates.append({
            "name": "periodic_particle_builtin",
            "hypothesis": "Use the validated periodic-background residual pipeline as a deterministic baseline.",
            "pipeline": {"name": "periodic_particle_builtin", "kind": "builtin_pipeline", "params": {"percentile": 97.0, "min_area": 20, "max_components": 3}},
            "source": {"type": "deterministic_fallback"},
        })
    segmentation = strategy.get("segmentation") or {}
    polarity = "dark" if segmentation.get("method") == "dark_threshold" else "bright"
    for candidate_polarity in (polarity, "bright" if polarity == "dark" else "dark"):
        fallback = {**strategy, "segmentation": {**segmentation, "method": f"{candidate_polarity}_threshold"}}
        candidates.append({
            "name": f"fallback_{candidate_polarity}_threshold",
            "hypothesis": f"Deterministic {candidate_polarity}-polarity threshold baseline.",
            "pipeline": strategy_to_pipeline(fallback, name=f"fallback_{candidate_polarity}_threshold"),
            "source": {"type": "deterministic_fallback"},
        })
    return candidates


def _deduplicate_candidates(candidates):
    unique = []
    seen = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        fingerprint = pipeline_fingerprint(candidate.get("pipeline"))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(candidate)
    return unique


def pipeline_fingerprint(pipeline):
    """Identify executable pipeline semantics while ignoring display names."""
    if not isinstance(pipeline, dict):
        return str(pipeline)
    generated = []
    for item in pipeline.get("generated_operators", []):
        if not isinstance(item, dict):
            continue
        generated.append({
            key: item.get(key)
            for key in ("name", "input_artifact", "output_artifact", "atomic", "source")
        })
    payload = {
        "kind": pipeline.get("kind"),
        "builtin_params": pipeline.get("params") if pipeline.get("kind") == "builtin_pipeline" else None,
        "steps": pipeline.get("steps", []),
        "generated_operators": generated,
    }
    try:
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(payload)


def _serializable_attempt(attempt):
    result = {
        "index": attempt["index"],
        "name": attempt["name"],
        "hypothesis": attempt["hypothesis"],
        "source": attempt.get("source", {"type": "qwen"}),
        "status": attempt["status"],
        "failure_type": attempt.get("failure_type"),
        "pipeline": attempt["pipeline"],
        "quality": attempt["quality"],
        "measurements": attempt.get("measurements", {}),
        "directory": attempt["directory"],
    }
    execution = attempt.get("execution")
    if execution is not None:
        result["operator_trace"] = list(execution.trace)
    return result


def _serializable_retrieved_algorithm(item):
    return {
        "algorithm_id": item.get("algorithm_id"),
        "name": item.get("name"),
        "score": item.get("score"),
        "match_reasons": item.get("match_reasons", []),
        "source_task_id": item.get("source_task_id"),
        "path": item.get("path"),
    }


def _serialize_contours(contours):
    return {
        "image_shape": list(contours.image_shape),
        "count": len(contours.contours),
        "contours": [contour.tolist() for contour in contours.contours],
        "metadata": contours.metadata,
    }


def _make_run_dir(output_root):
    run_dir = Path(output_root) / f"{datetime.now():%Y%m%d_%H%M%S_%f}_agent_v2"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
