from core.experiments.runner import (
    run_candidate, apply_user_constraints, pipeline_fingerprint,
    _serializable_attempt, _write_json, _apply_feedback_quality, _serialize_contours,
)
from datetime import datetime
from copy import deepcopy
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from core.experiments.artifacts import record_experiments
from core.agent_events import emit_thinking
from core.measurement.area import measure_components
from core.measurement.evaluation import meets_ground_truth_gate
from core.pipelines.dsl import normalize_pipeline, strategy_to_pipeline
from core.preprocessing import load_grayscale
from core.runtime_metadata import runtime_metadata
from core.pipelines.periodic_template import periodic_segmentation_template
from core.visualization import save_annotated_image, save_mask_image


def _area_unit_label(unit):
    """把内部单位名转成面向用户的显示单位。"""
    return "px²" if str(unit or "").lower() in {"pixel", "px", "px2"} else unit


def run_planned_agent(
    target_image_path,
    description,
    understanding,
    output_root="outputs",
    unit="pixel",
    max_candidates=1,
    previous_state=None,
    retrieved_algorithms=None,
    planned_candidates=None,
    run_dir=None,
    return_failure_state=False,
    ground_truth_mask_path=None,
    max_calibration_candidates=0,
    task_contract=None,
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
    _write_json(iteration_dir / "runtime_environment.json", runtime_metadata())
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
    candidates = _fit_candidate_budget(candidates, max_candidates)
    rendering = understanding.get("rendering") or {}
    target_constraints = understanding.get("target_constraints") or {}
    ground_truth_mask = _load_ground_truth_mask(ground_truth_mask_path, image.shape)
    budget = {
        "max_candidates": max(1, int(max_candidates)),
        "max_calibration_candidates": max(0, int(max_calibration_candidates)),
        "planned_count": len(candidates),
    }
    attempts = []
    executed_fingerprints = set()

    def pending_candidates():
        yield from candidates
        # Calibration is only planned after the ordinary candidates fall short.
        usable = [item for item in attempts if item["status"] == "selected_for_review"]
        if ground_truth_mask is not None and usable and int(max_calibration_candidates) > 0:
            best = select_best_candidate(usable, prefer_evaluation=True)
            variants = _ground_truth_calibration_candidates(
                [best], ground_truth_mask, limit=int(max_calibration_candidates),
            )
            budget["planned_count"] += len(variants)
            yield from variants

    for index, candidate in enumerate(pending_candidates()):
        attempt = run_candidate(
            candidate, image_path, image, iteration_dir / f"candidate_{index}",
            index=index, previous_state=previous_state, rendering=rendering,
            target_constraints=target_constraints, ground_truth_mask=ground_truth_mask,
            unit=unit, executed_fingerprints=executed_fingerprints,
        )
        attempt['change_reason'] = candidate.get('change_reason') or candidate.get('hypothesis', '')
        attempt['expected_change'] = candidate.get('expected_change', '')
        attempts.append(attempt)
        if meets_ground_truth_gate(attempt["quality"].get("evaluation")):
            break

    from core.experiments.artifacts import experiment_scope
    scope = experiment_scope(image_path, task_contract, {
        **(previous_state or {}), 'ground_truth_mask_path': ground_truth_mask_path,
    })
    record_experiments(attempts, image_path, previous_state, iteration, scope=scope)
    budget["attempted_count"] = len(attempts)
    budget["stopped_on_gate"] = any(
        meets_ground_truth_gate(item.get("quality", {}).get("evaluation")) for item in attempts
    )
    _write_json(iteration_dir / "candidate_budget.json", budget)
    completed = [attempt for attempt in attempts if attempt["status"] == "selected_for_review"]
    if not completed:
        emit_thinking("实验未生成可用结果，保留错误记录供复查", "all_failed")
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
    selected = completed[0] if len(completed) == 1 else select_best_candidate(completed, prefer_evaluation=ground_truth_mask is not None)
    emit_thinking(f"实验执行完成，等待复查: {selected['name']}", "experiment_completed")
    execution = selected["execution"]
    measurements = selected["measurements"]

    emit_thinking("保存最终结果", "save_final_results")
    mask_path = iteration_dir / "mask.png"
    result_path = iteration_dir / "result_annotated.png"
    if execution.mask is None:
        shutil.copy2(Path(selected["directory"]) / "result_annotation.png", result_path)
        shutil.copy2(Path(selected["directory"]) / "outputs.json", iteration_dir / "outputs.json")
    else:
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
    if (Path(selected["directory"]) / "outputs.json").exists():
        shutil.copy2(Path(selected["directory"]) / "outputs.json", iteration_dir / "outputs.json")
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
    from core.input_contract import input_identity
    state = {
        **input_identity(image_path),
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
        "selected_experiment_id": selected.get("experiment_id"),
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
        "predicted_mask_path": str(mask_path) if execution.mask is not None else None,
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
                    "Agent 已生成并执行识别算法。"
                    if int(max_candidates) == 1
                    else f"已执行并比较 {len(attempts)} 个候选，保留了效果最好的一版。"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "本轮已生成结构化结果，详见量测数据。" if execution.mask is None else
                    f"本轮标出 {summary['count']} 个区域，总面积 "
                    f"{summary['total_area']} {_area_unit_label(summary['unit'])}。"
                ),
            },
        ],
    }
    state = promote_candidate_result(state, state.get("selected_candidate"))
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
        "这次实验没有标出目标，需要检查中间结果并定位原因。"
        if had_empty_mask
        else (
            "这次实验产生了不可靠的标注范围，需要检查相关处理步骤。"
            if had_health_failure else "这次实验没有运行成功，需要根据执行记录修复问题。"
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
        "selected_experiment_id": last_attempt.get("experiment_id"),
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
        return (meets_ground_truth_gate(evaluation),) + tuple(float(evaluation.get(key, -1.0)) for key in (
            "dice", "boundary_f1", "recall", "precision", "iou",
        ))

    return max(attempts, key=rank)


def _ground_truth_calibration_candidates(candidates, ground_truth_mask, limit=6):
    """Create a bounded local parameter search around a residual pipeline.

    Ground Truth is used only to set aggregate component/area constraints and
    rank the executed variants. Pixel coordinates are never copied into the
    generated algorithm, so the resulting Pipeline remains replayable.
    """
    if int(limit) <= 0:
        return []
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
            existing = next((step for step in steps if step.get("id") == filter_step.get("input")
                             and step.get("op") == "morphology"
                             and step.get("id", "").startswith("gt_calibration_dilate")
                             and step.get("params", {}).get("method") == "dilate"), None)
            dilate_id = existing["id"] if existing else "gt_calibration_dilate"
            used_ids = {step["id"] for step in steps}
            suffix = 1
            while existing is None and dilate_id in used_ids:
                dilate_id = f"gt_calibration_dilate_{suffix}"
                suffix += 1
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
            if existing:
                existing["params"] = {**existing.get("params", {}), "radius": radius}
            else:
                steps.insert(filter_index, dilate_step)
            if sigma is not None:
                for step in steps:
                    if step.get("op") == "local_background_residual":
                        step["params"] = {**(step.get("params") or {}), "sigma": round(sigma, 3)}
            pipeline["steps"] = steps
            # This is a new graph derived from the baseline, with different operators.
            pipeline.pop("operator_versions", None)
            pipeline.pop("version_provenance", None)
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
        ("outputs.json", "outputs.json"),
        ("raw_outputs.json", "raw_outputs.json"),
        ("execution_spec.json", "execution_spec.json"),
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
        else:
            (iteration_dir / target_name).unlink(missing_ok=True)
    updated = {
        **state,
        "selected_candidate": selected["name"],
        "selected_experiment_id": selected.get("experiment_id"),
        "pipeline": selected["pipeline"],
        "quality_report": selected.get("quality", {}),
        "evaluation_report": (selected.get("quality") or {}).get("evaluation"),
        "measurements": selected.get("measurements", state.get("measurements", {})),
        "annotated_image_path": str(iteration_dir / "result_annotated.png"),
        "predicted_mask_path": str(iteration_dir / "mask.png") if (candidate_dir / "mask.png").exists() else None,
        "contours_path": str(iteration_dir / "contours.json") if (candidate_dir / "contours.json").exists() else None,
        "pipeline_diff": _pipeline_diff(
            (state.get("previous_state") or {}).get("pipeline"),
            selected["pipeline"],
        ),
    }
    summary = updated['measurements'].get('summary') or {}
    updated['conversation'] = [
        {'role': 'assistant', 'content': (
            f"已执行实验 {selected['name']}，验收结论见复查记录。" if len(attempts) == 1
            else f"已执行并比较 {len(attempts)} 个方案，选定 {selected['name']}。")},
        {'role': 'assistant', 'content': (
            f"本轮标出 {summary.get('count', 0)} 个区域，总面积 {summary.get('total_area', 0)} {_area_unit_label(summary.get('unit'))}。"
            if updated.get('predicted_mask_path') else '本轮已生成结构化结果，详见量测数据。')},
    ]
    from core.experiments.delivery import write_manifest
    if state.get('target_image_path'):
        manifest = write_manifest(iteration_dir, state['target_image_path'], state.get('rendering') or {},
                                 updated['quality_report'].get('invalidated_outputs') or [])
    _write_json(iteration_dir / 'graph_state.json', {
        key: value for key, value in updated.items()
        if key not in {'__interrupt__', 'previous_state', 'planned_candidates', 'output_root'}
    })
    return updated








def plan_candidate_definitions(
    understanding,
    previous_state=None,
    retrieved_algorithms=None,
    max_candidates=1,
):
    strategy = understanding.get("recommended_strategy") or {}
    candidates = _candidate_definitions(
        understanding,
        strategy,
        retrieved_algorithms=retrieved_algorithms,
    )
    # History is evidence for revision/optional comparison, never another
    # mandatory execution. A candidate limit is an upper bound, not a quota.
    candidates = _deduplicate_candidates(candidates)
    return _fit_candidate_budget(candidates, max_candidates)


def _fit_candidate_budget(candidates, max_candidates):
    """An explicit caller may raise the limit; no baseline slot is reserved.

    The model's executed-and-submitted experiment ranks first: it is evidence
    for this exact input, and dropping it would both waste the understanding
    phase's sandbox run and trigger a duplicate execution. Historical replays
    come next; fresh proposals last.
    """
    limit = max(1, int(max_candidates))

    def priority(item):
        if item.get("reused_execution"):
            return 0
        if (item.get("source") or {}).get("type") == "accepted_algorithm":
            return 1
        return 2

    candidates = sorted(candidates, key=priority)
    return candidates[:limit]


def _pipeline_diff(previous, current):
    if not isinstance(previous, dict):
        return {"status": "initial", "changes": []}
    previous_steps = {
        step.get("id"): step
        for step in (previous.get("nodes") if previous.get("nodes") is not None else previous.get("steps", []))
        if isinstance(step, dict)
    }
    current_steps = {
        step.get("id"): step
        for step in (current.get("nodes") if current.get("nodes") is not None else current.get("steps", []))
        if isinstance(step, dict)
    }
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
        before_operator = before.get("operator") or before.get("tool") or before.get("op")
        after_operator = after.get("operator") or after.get("tool") or after.get("op")
        if before_operator != after_operator:
            changes.append({
                "step": step_id,
                "change": "operator_changed",
                "before": before_operator,
                "after": after_operator,
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
            "name": "periodic_segmentation_baseline",
            "hypothesis": "Run the periodic-background segmentation baseline.",
            "pipeline": periodic_segmentation_template(),
            "source": {"type": "deterministic_fallback", "baseline": "periodic_segmentation", "version": "1.0.0"},
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






def _serializable_retrieved_algorithm(item):
    return {
        "algorithm_id": item.get("algorithm_id"),
        "name": item.get("name"),
        "score": item.get("score"),
        "match_reasons": item.get("match_reasons", []),
        "source_task_id": item.get("source_task_id"),
        "path": item.get("path"),
    }




def _make_run_dir(output_root):
    run_dir = Path(output_root) / f"{datetime.now():%Y%m%d_%H%M%S_%f}_agent_v2"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
