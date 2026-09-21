"""Single-candidate execution shared by workflow orchestration and Agent Tools."""

import json
import shutil
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image

from core.experiments.delivery import finalize_mask_outputs, write_outputs, write_manifest
from core.input_contract import display_image, load_pixels, input_identity, input_metadata
from core.runtime_logging import logger
from core.agent_events import emit_thinking, emit_tool_call, emit_tool_result
from core.experiments.artifacts import persist_artifacts, export_artifacts
from core.experiments.serialization import to_jsonable
from core.measurement.area import measure_components
from core.measurement.evaluation import evaluate_prediction
from core.operators import ContourArtifact, MaskArtifact
from core.pipelines.dsl import PipelineExecutionResult, normalize_pipeline, pin_pipeline_operator_versions
from core.quality import evaluate_mask_quality, inspect_mask_health
from core.sandbox import execute_pipeline_sandbox

from core.visualization import save_annotated_image, save_mask_image


# Artifacts copied from a replayed experiment so the new attempt directory is
# self-contained and promote_candidate_result can promote it like any other.
_REUSE_COPY_FILES = (
    'mask.png', 'outputs.json', 'raw_outputs.json', 'execution_spec.json',
    'result_annotation.png', 'operator_trace.json', 'quality_report.json',
    'evaluation_report.json', 'measurements.json', 'contours.json',
)


def run_candidate(candidate, image_path, image, candidate_dir, *, index=0,
                  previous_state=None, rendering=None, target_constraints=None,
                  ground_truth_mask=None, unit="pixel", executed_fingerprints=None):
    if previous_state and previous_state.get('input_sha256') and previous_state['input_sha256'] != input_identity(image_path)['input_sha256']:
        previous_state = None
    rendering = rendering or {}
    target_constraints = target_constraints or {}
    executed_fingerprints = executed_fingerprints if executed_fingerprints is not None else set()
    attempts = []
    candidate_name = candidate.get("name") or f"candidate_{index + 1}"
    emit_thinking(f"正在执行实验: {candidate_name}", "execute_candidate")

    candidate_dir = Path(candidate_dir)
    candidate_dir.mkdir()
    raw_pipeline = candidate.get("pipeline")
    try:
        pipeline = normalize_pipeline(raw_pipeline, name=candidate_name)
        pipeline = pin_pipeline_operator_versions(pipeline)
    except Exception as exc:
        logger.warning("Candidate execution failed", exc_info=True)
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
        return attempts[0]
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
        return attempts[0]
    executed_fingerprints.add(fingerprint)

    replayed = _replay_submitted_execution(
        candidate, candidate_name, pipeline, index, image_path, image, candidate_dir)
    if replayed is not None:
        return replayed

    emit_tool_call("execute_pipeline_sandbox", {"pipeline": candidate_name, "steps": len(pipeline.get("steps", []))})

    try:
        external = {}
        for input_name, kind in pipeline.get("input_types", {}).items():
            if input_name == "$rgb" and kind == "ImageArtifact":
                external[input_name] = np.asarray(Image.open(image_path).convert("RGB"))
            else:
                available = (previous_state or {}).get("algorithm_inputs", {})
                if input_name not in available:
                    raise ValueError(f"external input is unavailable: {input_name}")
                external[input_name] = available[input_name]
        execution = execute_pipeline_sandbox(image, pipeline, inputs=external) if external else execute_pipeline_sandbox(image, pipeline)
        emit_tool_result("execute_pipeline_sandbox", "流水线执行成功", success=True)

        artifact_records = persist_artifacts(execution, candidate_dir)
        _write_json(candidate_dir / 'execution_spec.json', {
            'schema_version': 1, **input_metadata(image_path), 'unit': unit,
            'rendering': rendering, 'target_constraints': target_constraints,
            'ground_truth_sha256': sha256(np.asarray(ground_truth_mask, dtype=bool).tobytes()).hexdigest() if ground_truth_mask is not None else None,
            'feedback': {key: (previous_state or {}).get(key) for key in ('human_feedback', 'include_mask_path', 'exclude_mask_path')},
        })
        # Reports also flow into tool replies and graph state, which require JSON values.
        write_outputs(execution, candidate_dir, "raw_outputs.json")
        output_data = write_outputs(execution, candidate_dir)
        if execution.mask is None:
            # Structured/image results are valid results, not empty segmentations.
            preview = display_image(load_pixels(image_path))
            from PIL import ImageDraw
            from core.operators import ImageArtifact, MetadataArtifact
            draw = ImageDraw.Draw(preview)
            for value in sorted(execution.outputs.values(), key=lambda item: 0 if isinstance(item, ImageArtifact) else 1):
                if isinstance(value, ImageArtifact):
                    preview = display_image(value.data, value.metadata)
                    draw = ImageDraw.Draw(preview)
                elif isinstance(value, MetadataArtifact):
                    for box in value.data.get("boxes", []):
                        draw.rectangle(tuple(box), outline=rendering.get("contour_color", "#ff4030"), width=int(rendering.get("contour_thickness", 2)))
                    for x, y in value.data.get("points", []):
                        draw.ellipse((x-3, y-3, x+3, y+3), fill=rendering.get("contour_color", "#ff4030"))
            preview.save(candidate_dir / "result_annotation.png")
            quality = {"issues": [], "output_kind": "structured", "outputs": output_data,
                       "health": {"usable_for_review": True, "issues": []},
                       "note": "No segmentation mask; pixel metrics and brush constraints are not applicable."}
            measurements = {"results": [], "summary": {"count": 0, "total_area": 0, "unit": unit},
                            "structured_outputs": output_data}
            _write_json(candidate_dir / "quality_report.json", quality)
            _write_json(candidate_dir / "measurements.json", measurements)
            _write_json(candidate_dir / "operator_trace.json", list(execution.trace))
            _write_json(candidate_dir / "contours.json", _serialize_contours(execution.contours))
            write_manifest(candidate_dir, image_path, rendering)
            return {"index": index, "name": candidate_name, "hypothesis": candidate.get("hypothesis", ""),
                    "source": candidate.get("source", {"type": "qwen"}), "status": "selected_for_review",
                    "pipeline": pipeline, "execution": execution, "artifacts": artifact_records,
                    "quality": quality, "measurements": measurements, "directory": str(candidate_dir)}
        emit_thinking("应用用户约束", "apply_constraints")
        constrained_mask, constraint_report = apply_user_constraints(
            execution.mask.data, previous_state,
        )
        invalidated_outputs = finalize_mask_outputs(execution, constrained_mask)
        output_data = write_outputs(execution, candidate_dir)

        emit_thinking("评估质量", "evaluate_quality")
        quality = evaluate_mask_quality(constrained_mask)
        if output_data:
            quality["outputs"] = output_data
        quality["user_constraints"] = constraint_report
        quality["invalidated_outputs"] = invalidated_outputs
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
        if output_data:
            measurements["structured_outputs"] = output_data

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
        write_manifest(candidate_dir, image_path, rendering, invalidated_outputs)
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
            "artifacts": artifact_records,
            "quality": quality,
            "measurements": measurements,
            "directory": str(candidate_dir),
        })
    except Exception as exc:
        logger.warning("Candidate execution failed", exc_info=True)
        emit_tool_result("execute_pipeline_sandbox", f"{type(exc).__name__}: {exc}", success=False)
        issue = getattr(exc, "code", "execution_failed")
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

    return attempts[0]


def _replay_submitted_execution(candidate, candidate_name, pipeline, index,
                                image_path, image, candidate_dir):
    """Replay a submitted experiment's persisted artifacts instead of re-executing.

    The understanding phase already executed the submitted pipeline in the
    sandbox; the graph's execute node replays those artifacts so the identical
    pipeline does not run twice. Reuse requires the identical input image, an
    identical pipeline fingerprint, and a completed reviewable record — anything
    else falls back to a normal sandbox execution.
    """
    reuse = candidate.get('reused_execution') if isinstance(candidate, dict) else None
    if not isinstance(reuse, dict):
        return None
    directory = Path(str(reuse.get('directory') or ''))
    if not directory.is_dir():
        return None
    try:
        record = json.loads((directory / 'experiment.json').read_text(encoding='utf-8'))
        stored_pipeline = json.loads((directory / 'pipeline.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if record.get('input_sha256') != input_identity(image_path)['input_sha256']:
        return None
    if (record.get('execution_status') != 'completed'
            or record.get('status') not in {'selected_for_review', 'completed'}
            or record.get('acceptance_status') == 'rejected'):
        return None
    if pipeline_fingerprint(stored_pipeline) != pipeline_fingerprint(pipeline):
        return None
    for file_name in ('quality_report.json', 'measurements.json', 'operator_trace.json'):
        if not (directory / file_name).is_file():
            return None
    try:
        trace = json.loads((directory / 'operator_trace.json').read_text(encoding='utf-8'))
        quality = json.loads((directory / 'quality_report.json').read_text(encoding='utf-8'))
        measurements = json.loads((directory / 'measurements.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if not isinstance(trace, list) or not all(isinstance(item, dict) for item in trace):
        return None
    mask = None
    contours = None
    mask_path = directory / 'mask.png'
    shape = tuple(np.asarray(image).shape[:2])
    if mask_path.is_file():
        contours_path = directory / 'contours.json'
        if not contours_path.is_file():
            return None
        with Image.open(mask_path) as stored_mask:
            mask = np.asarray(stored_mask.convert('L')) > 0
        if mask.ndim != 2 or tuple(mask.shape) != shape:
            return None
        try:
            stored_contours = json.loads(contours_path.read_text(encoding='utf-8'))
            if not isinstance(stored_contours, dict):
                return None
            contours = ContourArtifact(
                tuple(np.asarray(item, dtype=np.float32) for item in stored_contours.get('contours') or []),
                tuple(stored_contours.get('image_shape') or shape),
                metadata=stored_contours.get('metadata') or {},
            )
        except (OSError, ValueError, TypeError):
            return None
        if tuple(contours.image_shape) != shape or any(
                not np.isfinite(item).all() for item in contours.contours):
            return None
    for file_name in _REUSE_COPY_FILES:
        source = directory / file_name
        if source.exists():
            shutil.copy2(source, candidate_dir / file_name)
    execution = PipelineExecutionResult(
        pipeline={}, mask=None if mask is None else MaskArtifact(mask, metadata={}),
        contours=contours, trace=tuple(trace), artifacts={}, outputs={})
    emit_tool_call('reuse_submitted_execution', {'experiment_id': reuse.get('experiment_id')})
    emit_tool_result('reuse_submitted_execution', '复用提交实验的执行产物，未重复执行沙箱', success=True)
    logger.info('Replayed submitted execution experiment_id=%s directory=%s',
                reuse.get('experiment_id'), directory)
    return {
        'index': index,
        'name': candidate_name,
        'hypothesis': candidate.get('hypothesis', ''),
        'source': candidate.get('source', {'type': 'qwen'}),
        'status': record.get('status') or 'selected_for_review',
        'pipeline': pipeline,
        'execution': execution,
        'artifacts': record.get('artifacts') or [],
        'quality': quality,
        'measurements': measurements,
        'directory': str(candidate_dir),
        'reused_from_experiment': reuse.get('experiment_id'),
    }


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
            for key in ("name", "input_artifact", "input_ports", "output_artifact", "atomic", "source")
        })
    payload = {
        "kind": pipeline.get("kind"),
        "builtin_params": pipeline.get("params") if pipeline.get("kind") == "builtin_pipeline" else None,
        "schema_version": pipeline.get("schema_version"),
        "input_types": pipeline.get("input_types", {}),
        "steps": pipeline.get("steps", []),
        "nodes": pipeline.get("nodes", []),
        "outputs": pipeline.get("outputs", {}),
        "generated_operators": generated,
    }
    try:
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(payload)


def _serializable_attempt(attempt):
    result = {
        "experiment_id": attempt.get("experiment_id"),
        "artifacts": attempt.get("artifacts", []),
        "index": attempt["index"],
        "name": attempt["name"],
        "hypothesis": attempt["hypothesis"],
        **{key: attempt[key] for key in (
            'change_reason', 'expected_change', 'parent_experiment_id', 'algorithm_version',
            'acceptance_status', 'review', 'scope', 'reused_from_experiment',
        ) if key in attempt},
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
    return to_jsonable(result)


def _serialize_contours(contours):
    if contours is None:
        return None
    return {
        "image_shape": list(contours.image_shape),
        "count": len(contours.contours),
        "contours": [contour.tolist() for contour in contours.contours],
        "metadata": contours.metadata,
    }


def _write_json(path, payload):
    Path(path).write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
