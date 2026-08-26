#!/usr/bin/env python3
"""Import explicitly accepted historical task pipelines into the registry.

This is intentionally conservative: ordinary task outputs and historical
``completed`` states are not enough. A task must have been explicitly accepted
and must contain a complete, schema-valid pipeline in its acceptance record.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.algorithm_registry import AlgorithmRegistry
from core.pipelines.dsl import normalize_pipeline, validate_pipeline


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def _pipeline_fingerprint(pipeline):
    return json.dumps(pipeline, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def migrate_accepted_algorithms(tasks_root, algorithm_root, *, dry_run=False):
    """Migrate accepted task records and return a deterministic import report."""
    tasks_root = Path(tasks_root)
    registry = AlgorithmRegistry(algorithm_root)
    existing = registry.list_algorithms(limit=100000)
    existing_tasks = {str(item.get("source_task_id")) for item in existing}
    existing_pipelines = {
        _pipeline_fingerprint(item.get("pipeline"))
        for item in existing
        if item.get("pipeline")
    }
    report = {"imported": [], "skipped": [], "dry_run": bool(dry_run)}

    for acceptance_path in sorted(tasks_root.glob("task_*/acceptance/algorithm.json")):
        task_dir = acceptance_path.parent.parent
        task_id = task_dir.name
        task = _read_json(task_dir / "task.json")
        latest = _read_json(task_dir / "acceptance/latest.json")
        algorithm = _read_json(acceptance_path)
        reason = None
        if not isinstance(task, dict) or task.get("status") != "accepted":
            reason = "task_not_explicitly_accepted"
        elif not isinstance(latest, dict) or not latest.get("accepted_at"):
            reason = "missing_acceptance_record"
        elif not isinstance(algorithm, dict):
            reason = "invalid_algorithm_record"
        elif _acceptance_quality_failed(latest):
            reason = "acceptance_quality_failed"
        elif task_id in existing_tasks:
            reason = "already_imported_task"
        else:
            raw_pipeline = algorithm.get("pipeline")
            try:
                pipeline = normalize_pipeline(raw_pipeline, name=algorithm.get("name") or task_id)
                validate_pipeline(pipeline)
            except (TypeError, ValueError) as exc:
                reason = f"invalid_pipeline:{exc}"
            else:
                if _pipeline_fingerprint(pipeline) in existing_pipelines:
                    reason = "duplicate_pipeline"

        if reason:
            report["skipped"].append({"task_id": task_id, "reason": reason})
            continue

        # Keep the historical acceptance payload as the source of truth while
        # supplying the state fields expected by AlgorithmRegistry.publish.
        state = {
            "selected_candidate": algorithm.get("name") or task_id,
            "description": algorithm.get("description") or task.get("title", ""),
            "strategy": algorithm.get("strategy") or {},
            "pipeline": pipeline,
            "understanding": {
                "output_requirements": algorithm.get("output_requirements") or [],
            },
            "rendering": algorithm.get("rendering") or {},
            "quality_report": latest.get("quality_report") or {},
            "measurements": {"summary": latest.get("measurement_summary") or {}},
            "run_dir": latest.get("run_dir"),
            "annotated_image_path": latest.get("annotated_image_path"),
            "predicted_mask_path": latest.get("predicted_mask_path"),
        }
        if dry_run:
            report["imported"].append({"task_id": task_id, "name": state["selected_candidate"], "dry_run": True})
            continue
        published = registry.publish(task_id, state, note=latest.get("note") or "历史用户确认算法迁移")
        existing_tasks.add(task_id)
        existing_pipelines.add(_pipeline_fingerprint(pipeline))
        report["imported"].append({
            "task_id": task_id,
            "name": state["selected_candidate"],
            "algorithm_id": published["id"],
            "path": published["path"],
        })
    return report


def _acceptance_quality_failed(latest):
    """Reject records that explicitly describe an empty or unhealthy result.

    Older acceptance records may not contain a quality status, so missing
    fields remain eligible when the user did explicitly accept a pipeline.
    """
    quality = latest.get("quality_report")
    if not isinstance(quality, dict):
        return False
    if str(quality.get("status", "")).lower() in {"failed", "health_failed", "no_annotation"}:
        return True
    issues = {str(item) for item in quality.get("issues", []) if item is not None}
    return bool(issues & {"empty_mask", "empty_annotation", "mask_health_failed"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-root", default="workspace/tasks")
    parser.add_argument("--algorithm-root", default="workspace/algorithms")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = migrate_accepted_algorithms(
        args.tasks_root,
        args.algorithm_root,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
