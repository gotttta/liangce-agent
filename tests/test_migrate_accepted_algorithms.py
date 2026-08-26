import json

from scripts.migrate_accepted_algorithms import migrate_accepted_algorithms


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _pipeline():
    return {
        "name": "accepted_threshold",
        "steps": [
            {"id": "final_mask", "op": "global_threshold", "input": "image", "params": {"polarity": "bright"}},
        ],
    }


def test_migration_requires_explicit_acceptance_and_is_idempotent(tmp_path):
    tasks = tmp_path / "tasks"
    algorithms = tmp_path / "algorithms"
    accepted = tasks / "task_accepted"
    _write(accepted / "task.json", {"id": "task_accepted", "status": "accepted", "title": "accepted"})
    _write(accepted / "acceptance/latest.json", {
        "accepted_at": "2026-08-20T00:00:00Z",
        "note": "用户确认",
        "quality_report": {"issues": []},
    })
    _write(accepted / "acceptance/algorithm.json", {
        "name": "accepted_threshold",
        "description": "历史确认算法",
        "pipeline": _pipeline(),
        "strategy": {"defect_type": "particle"},
    })

    unaccepted = tasks / "task_unaccepted"
    _write(unaccepted / "task.json", {"id": "task_unaccepted", "status": "completed"})
    _write(unaccepted / "acceptance/latest.json", {"accepted_at": "2026-08-20T00:00:00Z"})
    _write(unaccepted / "acceptance/algorithm.json", {"pipeline": _pipeline()})

    first = migrate_accepted_algorithms(tasks, algorithms)
    assert len(first["imported"]) == 1
    assert first["skipped"] == [{"task_id": "task_unaccepted", "reason": "task_not_explicitly_accepted"}]
    second = migrate_accepted_algorithms(tasks, algorithms)
    assert second["imported"] == []
    assert {item["reason"] for item in second["skipped"]} == {"already_imported_task", "task_not_explicitly_accepted"}


def test_migration_dry_run_does_not_write_registry(tmp_path):
    task = tmp_path / "tasks/task_accepted"
    _write(task / "task.json", {"id": "task_accepted", "status": "accepted"})
    _write(task / "acceptance/latest.json", {"accepted_at": "2026-08-20T00:00:00Z"})
    _write(task / "acceptance/algorithm.json", {"name": "x", "pipeline": _pipeline()})
    report = migrate_accepted_algorithms(tmp_path / "tasks", tmp_path / "algorithms", dry_run=True)
    assert report["imported"][0]["dry_run"] is True
    assert list((tmp_path / "algorithms").glob("algorithm_*/algorithm.json")) == []


def test_migration_skips_failed_acceptance_quality(tmp_path):
    task = tmp_path / "tasks/task_failed"
    _write(task / "task.json", {"id": "task_failed", "status": "accepted"})
    _write(task / "acceptance/latest.json", {
        "accepted_at": "2026-08-20T00:00:00Z",
        "quality_report": {"status": "failed", "issues": ["empty_mask"]},
    })
    _write(task / "acceptance/algorithm.json", {"name": "x", "pipeline": _pipeline()})
    report = migrate_accepted_algorithms(tmp_path / "tasks", tmp_path / "algorithms")
    assert report["imported"] == []
    assert report["skipped"][0]["reason"] == "acceptance_quality_failed"
