from datetime import datetime

from core.iteration_tracker import IterationTracker
from core.scoring import ImageScore, RunScore


def _run_score(composite_mean: float, image_id: str = "img_a") -> RunScore:
    return RunScore(
        image_scores=[
            ImageScore(
                image_id=image_id,
                iou_mean=composite_mean,
                false_positive_count=0,
                false_negative_count=0,
                ref_count=1,
                composite=composite_mean,
            )
        ],
        composite_mean=composite_mean,
    )


def _add(tracker: IterationTracker, composite_mean: float) -> None:
    tracker.record(_run_score(composite_mean), {"pipeline": [f"step_{composite_mean}"]})


def test_record_appends_and_assigns_iteration_numbers(tmp_path):
    tracker = IterationTracker(tmp_path / "run")
    first = tracker.record(_run_score(0.5), {"pipeline": ["a"]})
    second = tracker.record(_run_score(0.6), {"pipeline": ["b"]})

    assert first.iteration == 0
    assert second.iteration == 1
    assert len(tracker.history) == 2
    assert [record.iteration for record in tracker.history] == [0, 1]


def test_record_timestamp_is_utc_iso8601(tmp_path):
    tracker = IterationTracker(tmp_path / "run")
    record = tracker.record(_run_score(0.5), {"pipeline": []})
    assert record.timestamp.endswith("Z")
    parsed = datetime.fromisoformat(record.timestamp.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None


def test_best_returns_highest_composite_mean(tmp_path):
    tracker = IterationTracker(tmp_path / "run")
    _add(tracker, 0.4)
    _add(tracker, 0.9)
    _add(tracker, 0.6)

    assert tracker.best is not None
    assert tracker.best.iteration == 1
    assert tracker.best.run_score.composite_mean == 0.9


def test_best_returns_none_when_empty(tmp_path):
    tracker = IterationTracker(tmp_path / "run")
    assert tracker.best is None


def test_should_stop_target_reached(tmp_path):
    tracker = IterationTracker(tmp_path / "run")
    _add(tracker, 0.5)
    _add(tracker, 0.87)

    assert tracker.should_stop(target=0.85) == (True, "target_reached")


def test_should_stop_max_iterations(tmp_path):
    tracker = IterationTracker(tmp_path / "run")
    for i in range(3):
        _add(tracker, 0.1 * (i + 1))

    assert tracker.should_stop(max_iter=3) == (True, "max_iterations")


def test_should_stop_target_takes_priority_over_max_iterations(tmp_path):
    tracker = IterationTracker(tmp_path / "run")
    for composite in (0.3, 0.5, 0.9):
        _add(tracker, composite)

    assert tracker.should_stop(target=0.85, max_iter=3) == (True, "target_reached")


def test_should_stop_no_improvement(tmp_path):
    tracker = IterationTracker(tmp_path / "run")
    for composite in (0.5, 0.503, 0.506, 0.508):
        _add(tracker, composite)

    assert tracker.should_stop(patience=4) == (True, "no_improvement")


def test_should_stop_no_improvement_also_triggered_on_consistent_decline(tmp_path):
    tracker = IterationTracker(tmp_path / "run")
    for composite in (0.5, 0.495, 0.490, 0.488):   # 持续小幅下降
        _add(tracker, composite)
    assert tracker.should_stop(patience=4) == (True, "no_improvement")


def test_should_stop_no_improvement_not_triggered_with_fewer_records(tmp_path):
    tracker = IterationTracker(tmp_path / "run")
    _add(tracker, 0.5)
    _add(tracker, 0.503)

    assert tracker.should_stop(patience=4) == (False, "")


def test_should_stop_no_improvement_ignored_when_recent_jump(tmp_path):
    tracker = IterationTracker(tmp_path / "run")
    for composite in (0.5, 0.503, 0.55, 0.56):
        _add(tracker, composite)

    assert tracker.should_stop(patience=4) == (False, "")


def test_should_stop_returns_false_and_empty_reason_when_continuing(tmp_path):
    tracker = IterationTracker(tmp_path / "run")
    _add(tracker, 0.4)
    _add(tracker, 0.5)

    assert tracker.should_stop() == (False, "")


def test_should_stop_on_empty_history_returns_false(tmp_path):
    tracker = IterationTracker(tmp_path / "run")
    assert tracker.should_stop() == (False, "")


def test_history_restored_from_file_after_reinstantiation(tmp_path):
    workspace = tmp_path / "run"
    tracker = IterationTracker(workspace)
    tracker.record(_run_score(0.5, "img_a"), {"pipeline": ["threshold"]})
    tracker.record(_run_score(0.7, "img_b"), {"pipeline": ["threshold", "morphology"]})

    restored = IterationTracker(workspace)
    history = restored.history
    assert len(history) == 2
    assert history[0].iteration == 0
    assert history[0].run_score.composite_mean == 0.5
    assert history[0].run_score.image_scores[0].image_id == "img_a"
    assert history[0].algorithm_spec == {"pipeline": ["threshold"]}
    assert history[1].run_score.image_scores[0].ref_count == 1
    assert restored.best is not None
    assert restored.best.iteration == 1


def test_missing_history_file_is_empty_history(tmp_path):
    tracker = IterationTracker(tmp_path / "fresh")
    assert tracker.history == []
    assert tracker.best is None


def test_workspace_dir_created_when_missing(tmp_path):
    workspace = tmp_path / "nested" / "run"
    IterationTracker(workspace)
    assert workspace.is_dir()
