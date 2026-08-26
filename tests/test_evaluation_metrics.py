import numpy as np

from core.measurement.evaluation import evaluate_prediction


def test_evaluate_prediction_computes_iou_count_error_and_area_error():
    predicted = np.zeros((6, 6), dtype=bool)
    predicted[1:3, 1:3] = True
    predicted[4:6, 4:6] = True

    reference = np.zeros((6, 6), dtype=bool)
    reference[1:3, 1:3] = True
    reference[0:2, 4:6] = True

    metrics = evaluate_prediction(predicted, reference)

    assert metrics["status"] == "ok"
    assert metrics["iou"] == round(4 / 12, 6)
    assert metrics["dice"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["predicted_count"] == 2
    assert metrics["reference_count"] == 2
    assert metrics["count_error"] == 0
    assert metrics["predicted_area"] == 8
    assert metrics["reference_area"] == 8
    assert metrics["area_error"] == 0
    assert metrics["relative_area_error"] == 0.0
    assert 0 <= metrics["boundary_f1"] <= 1


def test_evaluate_prediction_rejects_shape_mismatch():
    predicted = np.zeros((4, 4), dtype=bool)
    reference = np.zeros((5, 4), dtype=bool)

    metrics = evaluate_prediction(predicted, reference)

    assert metrics["status"] == "invalid"
    assert metrics["reason"] == "shape_mismatch"


def test_evaluate_prediction_reports_perfect_region_and_boundary_metrics():
    mask = np.zeros((12, 12), dtype=bool)
    mask[3:9, 4:10] = True

    metrics = evaluate_prediction(mask, mask, boundary_tolerance=0)

    assert metrics["iou"] == 1.0
    assert metrics["dice"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["boundary_f1"] == 1.0
    assert metrics["average_symmetric_surface_distance"] == 0.0


def test_evaluate_prediction_measures_shifted_boundary_with_tolerance():
    reference = np.zeros((20, 20), dtype=bool)
    reference[5:15, 5:15] = True
    predicted = np.zeros((20, 20), dtype=bool)
    predicted[5:15, 6:16] = True

    strict = evaluate_prediction(predicted, reference, boundary_tolerance=0)
    tolerant = evaluate_prediction(predicted, reference, boundary_tolerance=1)

    assert strict["boundary_f1"] < 1.0
    assert tolerant["boundary_f1"] == 1.0
    assert tolerant["average_symmetric_surface_distance"] > 0


def test_evaluate_prediction_excludes_ignore_pixels():
    reference = np.zeros((8, 8), dtype=bool)
    reference[2:5, 2:5] = True
    predicted = reference.copy()
    predicted[0:2, 0:2] = True
    ignored = np.zeros((8, 8), dtype=bool)
    ignored[0:2, 0:2] = True

    metrics = evaluate_prediction(predicted, reference, ignore_mask=ignored)

    assert metrics["precision"] == 1.0
    assert metrics["iou"] == 1.0
    assert metrics["ignored_pixels"] == 4


def test_evaluate_prediction_handles_empty_masks_and_invalid_ignore_shape():
    empty = np.zeros((6, 6), dtype=bool)
    defect = empty.copy()
    defect[2:4, 2:4] = True

    both_empty = evaluate_prediction(empty, empty)
    one_empty = evaluate_prediction(empty, defect)
    invalid_ignore = evaluate_prediction(empty, empty, ignore_mask=np.zeros((5, 5)))

    assert both_empty["iou"] == 1.0
    assert both_empty["boundary_f1"] == 1.0
    assert one_empty["recall"] == 0.0
    assert one_empty["boundary_f1"] == 0.0
    assert one_empty["average_symmetric_surface_distance"] is None
    assert invalid_ignore["status"] == "invalid"
    assert invalid_ignore["reason"] == "ignore_shape_mismatch"
