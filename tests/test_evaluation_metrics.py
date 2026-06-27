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
    assert metrics["predicted_count"] == 2
    assert metrics["reference_count"] == 2
    assert metrics["count_error"] == 0
    assert metrics["predicted_area"] == 8
    assert metrics["reference_area"] == 8
    assert metrics["area_error"] == 0


def test_evaluate_prediction_rejects_shape_mismatch():
    predicted = np.zeros((4, 4), dtype=bool)
    reference = np.zeros((5, 4), dtype=bool)

    metrics = evaluate_prediction(predicted, reference)

    assert metrics["status"] == "invalid"
    assert metrics["reason"] == "shape_mismatch"
