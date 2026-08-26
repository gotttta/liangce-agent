import numpy as np

from core.measurement.area import measure_components


def evaluate_prediction(
    predicted_mask,
    reference_mask,
    min_area=1,
    ignore_mask=None,
    boundary_tolerance=2,
):
    predicted = np.asarray(predicted_mask, dtype=bool)
    reference = np.asarray(reference_mask, dtype=bool)

    if predicted.shape != reference.shape:
        return {
            "status": "invalid",
            "reason": "shape_mismatch",
            "predicted_shape": list(predicted.shape),
            "reference_shape": list(reference.shape),
        }
    if not isinstance(boundary_tolerance, int) or boundary_tolerance < 0:
        raise ValueError("boundary_tolerance must be a non-negative integer")

    if ignore_mask is None:
        ignored = np.zeros(predicted.shape, dtype=bool)
    else:
        ignored = np.asarray(ignore_mask, dtype=bool)
        if ignored.shape != predicted.shape:
            return {
                "status": "invalid",
                "reason": "ignore_shape_mismatch",
                "predicted_shape": list(predicted.shape),
                "ignore_shape": list(ignored.shape),
            }

    valid = ~ignored
    predicted_valid = predicted & valid
    reference_valid = reference & valid

    true_positive = int(np.count_nonzero(predicted_valid & reference_valid))
    false_positive = int(np.count_nonzero(predicted_valid & ~reference_valid))
    false_negative = int(np.count_nonzero(~predicted_valid & reference_valid))
    union = true_positive + false_positive + false_negative
    predicted_area = int(np.count_nonzero(predicted_valid))
    reference_area = int(np.count_nonzero(reference_valid))
    predicted_count = measure_components(predicted_valid, min_area=min_area)["summary"]["count"]
    reference_count = measure_components(reference_valid, min_area=min_area)["summary"]["count"]

    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    dice = _safe_ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative)
    boundary_metrics = _boundary_metrics(
        predicted,
        reference,
        valid,
        tolerance=boundary_tolerance,
    )

    return {
        "status": "ok",
        "iou": round(true_positive / union, 6) if union else 1.0,
        "dice": round(dice, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "predicted_count": predicted_count,
        "reference_count": reference_count,
        "count_error": predicted_count - reference_count,
        "predicted_area": predicted_area,
        "reference_area": reference_area,
        "area_error": predicted_area - reference_area,
        "relative_area_error": (
            round((predicted_area - reference_area) / reference_area, 6)
            if reference_area
            else (0.0 if predicted_area == 0 else None)
        ),
        "ignored_pixels": int(np.count_nonzero(ignored)),
        **boundary_metrics,
    }


def _safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else 1.0


def _boundary_metrics(predicted, reference, valid, tolerance):
    from scipy.ndimage import binary_erosion, distance_transform_edt

    predicted_boundary = (predicted & ~binary_erosion(predicted)) & valid
    reference_boundary = (reference & ~binary_erosion(reference)) & valid
    predicted_size = int(np.count_nonzero(predicted_boundary))
    reference_size = int(np.count_nonzero(reference_boundary))

    if predicted_size == 0 and reference_size == 0:
        return {
            "boundary_precision": 1.0,
            "boundary_recall": 1.0,
            "boundary_f1": 1.0,
            "average_symmetric_surface_distance": 0.0,
            "boundary_tolerance_px": tolerance,
        }
    if predicted_size == 0 or reference_size == 0:
        return {
            "boundary_precision": 0.0 if predicted_size else 1.0,
            "boundary_recall": 0.0 if reference_size else 1.0,
            "boundary_f1": 0.0,
            "average_symmetric_surface_distance": None,
            "boundary_tolerance_px": tolerance,
        }

    distance_to_reference = distance_transform_edt(~reference_boundary)
    distance_to_prediction = distance_transform_edt(~predicted_boundary)
    predicted_distances = distance_to_reference[predicted_boundary]
    reference_distances = distance_to_prediction[reference_boundary]
    boundary_precision = float(np.mean(predicted_distances <= tolerance))
    boundary_recall = float(np.mean(reference_distances <= tolerance))
    denominator = boundary_precision + boundary_recall
    boundary_f1 = 2 * boundary_precision * boundary_recall / denominator if denominator else 0.0
    assd = (float(np.mean(predicted_distances)) + float(np.mean(reference_distances))) / 2
    return {
        "boundary_precision": round(boundary_precision, 6),
        "boundary_recall": round(boundary_recall, 6),
        "boundary_f1": round(boundary_f1, 6),
        "average_symmetric_surface_distance": round(assd, 6),
        "boundary_tolerance_px": tolerance,
    }
