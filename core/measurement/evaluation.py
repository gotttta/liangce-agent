import numpy as np

from core.measurement.area import measure_components


def evaluate_prediction(predicted_mask, reference_mask, min_area=1):
    predicted = np.asarray(predicted_mask, dtype=bool)
    reference = np.asarray(reference_mask, dtype=bool)

    if predicted.shape != reference.shape:
        return {
            "status": "invalid",
            "reason": "shape_mismatch",
            "predicted_shape": list(predicted.shape),
            "reference_shape": list(reference.shape),
        }

    intersection = int(np.count_nonzero(predicted & reference))
    union = int(np.count_nonzero(predicted | reference))
    predicted_area = int(np.count_nonzero(predicted))
    reference_area = int(np.count_nonzero(reference))
    predicted_count = measure_components(predicted, min_area=min_area)["summary"]["count"]
    reference_count = measure_components(reference, min_area=min_area)["summary"]["count"]

    return {
        "status": "ok",
        "iou": round(intersection / union, 6) if union else 1.0,
        "predicted_count": predicted_count,
        "reference_count": reference_count,
        "count_error": predicted_count - reference_count,
        "predicted_area": predicted_area,
        "reference_area": reference_area,
        "area_error": predicted_area - reference_area,
    }
