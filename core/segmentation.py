import numpy as np


def segment_anomalies(image, sensitivity=1.8):
    """Segment bright or dark outliers, choosing the cleaner candidate mask."""
    mean = float(np.mean(image))
    std = float(np.std(image))
    if std < 1e-6:
        return np.zeros(image.shape, dtype=bool), {
            "selected": "none",
            "mean": mean,
            "std": std,
            "threshold": None,
            "coverage": 0.0,
        }

    bright_threshold = mean + sensitivity * std
    dark_threshold = mean - sensitivity * std
    bright = image >= bright_threshold
    dark = image <= dark_threshold

    candidates = [
        ("bright", bright, bright_threshold),
        ("dark", dark, dark_threshold),
    ]
    selected_name, selected_mask, selected_threshold = min(
        candidates,
        key=lambda item: _coverage_score(item[1]),
    )
    coverage = float(np.count_nonzero(selected_mask) / selected_mask.size)

    if coverage <= 0.0 or coverage > 0.35:
        selected_name = "none"
        selected_mask = np.zeros(image.shape, dtype=bool)
        selected_threshold = None
        coverage = 0.0

    return selected_mask, {
        "selected": selected_name,
        "mean": mean,
        "std": std,
        "threshold": None if selected_threshold is None else float(selected_threshold),
        "coverage": coverage,
    }


def _coverage_score(mask):
    coverage = np.count_nonzero(mask) / mask.size
    if coverage == 0:
        return 1.0
    if coverage > 0.35:
        return 1.0 + coverage
    return abs(coverage - 0.03)
