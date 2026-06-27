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


def segment_with_strategy(image, strategy):
    segmentation = (strategy or {}).get("segmentation", {})
    method = segmentation.get("method", "auto_bright_dark_threshold")
    sensitivity = float(segmentation.get("sensitivity", 1.8))
    morphology = segmentation.get("morphology", "open_then_close")

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

    if method == "bright_threshold":
        threshold = mean + sensitivity * std
        mask = image >= threshold
        selected = "bright"
    elif method == "dark_threshold":
        threshold = mean - sensitivity * std
        mask = image <= threshold
        selected = "dark"
    else:
        mask, meta = segment_anomalies(image, sensitivity=sensitivity)
        mask = _apply_morphology(mask, morphology)
        meta["method"] = method
        return mask, meta

    mask = _apply_morphology(mask, morphology)
    coverage = float(np.count_nonzero(mask) / mask.size)
    if coverage > 0.35:
        mask = np.zeros(image.shape, dtype=bool)
        selected = "none"
        threshold_value = None
        coverage = 0.0
    else:
        threshold_value = float(threshold)

    return mask, {
        "selected": selected,
        "method": method,
        "mean": mean,
        "std": std,
        "threshold": threshold_value,
        "coverage": coverage,
    }


def _apply_morphology(mask, morphology):
    if morphology == "none":
        return mask.astype(bool)

    try:
        from skimage.morphology import binary_closing, binary_opening, disk
    except ImportError:
        return mask.astype(bool)

    footprint = disk(1)
    if morphology == "open":
        return binary_opening(mask, footprint).astype(bool)
    if morphology == "close":
        return binary_closing(mask, footprint).astype(bool)
    if morphology == "open_then_close":
        return binary_closing(binary_opening(mask, footprint), footprint).astype(bool)
    if morphology == "close_then_open":
        return binary_opening(binary_closing(mask, footprint), footprint).astype(bool)
    return mask.astype(bool)
