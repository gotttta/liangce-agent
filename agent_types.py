from copy import deepcopy


def default_strategy():
    return {
        "defect_type": "particle_residue",
        "measurement_type": "area_count",
        "visual_observation": {
            "defect_appearance": "small anomaly regions",
            "background_pattern": "unknown",
            "polarity": "bright_on_dark",
        },
        "segmentation": {
            "method": "bright_threshold",
            "sensitivity": 1.8,
            "min_area_px": 20,
            "max_area_px": None,
            "morphology": "open_then_close",
        },
        "measurement": {
            "metrics": ["count", "area", "bbox", "area_ratio"],
            "unit": "pixel",
        },
        "confidence": 0.5,
        "notes": ["Default strategy used for area/count baseline."],
    }


def normalize_strategy(raw_strategy):
    strategy = default_strategy()
    raw = raw_strategy or {}

    for key in ("defect_type", "measurement_type", "confidence", "notes"):
        if key in raw:
            strategy[key] = raw[key]

    for section in ("visual_observation", "segmentation", "measurement"):
        if isinstance(raw.get(section), dict):
            merged = deepcopy(strategy[section])
            merged.update(raw[section])
            strategy[section] = merged

    return strategy
