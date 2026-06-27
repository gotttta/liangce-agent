from agent_types import default_strategy, normalize_strategy


def test_default_strategy_is_area_count_bright_threshold():
    strategy = default_strategy()

    assert strategy["measurement_type"] == "area_count"
    assert strategy["visual_observation"]["polarity"] == "bright_on_dark"
    assert strategy["segmentation"]["method"] == "bright_threshold"
    assert strategy["segmentation"]["min_area_px"] == 20
    assert strategy["measurement"]["metrics"] == ["count", "area", "bbox", "area_ratio"]


def test_normalize_strategy_fills_missing_fields_without_losing_llm_values():
    raw = {
        "defect_type": "residue",
        "segmentation": {
            "sensitivity": 2.4,
            "min_area_px": 12,
        },
        "confidence": 0.81,
    }

    strategy = normalize_strategy(raw)

    assert strategy["defect_type"] == "residue"
    assert strategy["measurement_type"] == "area_count"
    assert strategy["segmentation"]["method"] == "bright_threshold"
    assert strategy["segmentation"]["sensitivity"] == 2.4
    assert strategy["segmentation"]["min_area_px"] == 12
    assert strategy["confidence"] == 0.81
