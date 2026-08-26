import numpy as np
import pytest

from core.pipelines.dsl import (
    execute_pipeline,
    normalize_pipeline,
    strategy_to_pipeline,
    validate_pipeline,
)
from core.quality import evaluate_mask_quality, inspect_mask_health


def test_strategy_pipeline_executes_deterministically_and_produces_trace():
    image = np.full((40, 40), 20, dtype=np.float32)
    image[8:32, 14:22] = 200
    strategy = {
        "segmentation": {
            "method": "bright_threshold",
            "sensitivity": 1.0,
            "min_area_px": 10,
            "morphology": "close",
        }
    }
    pipeline = strategy_to_pipeline(strategy)

    first = execute_pipeline(image, pipeline)
    second = execute_pipeline(image, pipeline)

    assert np.array_equal(first.mask.data, second.mask.data)
    assert first.mask.data[20, 18]
    assert len(first.contours.contours) == 1
    assert [item["operator"] for item in first.trace] == [
        "normalize",
        "global_threshold",
        "morphology",
        "fill_holes",
        "filter_components",
        "extract_contours",
    ]


def test_pipeline_validation_rejects_unknown_operator_and_invalid_order():
    with pytest.raises(ValueError, match="not allowed"):
        validate_pipeline({
            "steps": [{"id": "bad", "op": "python", "input": "image", "params": {}}]
        })


@pytest.mark.parametrize("method_alias", ["operation", "op"])
def test_normalize_pipeline_accepts_qwen_morphology_parameter_aliases(method_alias):
    pipeline = normalize_pipeline({
        "name": "qwen aliases",
        "steps": [
            {"id": "normalized", "op": "normalize", "input": "image", "params": {}},
            {
                "id": "threshold_mask",
                "op": "global_threshold",
                "input": "normalized",
                "params": {"polarity": "bright", "sensitivity": 1.5},
            },
            {
                "id": "morphology_mask",
                "op": "morphology",
                "input": "threshold_mask",
                "params": {method_alias: "opening", "kernel_size": 3},
            },
            {
                "id": "final_mask",
                "op": "filter_components",
                "input": "morphology_mask",
                "params": {"min_area": 1},
            },
            {
                "id": "contours",
                "op": "extract_contours",
                "input": "final_mask",
                "params": {},
            },
        ],
    })

    morphology_step = pipeline["steps"][2]

    assert morphology_step["params"] == {"method": "open", "radius": 1}
    validate_pipeline(pipeline)

    with pytest.raises(ValueError, match="expects MaskArtifact"):
        validate_pipeline({
            "steps": [
                {"id": "bad_mask", "op": "morphology", "input": "image", "params": {}},
                {"id": "contours", "op": "extract_contours", "input": "bad_mask", "params": {}},
            ]
        })


def test_mask_report_contains_factual_statistics_without_quality_status():
    mask = np.zeros((30, 30), dtype=bool)
    mask[5:25, 8:12] = True
    report = evaluate_mask_quality(mask, {"expected_shape": "elongated"})
    empty = evaluate_mask_quality(np.zeros_like(mask))

    assert report["component_count"] == 1
    assert report["coverage"] > 0
    assert empty["component_count"] == 0
    assert empty["coverage"] == 0
    assert "status" not in report
    assert "status" not in empty


def test_pipeline_trace_records_mask_statistics_and_standard_warnings():
    image = np.zeros((20, 20), dtype=np.float32)
    pipeline = {
        "name": "empty_after_component_filter",
        "steps": [
            {"id": "initial", "op": "global_threshold", "input": "image", "params": {"polarity": "bright", "sensitivity": 10.0}},
            {"id": "final_mask", "op": "filter_components", "input": "initial", "params": {"min_area": 1}},
        ],
    }

    result = execute_pipeline(image, pipeline)

    assert result.trace[0]["mask_statistics"]["coverage"] == 0
    assert "empty_mask" in result.trace[0]["warnings"]
    assert result.trace[1]["metadata"]["kept_components"] == 0
    assert "kept_components=0" in result.trace[1]["warnings"]


def test_mask_health_rejects_empty_and_unbounded_masks_without_quality_score():
    empty = inspect_mask_health(np.zeros((10, 10), dtype=bool))
    full = inspect_mask_health(np.ones((10, 10), dtype=bool))

    assert empty["issues"] == ["empty_mask"]
    assert not empty["usable_for_review"]
    assert "coverage_too_large" in full["issues"]
    border = np.zeros((10, 10), dtype=bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    assert "border_dominated" in inspect_mask_health(border)["issues"]
    assert not full["usable_for_review"]


def test_mask_health_uses_explicit_expected_count_only():
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:3, 1:3] = True
    mask[6:8, 6:8] = True

    exact = inspect_mask_health(mask, {
        "expected_count": 3,
        "count_source": "user_explicit",
    })
    observed = inspect_mask_health(mask, {
        "observed_count": 3,
        "count_source": "model_observed",
    })

    assert "component_count_mismatch" in exact["issues"]
    assert "component_count_mismatch" not in observed["issues"]


def test_periodic_builtin_pipeline_is_executable_through_common_executor():
    image = np.tile(np.array([0.0, 20.0, 0.0, 20.0] * 20, dtype=np.float32), (80, 1))
    image[20:28, 30:38] += 100
    pipeline = normalize_pipeline({
        "name": "periodic_particle_builtin",
        "kind": "builtin_pipeline",
        "params": {"percentile": 95.0, "min_area": 2, "max_components": 3},
    })

    result = execute_pipeline(image, pipeline)

    assert result.mask.data.shape == image.shape
    assert result.trace
    assert result.trace[-1]["operator"] == "extract_contours"
