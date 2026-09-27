import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from core.pipelines.dsl import (
    execute_pipeline,
    is_v3_pipeline,
    normalize_pipeline,
    pin_pipeline_operator_versions,
    strategy_to_pipeline,
    validate_pipeline,
)
from core.quality import evaluate_mask_quality, inspect_mask_health


def _fixture_pipeline():
    return json.loads((Path(__file__).parent / "fixtures" / "legacy_skill.json").read_text())["pipeline_template"]


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


def test_mask_health_keeps_empty_and_large_masks_as_diagnostics():
    empty = inspect_mask_health(np.zeros((10, 10), dtype=bool))
    full = inspect_mask_health(np.ones((10, 10), dtype=bool))

    assert empty["issues"] == ["empty_mask"]
    assert empty["usable_for_review"]
    assert "coverage_too_large" in full["issues"]
    border = np.zeros((10, 10), dtype=bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    assert "border_dominated" in inspect_mask_health(border)["issues"]
    assert full["usable_for_review"]


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


def test_v3_dag_executes_multi_input_periodic_baseline_deterministically():
    from core.pipelines.periodic_template import periodic_segmentation_template

    height, width, period = 96, 144, 12
    x = np.arange(width)
    image = np.tile(55 + 30 * np.cos(2 * np.pi * x / period), (height, 1)).astype(np.float32)
    image[35:60, 62:86] += 100
    pipeline = periodic_segmentation_template()

    first = execute_pipeline(image, pipeline)
    second = execute_pipeline(image, pipeline)

    assert is_v3_pipeline(pipeline)
    assert np.array_equal(first.mask.data, second.mask.data)
    assert first.mask.data[45, 72]
    assert first.artifacts["background"].data.shape == image.shape
    assert first.trace[2]["inputs"] == {"image": "denoised", "period": "period"}


def test_v3_dag_rejects_type_mismatch_and_cycles():
    type_mismatch = {
        "schema_version": 3,
        "nodes": [{
            "id": "bad",
            "operator": "morphology",
            "inputs": {"mask": "$image"},
            "params": {},
        }],
        "outputs": {"mask": "bad"},
    }
    with pytest.raises(ValueError, match="expects MaskArtifact"):
        validate_pipeline(type_mismatch)

    cyclic = {
        "schema_version": 3,
        "nodes": [
            {"id": "a", "operator": "normalize", "inputs": {"image": "b"}, "params": {}},
            {"id": "b", "operator": "normalize", "inputs": {"image": "a"}, "params": {}},
            {"id": "mask", "operator": "global_threshold", "inputs": {"image": "a"}, "params": {}},
        ],
        "outputs": {"mask": "mask"},
    }
    with pytest.raises(ValueError, match="contains a cycle"):
        validate_pipeline(cyclic)


def test_builtin_image_declaration_error_repairs_without_changing_grayscale_input(tmp_path):
    from core.experiments.drafts import DraftStore
    pipeline = {
        'schema_version': 3,
        'input_types': {'$image': 'ImageArtifact'},
        'nodes': [{'id': 'mask', 'operator': 'global_threshold',
                   'inputs': {'image': '$image'}, 'params': {'polarity': 'bright', 'sensitivity': 1}}],
        'outputs': {'mask': 'mask'},
    }
    original = deepcopy(pipeline)
    draft = DraftStore(tmp_path).create(pipeline)
    assert not draft['validation']['valid']
    diagnostic = draft['validation']['error']['message']
    assert "Remove only input_types['$image']" in diagnostic
    assert 'keep node inputs referencing $image unchanged' in diagnostic
    assert '$rgb is a separate RGB input' in diagnostic
    assert pipeline == original

    repaired = deepcopy(pipeline)
    repaired.pop('input_types')
    validate_pipeline(repaired)
    gray = np.zeros((8, 8), dtype=np.float32)
    gray[2:6, 2:6] = 255
    result = execute_pipeline(gray, repaired)
    assert result.mask.data.shape == gray.shape
    assert result.mask.data[3, 3]
    assert result.trace[0]['inputs'] == {'image': '$image'}
    assert repaired['nodes'] == original['nodes']


def test_declared_rgb_remains_separate_from_builtin_primary_image():
    pipeline = {'schema_version': 3, 'input_types': {'$rgb': 'ImageArtifact'},
                'nodes': [{'id': 'color', 'operator': 'normalize',
                           'inputs': {'image': '$rgb'}, 'params': {}}],
                'outputs': {'color_image': 'color'}}
    gray = np.zeros((8, 8), dtype=np.float32)
    rgb = np.zeros((8, 8, 3), dtype=np.float32)
    rgb[:, :, 0] = 255
    result = execute_pipeline(gray, pipeline, inputs={'$rgb': rgb})
    assert result.artifacts['$image'].data.shape == (8, 8)
    assert result.artifacts['$rgb'].data.shape == (8, 8, 3)
    assert result.outputs['color_image'].data.shape == (8, 8, 3)


def test_pipeline_replay_record_pins_operator_versions():
    pipeline = strategy_to_pipeline({"segmentation": {"min_area_px": 2}})
    pinned = pin_pipeline_operator_versions(pipeline)

    assert pinned["operator_versions"]["normalize"] == "1.0.0"
    assert pinned["operator_versions"]["filter_components"] == "1.0.0"


@pytest.mark.parametrize("kind", ["legacy", "v3", "builtin"])
def test_replay_preserves_metadata_and_rejects_incompatible_versions(kind):

    pipeline = {
        "legacy": strategy_to_pipeline({}),
        "v3": _fixture_pipeline(),
        "builtin": {"kind": "builtin_pipeline", "name": "periodic_particle_builtin", "params": {}},
    }[kind]
    pinned = pin_pipeline_operator_versions(pipeline)
    restored = normalize_pipeline(pinned)
    assert restored == pinned
    assert pin_pipeline_operator_versions(restored) == pinned
    image = np.tile(np.array([0.0, 20.0, 0.0, 20.0] * 20, dtype=np.float32), (80, 1))
    image[20:28, 30:38] += 100
    assert np.array_equal(execute_pipeline(image, restored).mask.data, execute_pipeline(image, pinned).mask.data)

    tool = next(iter(restored["operator_versions"]))
    restored["operator_versions"][tool] = "999.0.0"
    assert pinned["operator_versions"][tool] != "999.0.0"
    for operation in (validate_pipeline, pin_pipeline_operator_versions, lambda value: execute_pipeline(image, value)):
        with pytest.raises(ValueError, match="version mismatch"):
            operation(restored)


@pytest.mark.parametrize("versions", [None, {}, {"unknown": "1.0.0"}])
def test_replay_rejects_incomplete_or_malformed_versions(versions):
    pipeline = strategy_to_pipeline({})
    pipeline["operator_versions"] = versions
    with pytest.raises(ValueError, match="operator_versions"):
        validate_pipeline(normalize_pipeline(pipeline))


def test_generated_operator_version_comes_from_embedded_definition():
    pipeline = {
        "generated_operators": [{
            "name": "custom_mask", "version": "2.3.0",
            "source": "def apply(data, params):\n    return data > np.mean(data)",
        }],
        "steps": [{"id": "mask", "op": "custom_mask", "input": "image", "params": {}}],
    }
    pinned = pin_pipeline_operator_versions(pipeline)
    assert pinned["operator_versions"]["custom_mask"] == "2.3.0"
    assert pinned["version_provenance"] == "recorded_at_execution"


@pytest.mark.parametrize("kind", ["legacy", "v3", "builtin"])
def test_old_operator_field_names_replay_without_losing_version_checks(kind):
    from copy import deepcopy

    pipeline = {
        "legacy": strategy_to_pipeline({}),
        "v3": _fixture_pipeline(),
        "builtin": {"kind": "builtin_pipeline", "name": "periodic_particle_builtin", "params": {}},
    }[kind]
    canonical = pin_pipeline_operator_versions(pipeline)
    old = deepcopy(canonical)
    old["tool_versions"] = old.pop("operator_versions")
    for node in old.get("nodes", []):
        node["tool"] = node.pop("operator")
    original = deepcopy(old)
    assert normalize_pipeline(old) == canonical
    assert pin_pipeline_operator_versions(old) == canonical
    validate_pipeline(old)
    image = np.tile(np.array([0, 20, 0, 20] * 20, dtype=np.float32), (80, 1))
    image[20:28, 30:38] += 100
    assert np.array_equal(execute_pipeline(image, old).mask.data, execute_pipeline(image, canonical).mask.data)
    assert old == original
    old["tool_versions"][next(iter(old["tool_versions"]))] = "999.0.0"
    for operation in (normalize_pipeline, validate_pipeline, pin_pipeline_operator_versions):
        if operation is normalize_pipeline:
            with pytest.raises(ValueError, match="version mismatch"):
                validate_pipeline(operation(old))
        else:
            with pytest.raises(ValueError, match="version mismatch"):
                operation(old)


@pytest.mark.parametrize("conflict", ["node", "versions"])
def test_conflicting_legacy_operator_fields_are_rejected(conflict):

    pipeline = pin_pipeline_operator_versions(_fixture_pipeline())
    if conflict == "node":
        pipeline["nodes"][0]["tool"] = "median_denoise"
    else:
        pipeline["tool_versions"] = {}
    for operation in (normalize_pipeline, validate_pipeline, pin_pipeline_operator_versions):
        with pytest.raises(ValueError, match="conflicting"):
            operation(pipeline)
