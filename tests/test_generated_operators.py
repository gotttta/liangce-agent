import numpy as np
import pytest

from core.operators import build_default_registry
from core.pipelines.dsl import execute_pipeline, validate_pipeline
from core.sandbox import SandboxExecutionError, execute_pipeline_sandbox


CUSTOM = {
    "name": "bright_custom",
    "input_artifact": "ImageArtifact",
    "output_artifact": "MaskArtifact",
    "description": "用均值和标准差提取亮点",
    "source": """
def apply(data, params):
    threshold = np.mean(data) + params.get("sensitivity", 1.0) * np.std(data)
    return data >= threshold
""",
}


def test_generated_operator_runs_only_through_validated_pipeline_and_sandbox():
    image = np.zeros((12, 12), dtype=np.float32)
    image[3:6, 4:7] = 10
    pipeline = {
        "name": "generated",
        "generated_operators": [CUSTOM],
        "steps": [
            {"id": "final_mask", "op": "bright_custom", "input": "image", "params": {"sensitivity": 0.5}},
        ],
    }

    validate_pipeline(pipeline)
    with pytest.raises(ValueError, match="inside the sandbox"):
        execute_pipeline(image, pipeline)
    isolated = execute_pipeline_sandbox(image, pipeline)

    assert isolated.mask.data.sum() == 9
    assert isolated.trace[0]["operator"] == "bright_custom"


def test_generated_operator_rejects_imports_and_unsafe_calls():
    unsafe = dict(CUSTOM, source="import os\ndef apply(data, params):\n    return data")
    with pytest.raises(ValueError):
        validate_pipeline({
            "generated_operators": [unsafe],
            "steps": [{"id": "final_mask", "op": "bright_custom", "input": "image", "params": {}}],
        })


def test_generated_operator_rejects_source_that_exceeds_sandbox_contract():
    unsafe = dict(CUSTOM, source="def apply(data, params):\n    return data.__class__(data)")
    with pytest.raises(ValueError):
        validate_pipeline({
            "generated_operators": [unsafe],
            "steps": [{"id": "final_mask", "op": "bright_custom", "input": "image", "params": {}}],
        })


def test_generated_operator_must_declare_atomic_stage():
    non_atomic = dict(CUSTOM, atomic=False)
    with pytest.raises(ValueError, match="atomic pipeline stages"):
        validate_pipeline({
            "generated_operators": [non_atomic],
            "steps": [{"id": "final_mask", "op": "bright_custom", "input": "image", "params": {}}],
        })


def test_generated_operator_allows_safe_array_methods_and_attributes():
    pipeline = {
        "name": "array_methods",
        "steps": [{"id": "final_mask", "op": "array_threshold", "input": "image", "params": {}}],
        "generated_operators": [{
            "name": "array_threshold",
            "input_artifact": "ImageArtifact",
            "output_artifact": "MaskArtifact",
            "atomic": True,
            "source": (
                "def apply(data, params):\n"
                "    copied = data.copy()\n"
                "    threshold = np.mean(copied)\n"
                "    return (copied > threshold).astype(np.bool_)"
            ),
        }],
    }

    execution = execute_pipeline(
        np.array([[0, 2]], dtype=np.float32),
        pipeline,
        allow_generated=True,
    )

    assert execution.mask.data.tolist() == [[False, True]]
