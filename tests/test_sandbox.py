import json
from pathlib import Path

import numpy as np
import pytest

from core.sandbox import SandboxExecutionError, SandboxLimits, execute_pipeline_sandbox


def _fixture_pipeline():
    return json.loads((Path(__file__).parent / "fixtures" / "legacy_skill.json").read_text())["pipeline_template"]


def _mask_only_pipeline():
    return {
        "name": "mask_only",
        "steps": [
            {"id": "normalized", "op": "normalize", "input": "image", "params": {}},
            {
                "id": "final_mask",
                "op": "global_threshold",
                "input": "normalized",
                "params": {"polarity": "bright", "sensitivity": 1.0},
            },
        ],
    }


def test_sandbox_executes_mask_only_pipeline_without_forcing_contours(docker_sandbox):
    image = np.zeros((20, 20), dtype=np.uint8)
    image[5:10, 6:12] = 255

    result = execute_pipeline_sandbox(image, _mask_only_pipeline())

    assert result.mask.data[7, 8]
    assert result.contours is not None
    assert result.trace[-1]["operator"] == "global_threshold"


def test_sandbox_rejects_pipeline_that_exceeds_step_limit():
    with pytest.raises(SandboxExecutionError, match="step sandbox limit"):
        execute_pipeline_sandbox(
            np.zeros((4, 4), dtype=np.uint8),
            _mask_only_pipeline(),
            SandboxLimits(max_steps=1),
        )


def test_sandbox_applies_custom_step_limit_to_v3_nodes():

    with pytest.raises(SandboxExecutionError, match="step sandbox limit"):
        execute_pipeline_sandbox(
            np.zeros((4, 4), dtype=np.uint8),
            _fixture_pipeline(),
            SandboxLimits(max_steps=1),
        )


def test_sandbox_returns_intermediate_arrays_for_diagnosis(docker_sandbox):
    from core.operators import ImageArtifact, MaskArtifact
    image = np.zeros((20, 20), dtype=np.float32)
    image[5:10, 6:12] = 255
    result = execute_pipeline_sandbox(image, _mask_only_pipeline())
    assert isinstance(result.artifacts['normalized'], ImageArtifact)
    assert isinstance(result.artifacts['final_mask'], MaskArtifact)
    np.testing.assert_array_equal(result.artifacts['final_mask'].data, result.mask.data)
    assert result.artifacts['normalized'].data[7, 8] == 1
