import numpy as np
import pytest

from core.sandbox import SandboxExecutionError, SandboxLimits, execute_pipeline_sandbox


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


def test_sandbox_executes_mask_only_pipeline_without_forcing_contours():
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
