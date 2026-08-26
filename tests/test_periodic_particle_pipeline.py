import numpy as np
import pytest

from core.operators import ImageArtifact, MaskArtifact, build_default_registry
from core.pipelines.periodic_particle import (
    PeriodicBackgroundUnavailable,
    run_periodic_particle_pipeline,
)


def _synthetic_periodic_image():
    height, width, period = 96, 144, 12
    x = np.arange(width)
    pattern = 55 + 30 * np.cos(2 * np.pi * x / period)
    image = np.tile(pattern, (height, 1)).astype(np.float32)
    image[35:60, 62:86] += 100
    ground_truth = np.zeros((height, width), dtype=bool)
    ground_truth[35:60, 62:86] = True
    return image, ground_truth


def test_periodic_background_residual_is_concentrated_on_synthetic_defect():
    registry = build_default_registry()
    image, ground_truth = _synthetic_periodic_image()
    source = ImageArtifact(image)

    background = registry.run(
        "periodic_background_model",
        source,
        axis="x",
        period_px=12,
        harmonic=1,
    )
    residual = registry.run(
        "periodic_background_residual",
        source,
        background=background.artifact,
    )

    assert residual.artifact.data[ground_truth].mean() > 80
    assert residual.artifact.data[~ground_truth].mean() < 1


def test_residual_threshold_ignores_excluded_high_value_border():
    registry = build_default_registry()
    residual = np.zeros((20, 20), dtype=np.float32)
    residual[:, -3:] = 255
    residual[8:12, 8:12] = 100
    valid = np.ones((20, 20), dtype=bool)
    valid[:, -3:] = False

    result = registry.run(
        "residual_threshold",
        ImageArtifact(residual),
        method="percentile",
        percentile=95,
        valid_mask=MaskArtifact(valid),
    )

    assert result.artifact.data[9, 9]
    assert not result.artifact.data[9, 19]


def test_fixed_periodic_particle_pipeline_finds_synthetic_defect_deterministically():
    image, ground_truth = _synthetic_periodic_image()

    first = run_periodic_particle_pipeline(image, percentile=95, min_area=20)
    second = run_periodic_particle_pipeline(image, percentile=95, min_area=20)

    assert np.array_equal(first.mask.data, second.mask.data)
    intersection = np.count_nonzero(first.mask.data & ground_truth)
    union = np.count_nonzero(first.mask.data | ground_truth)
    assert intersection / union > 0.8
    assert len(first.contours.contours) == 1
    assert [item["operator"] for item in first.trace] == [
        "normalize",
        "gaussian_denoise",
        "period_estimation",
        "periodic_background_model",
        "periodic_background_residual",
        "exclude_regions",
        "residual_threshold",
        "apply_valid_mask",
        "morphology",
        "morphology",
        "fill_holes",
        "filter_components",
        "extract_contours",
    ]


def test_fixed_pipeline_fails_explicitly_when_period_is_unavailable():
    image = np.zeros((40, 40), dtype=np.float32)

    with pytest.raises(PeriodicBackgroundUnavailable, match="no stable image period"):
        run_periodic_particle_pipeline(image)


def test_fixed_pipeline_validates_roi_bounds():
    image, _ = _synthetic_periodic_image()

    with pytest.raises(ValueError, match="roi exceeds image bounds"):
        run_periodic_particle_pipeline(image, roi=[130, 10, 30, 30])
