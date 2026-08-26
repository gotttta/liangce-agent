import numpy as np
import pytest
from pathlib import Path

from core.operators import (
    ContourArtifact,
    ImageArtifact,
    MaskArtifact,
    MetadataArtifact,
    build_default_registry,
)
from core.pipelines.dsl import pipeline_operator_catalog
from core.preprocessing import load_grayscale


def test_registry_exposes_only_registered_operators_and_validates_input_type():
    registry = build_default_registry()

    assert registry.names() == (
        "adaptive_threshold",
        "apply_valid_mask",
        "bilateral_denoise",
        "component_statistics",
        "convex_hull",
        "exclude_regions",
        "extract_contours",
        "fill_holes",
        "filter_components",
        "gaussian_denoise",
        "global_threshold",
        "hysteresis_threshold",
        "invert_intensity",
        "local_background_residual",
        "local_contrast",
        "median_denoise",
        "morphological_residual",
        "morphology",
        "normalize",
        "percentile_clip",
        "period_estimation",
        "periodic_background_model",
        "periodic_background_residual",
        "remove_border_components",
        "remove_small_objects",
        "residual_threshold",
        "statistical_threshold",
        "unsharp_enhance",
    )
    with pytest.raises(KeyError, match="unknown operator"):
        registry.run("not_registered", ImageArtifact(np.ones((4, 4))))
    with pytest.raises(TypeError, match="expects ImageArtifact"):
        registry.run("normalize", MaskArtifact(np.ones((4, 4))))


def test_normalize_is_deterministic_and_handles_constant_image():
    registry = build_default_registry()
    image = ImageArtifact(np.arange(100, dtype=np.float32).reshape(10, 10))

    first = registry.run("normalize", image, lower_percentile=10, upper_percentile=90)
    second = registry.run("normalize", image, lower_percentile=10, upper_percentile=90)

    assert isinstance(first.artifact, ImageArtifact)
    assert np.array_equal(first.artifact.data, second.artifact.data)
    assert first.artifact.data.min() == 0
    assert first.artifact.data.max() == 1

    constant = registry.run("normalize", ImageArtifact(np.full((4, 4), 7)))
    assert np.count_nonzero(constant.artifact.data) == 0
    assert constant.warnings == ("constant_or_low_contrast_image",)

    with pytest.raises(ValueError, match="percentiles"):
        registry.run("normalize", image, lower_percentile=90, upper_percentile=10)


def test_gaussian_denoise_preserves_shape_and_reduces_impulse():
    registry = build_default_registry()
    data = np.zeros((9, 9), dtype=np.float32)
    data[4, 4] = 1

    result = registry.run("gaussian_denoise", ImageArtifact(data), sigma=1.0)

    assert result.artifact.data.shape == data.shape
    assert 0 < result.artifact.data[4, 4] < 1
    assert np.array_equal(
        result.artifact.data,
        registry.run("gaussian_denoise", ImageArtifact(data), sigma=1.0).artifact.data,
    )


def test_defect_preprocessing_operators_preserve_shape_and_polarity():
    registry = build_default_registry()
    data = np.full((31, 31), 100, dtype=np.float32)
    data[14:17, 14:17] = 20
    image = ImageArtifact(data)

    denoised = registry.run("median_denoise", image, size=3)
    residual = registry.run("local_background_residual", denoised.artifact, sigma=3, polarity="dark")
    top_hat = registry.run("morphological_residual", image, radius=3, polarity="dark")

    assert denoised.artifact.data.shape == data.shape
    assert residual.artifact.data[15, 15] > residual.artifact.data[0, 0]
    assert top_hat.artifact.data[15, 15] > top_hat.artifact.data[0, 0]


def test_contrast_threshold_and_candidate_cleanup_operators():
    registry = build_default_registry()
    data = np.full((32, 32), 50, dtype=np.float32)
    data[12:20, 12:20] = 180
    data[0, 0] = 255
    image = ImageArtifact(data)

    clipped = registry.run("percentile_clip", image, lower=1, upper=99)
    contrasted = registry.run("local_contrast", clipped.artifact, clip_limit=0.05, tile_grid=4)
    sharpened = registry.run("unsharp_enhance", contrasted.artifact, radius=1, amount=1)
    bilateral = registry.run("bilateral_denoise", sharpened.artifact, sigma_spatial=1, sigma_color=0.2)
    inverted = registry.run("invert_intensity", bilateral.artifact)

    assert inverted.artifact.data.shape == data.shape
    statistical = registry.run("statistical_threshold", ImageArtifact(data), method="otsu")
    hysteresis = registry.run("hysteresis_threshold", ImageArtifact(data / 255), low=0.2, high=0.6)
    small = registry.run("remove_small_objects", statistical.artifact, min_size=10)
    hull = registry.run("convex_hull", small.artifact)

    assert statistical.artifact.data[15, 15]
    assert hysteresis.artifact.data[15, 15]
    assert not small.artifact.data[0, 0]
    assert hull.artifact.data[15, 15]


def test_adaptive_threshold_border_cleanup_and_component_statistics():
    registry = build_default_registry()
    data = np.full((20, 20), 10, dtype=np.float32)
    data[5:9, 5:9] = 100
    data[:3, :3] = 100
    threshold = registry.run(
        "adaptive_threshold",
        ImageArtifact(data),
        block_size=5,
        polarity="bright",
    )
    cleaned = registry.run("remove_border_components", threshold.artifact)
    stats = registry.run("component_statistics", cleaned.artifact)

    assert isinstance(threshold.artifact, MaskArtifact)
    assert not cleaned.artifact.data[1, 1]
    assert cleaned.artifact.data[6, 6]
    assert isinstance(stats.artifact, MetadataArtifact)
    assert len(stats.artifact.data["components"]) == 1
    assert stats.artifact.data["components"][0]["area"] == 16


def test_exclude_regions_builds_validity_mask():
    registry = build_default_registry()
    result = registry.run(
        "exclude_regions",
        ImageArtifact(np.zeros((8, 10))),
        border_px=1,
        rectangles=[[3, 2, 2, 3]],
    )

    assert isinstance(result.artifact, MaskArtifact)
    assert not result.artifact.data[0, 5]
    assert not result.artifact.data[3, 4]
    assert result.artifact.data[1, 1]
    with pytest.raises(ValueError, match="bounds"):
        registry.run("exclude_regions", ImageArtifact(np.zeros((8, 10))), rectangles=[[9, 1, 2, 2]])


def test_threshold_and_morphology_find_bright_region():
    registry = build_default_registry()
    data = np.full((20, 20), 20, dtype=np.float32)
    data[6:14, 7:15] = 150

    threshold = registry.run(
        "global_threshold",
        ImageArtifact(data),
        polarity="bright",
        sensitivity=1.0,
    )
    cleaned = registry.run("morphology", threshold.artifact, method="open_then_close", radius=1)

    assert cleaned.artifact.data[9, 10]
    assert not cleaned.artifact.data[0, 0]


def test_fill_holes_and_filter_components():
    registry = build_default_registry()
    mask = np.zeros((12, 12), dtype=bool)
    mask[2:9, 2:9] = True
    mask[4:7, 4:7] = False
    mask[10, 10] = True

    filled = registry.run("fill_holes", MaskArtifact(mask))
    filtered = registry.run("filter_components", filled.artifact, min_area=10)

    assert filled.artifact.data[5, 5]
    assert not filtered.artifact.data[10, 10]
    assert filtered.metadata["kept_components"] == 1


def test_filter_components_can_keep_only_largest_candidates():
    registry = build_default_registry()
    mask = np.zeros((12, 12), dtype=bool)
    mask[1:6, 1:6] = True
    mask[8:11, 8:11] = True

    result = registry.run(
        "filter_components",
        MaskArtifact(mask),
        min_area=1,
        max_components=1,
    )

    assert result.metadata["kept_components"] == 1
    assert result.artifact.data[2, 2]
    assert not result.artifact.data[9, 9]


def test_extract_contours_returns_closed_xy_coordinates():
    registry = build_default_registry()
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:7, 3:8] = True

    result = registry.run("extract_contours", MaskArtifact(mask))

    assert isinstance(result.artifact, ContourArtifact)
    assert len(result.artifact.contours) == 1
    contour = result.artifact.contours[0]
    assert np.array_equal(contour[0], contour[-1])
    assert result.artifact.image_shape == (10, 10)


def test_pipeline_operator_catalog_exposes_contour_tool_contract():
    contour_tool = next(
        item for item in pipeline_operator_catalog()
        if item["name"] == "extract_contours"
    )

    assert contour_tool["input_artifact"] == "MaskArtifact"
    assert contour_tool["output_artifact"] == "ContourArtifact"
    assert "闭合" in contour_tool["description"]


@pytest.mark.parametrize(
    ("axis", "shape", "period"),
    [("x", (80, 120), 12), ("y", (120, 80), 15)],
)
def test_period_estimation_finds_synthetic_period(axis, shape, period):
    registry = build_default_registry()
    coordinates = np.arange(shape[1] if axis == "x" else shape[0])
    profile = np.sin(2 * np.pi * coordinates / period)
    data = np.tile(profile, (shape[0], 1)) if axis == "x" else np.tile(profile[:, None], (1, shape[1]))

    result = registry.run(
        "period_estimation",
        ImageArtifact(data),
        axis=axis,
        min_period=4,
        max_period=30,
    )

    assert isinstance(result.artifact, MetadataArtifact)
    assert abs(result.artifact.data["period_px"] - period) <= 1
    assert result.artifact.data["confidence"] > 0.5


@pytest.mark.parametrize(
    ("filename", "expected_axis", "expected_period"),
    [
        ("in_film_particle_left_pattern.jpg", "x", 15),
        ("in_film_particle_middle_defect.jpg", "x", 19),
        ("in_film_particle_middle_defect_tight.jpg", "x", 20),
        ("in_film_particle_right_defect_tight.jpg", "x", None),
        ("in_film_particle_right_zoom.jpg", "y", 97),
    ],
)
def test_period_estimation_is_stable_on_gate_zero_samples(filename, expected_axis, expected_period):
    registry = build_default_registry()
    samples_root = Path(__file__).parents[1] / "data" / "samples"
    sample = samples_root / filename
    if not sample.exists():
        sample = samples_root / "outline" / filename

    first = registry.run("period_estimation", ImageArtifact(load_grayscale(sample)), axis="auto")
    second = registry.run("period_estimation", ImageArtifact(load_grayscale(sample)), axis="auto")

    assert first.artifact.data == second.artifact.data
    assert first.artifact.data["axis"] == expected_axis
    assert first.artifact.data["period_px"] == expected_period
    if expected_period is None:
        assert first.warnings == ("period_not_found",)
