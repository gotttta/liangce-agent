from dataclasses import dataclass

import numpy as np

from core.operators import ContourArtifact, ImageArtifact, MaskArtifact, build_default_registry
from core.quality import mask_statistics


class PeriodicBackgroundUnavailable(ValueError):
    pass


@dataclass
class PeriodicParticleResult:
    mask: MaskArtifact
    contours: ContourArtifact
    trace: tuple
    debug_images: dict
    period: dict


def run_periodic_particle_pipeline(
    image,
    percentile=97.0,
    min_area=20,
    border_px=1,
    max_components=1,
    roi=None,
):
    registry = build_default_registry()
    source = ImageArtifact(np.asarray(image, dtype=np.float32))
    trace = []
    debug_images = {"source": source.data}

    normalized = _run(registry, trace, "normalize", source)
    denoised = _run(registry, trace, "gaussian_denoise", normalized.artifact, sigma=0.8)
    period = _run(registry, trace, "period_estimation", denoised.artifact, axis="auto")
    period_data = period.artifact.data
    if period_data["period_px"] is None:
        raise PeriodicBackgroundUnavailable("no stable image period was found")

    try:
        background = _run(
            registry,
            trace,
            "periodic_background_model",
            denoised.artifact,
            axis=period_data["axis"],
            period_px=period_data["period_px"],
            harmonic="auto",
        )
    except ValueError as exc:
        raise PeriodicBackgroundUnavailable(str(exc)) from exc
    if background.metadata["repeat_count"] < 3.0:
        raise PeriodicBackgroundUnavailable("fewer than three background repeats are available")
    residual = _run(
        registry,
        trace,
        "periodic_background_residual",
        denoised.artifact,
        background=background.artifact,
        mode="absolute",
    )
    edge_margin = period_data["period_px"]
    height, width = source.data.shape
    if period_data["axis"] == "x":
        edge_rectangles = [[0, 0, edge_margin, height], [width - edge_margin, 0, edge_margin, height]]
    else:
        edge_rectangles = [[0, 0, width, edge_margin], [0, height - edge_margin, width, edge_margin]]
    edge_rectangles.extend(_outside_roi_rectangles(roi, width, height))
    valid = _run(
        registry,
        trace,
        "exclude_regions",
        source,
        border_px=border_px,
        rectangles=edge_rectangles,
    )
    threshold = _run(
        registry,
        trace,
        "residual_threshold",
        residual.artifact,
        method="percentile",
        percentile=percentile,
        valid_mask=valid.artifact,
    )
    constrained = _run(
        registry,
        trace,
        "apply_valid_mask",
        threshold.artifact,
        valid_mask=valid.artifact,
    )
    opened = _run(registry, trace, "morphology", constrained.artifact, method="open", radius=1)
    closed = _run(registry, trace, "morphology", opened.artifact, method="close", radius=3)
    filled = _run(registry, trace, "fill_holes", closed.artifact, max_hole_area=None)
    filtered = _run(
        registry,
        trace,
        "filter_components",
        filled.artifact,
        min_area=min_area,
        min_aspect_ratio=0.2,
        max_aspect_ratio=5.0,
        max_components=max_components,
    )
    contours = _run(registry, trace, "extract_contours", filtered.artifact)

    debug_images.update({
        "normalized": normalized.artifact.data,
        "background": background.artifact.data,
        "residual": residual.artifact.data,
        "candidate_mask": threshold.artifact.data,
        "final_mask": filtered.artifact.data,
    })
    return PeriodicParticleResult(
        mask=filtered.artifact,
        contours=contours.artifact,
        trace=tuple(trace),
        debug_images=debug_images,
        period=period_data,
    )


def _run(registry, trace, name, artifact, **params):
    result = registry.run(name, artifact, **params)
    trace_item = {
        "step_id": name,
        "operator": name,
        "params": {key: _serializable_param(value) for key, value in params.items()},
        "metadata": result.metadata,
        "warnings": list(result.warnings),
    }
    if isinstance(result.artifact, MaskArtifact):
        facts = mask_statistics(result.artifact.data)
        trace_item["mask_statistics"] = facts
        if facts["coverage"] == 0:
            trace_item["warnings"].append("empty_mask")
        if facts["coverage"] > 0.35:
            trace_item["warnings"].append("coverage_exceeded")
        if result.metadata.get("kept_components") == 0:
            trace_item["warnings"].append("kept_components=0")
        trace_item["warnings"] = list(dict.fromkeys(trace_item["warnings"]))
    trace.append(trace_item)
    return result


def _serializable_param(value):
    if isinstance(value, (ImageArtifact, MaskArtifact)):
        return {"artifact": type(value).__name__, "shape": list(value.data.shape)}
    return value


def _outside_roi_rectangles(roi, image_width, image_height):
    if roi is None:
        return []
    if len(roi) != 4:
        raise ValueError("roi must be [x, y, width, height]")
    x, y, width, height = (int(value) for value in roi)
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("roi coordinates and dimensions are invalid")
    if x + width > image_width or y + height > image_height:
        raise ValueError("roi exceeds image bounds")

    rectangles = []
    if x:
        rectangles.append([0, 0, x, image_height])
    if x + width < image_width:
        rectangles.append([x + width, 0, image_width - (x + width), image_height])
    if y:
        rectangles.append([x, 0, width, y])
    if y + height < image_height:
        rectangles.append([x, y + height, width, image_height - (y + height)])
    return rectangles
