import numpy as np


def mask_statistics(mask):
    """Return reproducible mask facts used in traces and health routing."""
    from skimage.measure import label, regionprops

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or binary.size == 0:
        raise ValueError("quality evaluator expects a non-empty 2D mask")
    coverage = float(np.mean(binary))
    labels = label(binary, connectivity=2)
    regions = regionprops(labels)
    component_count = len(regions)
    border = np.zeros_like(binary)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    border_pixels = int(np.count_nonzero(binary & border))
    total_pixels = int(np.count_nonzero(binary))
    border_fraction = border_pixels / total_pixels if total_pixels else 0.0
    largest_component_fraction = float(
        max((region.area for region in regions), default=0) / binary.size
    )

    return {
        "coverage": round(coverage, 6),
        "component_count": component_count,
        "border_fraction": round(border_fraction, 6),
        "largest_component_fraction": round(largest_component_fraction, 6),
    }


def inspect_mask_health(mask, constraints=None):
    """Classify execution health without claiming that a mask is visually correct."""
    facts = mask_statistics(mask)
    constraints = constraints if isinstance(constraints, dict) else {}
    try:
        max_coverage = float(constraints.get("max_coverage", 0.35))
    except (TypeError, ValueError):
        max_coverage = 0.35
    max_coverage = min(1.0, max(0.0, max_coverage))
    try:
        max_components = constraints.get("max_components")
        max_components = int(max_components) if max_components is not None else None
    except (TypeError, ValueError):
        max_components = None
    try:
        expected_count = int(constraints.get("expected_count"))
        expected_count = expected_count if expected_count > 0 else None
    except (TypeError, ValueError):
        expected_count = None
    count_source = str(constraints.get("count_source") or "").strip().lower()
    try:
        count_tolerance = max(0, int(constraints.get("count_tolerance", 0)))
    except (TypeError, ValueError):
        count_tolerance = 0
    issues = []
    if facts["coverage"] == 0:
        issues.append("empty_mask")
    if facts["coverage"] > max_coverage:
        issues.append("coverage_too_large")
    if facts["border_fraction"] > 0.50:
        issues.append("border_dominated")
    if facts["largest_component_fraction"] > max_coverage:
        issues.append("dominant_component")
    if max_components is not None and max_components >= 0 and facts["component_count"] > max_components:
        issues.append("too_many_components")
    if (
        expected_count is not None
        and count_source == "user_explicit"
        and abs(facts["component_count"] - expected_count) > count_tolerance
    ):
        issues.append("component_count_mismatch")
    return {
        **facts,
        "issues": issues,
        "usable_for_review": not issues,
    }


def evaluate_mask_quality(mask, target_constraints=None):
    """Backward-compatible factual report; health routing uses inspect_mask_health."""
    return mask_statistics(mask)
