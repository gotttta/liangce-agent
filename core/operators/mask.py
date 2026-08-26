import numpy as np

from core.operators.types import ContourArtifact, MaskArtifact, OperatorResult


def morphology(mask, method="open_then_close", radius=1):
    if method not in {"none", "open", "close", "open_then_close", "close_then_open", "dilate", "erode"}:
        raise ValueError(f"unsupported morphology method: {method}")
    if not isinstance(radius, int) or not 0 <= radius <= 50:
        raise ValueError("radius must be an integer between 0 and 50")
    if method == "none" or radius == 0:
        result = mask.data.copy()
    else:
        from skimage.morphology import (
            binary_closing,
            binary_dilation,
            binary_erosion,
            binary_opening,
            disk,
        )

        footprint = disk(radius)
        if method == "open":
            result = binary_opening(mask.data, footprint)
        elif method == "close":
            result = binary_closing(mask.data, footprint)
        elif method == "open_then_close":
            result = binary_closing(binary_opening(mask.data, footprint), footprint)
        elif method == "close_then_open":
            result = binary_opening(binary_closing(mask.data, footprint), footprint)
        elif method == "dilate":
            result = binary_dilation(mask.data, footprint)
        else:
            result = binary_erosion(mask.data, footprint)
    metadata = {"operator": "morphology", "method": method, "radius": radius}
    return OperatorResult(MaskArtifact(result, metadata=metadata), metadata)


def fill_holes(mask, max_hole_area=None):
    if max_hole_area is not None and (not isinstance(max_hole_area, int) or max_hole_area < 1):
        raise ValueError("max_hole_area must be a positive integer or None")
    from scipy.ndimage import binary_fill_holes
    from skimage.morphology import remove_small_holes

    if max_hole_area is None:
        filled = binary_fill_holes(mask.data)
    else:
        filled = remove_small_holes(mask.data, area_threshold=max_hole_area + 1, connectivity=2)
    metadata = {"operator": "fill_holes", "max_hole_area": max_hole_area}
    return OperatorResult(MaskArtifact(filled, metadata=metadata), metadata)


def filter_components(
    mask,
    min_area=1,
    max_area=None,
    min_aspect_ratio=None,
    max_aspect_ratio=None,
    max_components=None,
):
    if not isinstance(min_area, int) or min_area < 1:
        raise ValueError("min_area must be a positive integer")
    if max_area is not None and (not isinstance(max_area, int) or max_area < min_area):
        raise ValueError("max_area must be an integer >= min_area or None")
    if min_aspect_ratio is not None and min_aspect_ratio < 0:
        raise ValueError("min_aspect_ratio must be non-negative or None")
    if max_aspect_ratio is not None and max_aspect_ratio <= 0:
        raise ValueError("max_aspect_ratio must be positive or None")
    if min_aspect_ratio is not None and max_aspect_ratio is not None and min_aspect_ratio > max_aspect_ratio:
        raise ValueError("min_aspect_ratio cannot exceed max_aspect_ratio")
    if max_components is not None and (not isinstance(max_components, int) or max_components < 1):
        raise ValueError("max_components must be a positive integer or None")

    from skimage.measure import label, regionprops

    labels = label(mask.data, connectivity=2)
    accepted_regions = []
    for region in regionprops(labels):
        min_row, min_col, max_row, max_col = region.bbox
        width = max_col - min_col
        height = max_row - min_row
        aspect_ratio = width / height if height else 0.0
        if region.area < min_area or (max_area is not None and region.area > max_area):
            continue
        if min_aspect_ratio is not None and aspect_ratio < min_aspect_ratio:
            continue
        if max_aspect_ratio is not None and aspect_ratio > max_aspect_ratio:
            continue
        accepted_regions.append(region)

    accepted_regions.sort(key=lambda region: region.area, reverse=True)
    if max_components is not None:
        accepted_regions = accepted_regions[:max_components]

    filtered = np.zeros_like(mask.data)
    for region in accepted_regions:
        filtered[labels == region.label] = True

    metadata = {
        "operator": "filter_components",
        "min_area": min_area,
        "max_area": max_area,
        "min_aspect_ratio": min_aspect_ratio,
        "max_aspect_ratio": max_aspect_ratio,
        "max_components": max_components,
        "kept_components": len(accepted_regions),
    }
    return OperatorResult(MaskArtifact(filtered, metadata=metadata), metadata)


def extract_contours(mask, level=0.5):
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    from skimage.measure import find_contours

    padded = np.pad(mask.data.astype(np.uint8), 1)
    contours = []
    for contour in find_contours(padded, level=level, fully_connected="high"):
        xy = np.column_stack((contour[:, 1] - 1, contour[:, 0] - 1)).astype(np.float32)
        if xy.shape[0] >= 3:
            contours.append(xy)
    metadata = {"operator": "extract_contours", "count": len(contours), "level": float(level)}
    artifact = ContourArtifact(tuple(contours), mask.data.shape, metadata=metadata)
    return OperatorResult(artifact, metadata)


def apply_valid_mask(mask, valid_mask):
    if not isinstance(valid_mask, MaskArtifact):
        raise TypeError("valid_mask must be a MaskArtifact")
    if valid_mask.data.shape != mask.data.shape:
        raise ValueError("valid_mask shape must match mask shape")
    constrained = mask.data & valid_mask.data
    metadata = {
        "operator": "apply_valid_mask",
        "removed_pixels": int(np.count_nonzero(mask.data & ~valid_mask.data)),
    }
    return OperatorResult(MaskArtifact(constrained, metadata=metadata), metadata)


def register_mask_operators(registry):
    registry.register("morphology", morphology, MaskArtifact, MaskArtifact)
    registry.register("fill_holes", fill_holes, MaskArtifact, MaskArtifact)
    registry.register("filter_components", filter_components, MaskArtifact, MaskArtifact)
    registry.register("extract_contours", extract_contours, MaskArtifact, ContourArtifact)
    registry.register("apply_valid_mask", apply_valid_mask, MaskArtifact, MaskArtifact)
