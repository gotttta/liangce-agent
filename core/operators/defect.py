"""Common deterministic operators for semiconductor defect imagery."""

import numpy as np

from core.operators.types import ImageArtifact, MaskArtifact, MetadataArtifact, OperatorResult


def invert_intensity(image):
    """Invert intensities after robustly scaling to the observed range."""
    low, high = np.percentile(image.data, [1, 99])
    if high <= low:
        result = np.zeros_like(image.data, dtype=np.float32)
    else:
        scaled = np.clip((image.data - low) / (high - low), 0, 1)
        result = (1 - scaled).astype(np.float32)
    metadata = {"operator": "invert_intensity", "low_value": float(low), "high_value": float(high)}
    return OperatorResult(ImageArtifact(result, metadata=metadata), metadata)


def percentile_clip(image, lower=1.0, upper=99.0):
    """Clip outlier intensities without changing image geometry."""
    if not 0 <= lower < upper <= 100:
        raise ValueError("percentiles must satisfy 0 <= lower < upper <= 100")
    low, high = np.percentile(image.data, [lower, upper])
    result = np.clip(image.data, low, high).astype(np.float32)
    metadata = {"operator": "percentile_clip", "lower": float(lower), "upper": float(upper), "low_value": float(low), "high_value": float(high)}
    return OperatorResult(ImageArtifact(result, metadata=metadata), metadata)


def local_contrast(image, clip_limit=0.01, tile_grid=8):
    """Improve local contrast for faint defects under uneven illumination."""
    if not 0 < clip_limit <= 1:
        raise ValueError("clip_limit must be in (0, 1]")
    if not isinstance(tile_grid, int) or not 1 <= tile_grid <= 64:
        raise ValueError("tile_grid must be an integer between 1 and 64")
    from skimage.exposure import equalize_adapthist

    low, high = np.percentile(image.data, [1, 99])
    scaled = np.zeros_like(image.data, dtype=np.float32) if high <= low else np.clip((image.data - low) / (high - low), 0, 1)
    result = equalize_adapthist(scaled, clip_limit=clip_limit, kernel_size=tile_grid).astype(np.float32)
    metadata = {"operator": "local_contrast", "clip_limit": float(clip_limit), "tile_grid": tile_grid}
    return OperatorResult(ImageArtifact(result, metadata=metadata), metadata)


def unsharp_enhance(image, radius=1.0, amount=1.0):
    """Enhance defect edges using an unsharp mask."""
    if not 0 <= radius <= 20 or not 0 <= amount <= 10:
        raise ValueError("radius must be in [0, 20] and amount in [0, 10]")
    from skimage.filters import unsharp_mask

    result = unsharp_mask(image.data, radius=radius, amount=amount, preserve_range=True).astype(np.float32)
    metadata = {"operator": "unsharp_enhance", "radius": float(radius), "amount": float(amount)}
    return OperatorResult(ImageArtifact(result, metadata=metadata), metadata)


def bilateral_denoise(image, sigma_spatial=3.0, sigma_color=0.1):
    """Denoise while preserving sharp defect boundaries."""
    if not 0 <= sigma_spatial <= 30 or not 0 <= sigma_color <= 1:
        raise ValueError("sigma_spatial must be in [0, 30] and sigma_color in [0, 1]")
    from skimage.restoration import denoise_bilateral

    result = denoise_bilateral(image.data, sigma_spatial=sigma_spatial, sigma_color=sigma_color, channel_axis=None).astype(np.float32)
    metadata = {"operator": "bilateral_denoise", "sigma_spatial": float(sigma_spatial), "sigma_color": float(sigma_color)}
    return OperatorResult(ImageArtifact(result, metadata=metadata), metadata)


def median_denoise(image, size=3):
    """Suppress isolated SEM/sensor speckle while preserving defect edges."""
    if not isinstance(size, int) or size < 1 or size > 15 or size % 2 == 0:
        raise ValueError("size must be an odd integer between 1 and 15")
    from scipy.ndimage import median_filter

    result = median_filter(image.data, size=size).astype(np.float32)
    metadata = {"operator": "median_denoise", "size": size}
    return OperatorResult(ImageArtifact(result, metadata=metadata), metadata)


def local_background_residual(image, sigma=10.0, polarity="dark"):
    """Extract defects as deviations from a smooth local background."""
    if not 0 < sigma <= 100:
        raise ValueError("sigma must be between 0 and 100")
    if polarity not in {"dark", "bright", "absolute"}:
        raise ValueError("polarity must be dark, bright, or absolute")
    from skimage.filters import gaussian

    background = gaussian(image.data, sigma=sigma, preserve_range=True)
    difference = image.data - background
    if polarity == "dark":
        residual = np.maximum(-difference, 0)
    elif polarity == "bright":
        residual = np.maximum(difference, 0)
    else:
        residual = np.abs(difference)
    residual = residual.astype(np.float32)
    metadata = {"operator": "local_background_residual", "sigma": float(sigma), "polarity": polarity}
    return OperatorResult(
        ImageArtifact(residual, metadata=metadata),
        metadata,
        debug_images={"background": background.astype(np.float32)},
    )


def morphological_residual(image, radius=5, polarity="dark"):
    """Use black/white top-hat filtering to isolate small defects."""
    if not isinstance(radius, int) or not 1 <= radius <= 100:
        raise ValueError("radius must be an integer between 1 and 100")
    if polarity not in {"dark", "bright"}:
        raise ValueError("polarity must be dark or bright")
    from skimage.morphology import black_tophat, disk, white_tophat

    footprint = disk(radius)
    result = black_tophat(image.data, footprint) if polarity == "dark" else white_tophat(image.data, footprint)
    result = result.astype(np.float32)
    metadata = {"operator": "morphological_residual", "radius": radius, "polarity": polarity}
    return OperatorResult(ImageArtifact(result, metadata=metadata), metadata)


def adaptive_threshold(image, block_size=31, offset=0.0, polarity="bright"):
    """Threshold against a local neighborhood for uneven wafer illumination."""
    if not isinstance(block_size, int) or block_size < 3 or block_size % 2 == 0:
        raise ValueError("block_size must be an odd integer >= 3")
    if polarity not in {"bright", "dark"}:
        raise ValueError("polarity must be bright or dark")
    from skimage.filters import threshold_local

    local = threshold_local(image.data, block_size=block_size, offset=float(offset))
    mask = image.data >= local if polarity == "bright" else image.data <= local
    metadata = {
        "operator": "adaptive_threshold",
        "block_size": block_size,
        "offset": float(offset),
        "polarity": polarity,
        "coverage": float(np.mean(mask)),
    }
    return OperatorResult(MaskArtifact(mask, metadata=metadata), metadata)


def statistical_threshold(image, method="otsu", polarity="bright"):
    """Generate a mask using a global Otsu/Yen/Li/Triangle/mean threshold."""
    if method not in {"otsu", "yen", "li", "triangle", "mean"}:
        raise ValueError("method must be otsu, yen, li, triangle, or mean")
    if polarity not in {"bright", "dark"}:
        raise ValueError("polarity must be bright or dark")
    from skimage.filters import threshold_li, threshold_mean, threshold_otsu, threshold_triangle, threshold_yen

    threshold = {"otsu": threshold_otsu, "yen": threshold_yen, "li": threshold_li, "triangle": threshold_triangle, "mean": threshold_mean}[method](image.data)
    mask = image.data >= threshold if polarity == "bright" else image.data <= threshold
    metadata = {"operator": "statistical_threshold", "method": method, "polarity": polarity, "threshold": float(threshold), "coverage": float(np.mean(mask))}
    return OperatorResult(MaskArtifact(mask, metadata=metadata), metadata)


def hysteresis_threshold(image, low=0.1, high=0.3):
    """Keep strong residuals and connected weak residuals."""
    if not 0 <= low < high <= 1:
        raise ValueError("low and high must satisfy 0 <= low < high <= 1")
    from skimage.filters import apply_hysteresis_threshold

    result = apply_hysteresis_threshold(image.data, low, high)
    metadata = {"operator": "hysteresis_threshold", "low": float(low), "high": float(high), "coverage": float(np.mean(result))}
    return OperatorResult(MaskArtifact(result, metadata=metadata), metadata)


def remove_border_components(mask, buffer=0):
    """Remove candidates touching the image border or a configurable border band."""
    if not isinstance(buffer, int) or not 0 <= buffer <= 1000:
        raise ValueError("buffer must be an integer between 0 and 1000")
    from skimage.segmentation import clear_border

    result = clear_border(mask.data, buffer_size=buffer)
    metadata = {
        "operator": "remove_border_components",
        "buffer": buffer,
        "removed_pixels": int(np.count_nonzero(mask.data & ~result)),
    }
    return OperatorResult(MaskArtifact(result, metadata=metadata), metadata)


def remove_small_objects(mask, min_size=10, connectivity=2):
    """Remove isolated pixel clusters below the expected defect size."""
    if not isinstance(min_size, int) or min_size < 1:
        raise ValueError("min_size must be a positive integer")
    if connectivity not in {1, 2}:
        raise ValueError("connectivity must be 1 or 2")
    from skimage.measure import label, regionprops

    labels = label(mask.data, connectivity=connectivity)
    result = np.zeros_like(mask.data, dtype=bool)
    for region in regionprops(labels):
        if region.area >= min_size:
            result[labels == region.label] = True
    metadata = {"operator": "remove_small_objects", "min_size": min_size, "connectivity": connectivity}
    return OperatorResult(MaskArtifact(result, metadata=metadata), metadata)


def convex_hull(mask):
    """Fill concavities in connected candidates, useful for chipped particles."""
    from skimage.morphology import convex_hull_image

    result = convex_hull_image(mask.data)
    metadata = {"operator": "convex_hull"}
    return OperatorResult(MaskArtifact(result, metadata=metadata), metadata)


def component_statistics(mask, connectivity=2):
    """Return measurable area, centroid, bbox, and shape statistics for candidates."""
    if connectivity not in {1, 2}:
        raise ValueError("connectivity must be 1 or 2")
    from skimage.measure import label, regionprops

    labels = label(mask.data, connectivity=connectivity)
    components = []
    for region in regionprops(labels):
        min_row, min_col, max_row, max_col = region.bbox
        width, height = max_col - min_col, max_row - min_row
        components.append({
            "label": int(region.label),
            "area": int(region.area),
            "centroid_xy": [float(region.centroid[1]), float(region.centroid[0])],
            "bbox_xywh": [int(min_col), int(min_row), int(width), int(height)],
            "aspect_ratio": float(width / height) if height else 0.0,
            "solidity": float(region.solidity),
            "eccentricity": float(region.eccentricity),
        })
    components.sort(key=lambda item: item["area"], reverse=True)
    data = {"operator": "component_statistics", "connectivity": connectivity, "components": components}
    return OperatorResult(MetadataArtifact(data), data)


def register_defect_operators(registry):
    registry.register("invert_intensity", invert_intensity, ImageArtifact, ImageArtifact)
    registry.register("percentile_clip", percentile_clip, ImageArtifact, ImageArtifact)
    registry.register("local_contrast", local_contrast, ImageArtifact, ImageArtifact)
    registry.register("unsharp_enhance", unsharp_enhance, ImageArtifact, ImageArtifact)
    registry.register("bilateral_denoise", bilateral_denoise, ImageArtifact, ImageArtifact)
    registry.register("median_denoise", median_denoise, ImageArtifact, ImageArtifact)
    registry.register("local_background_residual", local_background_residual, ImageArtifact, ImageArtifact)
    registry.register("morphological_residual", morphological_residual, ImageArtifact, ImageArtifact)
    registry.register("adaptive_threshold", adaptive_threshold, ImageArtifact, MaskArtifact)
    registry.register("statistical_threshold", statistical_threshold, ImageArtifact, MaskArtifact)
    registry.register("hysteresis_threshold", hysteresis_threshold, ImageArtifact, MaskArtifact)
    registry.register("remove_border_components", remove_border_components, MaskArtifact, MaskArtifact)
    registry.register("remove_small_objects", remove_small_objects, MaskArtifact, MaskArtifact)
    registry.register("convex_hull", convex_hull, MaskArtifact, MaskArtifact)
    registry.register("component_statistics", component_statistics, MaskArtifact, MetadataArtifact)
