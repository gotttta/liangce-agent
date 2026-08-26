import numpy as np

from core.operators.types import ImageArtifact, MaskArtifact, MetadataArtifact, OperatorResult


def normalize(image, lower_percentile=1.0, upper_percentile=99.0):
    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError("percentiles must satisfy 0 <= lower < upper <= 100")
    low, high = np.percentile(image.data, [lower_percentile, upper_percentile])
    if high - low < 1e-6:
        normalized = np.zeros(image.data.shape, dtype=np.float32)
        warnings = ("constant_or_low_contrast_image",)
    else:
        normalized = np.clip((image.data - low) / (high - low), 0.0, 1.0).astype(np.float32)
        warnings = ()
    metadata = {
        "operator": "normalize",
        "lower_percentile": float(lower_percentile),
        "upper_percentile": float(upper_percentile),
        "low_value": float(low),
        "high_value": float(high),
    }
    return OperatorResult(ImageArtifact(normalized, metadata=metadata), metadata, warnings)


def gaussian_denoise(image, sigma=1.0):
    if not 0 <= sigma <= 20:
        raise ValueError("sigma must be between 0 and 20")
    if sigma == 0:
        denoised = image.data.copy()
    else:
        from skimage.filters import gaussian

        denoised = gaussian(image.data, sigma=sigma, preserve_range=True).astype(np.float32)
    metadata = {"operator": "gaussian_denoise", "sigma": float(sigma)}
    return OperatorResult(ImageArtifact(denoised, metadata=metadata), metadata)


def exclude_regions(image, rectangles=(), border_px=0):
    if not isinstance(border_px, int) or border_px < 0:
        raise ValueError("border_px must be a non-negative integer")
    height, width = image.data.shape
    if border_px * 2 >= min(height, width):
        raise ValueError("border_px excludes the entire image")

    valid = np.ones((height, width), dtype=bool)
    if border_px:
        valid[:border_px, :] = False
        valid[-border_px:, :] = False
        valid[:, :border_px] = False
        valid[:, -border_px:] = False

    normalized_rectangles = []
    for rectangle in rectangles:
        if len(rectangle) != 4:
            raise ValueError("each rectangle must be [x, y, width, height]")
        x, y, rect_width, rect_height = (int(value) for value in rectangle)
        if x < 0 or y < 0 or rect_width <= 0 or rect_height <= 0:
            raise ValueError("rectangle coordinates and dimensions are invalid")
        if x + rect_width > width or y + rect_height > height:
            raise ValueError("rectangle exceeds image bounds")
        valid[y:y + rect_height, x:x + rect_width] = False
        normalized_rectangles.append([x, y, rect_width, rect_height])

    metadata = {
        "operator": "exclude_regions",
        "border_px": border_px,
        "rectangles": normalized_rectangles,
        "valid_fraction": float(np.count_nonzero(valid) / valid.size),
    }
    return OperatorResult(MaskArtifact(valid, metadata=metadata), metadata)


def global_threshold(image, polarity="bright", sensitivity=1.8, max_coverage=0.35):
    if polarity not in {"bright", "dark"}:
        raise ValueError("polarity must be bright or dark")
    if not 0 <= sensitivity <= 10:
        raise ValueError("sensitivity must be between 0 and 10")
    if max_coverage is not None and not 0 < max_coverage <= 1:
        raise ValueError("max_coverage must be in (0, 1] or None")

    mean = float(np.mean(image.data))
    std = float(np.std(image.data))
    warnings = []
    if std < 1e-6:
        mask = np.zeros(image.data.shape, dtype=bool)
        threshold = None
        warnings.append("constant_image")
    else:
        threshold = mean + sensitivity * std if polarity == "bright" else mean - sensitivity * std
        mask = image.data >= threshold if polarity == "bright" else image.data <= threshold

    coverage = float(np.count_nonzero(mask) / mask.size)
    if max_coverage is not None and coverage > max_coverage:
        mask = np.zeros(image.data.shape, dtype=bool)
        warnings.append("coverage_exceeded")
        coverage = 0.0

    metadata = {
        "operator": "global_threshold",
        "polarity": polarity,
        "sensitivity": float(sensitivity),
        "mean": mean,
        "std": std,
        "threshold": None if threshold is None else float(threshold),
        "coverage": coverage,
    }
    return OperatorResult(MaskArtifact(mask, metadata=metadata), metadata, warnings)


def period_estimation(image, axis="auto", min_period=4, max_period=None):
    if axis not in {"auto", "x", "y"}:
        raise ValueError("axis must be auto, x, or y")
    if not isinstance(min_period, int) or min_period < 2:
        raise ValueError("min_period must be an integer >= 2")

    estimates = {}
    axes = ("x", "y") if axis == "auto" else (axis,)
    for candidate_axis in axes:
        profile = np.mean(image.data, axis=0 if candidate_axis == "x" else 1)
        estimates[candidate_axis] = _estimate_profile_period(profile, min_period, max_period)

    selected_axis = max(estimates, key=lambda item: estimates[item]["confidence"])
    selected = estimates[selected_axis]
    data = {
        "axis": selected_axis,
        "period_px": selected["period_px"],
        "confidence": selected["confidence"],
        "candidates": estimates,
    }
    warnings = () if selected["period_px"] is not None else ("period_not_found",)
    return OperatorResult(MetadataArtifact(data), data, warnings)


def periodic_background_model(image, axis, period_px, harmonic="auto", max_harmonic=3):
    if axis not in {"x", "y"}:
        raise ValueError("axis must be x or y")
    if not isinstance(period_px, int) or period_px < 2:
        raise ValueError("period_px must be an integer >= 2")
    if harmonic != "auto" and (not isinstance(harmonic, int) or harmonic < 1):
        raise ValueError("harmonic must be auto or a positive integer")
    if not isinstance(max_harmonic, int) or not 1 <= max_harmonic <= 8:
        raise ValueError("max_harmonic must be an integer between 1 and 8")

    axis_length = image.data.shape[1] if axis == "x" else image.data.shape[0]
    harmonics = range(1, max_harmonic + 1) if harmonic == "auto" else (harmonic,)
    candidates = []
    for candidate_harmonic in harmonics:
        candidate_period = period_px * candidate_harmonic
        if axis_length / candidate_period < 2.5:
            continue
        background = _phase_median_background(image.data, axis, candidate_period)
        residual = np.abs(image.data - background)
        score = float(np.median(residual))
        candidates.append((score, candidate_period, candidate_harmonic, background))
    if not candidates:
        raise ValueError("image does not contain enough repeated periods for background modeling")

    score, selected_period, selected_harmonic, background = min(candidates, key=lambda item: item[0])
    metadata = {
        "operator": "periodic_background_model",
        "axis": axis,
        "base_period_px": period_px,
        "selected_period_px": selected_period,
        "selected_harmonic": selected_harmonic,
        "repeat_count": float(axis_length / selected_period),
        "reconstruction_error": score,
        "candidates": [
            {"period_px": candidate[1], "harmonic": candidate[2], "error": candidate[0]}
            for candidate in candidates
        ],
    }
    artifact = ImageArtifact(background, metadata=metadata)
    return OperatorResult(artifact, metadata, debug_images={"background": background})


def periodic_background_residual(image, background, mode="absolute"):
    if not isinstance(background, ImageArtifact):
        raise TypeError("background must be an ImageArtifact")
    if background.data.shape != image.data.shape:
        raise ValueError("background shape must match image shape")
    if mode not in {"absolute", "positive", "negative"}:
        raise ValueError("mode must be absolute, positive, or negative")

    difference = image.data - background.data
    if mode == "absolute":
        residual = np.abs(difference)
    elif mode == "positive":
        residual = np.maximum(difference, 0)
    else:
        residual = np.maximum(-difference, 0)
    residual = residual.astype(np.float32)
    metadata = {
        "operator": "periodic_background_residual",
        "mode": mode,
        "mean": float(np.mean(residual)),
        "max": float(np.max(residual)),
    }
    return OperatorResult(
        ImageArtifact(residual, metadata=metadata),
        metadata,
        debug_images={"residual": residual},
    )


def residual_threshold(
    image,
    method="percentile",
    percentile=97.0,
    sensitivity=3.0,
    valid_mask=None,
):
    if method not in {"percentile", "otsu", "mad"}:
        raise ValueError("method must be percentile, otsu, or mad")
    if not 0 < percentile < 100:
        raise ValueError("percentile must be between 0 and 100")
    if not 0 <= sensitivity <= 20:
        raise ValueError("sensitivity must be between 0 and 20")
    if valid_mask is not None:
        if not isinstance(valid_mask, MaskArtifact):
            raise TypeError("valid_mask must be a MaskArtifact")
        if valid_mask.data.shape != image.data.shape:
            raise ValueError("valid_mask shape must match image shape")
        values = image.data[valid_mask.data]
        if values.size == 0:
            raise ValueError("valid_mask excludes the entire image")
    else:
        values = image.data

    if method == "percentile":
        threshold = float(np.percentile(values, percentile))
    elif method == "otsu":
        from skimage.filters import threshold_otsu

        threshold = float(threshold_otsu(values))
    else:
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        threshold = median + sensitivity * 1.4826 * mad
    mask = image.data >= threshold
    if valid_mask is not None:
        mask &= valid_mask.data
    coverage = float(np.count_nonzero(mask) / mask.size)
    warnings = ("empty_mask",) if not np.any(mask) else ()
    metadata = {
        "operator": "residual_threshold",
        "method": method,
        "threshold": threshold,
        "percentile": float(percentile),
        "sensitivity": float(sensitivity),
        "coverage": coverage,
        "valid_fraction": 1.0 if valid_mask is None else float(np.mean(valid_mask.data)),
    }
    return OperatorResult(MaskArtifact(mask, metadata=metadata), metadata, warnings)


def _estimate_profile_period(profile, min_period, max_period):
    centered = np.asarray(profile, dtype=np.float64) - float(np.mean(profile))
    length = centered.size
    upper = min(max_period or length // 3, length - 2)
    if upper < min_period or float(np.std(centered)) < 1e-8:
        return {"period_px": None, "confidence": 0.0}

    correlation = np.correlate(centered, centered, mode="full")[length - 1:]
    if correlation[0] <= 0:
        return {"period_px": None, "confidence": 0.0}
    correlation = correlation / correlation[0]
    candidates = [
        lag
        for lag in range(min_period, upper + 1)
        if correlation[lag] >= correlation[lag - 1] and correlation[lag] >= correlation[lag + 1]
    ]
    if not candidates:
        return {"period_px": None, "confidence": 0.0}

    scored_candidates = []
    for lag in candidates:
        harmonic_values = []
        multiple = lag
        while multiple <= upper:
            start = max(min_period, multiple - 2)
            stop = min(upper, multiple + 2)
            harmonic_values.append(max(float(correlation[index]) for index in range(start, stop + 1)))
            multiple += lag
        support = sum(value >= 0.1 for value in harmonic_values)
        harmonic_score = sum(max(0.0, value) for value in harmonic_values) / len(harmonic_values)
        primary_correlation = max(0.0, float(correlation[lag]))
        combined_score = harmonic_score * 0.7 + primary_correlation * 0.3
        scored_candidates.append((combined_score, support, primary_correlation, -lag, lag))

    _, _, _, _, best_lag = max(scored_candidates)
    confidence = max(0.0, min(1.0, float(correlation[best_lag])))
    support = next(item[1] for item in scored_candidates if item[-1] == best_lag)
    return {
        "period_px": int(best_lag),
        "confidence": round(confidence, 6),
        "harmonic_support": int(support),
    }


def _phase_median_background(image, axis, period):
    if axis == "x":
        phase_template = np.stack(
            [np.median(image[:, phase::period], axis=1) for phase in range(period)],
            axis=1,
        )
        repeats = (image.shape[1] + period - 1) // period
        return np.tile(phase_template, (1, repeats))[:, :image.shape[1]].astype(np.float32)

    phase_template = np.stack(
        [np.median(image[phase::period, :], axis=0) for phase in range(period)],
        axis=0,
    )
    repeats = (image.shape[0] + period - 1) // period
    return np.tile(phase_template, (repeats, 1))[:image.shape[0], :].astype(np.float32)


def register_image_operators(registry):
    registry.register("normalize", normalize, ImageArtifact, ImageArtifact)
    registry.register("gaussian_denoise", gaussian_denoise, ImageArtifact, ImageArtifact)
    registry.register("exclude_regions", exclude_regions, ImageArtifact, MaskArtifact)
    registry.register("global_threshold", global_threshold, ImageArtifact, MaskArtifact)
    registry.register("period_estimation", period_estimation, ImageArtifact, MetadataArtifact)
    registry.register("periodic_background_model", periodic_background_model, ImageArtifact, ImageArtifact)
    registry.register("periodic_background_residual", periodic_background_residual, ImageArtifact, ImageArtifact)
    registry.register("residual_threshold", residual_threshold, ImageArtifact, MaskArtifact)
