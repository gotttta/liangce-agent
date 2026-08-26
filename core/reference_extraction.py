"""Extraction of colored Handbook annotations and binary Ground Truth masks."""

from pathlib import Path

import cv2
import numpy as np


def extract_annotation_from_reference(reference_image_path, color_ranges=None):
    """Return a boolean color-annotation mask and a conservative confidence."""
    ranges = color_ranges or [
        ((0, 100, 100), (10, 255, 255)),
        ((170, 100, 100), (180, 255, 255)),
        ((40, 100, 100), (80, 255, 255)),
        ((80, 100, 100), (100, 255, 255)),
    ]
    path = Path(reference_image_path)
    image = cv2.imread(str(path))
    if image is None:
        return None, 0.0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(image.shape[:2], dtype=bool)
    for lower, upper in ranges:
        mask |= cv2.inRange(hsv, np.array(lower), np.array(upper)) > 0
    coverage = float(np.mean(mask))
    if 0.001 < coverage < 0.05:
        confidence = 1.0
    elif 0 < coverage < 0.15:
        confidence = 0.5
    else:
        confidence = 0.0
    return mask, confidence


def extract_ground_truth_mask(annotation_image_path, expected_shape=None, color_ranges=None):
    """Convert a Handbook annotation into a filled binary Ground Truth mask.

    Handbook files normally contain thin colored contours.  Those contours are
    useful for people but not for pixel metrics, so this function extracts the
    ink, closes small anti-aliasing gaps, and fills each outer contour.
    """
    path = Path(annotation_image_path)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"无法读取 Ground Truth 标注图：{path}")
    if image.ndim == 2:
        source = image
        color_mask = _binary_mask(source)
    else:
        bgr = image[:, :, :3]
        color_mask = _colored_annotation_mask(bgr, color_ranges=color_ranges)
        if not np.any(color_mask):
            color_mask = _binary_mask(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    if expected_shape is not None and tuple(color_mask.shape) != tuple(expected_shape):
        raise ValueError(
            "Ground Truth 标注图尺寸与当前原图不一致："
            f"{list(color_mask.shape)} != {list(expected_shape)}"
        )
    if not np.any(color_mask):
        raise ValueError("未在 Ground Truth 标注图中识别到彩色轮廓或二值 Mask")

    # A small close connects anti-aliased/JPEG contour breaks without changing
    # the intended object boundary materially.
    closed = cv2.morphologyEx(
        color_mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(closed, dtype=np.uint8)
    retained = [contour for contour in contours if cv2.contourArea(contour) >= 4.0]
    if retained:
        cv2.drawContours(filled, retained, -1, color=1, thickness=cv2.FILLED)
    if not np.any(filled):
        raise ValueError("Ground Truth 轮廓无法形成有效的闭合目标区域")
    mask = filled.astype(bool)
    return mask, {
        "annotation_shape": list(mask.shape),
        "contour_count": len(retained),
        "annotation_pixel_count": int(np.count_nonzero(color_mask)),
        "filled_pixel_count": int(np.count_nonzero(mask)),
        "coverage": round(float(np.mean(mask)), 6),
    }


def save_ground_truth_mask(mask, output_path):
    """Persist a metric-ready black/white mask without lossy JPEG encoding."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.asarray(mask, dtype=np.uint8) * 255):
        raise OSError(f"无法写入 Ground Truth Mask：{path}")
    return str(path)


def _colored_annotation_mask(bgr, color_ranges=None):
    ranges = color_ranges or [
        ((0, 80, 80), (12, 255, 255)),
        ((165, 80, 80), (180, 255, 255)),
        ((35, 70, 70), (95, 255, 255)),
    ]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=bool)
    for lower, upper in ranges:
        mask |= cv2.inRange(hsv, np.array(lower), np.array(upper)) > 0
    return mask


def _binary_mask(gray):
    """Accept an already-exported mask, while rejecting ordinary photographs."""
    values = np.unique(np.asarray(gray))
    if len(values) > 4:
        return np.zeros(gray.shape, dtype=bool)
    return np.asarray(gray) > 0


def reference_mask_stats(mask):
    """Return serializable region statistics for a reference annotation mask."""
    binary = np.asarray(mask, dtype=np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return {
        "region_count": len(contours),
        "total_area_px": int(np.count_nonzero(binary)),
        "coverage": round(float(np.mean(binary > 0)), 6),
    }
