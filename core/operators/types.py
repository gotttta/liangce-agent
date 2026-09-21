from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _validate_2d(data, name):
    array = np.asarray(data)
    if array.ndim != 2:
        raise ValueError(f"{name} data must be a 2D array")
    if array.size == 0:
        raise ValueError(f"{name} data must not be empty")
    return array


@dataclass
class ImageArtifact:
    data: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        array = np.asarray(self.data)
        if not array.size or not (array.ndim == 2 or (array.ndim == 3 and array.shape[2] in (3, 4))):
            raise ValueError("image must be grayscale, RGB or RGBA")
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError("image data must be numeric")
        if not np.isfinite(array).all():
            raise ValueError("image data must be finite")
        self.metadata = dict(self.metadata)
        self.metadata.setdefault("color_space", "RGB" if array.ndim == 3 else "gray")
        self.metadata.setdefault("purpose", "display" if array.dtype == np.uint8 or array.ndim == 3 else "intensity")
        if array.dtype == np.uint8 or array.ndim == 3:
            self.metadata.setdefault("display_range", [0, 255])
        self.data = array.astype(np.float32, copy=False)


@dataclass
class MaskArtifact:
    data: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        array = _validate_2d(self.data, "mask")
        if not np.isfinite(array).all() or not np.isin(array, [0, 1]).all():
            raise ValueError("mask data must be finite and binary")
        self.data = array.astype(bool, copy=False)


@dataclass
class ContourArtifact:
    contours: tuple
    image_shape: tuple
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if len(self.image_shape) != 2:
            raise ValueError("contour image_shape must contain height and width")
        self.contours = tuple(np.asarray(contour, dtype=np.float32) for contour in self.contours)
        for contour in self.contours:
            if contour.ndim != 2 or contour.shape[1] != 2:
                raise ValueError("each contour must be an Nx2 array of x/y coordinates")


@dataclass
class MetadataArtifact:
    data: dict


@dataclass
class OperatorResult:
    artifact: Any
    metadata: dict = field(default_factory=dict)
    warnings: tuple = ()
    debug_images: dict = field(default_factory=dict)

    def __post_init__(self):
        self.warnings = tuple(self.warnings)
