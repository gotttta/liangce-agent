from pathlib import Path

import numpy as np
from PIL import Image


def load_grayscale(path):
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    from core.input_contract import load_pixels
    pixels = load_pixels(image_path)
    array = pixels.astype(np.float32)
    if array.ndim == 3:
        array = np.asarray(Image.fromarray(pixels).convert("L"), dtype=np.float32) if pixels.dtype == np.uint8 else array[:, :, :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    if array.size == 0:
        raise ValueError(f"Empty image: {image_path}")
    return array
