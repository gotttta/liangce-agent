from pathlib import Path

import numpy as np
from PIL import Image


def load_grayscale(path):
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("L")
    array = np.asarray(image, dtype=np.float32)
    if array.size == 0:
        raise ValueError(f"Empty image: {image_path}")
    return array
