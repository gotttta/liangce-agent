"""Identity and display mapping for pixel-aligned evidence (stored coordinates)."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image

COORDINATE_VERSION = 'stored-pixels-v1'


def input_identity(path):
    return {'input_sha256': sha256(Path(path).read_bytes()).hexdigest(),
            'coordinate_version': COORDINATE_VERSION}


def evidence_matches(evidence, path):
    if not evidence or not path or not Path(path).is_file():
        return False
    identity = input_identity(path)
    return all(evidence.get(key) == value for key, value in identity.items())


def load_pixels(path):
    with Image.open(path) as image:
        if getattr(image, 'n_frames', 1) != 1:
            raise ValueError('Multi-frame images are unsupported; select one frame first')
        if image.mode not in {'1', 'L', 'I', 'F', 'I;16', 'I;16B', 'I;16L', 'RGB', 'RGBA', 'P'}:
            raise ValueError(f'Unsupported image mode: {image.mode}')
        pixels = np.array(image.convert('RGB') if image.mode == 'P' else image)
    if not pixels.size or not np.isfinite(pixels).all():
        raise ValueError('Image must contain finite pixels')
    return pixels


def display_image(pixels, metadata=None):
    pixels = np.asarray(pixels)
    metadata = metadata or {}
    bounds = metadata.get('display_range')
    if bounds is None:
        bounds = [0, 255] if pixels.dtype == np.uint8 or pixels.ndim == 3 else [float(pixels.min()), float(pixels.max())]
    low, high = bounds
    if not np.isfinite([low, high]).all() or high < low:
        raise ValueError('Invalid display range')
    scaled = (pixels.astype(np.float64) - low) * 255 / (high - low) if high > low else np.zeros_like(pixels)
    rendered = Image.fromarray(np.clip(np.rint(scaled), 0, 255).astype(np.uint8))
    return rendered.convert('RGB')


def preview_png(path):
    image = display_image(load_pixels(path))
    image.thumbnail((2048, 2048))
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


def input_metadata(path):
    pixels = load_pixels(path)
    return {**input_identity(path), 'shape': list(pixels.shape), 'dtype': str(pixels.dtype),
            'coordinate_transform': {'orientation': 'stored', 'scale_xy': [1, 1], 'offset_xy': [0, 0]},
            'preview_intensity_mapping': {'method': 'linear', 'range': [0, 255] if pixels.dtype == np.uint8 else [float(pixels.min()), float(pixels.max())]}}
