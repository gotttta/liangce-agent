import numpy as np
from PIL import Image

from core.reference_extraction import extract_annotation_from_reference, reference_mask_stats


def test_extract_annotation_from_reference_detects_red_overlay(tmp_path):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[20:30, 40:50] = (255, 0, 0)
    path = tmp_path / "reference.png"
    Image.fromarray(image).save(path)

    mask, confidence = extract_annotation_from_reference(path)

    assert confidence == 1.0
    assert mask[25, 45]
    assert reference_mask_stats(mask)["region_count"] == 1
