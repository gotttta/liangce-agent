import numpy as np

from core.segmentation import segment_with_strategy


def test_segment_with_strategy_finds_bright_defect():
    image = np.full((20, 20), 50, dtype=np.float32)
    image[5:9, 6:10] = 120
    strategy = {
        "segmentation": {
            "method": "bright_threshold",
            "sensitivity": 1.0,
            "min_area_px": 5,
            "morphology": "none",
        }
    }

    mask, meta = segment_with_strategy(image, strategy)

    assert meta["selected"] == "bright"
    assert mask[6, 7]
    assert not mask[0, 0]


def test_segment_with_strategy_finds_dark_defect():
    image = np.full((20, 20), 100, dtype=np.float32)
    image[11:15, 12:16] = 10
    strategy = {
        "segmentation": {
            "method": "dark_threshold",
            "sensitivity": 1.0,
            "min_area_px": 5,
            "morphology": "none",
        }
    }

    mask, meta = segment_with_strategy(image, strategy)

    assert meta["selected"] == "dark"
    assert mask[12, 13]
    assert not mask[0, 0]
