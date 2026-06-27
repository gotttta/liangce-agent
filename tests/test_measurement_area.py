import numpy as np

from core.measurement.area import measure_components


def test_measure_components_reports_count_area_bbox_and_area_ratio():
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:3, 2:5] = True
    mask[6:8, 7:9] = True

    result = measure_components(mask, min_area=1, unit="pixel")

    assert result["summary"] == {
        "count": 2,
        "total_area": 10,
        "unit": "pixel",
        "area_ratio": 0.1,
    }
    assert result["results"][0]["bbox"] == [2, 1, 3, 2]
    assert result["results"][0]["area"] == 6
    assert result["results"][1]["bbox"] == [7, 6, 2, 2]
    assert result["results"][1]["area"] == 4


def test_measure_components_filters_small_noise():
    mask = np.zeros((8, 8), dtype=bool)
    mask[1, 1] = True
    mask[3:6, 3:6] = True

    result = measure_components(mask, min_area=4, unit="pixel")

    assert result["summary"]["count"] == 1
    assert result["summary"]["total_area"] == 9
    assert result["summary"]["area_ratio"] == round(9 / 64, 6)
