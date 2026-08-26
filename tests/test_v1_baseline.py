import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core.measurement.area import measure_components
from core.operators import ImageArtifact, build_default_registry
from core.preprocessing import load_grayscale
from core.segmentation import segment_with_strategy
from core.visualization import save_annotated_image, save_mask_image
from scripts.freeze_v1_baseline import FROZEN_STRATEGY, ROOT


MANIFEST_PATH = ROOT / "data" / "baselines" / "v1_threshold" / "manifest.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _assert_segmentation_metadata(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if key in {"mean", "std", "threshold", "coverage"} and value is not None:
            assert actual[key] == pytest.approx(value, abs=1e-5)
        else:
            assert actual[key] == value


def _image_pixels(path):
    with Image.open(path) as image:
        return np.asarray(image)


def test_frozen_v1_baseline_is_complete_and_reproducible(tmp_path):
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["baseline_id"] == "v1-threshold-2026-07"
    assert len(manifest["samples"]) == 5
    assert manifest["known_limitations"]
    assert (ROOT / manifest["observations_file"]).exists()
    frozen_strategy = json.loads((ROOT / manifest["strategy_file"]).read_text(encoding="utf-8"))
    assert frozen_strategy == FROZEN_STRATEGY

    for record in manifest["samples"]:
        source = ROOT / record["sample"]
        assert _sha256(source) == record["source_sha256"]

        image = load_grayscale(source)
        first_mask, first_meta = segment_with_strategy(image, FROZEN_STRATEGY)
        second_mask, second_meta = segment_with_strategy(image, FROZEN_STRATEGY)
        assert first_mask.tobytes() == second_mask.tobytes()
        assert first_meta == second_meta
        _assert_segmentation_metadata(first_meta, record["segmentation"])

        registry = build_default_registry()
        threshold = registry.run(
            "global_threshold",
            ImageArtifact(image),
            polarity="bright",
            sensitivity=1.8,
            max_coverage=None,
        )
        operator_mask = registry.run(
            "morphology",
            threshold.artifact,
            method="open_then_close",
            radius=1,
        ).artifact.data
        assert operator_mask.tobytes() == first_mask.tobytes()

        regenerated_mask = tmp_path / f"{source.stem}_mask.png"
        save_mask_image(first_mask, regenerated_mask)
        assert np.array_equal(
            _image_pixels(regenerated_mask),
            _image_pixels(ROOT / record["artifacts"]["mask"]),
        )

        measurements = measure_components(first_mask, min_area=20, unit="pixel")
        assert measurements["summary"] == record["summary"]
        stored_measurements = json.loads(
            (ROOT / record["artifacts"]["measurements"]).read_text(encoding="utf-8")
        )
        assert measurements == stored_measurements

        regenerated_contour = tmp_path / f"{source.stem}_contour.png"
        save_annotated_image(source, measurements["results"], regenerated_contour, mask=first_mask)
        assert np.array_equal(
            _image_pixels(regenerated_contour),
            _image_pixels(ROOT / record["artifacts"]["contour"]),
        )

        stored_segmentation = json.loads(
            (ROOT / record["artifacts"]["segmentation"]).read_text(encoding="utf-8")
        )
        _assert_segmentation_metadata(first_meta, stored_segmentation)

        for artifact in record["artifacts"].values():
            assert (ROOT / artifact).exists()
