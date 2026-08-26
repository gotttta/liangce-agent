import argparse
import hashlib
import json
import time
from pathlib import Path

from core.measurement.area import measure_components
from core.preprocessing import load_grayscale
from core.segmentation import segment_with_strategy
from core.visualization import save_annotated_image, save_mask_image


ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = ROOT / "data" / "samples"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "baselines" / "v1_threshold"

FROZEN_STRATEGY = {
    "name": "v1_global_bright_threshold",
    "version": 1,
    "segmentation": {
        "method": "bright_threshold",
        "sensitivity": 1.8,
        "min_area_px": 20,
        "max_area_px": None,
        "morphology": "open_then_close",
    },
    "measurement": {
        "metrics": ["count", "area", "bbox", "area_ratio"],
        "unit": "pixel",
    },
}

KNOWN_LIMITATIONS = [
    "Global brightness cannot distinguish normal bright periodic lines from defects.",
    "Image borders and scale bars are not excluded before segmentation.",
    "The saved mask contains components below min_area_px while measurements filter them out.",
    "The baseline has no periodic-background model or semantic defect understanding.",
]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_sample(sample_path, output_dir):
    started = time.perf_counter()
    image = load_grayscale(sample_path)
    mask, segmentation = segment_with_strategy(image, FROZEN_STRATEGY)
    measurements = measure_components(
        mask,
        min_area=FROZEN_STRATEGY["segmentation"]["min_area_px"],
        unit="pixel",
    )

    sample_dir = output_dir / sample_path.stem
    sample_dir.mkdir(parents=True, exist_ok=True)
    mask_path = sample_dir / "mask.png"
    contour_path = sample_dir / "result_contour.png"
    save_mask_image(mask, mask_path)
    save_annotated_image(sample_path, measurements["results"], contour_path, mask=mask)
    write_json(sample_dir / "segmentation.json", segmentation)
    write_json(sample_dir / "measurements.json", measurements)

    return {
        "sample": str(sample_path.relative_to(ROOT)),
        "source_sha256": sha256_file(sample_path),
        "mask_sha256": sha256_file(mask_path),
        "contour_sha256": sha256_file(contour_path),
        "summary": measurements["summary"],
        "segmentation": segmentation,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "artifacts": {
            "mask": str(mask_path.relative_to(ROOT)),
            "contour": str(contour_path.relative_to(ROOT)),
            "measurements": str((sample_dir / "measurements.json").relative_to(ROOT)),
            "segmentation": str((sample_dir / "segmentation.json").relative_to(ROOT)),
        },
    }


def freeze_baseline(output_dir=DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = sorted(SAMPLES_DIR.glob("*.jpg"))
    if not samples:
        raise FileNotFoundError(f"No baseline samples found in {SAMPLES_DIR}")

    records = [freeze_sample(sample, output_dir) for sample in samples]
    write_json(output_dir / "strategy.json", FROZEN_STRATEGY)
    manifest = {
        "baseline_id": "v1-threshold-2026-07",
        "purpose": "Deterministic comparison point before v2 operator development.",
        "strategy_file": str((output_dir / "strategy.json").relative_to(ROOT)),
        "observations_file": str((output_dir / "observations.md").relative_to(ROOT)),
        "known_limitations": KNOWN_LIMITATIONS,
        "samples": records,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Freeze the deterministic v1 threshold baseline")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main():
    manifest = freeze_baseline(parse_args().output_dir)
    print(f"Frozen {len(manifest['samples'])} samples for {manifest['baseline_id']}")


if __name__ == "__main__":
    main()
