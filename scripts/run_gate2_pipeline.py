import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from core.measurement.area import measure_components
from core.pipelines.periodic_particle import (
    PeriodicBackgroundUnavailable,
    run_periodic_particle_pipeline,
)
from core.preprocessing import load_grayscale
from core.visualization import save_annotated_image, save_mask_image


ROOT = Path(__file__).resolve().parents[1]

GATE2_SAMPLE_CONFIG = {
    "in_film_particle_left_pattern.jpg": {"roi": [105, 175, 65, 70], "percentile": 70.0},
    "in_film_particle_middle_defect.jpg": {"roi": [90, 140, 110, 150], "percentile": 70.0},
    "in_film_particle_right_zoom.jpg": {"roi": [90, 70, 250, 250], "percentile": 70.0},
}


def run_samples(samples_dir, output_dir):
    samples_dir = Path(samples_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"pipeline": "gate2_fixed_periodic_particle", "samples": []}
    for sample in sorted(samples_dir.glob("*.jpg")):
        sample_dir = output_dir / sample.stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        config = GATE2_SAMPLE_CONFIG.get(sample.name, {})
        try:
            result = run_periodic_particle_pipeline(load_grayscale(sample), **config)
        except PeriodicBackgroundUnavailable as exc:
            report["samples"].append({"sample": sample.name, "status": "unsupported", "reason": str(exc)})
            continue

        mask_path = sample_dir / "mask.png"
        contour_path = sample_dir / "result_contour.png"
        save_mask_image(result.mask.data, mask_path)
        measurements = measure_components(result.mask.data, min_area=1, unit="pixel")
        save_annotated_image(sample, measurements["results"], contour_path, mask=result.mask.data)
        for name, debug_image in result.debug_images.items():
            _save_debug_image(debug_image, sample_dir / f"{name}.png")
        _write_json(sample_dir / "trace.json", list(result.trace))
        _write_json(sample_dir / "period.json", result.period)
        _write_json(sample_dir / "measurements.json", measurements)
        report["samples"].append({
            "sample": sample.name,
            "status": "ok",
            "period": result.period,
            "config": config,
            "summary": measurements["summary"],
            "artifacts": {
                "mask": _display_path(mask_path),
                "contour": _display_path(contour_path),
                "trace": _display_path(sample_dir / "trace.json"),
            },
        })
    _write_json(output_dir / "report.json", report)
    return report


def _save_debug_image(array, path):
    data = np.asarray(array)
    if data.dtype == bool:
        rendered = data.astype(np.uint8) * 255
    else:
        minimum = float(np.min(data))
        maximum = float(np.max(data))
        if maximum - minimum < 1e-6:
            rendered = np.zeros(data.shape, dtype=np.uint8)
        else:
            rendered = np.clip((data - minimum) / (maximum - minimum) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(rendered, mode="L").save(path)


def _display_path(path):
    """Keep reports readable for both repo-local and temporary output dirs."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Gate 2 fixed periodic-particle pipeline")
    parser.add_argument("--samples-dir", type=Path, default=ROOT / "data" / "samples")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "gate2_fixed")
    return parser.parse_args()


def main():
    args = parse_args()
    report = run_samples(args.samples_dir, args.output_dir)
    succeeded = sum(item["status"] == "ok" for item in report["samples"])
    print(f"Gate 2 fixed pipeline completed {succeeded}/{len(report['samples'])} samples")


if __name__ == "__main__":
    main()
