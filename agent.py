import argparse
import json
from datetime import datetime
from types import SimpleNamespace
from pathlib import Path

from core.measurement.area import measure_components
from core.preprocessing import load_grayscale
from core.segmentation import segment_anomalies
from core.visualization import save_annotated_image, save_mask_image


def build_strategy(description, unit="pixel"):
    text = (description or "").lower()
    measurement_type = "area_count"
    if any(word in text for word in ["bridge", "open", "gap", "断线", "桥连", "缺口"]):
        measurement_type = "bridge_open"
    elif any(word in text for word in ["cd", "width", "space", "线宽", "间距"]):
        measurement_type = "cd_space"

    return {
        "defect_type": "unknown",
        "target_structure": "unknown",
        "measurement_type": measurement_type,
        "preprocess": {
            "grayscale": True,
            "normalize": True,
        },
        "segmentation": {
            "method": "auto_bright_dark_threshold",
            "sensitivity": 1.8,
        },
        "measurement": {
            "min_area_px": 20,
            "unit": unit,
        },
        "notes": [
            "MVP baseline: no LLM or SAM is connected yet.",
            "If no calibration is provided, results are reported in pixels.",
        ],
    }


def make_run_dir(output_root, defect_type):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_defect = "".join(ch if ch.isalnum() else "_" for ch in defect_type).strip("_") or "unknown"
    run_dir = Path(output_root) / f"{timestamp}_{safe_defect}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_algorithm_stub(run_dir, args):
    script = f'''"""Generated MVP runner record.

This file records the command shape used for this run. In the current MVP,
the reusable implementation lives in project modules under core/.
"""

COMMAND = {{
    "target": {str(args.target)!r},
    "reference": {str(args.reference) if args.reference else None!r},
    "description": {args.description!r},
    "output_dir": {str(run_dir)!r},
}}
'''
    (run_dir / "algorithm.py").write_text(script, encoding="utf-8")


def run(args):
    strategy = build_strategy(args.description, unit=args.unit)
    run_dir = make_run_dir(args.output_root, strategy["defect_type"])

    image = load_grayscale(args.target)
    mask, segmentation_meta = segment_anomalies(
        image,
        sensitivity=strategy["segmentation"]["sensitivity"],
    )
    measurements = measure_components(
        mask,
        min_area=strategy["measurement"]["min_area_px"],
        unit=strategy["measurement"]["unit"],
    )

    notes = list(strategy["notes"])
    if args.reference is None:
        notes.append("No reference image was provided for this run.")
    if args.unit == "pixel":
        notes.append("No physical calibration was provided; values use pixel units.")
    if not measurements["results"]:
        notes.append("No defect region passed the current area filter.")

    output = {
        "defect_type": strategy["defect_type"],
        "measurement_type": strategy["measurement_type"],
        "unit": strategy["measurement"]["unit"],
        "status": "ok" if measurements["results"] else "empty",
        "results": measurements["results"],
        "summary": measurements["summary"],
        "segmentation": segmentation_meta,
        "notes": notes,
    }

    save_mask_image(mask, run_dir / "mask.png")
    save_annotated_image(args.target, measurements["results"], run_dir / "result_annotated.png")
    (run_dir / "strategy.json").write_text(json.dumps(strategy, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "measurements.json").write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    write_algorithm_stub(run_dir, args)

    return run_dir, output


def parse_args():
    parser = argparse.ArgumentParser(description="DRAM defect metrology MVP runner")
    parser.add_argument("--target", required=True, help="Path to the unannotated target/wafer image")
    parser.add_argument("--description", required=True, help="Natural language defect description")
    parser.add_argument("--reference", help="Optional handbook/reference image path")
    parser.add_argument("--unit", default="pixel", help="Metrology unit, defaults to pixel")
    parser.add_argument("--output-root", default="outputs", help="Directory for run outputs")
    return parser.parse_args()


def run_from_paths(target, description, reference=None, unit="pixel", output_root="outputs"):
    args = SimpleNamespace(
        target=target,
        description=description,
        reference=reference,
        unit=unit,
        output_root=output_root,
    )
    return run(args)


def main():
    run_dir, output = run(parse_args())
    print(f"Output written to: {run_dir}")
    print(json.dumps(output["summary"], indent=2, ensure_ascii=False))
