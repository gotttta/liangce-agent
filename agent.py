import argparse
import json
from types import SimpleNamespace
from pathlib import Path


def run(args):
    from graph_workflow import run_graph
    from providers.vision import MockVisionProvider

    state = run_graph(
        target_image_path=args.target,
        description=args.description,
        reference_annotation_path=args.reference,
        output_root=args.output_root,
        provider=MockVisionProvider(),
        unit=args.unit,
    )
    return Path(state["run_dir"]), state_to_output(state)


def state_to_output(state):
    return {
        "defect_type": state["strategy"]["defect_type"],
        "measurement_type": state["strategy"]["measurement_type"],
        "unit": state["measurements"]["summary"]["unit"],
        "status": state["status"],
        "results": state["measurements"]["results"],
        "summary": state["measurements"]["summary"],
        "segmentation": state.get("segmentation", {}),
        "metrics": state.get("metrics", {}),
        "notes": state["strategy"].get("notes", []),
        "conversation": state.get("conversation", []),
        "latest_iteration": state.get("iteration", 0),
        "latest_annotated_image": state.get("annotated_image_path"),
        "latest_mask": state.get("predicted_mask_path"),
    }


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
