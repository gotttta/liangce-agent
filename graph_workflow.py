import json
from datetime import datetime
from pathlib import Path

from langgraph.graph import END, StateGraph

from agent_types import AgentState
from core.measurement.area import measure_components
from core.measurement.evaluation import evaluate_prediction
from core.preprocessing import load_grayscale
from core.segmentation import segment_with_strategy
from core.visualization import save_annotated_image, save_mask_image
from providers.vision import build_runtime_provider


def run_graph(
    target_image_path,
    description,
    output_root="outputs",
    reference_annotation_path=None,
    feedback_brush_path=None,
    feedback_text=None,
    provider=None,
    unit="pixel",
    max_iterations=2,
    existing_run_dir=None,
    initial_iteration=0,
):
    graph = build_graph(provider=provider or build_runtime_provider(), max_iterations=max_iterations)
    initial_state = {
        "target_image_path": str(target_image_path),
        "description": description,
        "reference_annotation_path": str(reference_annotation_path) if reference_annotation_path else None,
        "feedback_brush_path": str(feedback_brush_path) if feedback_brush_path else None,
        "feedback_text": feedback_text,
        "run_dir": str(existing_run_dir or make_run_dir(output_root)),
        "iteration": initial_iteration,
        "conversation": [],
        "run_history": [],
        "status": "pending",
        "errors": [],
        "unit": unit,
    }
    return graph.invoke(initial_state)


def build_graph(provider, max_iterations=2):
    workflow = StateGraph(AgentState)
    workflow.add_node("prepare_inputs", prepare_inputs)
    workflow.add_node("vision_strategy", make_vision_strategy_node(provider))
    workflow.add_node("segment_defects", segment_defects)
    workflow.add_node("measure_defects", measure_defects)
    workflow.add_node("render_outputs", render_outputs)
    workflow.add_node("prepare_feedback_iteration", prepare_feedback_iteration)

    workflow.set_entry_point("prepare_inputs")
    workflow.add_edge("prepare_inputs", "vision_strategy")
    workflow.add_edge("vision_strategy", "segment_defects")
    workflow.add_edge("segment_defects", "measure_defects")
    workflow.add_edge("measure_defects", "render_outputs")
    workflow.add_conditional_edges(
        "render_outputs",
        lambda state: route_after_render(state, max_iterations=max_iterations),
        {"rerun": "prepare_feedback_iteration", "end": END},
    )
    workflow.add_edge("prepare_feedback_iteration", "vision_strategy")
    return workflow.compile()


def make_run_dir(output_root):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(output_root) / f"{timestamp}_agent"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def prepare_inputs(state):
    target = Path(state["target_image_path"])
    if not target.exists():
        raise FileNotFoundError(f"Image not found: {target}")
    if not state.get("description"):
        raise ValueError("缺少缺陷描述")
    Path(state["run_dir"]).mkdir(parents=True, exist_ok=True)
    state["conversation"].append({"role": "assistant", "content": "我先看图并生成一版量测策略。"})
    return state


def make_vision_strategy_node(provider):
    def vision_strategy(state):
        previous_state = {
            "strategy": state.get("strategy"),
            "measurements": state.get("measurements"),
            "feedback_text": state.get("feedback_text"),
            "feedback_brush_path": state.get("feedback_brush_path"),
        }
        strategy = provider.create_strategy(
            target_image_path=state["target_image_path"],
            description=state["description"],
            reference_annotation_path=state.get("reference_annotation_path"),
            previous_state=previous_state if state.get("iteration", 0) > 0 else None,
        )
        state["strategy"] = strategy
        state["conversation"].append({
            "role": "assistant",
            "content": f"我会使用{strategy['segmentation']['method']}，最小面积阈值为{strategy['segmentation']['min_area_px']} px。",
        })
        return state

    return vision_strategy


def segment_defects(state):
    image = load_grayscale(state["target_image_path"])
    mask, meta = segment_with_strategy(image, state["strategy"])
    iteration_dir = current_iteration_dir(state)
    mask_path = iteration_dir / "mask.png"
    save_mask_image(mask, mask_path)
    state["predicted_mask_path"] = str(mask_path)
    state["segmentation"] = meta
    state["_mask_array"] = mask
    return state


def measure_defects(state):
    min_area = int(state["strategy"]["segmentation"].get("min_area_px", 20))
    unit = state.get("unit") or state["strategy"]["measurement"].get("unit", "pixel")
    measurements = measure_components(state["_mask_array"], min_area=min_area, unit=unit)
    state["measurements"] = measurements

    reference_path = state.get("reference_annotation_path")
    if reference_path:
        reference = load_grayscale(reference_path) > 0
        state["metrics"] = evaluate_prediction(state["_mask_array"], reference, min_area=min_area)
    else:
        state["metrics"] = {"status": "skipped", "reason": "no_reference_annotation"}
    return state


def render_outputs(state):
    iteration_dir = current_iteration_dir(state)
    annotated_path = iteration_dir / "result_annotated.png"
    save_annotated_image(state["target_image_path"], state["measurements"]["results"], annotated_path)
    state["annotated_image_path"] = str(annotated_path)
    state["status"] = "ok"

    write_json(iteration_dir / "strategy.json", state["strategy"])
    write_json(iteration_dir / "measurements.json", state["measurements"])
    write_json(iteration_dir / "metrics.json", state["metrics"])

    serializable = {key: value for key, value in state.items() if key != "_mask_array"}
    write_json(iteration_dir / "graph_state.json", serializable)

    summary = state["measurements"]["summary"]
    state["conversation"].append({
        "role": "assistant",
        "content": f"这一轮标出 {summary['count']} 个区域，总面积 {summary['total_area']} {summary['unit']}。",
    })
    state["run_history"].append({
        "iteration": state["iteration"],
        "directory": str(iteration_dir),
        "summary": summary,
        "metrics": state["metrics"],
    })
    state.pop("_mask_array", None)
    return state


def route_after_render(state, max_iterations):
    has_feedback = bool(state.get("feedback_text") or state.get("feedback_brush_path"))
    if has_feedback and state.get("iteration", 0) + 1 < max_iterations:
        return "rerun"
    return "end"


def prepare_feedback_iteration(state):
    state["iteration"] = state.get("iteration", 0) + 1
    state["conversation"].append({
        "role": "assistant",
        "content": "收到你的画笔和文字反馈，我会修订策略后重新跑一轮。",
    })
    return state


def current_iteration_dir(state):
    iteration_dir = Path(state["run_dir"]) / f"iteration_{state.get('iteration', 0)}"
    iteration_dir.mkdir(parents=True, exist_ok=True)
    return iteration_dir


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
