from pathlib import Path


def state_to_chat_messages(state):
    return [
        {"role": item.get("role", "assistant"), "content": item.get("content", "")}
        for item in state.get("conversation", [])
    ]


def measurement_rows(state):
    rows = []
    for item in state.get("measurements", {}).get("results", []):
        rows.append([
            item.get("id"),
            item.get("area"),
            item.get("width"),
            item.get("height"),
            item.get("aspect_ratio"),
            str(item.get("bbox")),
        ])
    return rows


def output_files(state):
    run_dir = Path(state["run_dir"])
    iteration_dir = run_dir / f"iteration_{state.get('iteration', 0)}"
    return [
        state["annotated_image_path"],
        state["predicted_mask_path"],
        str(iteration_dir / "measurements.json"),
        str(iteration_dir / "strategy.json"),
        str(iteration_dir / "graph_state.json"),
    ]


def save_feedback_layer(editor_value, run_dir):
    if not editor_value:
        return None
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "feedback_brush.png"

    if isinstance(editor_value, dict):
        composite = editor_value.get("composite")
        if composite is not None:
            return _save_editor_image(composite, output_path)
        layers = editor_value.get("layers") or []
        if layers:
            return _save_editor_image(layers[-1], output_path)
    return None


def _save_editor_image(value, output_path):
    if isinstance(value, str):
        return value

    from PIL import Image

    Image.fromarray(value).save(output_path)
    return str(output_path)
