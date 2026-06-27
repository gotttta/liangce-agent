from pathlib import Path

from ui.gradio_adapters import (
    measurement_rows,
    output_files,
    save_feedback_layer,
    state_to_chat_messages,
)


def test_measurement_rows_convert_results_to_table_rows():
    state = {
        "measurements": {
            "results": [
                {"id": 1, "area": 12, "bbox": [2, 3, 4, 5], "width": 4, "height": 5, "aspect_ratio": 0.8}
            ]
        }
    }

    rows = measurement_rows(state)

    assert rows == [[1, 12, 4, 5, 0.8, "[2, 3, 4, 5]"]]


def test_state_to_chat_messages_uses_openai_role_content_format():
    state = {
        "conversation": [
            {"role": "assistant", "content": "我先看图。"},
            {"role": "assistant", "content": "这一轮标出 1 个区域。"},
        ]
    }

    messages = state_to_chat_messages(state)

    assert messages == [
        {"role": "assistant", "content": "我先看图。"},
        {"role": "assistant", "content": "这一轮标出 1 个区域。"},
    ]


def test_output_files_returns_latest_artifacts(tmp_path):
    run_dir = tmp_path / "outputs" / "run_001"
    iteration_dir = run_dir / "iteration_0"
    iteration_dir.mkdir(parents=True)
    state = {
        "run_dir": str(run_dir),
        "iteration": 0,
        "annotated_image_path": str(iteration_dir / "result_annotated.png"),
        "predicted_mask_path": str(iteration_dir / "mask.png"),
    }

    files = output_files(state)

    assert files == [
        str(iteration_dir / "result_annotated.png"),
        str(iteration_dir / "mask.png"),
        str(iteration_dir / "measurements.json"),
        str(iteration_dir / "strategy.json"),
        str(iteration_dir / "graph_state.json"),
    ]


def test_save_feedback_layer_writes_composite_image(tmp_path):
    import numpy as np

    image = np.zeros((2, 2, 4), dtype=np.uint8)
    image[:, :, 3] = 255

    path = save_feedback_layer({"composite": image}, tmp_path)

    assert Path(path).exists()
    assert Path(path).suffix == ".png"
