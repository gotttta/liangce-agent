from pathlib import Path

import numpy as np
from PIL import Image

from ui.app import run_feedback, run_initial
from providers.vision import MockVisionProvider


def test_run_initial_returns_chat_images_table_and_state(tmp_path, monkeypatch):
    target = tmp_path / "target.png"
    image = np.full((24, 24), 40, dtype=np.uint8)
    image[8:14, 9:15] = 180
    Image.fromarray(image, mode="L").save(target)

    monkeypatch.setattr("ui.app.OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr("ui.app.build_runtime_provider", lambda: MockVisionProvider())

    chat, annotated, mask, editor_bg, rows, strategy, metrics, files, state = run_initial(
        str(target),
        "找亮色残留，量面积和数量",
        None,
        "pixel",
    )

    assert chat
    assert Path(annotated).exists()
    assert Path(mask).exists()
    assert editor_bg == annotated
    assert rows[0][1] > 0
    assert strategy["measurement_type"] == "area_count"
    assert metrics["status"] == "skipped"
    assert len(files) == 5
    assert state["status"] == "ok"


def test_run_feedback_appends_second_iteration(tmp_path, monkeypatch):
    target = tmp_path / "target.png"
    image = np.full((24, 24), 40, dtype=np.uint8)
    image[8:14, 9:15] = 180
    Image.fromarray(image, mode="L").save(target)
    monkeypatch.setattr("ui.app.OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr("ui.app.build_runtime_provider", lambda: MockVisionProvider())

    *_, state = run_initial(str(target), "找亮色残留，量面积和数量", None, "pixel")
    feedback_image = np.zeros((24, 24, 4), dtype=np.uint8)
    feedback_image[10:18, 10:18, 0] = 255
    feedback_image[10:18, 10:18, 3] = 255

    *_, new_state = run_feedback(
        state,
        {"composite": feedback_image},
        "这里漏了，重新调高灵敏度",
    )

    assert new_state["iteration"] == 1
    assert Path(new_state["run_dir"], "iteration_1", "graph_state.json").exists()
