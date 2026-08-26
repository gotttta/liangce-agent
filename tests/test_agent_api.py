from pathlib import Path

import numpy as np
from PIL import Image

from agent import run_from_paths
from providers.vision import MockVisionProvider


def test_run_from_paths_returns_latest_iteration_result(tmp_path, monkeypatch):
    target = tmp_path / "target.png"
    image = np.full((24, 24), 30, dtype=np.uint8)
    image[8:13, 9:14] = 160
    Image.fromarray(image, mode="L").save(target)
    monkeypatch.setattr("agent.build_runtime_provider", lambda: MockVisionProvider())

    run_dir, output = run_from_paths(
        target=target,
        description="找亮色残留，统计面积和数量",
        output_root=tmp_path / "outputs",
    )

    assert Path(run_dir).exists()
    assert output["status"] == "ok"
    assert output["summary"]["count"] == 1
    assert (Path(run_dir) / "iteration_0" / "result_annotated.png").exists()


def test_run_from_paths_uses_runtime_provider_factory(tmp_path, monkeypatch):
    target = tmp_path / "target.png"
    image = np.full((24, 24), 30, dtype=np.uint8)
    image[8:13, 9:14] = 160
    Image.fromarray(image, mode="L").save(target)

    calls = []

    def fake_runtime_provider():
        calls.append("called")
        return MockVisionProvider()

    monkeypatch.setattr("agent.build_runtime_provider", fake_runtime_provider)

    run_from_paths(
        target=target,
        description="找亮色残留，统计面积和数量",
        output_root=tmp_path / "outputs",
    )

    assert calls == ["called"]


def test_run_from_paths_persists_handbook_examples_as_few_shot_context(tmp_path, monkeypatch):
    target = tmp_path / "target.png"
    handbook = tmp_path / "handbook.png"
    image = np.full((24, 24), 30, dtype=np.uint8)
    image[8:13, 9:14] = 160
    Image.fromarray(image, mode="L").save(target)
    Image.fromarray(image, mode="L").save(handbook)
    monkeypatch.setattr("agent.build_runtime_provider", lambda: MockVisionProvider())

    run_dir, _ = run_from_paths(
        target=target,
        description="参考甲方示例提取轮廓",
        reference=[str(handbook)],
        output_root=tmp_path / "outputs",
    )

    state = __import__("json").loads(
        (Path(run_dir) / "iteration_0" / "graph_state.json").read_text(encoding="utf-8")
    )
    assert state["reference_examples"][0]["image_path"] == str(handbook)
