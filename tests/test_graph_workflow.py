from pathlib import Path

import numpy as np
from PIL import Image

from graph_workflow import run_graph
from providers.vision import MockVisionProvider


def test_run_graph_writes_iteration_outputs(tmp_path):
    target = tmp_path / "target.png"
    image = np.full((30, 30), 40, dtype=np.uint8)
    image[10:16, 12:18] = 180
    Image.fromarray(image, mode="L").save(target)

    state = run_graph(
        target_image_path=target,
        description="找亮色残留，量面积和数量",
        output_root=tmp_path / "outputs",
        provider=MockVisionProvider(),
    )

    assert state["status"] == "ok"
    assert state["iteration"] == 0
    assert Path(state["predicted_mask_path"]).exists()
    assert Path(state["annotated_image_path"]).exists()
    assert state["measurements"]["summary"]["count"] == 1
    assert state["conversation"]
