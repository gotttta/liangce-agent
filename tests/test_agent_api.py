from pathlib import Path

import numpy as np
from PIL import Image

from agent import run_from_paths


def test_run_from_paths_returns_latest_iteration_result(tmp_path):
    target = tmp_path / "target.png"
    image = np.full((24, 24), 30, dtype=np.uint8)
    image[8:13, 9:14] = 160
    Image.fromarray(image, mode="L").save(target)

    run_dir, output = run_from_paths(
        target=target,
        description="找亮色残留，统计面积和数量",
        output_root=tmp_path / "outputs",
    )

    assert Path(run_dir).exists()
    assert output["status"] == "ok"
    assert output["summary"]["count"] == 1
    assert (Path(run_dir) / "iteration_0" / "result_annotated.png").exists()
