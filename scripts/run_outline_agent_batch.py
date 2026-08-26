"""Run the vision agent on the five outline samples."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.agent_graph import run_agent_graph
from providers.vision import AliyunVisionProvider, load_env_file


DESCRIPTION = (
    "提取当前显微图中需要标注的缺陷目标轮廓并用荧光绿色显示。"
    "对于椭圆纳米孔或椭圆孔阵列，标注目标椭圆孔的完整外轮廓；"
    "对于规则线路图，标注中央异常颗粒或破坏区域的完整外轮廓。"
    "排除正常周期背景、文字、边框和比例尺。请根据当前图片自行选择和组合最合适的图像算子，"
    "不要套用固定默认流程；输出轮廓标注图和对应Mask。"
)


class VisionPlanOnlyProvider(AliyunVisionProvider):
    """Use remote visual planning and deterministic local candidate selection."""

    def review_candidates(self, target_image_path, description, candidates, reference_examples=None):
        completed = [item for item in candidates if item.get("status") == "completed"]
        if not completed:
            return {"decision": "cannot_determine", "selected_candidate": None, "reason": "没有可复查的候选"}
        return {
            "decision": "present",
            "selected_candidate": completed[0].get("name"),
            "reason": "批处理测试使用本地复查，避免重复调用远程模型。",
        }

SAMPLES = (
    "01_elliptical_nanohole_array_sem_a.jpg",
    "01_elliptical_nanohole_array_sem_b.jpg",
    "04_triangular_elliptical_hole_array_sem.jpg",
    "in_film_particle_left_pattern.jpg",
    "in_film_particle_middle_defect.jpg",
)


def main():
    root = Path("data/samples/outline")
    output = Path("outputs/outline_agent_test_v2")
    output.mkdir(parents=True, exist_ok=True)
    load_env_file()
    provider = VisionPlanOnlyProvider(timeout_seconds=45, max_retries=0)
    summary = []
    for filename in SAMPLES:
        path = root / filename
        print(f"RUN {filename}", flush=True)
        try:
            state = run_agent_graph(
                target_image_path=str(path),
                description=DESCRIPTION,
                output_root=str(output),
                unit="pixel",
                max_candidates=3,
                max_auto_revisions=1,
                provider=provider,
            )
            item = {
                "input": str(path),
                "status": state.get("status"),
                "agent_status": state.get("agent_status"),
                "run_dir": state.get("run_dir"),
                "selected_candidate": state.get("selected_candidate"),
                "annotated_image_path": state.get("annotated_image_path"),
                "predicted_mask_path": state.get("predicted_mask_path"),
                "pipeline": state.get("pipeline"),
                "quality_report": state.get("quality_report"),
                "review": state.get("review"),
                "interrupted": bool(state.get("__interrupt__")),
            }
        except Exception as exc:
            item = {
                "input": str(path),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        summary.append(item)
        print(json.dumps(item, ensure_ascii=False, indent=2, default=str), flush=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"SUMMARY {output / 'summary.json'}")


if __name__ == "__main__":
    main()
