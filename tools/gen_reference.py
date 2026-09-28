# 需要在 tmp/sam-venv 下运行：
#   tmp/sam-venv/bin/python tools/gen_reference.py ...
# 依赖：transformers, torch, openai（已在 tmp/sam-venv 里安装）
import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

# 以 python tools/gen_reference.py 方式运行时 sys.path[0] 是 tools/，补上项目根目录以导入 core
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.reference_store import ReferenceStore
from core.sam_session import (
    MIN_MEAN_IOU,
    SamSession,
    _select_array_masks,
    _select_defect_masks,
    _tint,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _load_env() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _make_llm() -> tuple[object, str]:
    from openai import OpenAI

    _load_env()
    client = OpenAI(
        api_key=os.environ["ALIYUN_API_KEY"],
        base_url=os.environ["ALIYUN_BASE_URL"],
        timeout=180,
    )
    return client, os.environ["ALIYUN_VISION_MODEL"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="为一张样本图生成 SAM 参考掩膜并写入 ReferenceStore（需在 tmp/sam-venv 下运行）"
    )
    parser.add_argument("--image", required=True, help="样本图路径")
    parser.add_argument("--task", required=True, help="标注任务描述")
    parser.add_argument("--target-type", required=True, choices=("defect", "array"), help="目标类型")
    parser.add_argument(
        "--workspace",
        default=str(PROJECT_ROOT / "workspace" / "references"),
        help="ReferenceStore 根目录（默认 workspace/references）",
    )
    parser.add_argument("--sam-model", default="facebook/sam-vit-base", help="SAM 模型名")
    parser.add_argument("--show", action="store_true", help="生成后用 cv2 展示叠加图，等待键盘确认")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    image_path = Path(args.image)
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"无法读取图片: {image_path}")

    client, model = _make_llm()
    session = SamSession(args.sam_model)
    print(f"SAM 模型 {args.sam_model} 运行在 {session.device}")

    if args.target_type == "array":
        parts = _select_array_masks(client, model, session, image, args.task)
    else:
        parts = _select_defect_masks(client, model, session, image, args.task)

    if not parts:
        raise SystemExit("没有选出任何目标掩膜，未写入参考")

    merged = np.zeros(image.shape[:2], dtype=bool)
    for _, mask in parts:
        merged |= mask
    mean_iou = float(np.mean([score for score, _ in parts]))
    skip = mean_iou < MIN_MEAN_IOU
    if skip:
        print(f"警告: SAM IoU 分数均值 {mean_iou:.3f} 低于 {MIN_MEAN_IOU}，已标记 skip_scoring")

    meta = {
        "image_id": image_path.stem,
        "image_path": str(image_path),
        "task": args.task,
        "target_type": args.target_type,
        "sam_iou_score": mean_iou,
        "skip_scoring": skip,
        "skip_reason": "" if not skip else f"SAM IoU 分数均值 {mean_iou:.3f} 低于 {MIN_MEAN_IOU}",
        "confirmed_at": _utc_now_iso(),
        "confirmed_by": "user",
    }
    meta_path = ReferenceStore(args.workspace).save(image_path.stem, merged, meta)
    print(f"已写入 {meta_path}（{len(parts)} 个掩膜，SAM IoU 均值 {mean_iou:.3f}）")

    if args.show:
        overlay = _tint(image, merged, (0, 255, 0))
        cv2.imshow("reference", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
