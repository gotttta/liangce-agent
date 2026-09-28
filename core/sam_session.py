# SAM 依赖（transformers, torch）只在 tmp/sam-venv 里有，不在主 venv 里。
# 所有 import 都在方法内部，主 venv 下 import 本文件不会报错。
import base64
import json
import re
from typing import Any

import cv2
import numpy as np
from PIL import Image

PALETTE = [
    (255, 80, 80), (80, 200, 255), (80, 255, 120), (255, 200, 60),
    (200, 120, 255), (255, 120, 200), (60, 255, 255),
]
LETTERS = "ABCDEF"
MIN_MEAN_IOU = 0.7
CANDIDATE_SCORE_MIN = 0.86
CANDIDATE_IOU_MAX = 0.6
OPTION_IOU_MAX = 0.9
MAX_CANDIDATES = 80
GRID_DIVISIONS = 24


def _encode_png(image: np.ndarray) -> str:
    return base64.b64encode(cv2.imencode(".png", image)[1].tobytes()).decode()


def _chat_json(client: Any, model: str, images: list[np.ndarray], prompt: str) -> dict:
    content = [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + _encode_png(image)}}
        for image in images
    ]
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[{"role": "user", "content": content + [{"type": "text", "text": prompt}]}],
    )
    text = response.choices[0].message.content or ""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


class SamSession:
    def __init__(self, model_name: str) -> None:
        import torch
        from transformers import SamModel, SamProcessor

        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.processor = SamProcessor.from_pretrained(model_name)
        self.model = SamModel.from_pretrained(model_name).to(self.device).eval()

    def embed(self, image_bgr: np.ndarray) -> tuple[Any, Any, Any]:
        inputs = self.processor(
            Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)), return_tensors="pt"
        )
        with self.torch.no_grad():
            embeddings = self.model.get_image_embeddings(
                inputs["pixel_values"].float().to(self.device)
            )
        return embeddings, inputs["original_sizes"], inputs["reshaped_input_sizes"]

    def predict(
        self,
        embeddings: Any,
        sizes: tuple[Any, Any],
        point_groups: list[list[list[float]]],
        box: list[float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """point_groups 是原图像素坐标；返回每组 3 个候选掩膜和对应分数。"""
        torch = self.torch
        scale = sizes[1][0][1].item() / sizes[0][0][1].item()
        points = torch.tensor(
            [[[[x * scale, y * scale] for x, y in group] for group in point_groups]],
            dtype=torch.float32,
        )
        labels = torch.tensor(
            [[[1] * len(group) for group in point_groups]], dtype=torch.int64
        )
        extra: dict = {}
        if box is not None:
            extra["input_boxes"] = torch.tensor(
                [[[v * scale for v in box]]], dtype=torch.float32
            ).to(self.device)
        with torch.no_grad():
            output = self.model(
                image_embeddings=embeddings,
                input_points=points.to(self.device),
                input_labels=labels.to(self.device),
                multimask_output=True,
                **extra,
            )
        masks = self.processor.post_process_masks(
            output.pred_masks.cpu(), sizes[0], sizes[1]
        )[0]
        return masks, output.iou_scores.cpu()[0]


def _grid_candidates(
    session: SamSession,
    embeddings: Any,
    sizes: tuple[Any, Any],
    image_shape: tuple[int, int],
    grid: int = GRID_DIVISIONS,
) -> list[tuple[float, np.ndarray]]:
    height, width = image_shape
    divisions = np.linspace(0, height - 1, grid + 2)[1:-1]
    columns = np.linspace(0, width - 1, grid + 2)[1:-1]
    points = [[[float(x), float(y)]] for y in divisions for x in columns]
    found: list[tuple[float, np.ndarray]] = []
    for start in range(0, len(points), 64):
        masks, scores = session.predict(embeddings, sizes, points[start:start + 64])
        for mask_set, score_set in zip(masks, scores):
            best = int(score_set.argmax())
            mask = mask_set[best].numpy().astype(bool)
            area = int(mask.sum())
            if float(score_set[best]) >= CANDIDATE_SCORE_MIN and max(20, 3e-4 * height * width) <= area <= 0.4 * height * width:
                found.append((float(score_set[best]), mask))
    found.sort(key=lambda item: -item[0])
    kept: list[tuple[float, np.ndarray]] = []
    for score, mask in found:
        if all(
            (mask & other).sum() / (mask | other).sum() < CANDIDATE_IOU_MAX
            for _, other in kept
        ):
            kept.append((score, mask))
    return kept[:MAX_CANDIDATES]


def _tint(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    layer = image.copy()
    layer[mask] = color
    return cv2.addWeighted(layer, alpha, image, 1 - alpha, 0)


def _outline(vis: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], label: str | None = None) -> None:
    big = cv2.resize(
        mask.astype(np.uint8), (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    contours, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(vis, contours, -1, color, 2)
    if label is not None and contours:
        x, y, _, _ = cv2.boundingRect(max(contours, key=cv2.contourArea))
        for thickness, stroke in ((4, (0, 0, 0)), (1, color)):
            cv2.putText(vis, label, (x + 2, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, stroke, thickness)


def _resized(image: np.ndarray, target: float) -> np.ndarray:
    height, width = image.shape[:2]
    factor = max(1.0, target / max(height, width))
    return cv2.resize(image, (int(width * factor), int(height * factor)), interpolation=cv2.INTER_CUBIC)


def _draw_numbered_candidates(image: np.ndarray, masks: list[np.ndarray]) -> tuple[np.ndarray, float]:
    vis = _resized(image, 900)
    factor = vis.shape[1] / image.shape[1]
    for index, mask in enumerate(masks):
        big = cv2.resize(
            mask.astype(np.uint8), (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        color = PALETTE[index % len(PALETTE)]
        contours, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, contours, -1, color, 2)
        dist = cv2.distanceTransform(big, cv2.DIST_L2, 3)
        y, x = np.unravel_index(dist.argmax(), dist.shape)
        for thickness, stroke in ((4, (0, 0, 0)), (1, color)):
            cv2.putText(vis, str(index), (int(x) - 8, int(y) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, stroke, thickness)
    return vis, factor


def _draw_review_overlay(image: np.ndarray, parts: list[tuple[float, np.ndarray]]) -> np.ndarray:
    vis = _resized(image, 900)
    for index, (_, mask) in enumerate(parts):
        color = PALETTE[index % len(PALETTE)]
        big = cv2.resize(
            mask.astype(np.uint8), (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        vis = _tint(vis, big, color, 0.3)
        _outline(vis, mask, color, f"R{index}")
    return vis


def _mask_from_point(
    client: Any,
    model: str,
    session: SamSession,
    image: np.ndarray,
    embeddings: Any,
    sizes: tuple[Any, Any],
    item: dict,
    task: str,
) -> tuple[tuple[float, np.ndarray] | None, str]:
    height, width = image.shape[:2]
    x, y = (float(value) for value in item["point"])
    box = item.get("box")
    box_variants = [box, None] if box and len(box) == 4 else [None]
    options: list[tuple[float, np.ndarray]] = []
    for candidate_box in box_variants:
        masks, scores = session.predict(
            embeddings,
            sizes,
            [[[x, y]]],
            box=[float(value) for value in candidate_box] if candidate_box else None,
        )
        for choice in range(3):
            mask = masks[0][choice].numpy().astype(bool)
            if 20 <= mask.sum() <= 0.4 * height * width and all(
                (mask & other).sum() / (mask | other).sum() < OPTION_IOU_MAX
                for _, other in options
            ):
                options.append((float(scores[0][choice]), mask))
    if not options:
        return None, "没有可用的 SAM 掩膜"
    panels = []
    for letter, (_, mask) in enumerate(options[:6]):
        panel = _resized(_tint(image, mask, (255, 0, 255), 0.35), 420)
        _outline(panel, mask, (255, 0, 255))
        for thickness, stroke in ((5, (0, 0, 0)), (2, (0, 255, 255))):
            cv2.putText(panel, LETTERS[letter], (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, stroke, thickness)
        panels.append(panel)
    sheet = cv2.hconcat(panels)
    prompt = (
        f"第一张是原图，第二张是{len(panels)}个候选掩膜（紫色，左上角字母编号）。任务：{task}。\n"
        f"要找的目标是：{item.get('note', '')}。\n选出与该目标边界最吻合的一个；都不合适就选 none。"
        '只输出JSON：{"choice":"A","reason":"一句话"}'
    )
    answer = _chat_json(client, model, [image, sheet], prompt)
    choice = str(answer.get("choice", "none")).strip().upper()[:1]
    index = LETTERS.find(choice)
    if 0 <= index < len(panels):
        return options[index], f"选择 {LETTERS[index]}: {answer.get('reason', '')}"
    return None, f"未选中（{choice or 'none'}）: {answer.get('reason', '')}"


def _review_parts(
    client: Any,
    model: str,
    image: np.ndarray,
    parts: list[tuple[float, np.ndarray]],
    task: str,
) -> list[tuple[float, np.ndarray]]:
    overlay = _draw_review_overlay(image, parts)
    prompt = (
        f"第一张是原图，第二张是当前的标注结果，每个区域标了R编号。任务：{task}。\n"
        '逐个检查：区域若主要是背景或正常图案（不属于目标）就删除。只输出JSON：{"remove":["R0"],"reason":"一句话"}'
    )
    answer = _chat_json(client, model, [image, overlay], prompt)
    drop = {int(value or -1) for value in (re.sub(r"\D", "", str(item)) for item in answer.get("remove", []))}
    kept = [part for index, part in enumerate(parts) if index not in drop]
    dropped = sorted(index for index in drop if 0 <= index < len(parts))
    print(f"review 移除 R{dropped}: {answer.get('reason', '')}")
    return kept


def _select_array_masks(
    client: Any,
    model: str,
    session: SamSession,
    image: np.ndarray,
    task: str,
) -> list[tuple[float, np.ndarray]]:
    embeddings, original_sizes, reshaped_sizes = session.embed(image)
    sizes = (original_sizes, reshaped_sizes)
    candidates = _grid_candidates(session, embeddings, sizes, image.shape[:2])
    print(f"全图候选 {len(candidates)} 个")
    if not candidates:
        return []
    overlay, factor = _draw_numbered_candidates(image, [mask for _, mask in candidates])
    height, width = image.shape[:2]
    prompt = (
        f"第一张是原图（宽{width}、高{height}像素，原点在左上角）。第二张是同一张图放大{factor:.2f}倍后叠加的自动分割候选，"
        f"共{len(candidates)}个，编号0到{len(candidates) - 1}。任务：{task}。\n"
        "1) 选出属于目标的候选编号。一个目标被分成多块时全部选上；包含大量背景或属于正常图案的不选。\n"
        "2) 如果某个目标没有被任何候选合适地覆盖，给出该目标内部的一个点和大致外接框，使用原图像素坐标。\n"
        '只输出JSON：{"selected":[编号],"missing":[{"point":[x,y],"box":[x1,y1,x2,y2],"note":"描述"}],"reason":"一句话"}'
    )
    answer = _chat_json(client, model, [image, overlay], prompt)
    selected = [
        index
        for index in answer.get("selected", [])
        if isinstance(index, int) and 0 <= index < len(candidates)
    ]
    print(f"模型选择 {selected}: {answer.get('reason', '')}")
    parts = [candidates[index] for index in selected]
    for item in answer.get("missing", []) or []:
        try:
            part, note = _mask_from_point(client, model, session, image, embeddings, sizes, item, task)
        except (KeyError, ValueError, TypeError) as exc:
            part, note = None, f"missing 条目无效 {item}: {exc}"
        print(f"missing {item.get('point')} -> {note}")
        if part is not None:
            parts.append(part)
    if parts:
        parts = _review_parts(client, model, image, parts, task)
    return parts


def _select_defect_masks(
    client: Any,
    model: str,
    session: SamSession,
    image: np.ndarray,
    task: str,
) -> list[tuple[float, np.ndarray]]:
    embeddings, original_sizes, reshaped_sizes = session.embed(image)
    sizes = (original_sizes, reshaped_sizes)
    height, width = image.shape[:2]
    prompt = (
        f"图片宽{width}、高{height}像素，原点在左上角。任务：{task}。"
        "列出每个目标：内部一个点和大致外接框（原图像素坐标，框宁可略大）。"
        '只输出JSON：{"targets":[{"point":[x,y],"box":[x1,y1,x2,y2],"note":"描述"}]}'
    )
    located = _chat_json(client, model, [image], prompt)
    targets = located.get("targets", []) or []
    print(f"定位到 {len(targets)} 个目标: {[item.get('note', '') for item in targets]}")
    parts = []
    for item in targets:
        try:
            part, note = _mask_from_point(client, model, session, image, embeddings, sizes, item, task)
        except (KeyError, ValueError, TypeError) as exc:
            part, note = None, f"定位条目无效 {item}: {exc}"
        print(f"目标 {item.get('note', '')} -> {note}")
        if part is not None:
            parts.append(part)
    if parts:
        parts = _review_parts(client, model, image, parts, task)
    return parts
