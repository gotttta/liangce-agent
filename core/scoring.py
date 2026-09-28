from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.measure import label as label_instances


def mask_iou(pred: np.ndarray, ref: np.ndarray) -> float:
    pred_mask = np.asarray(pred) != 0
    ref_mask = np.asarray(ref) != 0
    union = np.logical_or(pred_mask, ref_mask).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred_mask, ref_mask).sum() / union)


def match_instances(
    pred_masks: list[np.ndarray],
    ref_masks: list[np.ndarray],
    iou_threshold: float = 0.3,
) -> dict:
    iou_matrix = np.zeros((len(pred_masks), len(ref_masks)), dtype=np.float64)
    for i, pred in enumerate(pred_masks):
        for j, ref in enumerate(ref_masks):
            iou_matrix[i, j] = mask_iou(pred, ref)

    matched = []
    if pred_masks and ref_masks:
        rows, cols = linear_sum_assignment(1.0 - iou_matrix)
        for i, j in zip(rows, cols):
            if iou_matrix[i, j] >= iou_threshold:
                matched.append((int(i), int(j), float(iou_matrix[i, j])))

    matched_pred = {i for i, _, _ in matched}
    matched_ref = {j for _, j, _ in matched}
    return {
        "matched": matched,
        "false_positives": [i for i in range(len(pred_masks)) if i not in matched_pred],
        "false_negatives": [j for j in range(len(ref_masks)) if j not in matched_ref],
    }


@dataclass
class ImageScore:
    image_id: str
    iou_mean: float
    false_positive_count: int
    false_negative_count: int
    ref_count: int
    composite: float


@dataclass
class RunScore:
    image_scores: list[ImageScore]
    composite_mean: float


def _composite(
    iou_mean: float,
    false_positive_count: int,
    false_negative_count: int,
    ref_count: int,
) -> float:
    # 双方都无目标视为完全一致，与 mask_iou 的空-空约定保持一致
    if ref_count == 0 and false_positive_count == 0:
        return 1.0
    fp_penalty = false_positive_count / max(ref_count, 1)
    fn_penalty = false_negative_count / max(ref_count, 1)
    return iou_mean * max(0.0, 1.0 - fp_penalty) * max(0.0, 1.0 - fn_penalty)


def score_image(
    image_id: str,
    pred_masks: list[np.ndarray],
    ref_whole_mask: np.ndarray,
) -> ImageScore:
    labels, instance_count = label_instances(
        np.asarray(ref_whole_mask) != 0, return_num=True
    )
    ref_masks = [labels == k for k in range(1, instance_count + 1)]

    result = match_instances(pred_masks, ref_masks)
    matched_ious = [iou for _, _, iou in result["matched"]]
    iou_mean = float(np.mean(matched_ious)) if matched_ious else 0.0
    false_positive_count = len(result["false_positives"])
    false_negative_count = len(result["false_negatives"])

    return ImageScore(
        image_id=image_id,
        iou_mean=iou_mean,
        false_positive_count=false_positive_count,
        false_negative_count=false_negative_count,
        ref_count=len(ref_masks),
        composite=_composite(
            iou_mean, false_positive_count, false_negative_count, len(ref_masks)
        ),
    )


def score_run(scores: list[ImageScore]) -> RunScore:
    composite_mean = (
        float(np.mean([score.composite for score in scores])) if scores else 0.0
    )
    return RunScore(image_scores=list(scores), composite_mean=composite_mean)
