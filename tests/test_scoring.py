import numpy as np

from core.scoring import (
    ImageScore,
    RunScore,
    mask_iou,
    match_instances,
    score_image,
    score_run,
)


def _block(shape: tuple[int, int], top: int, left: int, height: int, width: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[top:top + height, left:left + width] = True
    return mask


def test_mask_iou_identical_masks_return_one():
    mask = _block((10, 10), 2, 3, 4, 5)
    assert mask_iou(mask, mask.copy()) == 1.0


def test_mask_iou_disjoint_masks_return_zero():
    pred = _block((10, 10), 0, 0, 4, 4)
    ref = _block((10, 10), 5, 5, 4, 4)
    assert mask_iou(pred, ref) == 0.0


def test_mask_iou_partial_overlap():
    pred = _block((10, 10), 0, 0, 2, 2)
    ref = _block((10, 10), 0, 1, 2, 2)
    assert mask_iou(pred, ref) == np.float64(2 / 6)


def test_mask_iou_both_empty_return_one():
    assert mask_iou(np.zeros((5, 5), dtype=bool), np.zeros((5, 5), dtype=bool)) == 1.0


def test_mask_iou_one_empty_one_nonempty_return_zero():
    empty = np.zeros((5, 5), dtype=bool)
    filled = _block((5, 5), 0, 0, 2, 2)
    assert mask_iou(empty, filled) == 0.0
    assert mask_iou(filled, empty) == 0.0


def test_mask_iou_converts_non_bool_input():
    pred = np.zeros((5, 5), dtype=np.uint8)
    pred[1:3, 1:3] = 7
    ref = _block((5, 5), 1, 1, 2, 2)
    assert mask_iou(pred, ref) == 1.0


def test_match_instances_perfect_single_match():
    mask = _block((10, 10), 2, 2, 3, 3)
    result = match_instances([mask], [mask.copy()])
    assert result["matched"] == [(0, 0, 1.0)]
    assert result["false_positives"] == []
    assert result["false_negatives"] == []


def test_match_instances_multiple_targets():
    pred_a = _block((10, 10), 0, 0, 3, 3)
    pred_b = _block((10, 10), 5, 5, 3, 3)
    ref_a = _block((10, 10), 0, 0, 3, 3)
    ref_b = _block((10, 10), 5, 5, 3, 3)
    result = match_instances([pred_a, pred_b], [ref_a, ref_b])
    assert {(i, j) for i, j, _ in result["matched"]} == {(0, 0), (1, 1)}
    assert all(iou == 1.0 for _, _, iou in result["matched"])
    assert result["false_positives"] == []
    assert result["false_negatives"] == []


def test_match_instances_assigns_crossed_pairs_optimally():
    ref_top = _block((10, 10), 0, 0, 3, 10)
    ref_bottom = _block((10, 10), 5, 0, 3, 10)
    result = match_instances([ref_bottom.copy(), ref_top.copy()], [ref_top, ref_bottom])
    assert {(i, j) for i, j, _ in result["matched"]} == {(0, 1), (1, 0)}
    assert result["false_positives"] == []
    assert result["false_negatives"] == []


def test_match_instances_all_false_positives():
    pred = _block((10, 10), 0, 0, 3, 3)
    ref = _block((10, 10), 6, 6, 3, 3)
    result = match_instances([pred, pred.copy()], [ref])
    assert result["matched"] == []
    assert result["false_positives"] == [0, 1]
    assert result["false_negatives"] == [0]


def test_match_instances_all_false_negatives():
    pred = _block((10, 10), 0, 0, 3, 3)
    ref = _block((10, 10), 6, 6, 3, 3)
    result = match_instances([pred], [ref, ref.copy()])
    assert result["matched"] == []
    assert result["false_positives"] == [0]
    assert result["false_negatives"] == [0, 1]


def test_match_instances_empty_inputs():
    mask = _block((5, 5), 0, 0, 2, 2)
    only_pred = match_instances([mask], [])
    assert only_pred == {"matched": [], "false_positives": [0], "false_negatives": []}
    only_ref = match_instances([], [mask])
    assert only_ref == {"matched": [], "false_positives": [], "false_negatives": [0]}
    both_empty = match_instances([], [])
    assert both_empty == {"matched": [], "false_positives": [], "false_negatives": []}


def test_match_instances_drops_pairs_below_threshold():
    ref = _block((10, 10), 0, 0, 10, 10)
    pred = _block((10, 10), 0, 0, 2, 10)
    assert mask_iou(pred, ref) == np.float64(0.2)
    result = match_instances([pred], [ref], iou_threshold=0.3)
    assert result["matched"] == []
    assert result["false_positives"] == [0]
    assert result["false_negatives"] == [0]


def test_score_image_perfect_single_target():
    ref_mask = _block((20, 20), 4, 4, 6, 6)
    score = score_image("img_a", [ref_mask.copy()], ref_mask)
    assert score.image_id == "img_a"
    assert score.ref_count == 1
    assert score.iou_mean == 1.0
    assert score.false_positive_count == 0
    assert score.false_negative_count == 0
    assert score.composite == 1.0


def test_score_image_false_positive_reduces_composite():
    ref_a = _block((20, 20), 2, 2, 4, 4)
    ref_b = _block((20, 20), 10, 10, 4, 4)
    ref_mask = ref_a | ref_b
    stray = _block((20, 20), 2, 16, 3, 3)
    score = score_image("img_a", [ref_a.copy(), ref_b.copy(), stray], ref_mask)
    assert score.ref_count == 2
    assert score.false_positive_count == 1
    assert score.false_negative_count == 0
    assert score.iou_mean == 1.0
    assert score.composite == np.float64(0.5)


def test_score_image_partial_overlap_reduces_iou_mean():
    ref_mask = _block((20, 20), 0, 0, 4, 4)
    pred = _block((20, 20), 0, 0, 2, 4)
    score = score_image("img_a", [pred], ref_mask)
    assert score.iou_mean == np.float64(0.5)
    assert score.composite == np.float64(0.5)


def test_score_image_splits_merged_mask_into_instances():
    ref_a = _block((20, 20), 2, 2, 4, 4)
    ref_b = _block((20, 20), 10, 10, 4, 4)
    ref_mask = ref_a | ref_b
    score = score_image("img_a", [ref_a.copy()], ref_mask)
    assert score.ref_count == 2
    assert score.false_negative_count == 1
    assert score.false_positive_count == 0
    assert score.composite == np.float64(0.5)


def test_score_image_empty_ref_and_empty_pred_is_perfect():
    score = score_image("img_a", [], np.zeros((20, 20), dtype=bool))
    assert score.ref_count == 0
    assert score.iou_mean == 0.0
    assert score.composite == 1.0


def test_score_image_empty_ref_with_predictions_scores_zero():
    pred = _block((20, 20), 2, 2, 3, 3)
    score = score_image("img_a", [pred], np.zeros((20, 20), dtype=bool))
    assert score.false_positive_count == 1
    assert score.composite == 0.0


def test_score_run_averages_composite():
    first = ImageScore("img_a", 1.0, 0, 0, 1, 1.0)
    second = ImageScore("img_b", 1.0, 0, 0, 2, 0.5)
    run = score_run([first, second])
    assert run.composite_mean == np.float64(0.75)
    assert run.image_scores == [first, second]
