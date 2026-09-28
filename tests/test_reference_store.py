import numpy as np
import pytest

from core.reference_store import ReferenceStore


def _meta(image_id: str = "img_a", **overrides) -> dict:
    meta = {
        "image_id": image_id,
        "image_path": f"data/samples/{image_id}.jpg",
        "task": "标出图中膜内的颗粒缺陷",
        "target_type": "defect",
        "sam_iou_score": 0.91,
        "skip_scoring": False,
        "skip_reason": "",
        "confirmed_at": "2026-09-28T12:00:00Z",
        "confirmed_by": "user",
    }
    meta.update(overrides)
    return meta


def test_save_then_load_roundtrip(tmp_path):
    store = ReferenceStore(tmp_path / "references")
    mask = np.zeros((12, 10), dtype=bool)
    mask[3:6, 4:8] = True

    store.save("img_a", mask, _meta())

    loaded_mask, loaded_meta = store.load("img_a")
    assert loaded_mask.dtype == np.bool_
    np.testing.assert_array_equal(loaded_mask, mask)
    assert loaded_meta == _meta()


def test_save_converts_non_bool_mask_to_bool(tmp_path):
    store = ReferenceStore(tmp_path / "references")
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:4, 2:4] = 1

    store.save("img_a", mask, _meta())

    loaded_mask, _ = store.load("img_a")
    assert loaded_mask.dtype == np.bool_
    np.testing.assert_array_equal(loaded_mask, mask.astype(bool))


def test_save_overwrites_existing_entry(tmp_path):
    store = ReferenceStore(tmp_path / "references")
    first = np.zeros((6, 6), dtype=bool)
    second = np.ones((6, 6), dtype=bool)

    store.save("img_a", first, _meta(sam_iou_score=0.5))
    store.save("img_a", second, _meta(sam_iou_score=0.9))

    loaded_mask, loaded_meta = store.load("img_a")
    np.testing.assert_array_equal(loaded_mask, second)
    assert loaded_meta["sam_iou_score"] == 0.9


def test_save_missing_required_field_raises(tmp_path):
    store = ReferenceStore(tmp_path / "references")
    meta = _meta()
    del meta["target_type"]

    with pytest.raises(ValueError, match="target_type"):
        store.save("img_a", np.zeros((4, 4), dtype=bool), meta)


def test_save_non_2d_mask_raises(tmp_path):
    store = ReferenceStore(tmp_path / "references")

    with pytest.raises(ValueError, match="2D"):
        store.save("img_a", np.zeros((4, 4, 3), dtype=bool), _meta())


def test_load_unknown_image_id_raises(tmp_path):
    store = ReferenceStore(tmp_path / "references")

    with pytest.raises(FileNotFoundError):
        store.load("nope")


def test_mark_skip_persists_flag_and_reason(tmp_path):
    store = ReferenceStore(tmp_path / "references")
    store.save("img_a", np.zeros((4, 4), dtype=bool), _meta())

    store.mark_skip("img_a", "SAM 分数过低")

    _, meta = store.load("img_a")
    assert meta["skip_scoring"] is True
    assert meta["skip_reason"] == "SAM 分数过低"


def test_mark_skip_unknown_image_id_raises(tmp_path):
    store = ReferenceStore(tmp_path / "references")

    with pytest.raises(FileNotFoundError):
        store.mark_skip("nope", "any")


def test_list_scoreable_returns_only_unskipped_sorted(tmp_path):
    store = ReferenceStore(tmp_path / "references")
    store.save("img_c", np.zeros((4, 4), dtype=bool), _meta("img_c"))
    store.save("img_a", np.zeros((4, 4), dtype=bool), _meta("img_a"))
    store.save("img_b", np.zeros((4, 4), dtype=bool), _meta("img_b"))
    store.mark_skip("img_b", "质量存疑")

    assert store.list_scoreable() == ["img_a", "img_c"]


def test_missing_optional_skip_fields_treated_as_scoreable(tmp_path):
    store = ReferenceStore(tmp_path / "references")
    meta = _meta()
    del meta["skip_scoring"]
    del meta["skip_reason"]
    store.save("img_a", np.zeros((4, 4), dtype=bool), meta)

    assert store.list_scoreable() == ["img_a"]


def test_root_directory_created_when_missing(tmp_path):
    root = tmp_path / "nested" / "references"
    ReferenceStore(root)
    assert root.is_dir()
