import json
from pathlib import Path

import numpy as np

REQUIRED_META_FIELDS = (
    "image_id",
    "image_path",
    "task",
    "target_type",
    "sam_iou_score",
    "confirmed_at",
    "confirmed_by",
)


class ReferenceStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, image_id: str) -> Path:
        return self.root / image_id

    def save(self, image_id: str, mask: np.ndarray, meta: dict) -> Path:
        missing = [field for field in REQUIRED_META_FIELDS if field not in meta]
        if missing:
            raise ValueError(f"meta missing required fields: {missing}")
        array = np.asarray(mask)
        if array.ndim != 2:
            raise ValueError("mask must be a 2D array")
        directory = self._dir(image_id)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "mask.npy", array.astype(bool))
        meta_path = directory / "meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return meta_path

    def load(self, image_id: str) -> tuple[np.ndarray, dict]:
        mask_path = self._dir(image_id) / "mask.npy"
        meta_path = self._dir(image_id) / "meta.json"
        if not mask_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(f"reference not found for image_id: {image_id}")
        mask = np.load(mask_path).astype(bool)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return mask, meta

    def list_scoreable(self) -> list[str]:
        image_ids = []
        for meta_path in sorted(self.root.glob("*/meta.json")):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not meta.get("skip_scoring", False):
                image_ids.append(meta_path.parent.name)
        return sorted(image_ids)

    def mark_skip(self, image_id: str, reason: str) -> None:
        meta_path = self._dir(image_id) / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"reference not found for image_id: {image_id}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["skip_scoring"] = True
        meta["skip_reason"] = reason
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
