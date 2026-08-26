"""File-backed library for accepted, reusable custom pipeline stages."""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from core.operators import normalize_generated_operator


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


class OperatorLibrary:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def publish(
        self,
        spec,
        source_task_id=None,
        *,
        user_tested=False,
        tested_by=None,
        test_note=None,
    ):
        if user_tested is not True:
            raise PermissionError(
                "operator publication requires explicit user-tested approval"
            )
        tested_by = str(tested_by or "").strip()
        if not tested_by:
            raise ValueError("tested_by is required for operator publication")
        normalized = normalize_generated_operator(spec)
        digest = hashlib.sha256(normalized.source.encode("utf-8")).hexdigest()[:12]
        safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", normalized.name).strip("_")
        operator_id = f"operator_{safe_name or 'custom'}_{digest}"
        operator_dir = self.root / operator_id
        operator_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "id": operator_id,
            "status": "accepted",
            "created_at": _utc_now(),
            "source_task_id": source_task_id,
            "approval": {
                "user_tested": True,
                "tested_by": tested_by,
                "test_note": str(test_note or "").strip(),
                "approved_at": _utc_now(),
            },
            **normalized.as_dict(),
        }
        path = operator_dir / "operator.json"
        self._write_json(path, payload)
        return {**payload, "path": str(path)}

    def list_operators(self, limit=200):
        operators = []
        for path in self.root.glob("operator_*/operator.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            approval = payload.get("approval") if isinstance(payload.get("approval"), dict) else {}
            if (
                payload.get("status") != "accepted"
                or approval.get("user_tested") is not True
                or not payload.get("source")
            ):
                continue
            operators.append({**payload, "path": str(path)})
        operators.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return operators[: max(1, int(limit))]

    def get(self, name):
        matches = [item for item in self.list_operators() if item.get("name") == name]
        return matches[0] if matches else None

    @staticmethod
    def _write_json(path, value):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        temporary.replace(path)
