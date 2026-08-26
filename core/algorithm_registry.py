import json
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from uuid import uuid4


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class AlgorithmRegistry:
    """File-backed registry for user-accepted, replayable pipelines."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def publish(self, task_id, state, note="用户确认当前结果"):
        strategy = state.get("strategy") if isinstance(state.get("strategy"), dict) else {}
        observation = (
            strategy.get("visual_observation")
            if isinstance(strategy.get("visual_observation"), dict)
            else {}
        )
        algorithm_id = f"algorithm_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
        algorithm_dir = self.root / algorithm_id
        algorithm_dir.mkdir(parents=True, exist_ok=False)
        payload = {
            "schema_version": 1,
            "id": algorithm_id,
            "status": "accepted",
            "created_at": utc_now(),
            "source_task_id": task_id,
            "name": state.get("selected_candidate") or "accepted_visual_algorithm",
            "description": state.get("description", ""),
            "defect_type": strategy.get("defect_type", "unknown"),
            "measurement_type": strategy.get("measurement_type", "unknown"),
            "background_type": observation.get("background_pattern", "unknown"),
            "polarity": observation.get("polarity", "unknown"),
            "pipeline": state.get("pipeline", {}),
            "strategy": strategy,
            "output_requirements": state.get("understanding", {}).get("output_requirements", []),
            "rendering": state.get("rendering", {}),
            "quality_report": state.get("quality_report", {}),
            "measurement_summary": state.get("measurements", {}).get("summary", {}),
            "acceptance_note": note,
            "artifacts": {
                "run_dir": state.get("run_dir"),
                "annotated_image_path": state.get("annotated_image_path"),
                "predicted_mask_path": state.get("predicted_mask_path"),
            },
        }
        path = algorithm_dir / "algorithm.json"
        self._write_json(path, payload)
        return {**payload, "path": str(path)}

    def list_algorithms(self, limit=100):
        algorithms = []
        for path in self.root.glob("algorithm_*/algorithm.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            if payload.get("status") != "accepted" or not payload.get("pipeline"):
                continue
            algorithms.append({**payload, "path": str(path)})
        algorithms.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return algorithms[: max(1, int(limit))]

    def search(self, understanding, limit=3, min_score=0.2):
        query = self._query_features(understanding)
        matches = []
        for algorithm in self.list_algorithms():
            score, reasons = self._score(query, algorithm)
            if score < float(min_score):
                continue
            matches.append({
                "algorithm_id": algorithm["id"],
                "name": algorithm.get("name") or algorithm["id"],
                "score": round(score, 6),
                "match_reasons": reasons,
                "pipeline": algorithm["pipeline"],
                "strategy": algorithm.get("strategy", {}),
                "source_task_id": algorithm.get("source_task_id"),
                "path": algorithm["path"],
            })
        matches.sort(key=lambda item: (item["score"], item["algorithm_id"]), reverse=True)
        return matches[: max(1, int(limit))]

    @staticmethod
    def _query_features(understanding):
        value = understanding if isinstance(understanding, dict) else {}
        strategy = value.get("recommended_strategy") if isinstance(value.get("recommended_strategy"), dict) else {}
        observation = strategy.get("visual_observation") if isinstance(strategy.get("visual_observation"), dict) else {}
        return {
            "defect_type": strategy.get("defect_type") or value.get("target_defect"),
            "measurement_type": strategy.get("measurement_type"),
            "background_type": observation.get("background_pattern") or value.get("normal_context"),
            "polarity": observation.get("polarity"),
            "text": " ".join(str(item or "") for item in (
                value.get("task_summary"),
                value.get("target_defect"),
                value.get("normal_context"),
            )),
        }

    @classmethod
    def _score(cls, query, algorithm):
        score = 0.0
        reasons = []
        for key, weight, label in (
            ("defect_type", 0.35, "defect_type"),
            ("background_type", 0.25, "background_type"),
            ("measurement_type", 0.15, "measurement_type"),
            ("polarity", 0.15, "polarity"),
        ):
            left = cls._normalize(query.get(key))
            right = cls._normalize(algorithm.get(key))
            if left and right and left not in {"unknown", "user_defined_defect"} and left == right:
                score += weight
                reasons.append(label)

        query_text = cls._normalize(query.get("text"))
        algorithm_text = cls._normalize(" ".join(str(item or "") for item in (
            algorithm.get("description"),
            algorithm.get("defect_type"),
            algorithm.get("background_type"),
        )))
        if query_text and algorithm_text:
            similarity = SequenceMatcher(None, query_text, algorithm_text).ratio()
            score += 0.1 * similarity
            if similarity >= 0.45:
                reasons.append("description_similarity")
        return min(1.0, score), reasons

    @staticmethod
    def _normalize(value):
        text = str(value or "").strip().lower()
        return re.sub(r"\s+", " ", text)

    @staticmethod
    def _write_json(path, value):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        temporary.replace(path)
