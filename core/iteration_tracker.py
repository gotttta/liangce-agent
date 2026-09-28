import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from core.scoring import ImageScore, RunScore

HISTORY_FILENAME = "iteration_history.jsonl"
NO_IMPROVEMENT_DELTA = 0.01  # 相邻轮次 composite_mean 变化量（含下降）小于此值视为无改善


@dataclass
class IterationRecord:
    iteration: int
    run_score: RunScore
    algorithm_spec: dict
    timestamp: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _record_to_json(record: IterationRecord) -> dict:
    return {
        "iteration": record.iteration,
        "composite_mean": record.run_score.composite_mean,
        "image_scores": [
            {
                "image_id": score.image_id,
                "iou_mean": score.iou_mean,
                "false_positive_count": score.false_positive_count,
                "false_negative_count": score.false_negative_count,
                "ref_count": score.ref_count,
                "composite": score.composite,
            }
            for score in record.run_score.image_scores
        ],
        "algorithm_spec": record.algorithm_spec,
        "timestamp": record.timestamp,
    }


def _record_from_json(data: dict) -> IterationRecord:
    return IterationRecord(
        iteration=data["iteration"],
        run_score=RunScore(
            image_scores=[ImageScore(**score) for score in data["image_scores"]],
            composite_mean=data["composite_mean"],
        ),
        algorithm_spec=data["algorithm_spec"],
        timestamp=data["timestamp"],
    )


class IterationTracker:
    def __init__(self, workspace_dir: str | Path) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.workspace_dir / HISTORY_FILENAME

    @property
    def history(self) -> list[IterationRecord]:
        if not self.history_path.is_file():
            return []
        records = []
        for line in self.history_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(_record_from_json(json.loads(line)))
        return records

    def record(self, run_score: RunScore, algorithm_spec: dict) -> IterationRecord:
        # "a" 模式不可读，须用 "a+"；追加写仍始终落在文件末尾
        with self.history_path.open("a+", encoding="utf-8") as handle:
            # 在同一个 open 调用内数行数，避免 TOCTOU
            handle.seek(0)
            iteration = sum(1 for line in handle if line.strip())
            record = IterationRecord(
                iteration=iteration,
                run_score=run_score,
                algorithm_spec=algorithm_spec,
                timestamp=_utc_now_iso(),
            )
            handle.seek(0, 2)  # 回到末尾再追加
            handle.write(json.dumps(_record_to_json(record), ensure_ascii=False) + "\n")
        return record

    @property
    def best(self) -> IterationRecord | None:
        records = self.history
        if not records:
            return None
        return max(records, key=lambda record: record.run_score.composite_mean)

    def should_stop(
        self,
        patience: int = 5,
        target: float = 0.85,
        max_iter: int = 100,
    ) -> tuple[bool, str]:
        """
        返回 (是否停止, 原因字符串)。
        原因字符串取值（按优先级，从高到低）：
          "target_reached"  最新记录的 composite_mean >= target
          "max_iterations"  历史条数 >= max_iter
          "no_improvement"  最近 patience 轮，每相邻两条 composite_mean 变化的绝对值 < 0.01（含下降）
          ""                不停止
        不足 patience 条时 "no_improvement" 不触发。
        """
        records = self.history
        if not records:
            return False, ""
        if records[-1].run_score.composite_mean >= target:
            return True, "target_reached"
        if len(records) >= max_iter:
            return True, "max_iterations"
        if len(records) >= patience:
            recent = records[-patience:]
            deltas = (
                recent[i].run_score.composite_mean - recent[i - 1].run_score.composite_mean
                for i in range(1, len(recent))
            )
            if all(abs(delta) < NO_IMPROVEMENT_DELTA for delta in deltas):
                return True, "no_improvement"
        return False, ""
