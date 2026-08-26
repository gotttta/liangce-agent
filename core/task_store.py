import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from core.algorithm_registry import AlgorithmRegistry
from core.operator_library import OperatorLibrary
from core.reference_extraction import extract_ground_truth_mask, save_ground_truth_mask


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_rejection_record(task_id, pipeline, rejection_reason, quality_report, task_root=None):
    """Persist a rejected pipeline with its user-provided reason."""
    root = Path(task_root) if task_root else Path("workspace/tasks")
    rejection_dir = root / str(task_id or "unknown") / "rejections"
    rejection_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "rejected_at": utc_now(),
        "pipeline": pipeline or {},
        "quality_report": quality_report or {},
        "rejection_reason": rejection_reason or "用户未说明原因",
    }
    path = rejection_dir / f"{datetime.now():%Y%m%d_%H%M%S_%f}_rejection.json"
    TaskStore._write_json(path, record)
    return record


class TaskStore:
    def __init__(self, root, algorithm_root=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.algorithm_registry = AlgorithmRegistry(
            algorithm_root or self.root.parent / "algorithms"
        )
        self.operator_library = OperatorLibrary(self.root.parent / "operators")

    def create_task(self, title="新缺陷检测任务"):
        task_id = f"task_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
        task_dir = self.root / task_id
        for name in ("samples", "references", "ground_truth", "nodes", "candidates", "feedback", "acceptance", "rejections"):
            (task_dir / name).mkdir(parents=True, exist_ok=True)
        task = {
            "id": task_id,
            "title": title,
            "status": "draft",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "samples": [],
            "reference_examples": [],
            "ground_truth": None,
            "current_node": None,
        }
        self._write_json(task_dir / "task.json", task)
        self.append_event(task_id, "task_created", {"title": title})
        return task

    def load_task(self, task_id):
        path = self.task_dir(task_id) / "task.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def list_tasks(self, limit=50):
        tasks = []
        for task_path in self.root.glob("task_*/task.json"):
            try:
                tasks.append(json.loads(task_path.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError):
                continue
        tasks.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return tasks[: max(1, int(limit))]

    def load_messages(self, task_id):
        path = self.task_dir(task_id) / "conversation.jsonl"
        if not path.exists():
            return []
        messages = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except (ValueError, TypeError):
                continue
            if item.get("role") in {"user", "assistant"}:
                role = item["role"]
                content = item.get("content", "")
                if role == "assistant":
                    content = _sanitize_legacy_message(content)
                messages.append({
                    "role": role,
                    "content": content,
                })
        return messages

    def save_memory(self, task_id, memory):
        payload = dict(memory or {})
        payload["updated_at"] = utc_now()
        path = self.task_dir(task_id) / "memory.json"
        self._write_json(path, payload)
        return payload

    def load_memory(self, task_id):
        path = self.task_dir(task_id) / "memory.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def load_latest_state(self, task_id):
        latest_path = self.task_dir(task_id) / "nodes" / "execute_candidate" / "latest.json"
        if not latest_path.exists():
            return None
        record = json.loads(latest_path.read_text(encoding="utf-8"))
        outputs = record.get("outputs") if isinstance(record.get("outputs"), dict) else {}
        annotated = outputs.get("annotated_image_path")
        if annotated:
            graph_state = Path(annotated).parent / "graph_state.json"
            if graph_state.exists():
                return json.loads(graph_state.read_text(encoding="utf-8"))
        return outputs or None

    def add_sample(self, task_id, source_path):
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(source)
        task_dir = self.task_dir(task_id)
        destination = task_dir / "samples" / f"{uuid4().hex[:8]}_{source.name}"
        shutil.copy2(source, destination)
        task = self.load_task(task_id)
        sample = {
            "path": str(destination),
            "source_name": source.name,
            "added_at": utc_now(),
        }
        task["samples"].append(sample)
        task["updated_at"] = utc_now()
        self._write_json(task_dir / "task.json", task)
        self.append_event(task_id, "sample_added", sample)
        return sample

    def remove_sample(self, task_id, sample_path):
        """Remove one uploaded sample and its task-local copy."""
        task_dir = self.task_dir(task_id)
        task = self.load_task(task_id)
        target = str(Path(sample_path)) if sample_path else ""
        removed = []
        remaining = []
        for sample in task.get("samples", []):
            if str(sample.get("path")) != target:
                remaining.append(sample)
                continue
            removed.append(sample)
            path = Path(sample.get("path", ""))
            if path.is_file() and path.parent == task_dir / "samples":
                path.unlink()
        if not removed:
            return task
        task["samples"] = remaining
        task["updated_at"] = utc_now()
        self._write_json(task_dir / "task.json", task)
        self.append_event(task_id, "sample_removed", {"path": target})
        return task

    def add_reference_example(self, task_id, source_path, description="甲方Handbook标注示例图"):
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        task_dir = self.task_dir(task_id)
        task = self.load_task(task_id)
        existing = task.get("reference_examples") or []
        for item in existing:
            if item.get("source_name") == source.name and Path(item.get("path", "")).is_file():
                return item
        destination = task_dir / "references" / f"{uuid4().hex[:8]}_{source.name}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        example = {
            "image_path": str(destination),
            "path": str(destination),
            "source_name": source.name,
            "description": str(description or "甲方Handbook标注示例图"),
            "added_at": utc_now(),
        }
        task.setdefault("reference_examples", []).append(example)
        task["updated_at"] = utc_now()
        self._write_json(task_dir / "task.json", task)
        self.append_event(task_id, "reference_example_added", example)
        return example

    def set_ground_truth_annotation(self, task_id, source_path, target_image_path):
        """Store a same-image Handbook annotation as a metric-ready GT mask."""
        source = Path(source_path)
        target = Path(target_image_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        if not target.is_file():
            raise FileNotFoundError(target)

        from PIL import Image

        with Image.open(target) as target_image:
            expected_shape = (target_image.height, target_image.width)
        mask, extraction = extract_ground_truth_mask(source, expected_shape=expected_shape)
        task_dir = self.task_dir(task_id)
        ground_truth_dir = task_dir / "ground_truth"
        destination = ground_truth_dir / f"{uuid4().hex[:8]}_{source.name}"
        shutil.copy2(source, destination)
        mask_path = ground_truth_dir / "ground_truth_mask.png"
        save_ground_truth_mask(mask, mask_path)
        ground_truth = {
            "type": "same_image_ground_truth",
            "annotation_path": str(destination),
            "mask_path": str(mask_path),
            "source_name": source.name,
            "target_image_path": str(target),
            "extraction": extraction,
            "added_at": utc_now(),
        }
        task = self.load_task(task_id)
        task["ground_truth"] = ground_truth
        task["updated_at"] = utc_now()
        self._write_json(task_dir / "task.json", task)
        self.append_event(task_id, "ground_truth_added", ground_truth)
        return ground_truth

    def set_title(self, task_id, title):
        task = self.load_task(task_id)
        task["title"] = str(title or task.get("title") or "缺陷检测任务")[:80]
        task["updated_at"] = utc_now()
        self._write_json(self.task_dir(task_id) / "task.json", task)
        return task

    def save_node_result(
        self,
        task_id,
        node_name,
        inputs,
        outputs,
        duration_seconds,
        status="completed",
        error=None,
    ):
        task_dir = self.task_dir(task_id)
        node_dir = task_dir / "nodes" / node_name
        node_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "node": node_name,
            "status": status,
            "started_at": utc_now(),
            "duration_seconds": round(float(duration_seconds), 4),
            "inputs": inputs,
            "outputs": outputs,
            "error": error,
        }
        self._write_json(node_dir / "latest.json", record)
        self._write_json(node_dir / f"run_{uuid4().hex[:8]}.json", record)
        task = self.load_task(task_id)
        task["current_node"] = node_name
        task["status"] = "failed" if status == "failed" else "in_progress"
        task["updated_at"] = utc_now()
        self._write_json(task_dir / "task.json", task)
        self.append_event(task_id, "node_finished", record)
        return record

    def append_message(self, task_id, role, content):
        message = {"timestamp": utc_now(), "role": role, "content": content}
        self._append_jsonl(self.task_dir(task_id) / "conversation.jsonl", message)
        return message

    def append_event(self, task_id, event_type, payload):
        event = {
            "timestamp": utc_now(),
            "type": event_type,
            "payload": payload,
        }
        self._append_jsonl(self.task_dir(task_id) / "events.jsonl", event)
        return event

    def accept_result(self, task_id, state, note="用户确认当前结果"):
        task_dir = self.task_dir(task_id)
        algorithm_path = task_dir / "acceptance" / "algorithm.json"
        self._write_json(algorithm_path, {
            "name": state.get("selected_candidate") or "accepted_visual_algorithm",
            "description": state.get("description", ""),
            "pipeline": state.get("pipeline", {}),
            "strategy": state.get("strategy", {}),
            "output_requirements": state.get("understanding", {}).get("output_requirements", []),
            "rendering": state.get("rendering", {}),
        })
        published_algorithm = self.algorithm_registry.publish(task_id, state, note=note)
        acceptance = {
            "accepted_at": utc_now(),
            "note": note,
            "algorithm_path": str(algorithm_path),
            "registry_algorithm_id": published_algorithm["id"],
            "registry_algorithm_path": published_algorithm["path"],
            # Accepting a complete algorithm does not approve its generated
            # stages for reuse. Operator approval is a separate user-tested act.
            "operator_library_paths": [],
            "iteration": state.get("iteration"),
            "run_dir": state.get("run_dir"),
            "annotated_image_path": state.get("annotated_image_path"),
            "predicted_mask_path": state.get("predicted_mask_path"),
            "measurement_summary": state.get("measurements", {}).get("summary", {}),
            "quality_report": state.get("quality_report", {}),
        }
        self._write_json(task_dir / "acceptance" / "latest.json", acceptance)
        task = self.load_task(task_id)
        task["status"] = "accepted"
        task["accepted_at"] = acceptance["accepted_at"]
        task["updated_at"] = utc_now()
        self._write_json(task_dir / "task.json", task)
        self.append_event(task_id, "result_accepted", acceptance)
        return acceptance

    def approve_tested_operator(
        self,
        spec,
        *,
        tested_by,
        test_note,
        source_task_id=None,
    ):
        """Publish one operator only after explicit user testing and approval."""
        return self.operator_library.publish(
            spec,
            source_task_id=source_task_id,
            user_tested=True,
            tested_by=tested_by,
            test_note=test_note,
        )

    def search_algorithms(self, understanding, limit=3, min_score=0.2):
        return self.algorithm_registry.search(
            understanding,
            limit=limit,
            min_score=min_score,
        )

    def exit_task(self, task_id, note="用户退出当前任务"):
        task = self.load_task(task_id)
        task["status"] = "exited"
        task["exit_note"] = note
        task["updated_at"] = utc_now()
        self._write_json(self.task_dir(task_id) / "task.json", task)
        self.append_event(task_id, "task_exited", {"note": note})
        return task

    def task_dir(self, task_id):
        task_dir = self.root / task_id
        if not task_dir.exists():
            raise FileNotFoundError(f"Unknown task: {task_id}")
        return task_dir

    @staticmethod
    def _write_json(path, value):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _append_jsonl(path, value):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


def _sanitize_legacy_message(content):
    """Keep old conversations readable after removing unsupported quality claims."""
    if not isinstance(content, str):
        return content
    text = content
    text = re.sub(r"质量状态\s*[^，。；\n]+[，。]?\s*得分\s*[0-9.]+[。]?", "本轮已完成算法执行。", text)
    text = re.sub(r"质量得分\s*[0-9.]+", "执行记录", text)
    text = text.replace("自动重规划", "自动调整算法")
    text = text.replace("候选结果仍有不确定项，需要用户文字或画笔反馈。", "标注结果已生成，等待用户检查。")
    text = text.replace("我会自动尝试不同处理方法并选择效果最好的一版。结果出来后，你只需要看图确认轮廓是否准确。", "我会直接生成并执行算法，再展示标注结果图。你只需要检查标注是否符合描述。")
    text = text.replace("我会自动生成并执行候选算法", "我会直接生成并执行算法")
    text = text.replace("轮廓结果", "标注结果")
    text = text.replace("检查轮廓是否准确", "检查标注是否准确")
    text = text.replace("检查轮廓有没有多圈或漏圈", "检查标注有没有多圈或漏标")
    text = text.replace("请只看轮廓是否圈得准确：没有多圈或漏圈", "请只看标注是否准确：没有多标或漏标")
    return text
