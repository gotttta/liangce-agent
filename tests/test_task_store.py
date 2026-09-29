import json
import pytest
from pathlib import Path

from core.task_store import TaskStore, save_rejection_record


def test_save_rejection_record_persists_reason_and_pipeline(tmp_path):
    record = save_rejection_record(
        "task_1",
        {"name": "bad-pipeline"},
        "比例尺被误标注",
        {"issues": ["scale_bar"]},
        task_root=tmp_path / "tasks",
    )
    files = list((tmp_path / "tasks" / "task_1" / "rejections").glob("*_rejection.json"))

    assert record["rejection_reason"] == "比例尺被误标注"
    assert len(files) == 1


def test_task_store_persists_samples_messages_and_node_runs(tmp_path):
    source = tmp_path / "sample.png"
    source.write_bytes(b"image")
    store = TaskStore(tmp_path / "tasks")

    task = store.create_task("bridge defect")
    sample = store.add_sample(task["id"], source)
    store.append_message(task["id"], "user", "find defect")
    record = store.save_node_result(
        task["id"],
        "understand_task",
        {"sample": sample["path"]},
        {"task_summary": "find defect"},
        1.25,
    )

    task_dir = tmp_path / "tasks" / task["id"]
    assert Path(sample["path"]).read_bytes() == b"image"
    assert json.loads((task_dir / "task.json").read_text())["current_node"] is None
    assert json.loads((task_dir / "nodes" / "understand_task" / "latest.json").read_text()) == record
    assert "find defect" in (task_dir / "conversation.jsonl").read_text()
    assert "node_finished" in (task_dir / "events.jsonl").read_text()


def test_task_store_lists_resumes_messages_and_structured_memory(tmp_path):
    store = TaskStore(tmp_path / "tasks")
    task = store.create_task()
    store.set_title(task["id"], "颗粒缺陷优化")
    store.append_message(task["id"], "user", "减少误检")
    store.append_message(task["id"], "assistant", "继续调整")
    memory = store.save_memory(task["id"], {
        "task_goal": "提取颗粒",
        "latest_iteration": 2,
    })

    listed = store.list_tasks()

    assert listed[0]["title"] == "颗粒缺陷优化"
    assert store.load_messages(task["id"])[-1]["content"] == "继续调整"
    assert store.load_memory(task["id"])["task_goal"] == "提取颗粒"
    assert memory["updated_at"]


def test_task_store_persists_handbook_reference_examples_separately(tmp_path):
    source = tmp_path / "handbook.png"
    source.write_bytes(b"annotated-example")
    store = TaskStore(tmp_path / "tasks")
    task = store.create_task()

    example = store.add_reference_example(task["id"], source, "甲方椭圆轮廓示例")
    loaded = store.load_task(task["id"])

    assert Path(example["image_path"]).read_bytes() == b"annotated-example"
    assert Path(example["image_path"]).parent.name == "references"
    assert loaded["reference_examples"][0]["description"] == "甲方椭圆轮廓示例"


def test_task_store_hides_legacy_quality_claims_when_loading_messages(tmp_path):
    store = TaskStore(tmp_path / "tasks")
    task = store.create_task()
    store.append_message(
        task["id"],
        "assistant",
        "质量状态 uncertain，得分 0.567。请只看轮廓是否圈得准确：没有多圈或漏圈。",
    )

    message = store.load_messages(task["id"])[0]["content"]

    assert "质量状态" not in message
    assert "得分" not in message
    assert "轮廓" not in message
    assert "标注" in message


def test_accepting_algorithm_does_not_publish_generated_operator(tmp_path):
    store = TaskStore(tmp_path / "tasks")
    task = store.create_task()
    generated = {
        "name": "candidate_mask",
        "source": "def apply(data, params):\n    return data > np.mean(data)",
        "input_artifact": "ImageArtifact",
        "output_artifact": "MaskArtifact",
        "atomic": True,
    }
    state = {
        "selected_candidate": "candidate",
        "description": "test",
        "pipeline": {
            "steps": [{"id": "final_mask", "op": "candidate_mask", "input": "image"}],
            "generated_operators": [generated],
        },
    }

    acceptance = store.accept_result(task["id"], state)

    assert acceptance["operator_library_paths"] == []
    assert store.operator_library.list_operators() == []

    approved = store.approve_tested_operator(
        generated,
        tested_by="user",
        test_note="manual test passed",
        source_task_id=task["id"],
    )
    assert approved["approval"]["user_tested"] is True
    assert store.operator_library.list_operators()[0]["name"] == "candidate_mask"

def test_delete_task_removes_directory(tmp_path):
    store = TaskStore(tmp_path / "tasks")
    task = store.create_task()
    assert (tmp_path / "tasks" / task["id"]).exists()

    assert store.delete_task(task["id"]) == task["id"]

    with pytest.raises(FileNotFoundError):
        store.task_dir(task["id"])
    # 越出任务根目录的 id 一律拒绝，防止路径穿越删除。
    with pytest.raises(FileNotFoundError):
        store.delete_task("../escape")


# ---- 运行状态持久化（原 tests/test_run_entry.py 中的 TaskStore 部分，UI 层删除后迁到这里）----

def run_state(task_id, **changes):
    return {
        "task_id": task_id,
        "run_id": "run_first",
        "graph_thread_id": "run_first",
        "run_started_at": "2026-09-21T12:00:00+00:00",
        "state_version": 0,
        "run_status": "running",
        "phase": "propose",
        "stop_reason": None,
        "budget": {"model_calls": 1, "executions": 0},
        **changes,
    }


@pytest.mark.parametrize("status, projected", [
    ("awaiting_feedback", "waiting_for_feedback"),
    ("completed", "completed"),
    ("stopped", "stopped"),
    ("failed", "failed"),
    ("cancelled", "cancelled"),
    ("interrupted", "interrupted"),
])
def test_node_diagnostics_cannot_overwrite_run_status(tmp_path, status, projected):
    store = TaskStore(tmp_path / "tasks")
    task_id = store.create_task()["id"]
    store.save_run_state(task_id, run_state(task_id))
    state = run_state(task_id, state_version=1, run_status=status,
                      phase="done", stop_reason="model_budget_exhausted")
    finished = store.save_run_state(task_id, state)

    store.save_node_result(task_id, "execute_candidate", {}, {}, 0.1)
    store.save_node_result(task_id, "late_diagnostic", {}, {}, 0.1, status="failed")

    assert store.load_task(task_id) == finished
    assert store.list_tasks()[0]["status"] == projected
    restored = store.load_latest_state(task_id)
    assert restored["run_status"] == status
    assert restored["stop_reason"] == "model_budget_exhausted"
    assert restored["budget"] == state["budget"]


def test_delayed_run_and_state_versions_do_not_replace_latest(tmp_path):
    store = TaskStore(tmp_path / "tasks")
    task_id = store.create_task()["id"]
    old = run_state(task_id)
    store.save_run_state(task_id, old)
    newer = run_state(task_id, run_id="run_second", graph_thread_id="run_second",
                      run_started_at="2026-09-21T12:01:00+00:00", state_version=4,
                      run_status="cancelled", phase="done")
    latest = store.save_run_state(task_id, newer)

    for stale in (
        {**old, "state_version": 100, "run_status": "awaiting_feedback"},
        {**newer, "state_version": 3, "run_status": "running"},
        {**newer, "run_status": "running"},
        {**old, "run_id": "unknown_old", "run_started_at": None},
    ):
        assert store.save_run_state(task_id, stale) == latest
    assert store.load_latest_state(task_id)["run_id"] == "run_second"
    assert store.load_latest_state(task_id)["run_status"] == "cancelled"


def test_run_projection_validates_task_and_path_scope(tmp_path):
    store = TaskStore(tmp_path / "tasks")
    task_id = store.create_task()["id"]
    with pytest.raises(ValueError, match="valid run_id"):
        store.save_run_state(task_id, run_state(task_id, run_id="../escape"))
    with pytest.raises(ValueError, match="another task"):
        store.save_run_state(task_id, run_state("other_task"))
    with pytest.raises(ValueError, match="valid run_status"):
        store.save_run_state(task_id, run_state(task_id, run_status="unknown"))
    store.save_run_state(task_id, run_state(task_id, run_id="run.v2"))
    assert store.load_latest_state(task_id)["run_id"] == "run.v2"
