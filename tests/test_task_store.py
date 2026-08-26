import json
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
    assert json.loads((task_dir / "task.json").read_text())["current_node"] == "understand_task"
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
