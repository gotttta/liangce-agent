"""Human decisions and algorithm publication recover across separate durable writes."""
import json

import pytest

from core.memory import MemoryService
from core.task_store import TaskStore


@pytest.fixture
def accepted_state(tmp_path):
    store = TaskStore(tmp_path / "tasks")
    task = store.create_task()
    state = {
        "task_id": task["id"], "run_id": "result_run", "state_version": 2,
        "agent_status": "accepted", "run_status": "completed", "iteration": 1,
        "pipeline": {"nodes": [{"id": "mask", "op": "threshold"}]},
        "annotated_image_path": str(tmp_path / "result.png"),
    }
    store.save_run_state(task["id"], state)
    return store, task["id"], state


def test_retry_after_memory_publication_failure_keeps_one_algorithm(accepted_state, monkeypatch):
    store, task_id, state = accepted_state
    original = MemoryService.publish

    def unavailable(*args, **kwargs):
        raise OSError("memory database unavailable")

    monkeypatch.setattr(MemoryService, "publish", unavailable)
    with pytest.raises(OSError, match="memory database unavailable"):
        store.accept_result(task_id, state)
    assert store.load_acceptance(task_id, state) is None
    assert len(list(store.algorithm_registry.root.glob("algorithm_*/algorithm.json"))) == 1

    monkeypatch.setattr(MemoryService, "publish", original)
    acceptance = store.accept_result(task_id, state)

    assert store.accept_result(task_id, state) == acceptance
    assert store.load_task(task_id)["accepted_at"] == acceptance["accepted_at"]
    assert len(list(store.algorithm_registry.root.glob("algorithm_*/algorithm.json"))) == 1


def test_retry_repairs_task_projection_without_republishing(accepted_state, monkeypatch):
    store, task_id, state = accepted_state
    original = store._write_json

    def fail_task_projection(path, value):
        if path.name == "task.json" and value.get("accepted_at"):
            raise OSError("task projection unavailable")
        return original(path, value)

    monkeypatch.setattr(store, "_write_json", fail_task_projection)
    with pytest.raises(OSError, match="task projection unavailable"):
        store.accept_result(task_id, state)
    acceptance = store.load_acceptance(task_id, state)
    assert acceptance is not None
    assert not store.load_task(task_id).get("accepted_at")

    monkeypatch.setattr(store, "_write_json", original)
    monkeypatch.setattr(store.algorithm_registry, "publish", lambda *a, **k: pytest.fail("algorithm republished"))
    assert store.accept_result(task_id, state) == acceptance
    assert store.load_task(task_id)["accepted_at"] == acceptance["accepted_at"]
    events = [json.loads(line) for line in (store.task_dir(task_id) / "events.jsonl").read_text().splitlines()]
    assert len([item for item in events if item["type"] == "result_accepted"]) == 1


def test_different_run_or_pipeline_cannot_reuse_acceptance(accepted_state):
    store, task_id, state = accepted_state
    store.accept_result(task_id, state)

    assert store.load_acceptance(task_id, {**state, "run_id": "another_run"}) is None
    assert store.load_acceptance(task_id, {**state, "pipeline": {"nodes": []}}) is None
