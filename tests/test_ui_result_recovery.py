"""Result rendering and terminal human decisions must survive stale browser state."""
from copy import deepcopy

from PIL import Image
import pytest

from core.task_store import TaskStore
from ui import annotation_app


@pytest.fixture
def saved_result(tmp_path, monkeypatch):
    monkeypatch.setattr(annotation_app, "TASK_ROOT", tmp_path / "tasks")
    store = TaskStore(tmp_path / "tasks")
    task = store.create_task()
    image = tmp_path / "result.png"
    Image.new("RGB", (16, 16), "white").save(image)
    state = {
        "task_id": task["id"], "run_id": "saved_run", "graph_thread_id": "saved_run",
        "run_started_at": "2026-09-21T12:00:00+00:00", "state_version": 1,
        "run_status": "awaiting_feedback", "agent_status": "waiting_for_feedback",
        "annotated_image_path": str(image), "selected_candidate": "candidate_1",
        "measurements": {"results": [], "summary": {"count": 1}},
        "conversation": [{"role": "assistant", "content": "Executed candidate_1; one region found."}],
        "decision": {"automatic_review_passed": False, "reason": "Boundary review did not pass."},
    }
    store.append_message(task["id"], "user", "Find all target boundaries")
    task = store.save_run_state(task["id"], state)
    return store, task, state


def test_restore_rebuilds_missing_latest_result_without_changing_conversation_file(saved_result):
    store, task, state = saved_result
    path = store.task_dir(task["id"]) / "conversation.jsonl"
    original = path.read_bytes()

    restored = annotation_app.resume_chat_task(task["id"])
    again = annotation_app.resume_chat_task(task["id"])

    assert restored[0][0] == {"role": "user", "content": "Find all target boundaries"}
    assert state["conversation"][0] in restored[0]
    assert {"role": "assistant", "content": state["decision"]["reason"]} in restored[0]
    assert restored[0][-1]["content"][0] == state["annotated_image_path"]
    assert restored[10]["visible"] is True
    assert restored[17]["visible"] is True
    assert again[0] == restored[0]
    assert path.read_bytes() == original


@pytest.mark.parametrize("message_format", ["tuple", "dict"])
def test_restore_does_not_duplicate_existing_latest_image(saved_result, message_format):
    store, task, state = saved_result
    content = (state["annotated_image_path"], "Result") if message_format == "tuple" else {
        "path": state["annotated_image_path"], "mime_type": "image/png",
    }
    store.append_message(task["id"], "assistant", content)
    original = store.load_messages(task["id"])

    assert annotation_app.resume_chat_task(task["id"])[0] == original


def test_restore_appends_latest_image_even_when_older_result_exists(saved_result, tmp_path):
    store, task, state = saved_result
    old = tmp_path / "old.png"
    Image.new("RGB", (16, 16), "black").save(old)
    store.append_message(task["id"], "assistant", (str(old), "Old result"))
    restored = annotation_app.resume_chat_task(task["id"])
    images = [item["content"][0] for item in restored[0] if isinstance(item.get("content"), (list, tuple))]
    assert images == [str(old), state["annotated_image_path"]]


def test_missing_result_file_never_exposes_acceptance_buttons(saved_result):
    store, task, state = saved_result
    from pathlib import Path
    Path(state["annotated_image_path"]).unlink()

    restored = annotation_app.resume_chat_task(task["id"])

    assert restored[10]["visible"] is False
    assert restored[17]["visible"] is False
    assert restored[0] == store.load_messages(task["id"])


@pytest.mark.parametrize("terminal_status,action", [("accepted", "accept"), ("exited", "exit")])
def test_duplicate_terminal_action_returns_saved_state_without_side_effects(saved_result, monkeypatch, terminal_status, action):
    store, task, displayed = saved_result
    finished = {**displayed, "state_version": 2, "agent_status": terminal_status,
                "run_status": "completed" if action == "accept" else "stopped"}
    store.save_run_state(task["id"], finished)
    if action == "accept":
        store.accept_result(task["id"], finished)
    else:
        store.exit_task(task["id"])
    latest = store.load_task(task["id"])
    original_messages = store.load_messages(task["id"])
    monkeypatch.setattr(annotation_app, "resume_agent_graph", lambda *a, **k: pytest.fail("terminal checkpoint resumed"))
    monkeypatch.setattr(TaskStore, "accept_result", lambda *a, **k: pytest.fail("algorithm published twice"))
    monkeypatch.setattr(TaskStore, "exit_task", lambda *a, **k: pytest.fail("exit repeated"))

    restored = annotation_app.handle_result_action(action, None, original_messages, task, deepcopy(displayed))

    assert restored[5]["agent_status"] == terminal_status
    assert restored[5]["state_version"] == 2
    assert restored[9]["visible"] is False
    assert store.load_task(task["id"]) == latest
    assert store.load_messages(task["id"]) == original_messages


def test_saved_accept_decision_can_complete_interrupted_publication(saved_result, monkeypatch):
    store, task, displayed = saved_result
    decided = {**displayed, "state_version": 2, "agent_status": "accepted", "run_status": "completed"}
    store.save_run_state(task["id"], decided)
    monkeypatch.setattr(annotation_app, "resume_agent_graph", lambda *a, **k: pytest.fail("terminal checkpoint resumed"))

    assert annotation_app.resume_chat_task(task["id"])[10]["visible"] is True
    completed = annotation_app.handle_result_action("accept", None, [], task, deepcopy(displayed))
    acceptance = store.load_acceptance(task["id"], decided)

    assert acceptance is not None
    assert completed[6]["accepted_at"] == acceptance["accepted_at"]
    assert annotation_app.resume_chat_task(task["id"])[10]["visible"] is False
    again = annotation_app.handle_result_action("accept", None, [], task, deepcopy(displayed))
    assert again[6]["accepted_at"] == acceptance["accepted_at"]
    assert len(list(store.algorithm_registry.root.glob("algorithm_*/algorithm.json"))) == 1


@pytest.mark.parametrize("terminal_status,action", [("accepted", "exit"), ("exited", "accept")])
def test_conflicting_terminal_action_from_old_page_is_rejected(saved_result, monkeypatch, terminal_status, action):
    store, task, displayed = saved_result
    finished = {**displayed, "state_version": 2, "agent_status": terminal_status,
                "run_status": "completed" if terminal_status == "accepted" else "stopped"}
    latest = store.save_run_state(task["id"], finished)
    monkeypatch.setattr(annotation_app, "resume_agent_graph", lambda *a, **k: pytest.fail("terminal checkpoint resumed"))
    monkeypatch.setattr(TaskStore, "accept_result", lambda *a, **k: pytest.fail("terminal result accepted again"))
    monkeypatch.setattr(TaskStore, "exit_task", lambda *a, **k: pytest.fail("accepted result exited"))

    with pytest.raises(Exception, match="不能执行相反操作"):
        annotation_app.handle_result_action(action, None, [], task, deepcopy(displayed))

    assert store.load_task(task["id"]) == latest


def test_checkpoint_terminal_conflict_cannot_publish_opposite_ui_action(saved_result, monkeypatch):
    store, task, displayed = saved_result
    monkeypatch.setattr(annotation_app, "resume_agent_graph", lambda *a, **k: {
        **displayed, "agent_status": "accepted", "run_status": "completed", "state_version": 2,
    })
    monkeypatch.setattr(TaskStore, "exit_task", lambda *a, **k: pytest.fail("accepted checkpoint exited"))
    monkeypatch.setattr(annotation_app, "save_rejection_record", lambda *a, **k: pytest.fail("accepted checkpoint rejected"))

    with pytest.raises(Exception, match="不能执行相反操作"):
        annotation_app.handle_result_action("exit", None, [], task, deepcopy(displayed))
