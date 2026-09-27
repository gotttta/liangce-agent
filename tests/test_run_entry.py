from pathlib import Path

from PIL import Image
import pytest

from core.task_store import TaskStore
from ui import annotation_app


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


def test_ui_calls_graph_once_and_preserves_prepared_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(annotation_app, "ROOT", tmp_path)
    monkeypatch.setattr(annotation_app, "TASK_ROOT", tmp_path / "tasks")
    source = tmp_path / "input.png"
    reference = tmp_path / "reference.png"
    Image.new("L", (8, 8)).save(source)
    Image.new("L", (8, 8)).save(reference)
    store = TaskStore(tmp_path / "tasks")
    task = store.create_task()
    history = [{"role": "user", "content": "prior instruction"}]

    class Provider:
        model = "single-entry-test"

        def understand_task(self, *args, **kwargs):
            pytest.fail("UI must not plan before invoking the graph")

    provider = Provider()
    monkeypatch.setattr(annotation_app, "build_runtime_provider", lambda: provider)
    from core.memory.service import MemoryService
    monkeypatch.setattr(MemoryService, "apply_updates", lambda *a, **k: pytest.fail("UI duplicated memory update"))
    monkeypatch.setattr(MemoryService, "record_result", lambda *a, **k: pytest.fail("UI duplicated result memory"))
    canvas = object()
    feedback_calls = []

    def save_feedback(editor, current, previous, green_editor_value=None):
        feedback_calls.append(editor)
        return {**previous, "feedback_pixel_count": 3, "include_mask_path": "saved-include.png"}

    monkeypatch.setattr(annotation_app, "save_canvas_feedback", save_feedback)
    graph_calls = []

    def graph(**kwargs):
        graph_calls.append(kwargs)
        state = run_state(task["id"], run_status="stopped", state_version=1,
                          stop_reason="model_budget_exhausted", phase="done",
                          measurements={"results": [], "summary": {}},
                          conversation=[{"role": "assistant", "content": "Call budget exhausted."}])
        kwargs["task_store"].save_run_state(task["id"], state)
        return state

    monkeypatch.setattr(annotation_app, "run_agent_graph", graph)
    result = annotation_app.run_chat_agent(
        str(source), "find target", history, task, editor_value=canvas,
        reference_example_paths=[str(reference)],
    )

    assert len(graph_calls) == 1
    call = graph_calls[0]
    assert "understanding" not in call
    assert call["provider"] is provider
    assert call["task_id"] == task["id"]
    assert call["task_store"].root == tmp_path / "tasks"
    assert call["memory_context"]["task_id"] == task["id"]
    assert call["memory_context"]["conversation"] == history
    assert call["previous_state"]["human_feedback"]["include_mask_path"] == "saved-include.png"
    assert len(call["reference_examples"]) == 1
    assert Path(call["reference_examples"][0]["image_path"]).is_file()
    assert feedback_calls == [canvas]
    assert result[0][:1] == history
    assert result[6]["status"] == "stopped"
    assert result[9]["visible"] is False
    assert result[13]["visible"] is False
    assert "Call budget exhausted." in result[0][-1]["content"]
    assert "任务完成" not in result[0][-1]["content"]
    assert annotation_app._task_rows()[0]["status"] == "已停止"


@pytest.mark.parametrize("status", ["stopped", "failed", "cancelled", "interrupted"])
def test_ui_can_display_terminal_run_without_understanding_or_messages(tmp_path, monkeypatch, status):
    monkeypatch.setattr(annotation_app, "ROOT", tmp_path)
    monkeypatch.setattr(annotation_app, "TASK_ROOT", tmp_path / "tasks")
    source = tmp_path / "input.png"
    Image.new("L", (8, 8)).save(source)
    task = TaskStore(tmp_path / "tasks").create_task()
    monkeypatch.setattr(annotation_app, "build_runtime_provider", lambda: object())
    monkeypatch.setattr(annotation_app, "run_agent_graph", lambda **kwargs: {
        "run_status": status, "measurements": {"results": []},
    })

    result = annotation_app.run_chat_agent(str(source), "find target", [], task)

    assert annotation_app._TASK_STATUS_LABELS[status] in result[0][-1]["content"]
    assert result[9]["visible"] is False


def test_ui_does_not_forward_ground_truth_invalidated_for_new_image(tmp_path, monkeypatch):
    from core.input_contract import input_identity

    monkeypatch.setattr(annotation_app, "ROOT", tmp_path)
    monkeypatch.setattr(annotation_app, "TASK_ROOT", tmp_path / "tasks")
    old_image, new_image, mask = (tmp_path / name for name in ("old.png", "new.png", "mask.png"))
    Image.new("L", (8, 8), 0).save(old_image)
    Image.new("L", (8, 8), 255).save(new_image)
    Image.new("L", (8, 8), 0).save(mask)
    store = TaskStore(tmp_path / "tasks")
    task = store.create_task()
    task["ground_truth"] = {**input_identity(old_image), "mask_path": str(mask)}
    store._write_json(store.task_dir(task["id"]) / "task.json", task)
    monkeypatch.setattr(annotation_app, "build_runtime_provider", lambda: object())
    calls = []

    def graph(**kwargs):
        calls.append(kwargs)
        return {"run_status": "stopped", "measurements": {"results": []}}

    monkeypatch.setattr(annotation_app, "run_agent_graph", graph)
    annotation_app.run_chat_agent(str(new_image), "find target", [], task)

    assert calls[0]["ground_truth_mask_path"] is None
    assert store.load_task(task["id"])["ground_truth"] is None


@pytest.mark.parametrize("action", ["accept", "exit"])
def test_stale_human_action_cannot_change_current_run(tmp_path, monkeypatch, action):
    monkeypatch.setattr(annotation_app, "TASK_ROOT", tmp_path / "tasks")
    store = TaskStore(tmp_path / "tasks")
    task = store.create_task()
    old = run_state(task["id"], run_status="awaiting_feedback")
    store.save_run_state(task["id"], old)
    current = run_state(task["id"], run_id="run_second", graph_thread_id="run_second",
                        run_started_at="2026-09-21T12:01:00+00:00", run_status="awaiting_feedback")
    latest = store.save_run_state(task["id"], current)
    monkeypatch.setattr(annotation_app, "resume_agent_graph", lambda *a, **k: pytest.fail("stale graph resumed"))
    monkeypatch.setattr(TaskStore, "accept_result", lambda *a, **k: pytest.fail("stale result published"))
    monkeypatch.setattr(TaskStore, "exit_task", lambda *a, **k: pytest.fail("current task exited"))

    for displayed in (old, {key: value for key, value in old.items() if key != "run_id"}):
        with pytest.raises(Exception, match="较早的结果"):
            annotation_app.handle_result_action(action, None, [], task, displayed)

    assert store.load_task(task["id"]) == latest


@pytest.mark.parametrize("error", [RuntimeError("checkpoint storage unavailable"), ValueError("未找到持久化工作流检查点")])
def test_current_checkpoint_failure_does_not_publish_result(tmp_path, monkeypatch, error):
    monkeypatch.setattr(annotation_app, "TASK_ROOT", tmp_path / "tasks")
    store = TaskStore(tmp_path / "tasks")
    task = store.create_task()
    state = run_state(task["id"], run_status="awaiting_feedback", interrupt=[{}],
                      annotated_image_path=str(tmp_path / "result.png"))
    latest = store.save_run_state(task["id"], state)

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(annotation_app, "resume_agent_graph", fail)
    monkeypatch.setattr(TaskStore, "accept_result", lambda *a, **k: pytest.fail("uncommitted result published"))
    with pytest.raises(Exception, match="无法保存本次结果操作"):
        annotation_app.handle_result_action("accept", None, [], task, state)
    assert store.load_task(task["id"]) == latest


def test_human_action_holds_task_lock_through_result_write(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from core.orchestration_runtime import TaskBusyError, task_lock

    monkeypatch.setattr(annotation_app, "TASK_ROOT", tmp_path / "tasks")
    store = TaskStore(tmp_path / "tasks")
    task = store.create_task()
    state = run_state(task["id"], run_status="awaiting_feedback")
    store.save_run_state(task["id"], state)

    def concurrent_run():
        with task_lock(store.root, task["id"]):
            pytest.fail("another run acquired the task during human action")

    def write_result(*args, **kwargs):
        with ThreadPoolExecutor(max_workers=1) as executor:
            with pytest.raises(TaskBusyError):
                executor.submit(concurrent_run).result(timeout=2)
        return "written"

    monkeypatch.setattr(annotation_app, "_handle_result_action", write_result)
    assert annotation_app.handle_result_action("accept", None, [], task, state) == "written"


def test_busy_checkpoint_resume_does_not_fall_through_to_acceptance(tmp_path, monkeypatch):
    from core.orchestration_runtime import TaskBusyError

    monkeypatch.setattr(annotation_app, "TASK_ROOT", tmp_path / "tasks")
    store = TaskStore(tmp_path / "tasks")
    task = store.create_task()
    state = run_state(task["id"], run_status="awaiting_feedback", interrupt=[{}],
                      annotated_image_path=str(tmp_path / "result.png"))
    store.save_run_state(task["id"], state)

    def busy(*args, **kwargs):
        raise TaskBusyError("already running")

    monkeypatch.setattr(annotation_app, "resume_agent_graph", busy)
    monkeypatch.setattr(TaskStore, "accept_result", lambda *a, **k: pytest.fail("busy run was accepted"))
    with pytest.raises(Exception, match="任务仍在运行"):
        annotation_app.handle_result_action("accept", None, [], task, state)


def test_internal_type_error_does_not_repeat_entire_ui_run(monkeypatch):
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise TypeError("operator() got an unexpected keyword argument 'source'")

    monkeypatch.setattr(annotation_app, "run_chat_agent", fail)
    updates = list(annotation_app.run_chat_agent_stream("input.png", "find target", [], None))

    assert calls == [1]
    assert "unexpected keyword argument" in updates[-1][0][-1]["content"]


def test_stream_error_restores_new_run_instead_of_previous_result(tmp_path, monkeypatch):
    monkeypatch.setattr(annotation_app, "TASK_ROOT", tmp_path / "tasks")
    store = TaskStore(tmp_path / "tasks")
    task = store.create_task()
    previous = run_state(task["id"], run_status="awaiting_feedback", annotated_image_path="old.png")
    store.save_run_state(task["id"], previous)
    failed = run_state(task["id"], run_id="run_failed", run_started_at="2026-09-21T12:01:00+00:00",
                       run_status="failed", stop_reason="receipt_corrupt", annotated_image_path="new.png",
                       measurements={"results": [], "summary": {"count": 3}})

    def fail(*args, **kwargs):
        store.save_run_state(task["id"], failed)
        raise ValueError("receipt corrupted")

    monkeypatch.setattr(annotation_app, "run_chat_agent", fail)
    updates = list(annotation_app.run_chat_agent_stream("input.png", "find target", [], task, previous))

    assert updates[-1][5]["run_id"] == "run_failed"
    assert updates[-1][6]["status"] == "failed"
    assert updates[-1][1]["value"] == "new.png"
    assert updates[-1][9]["visible"] is True


def test_error_response_survives_unavailable_task_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(annotation_app, "TASK_ROOT", tmp_path / "tasks")
    task = TaskStore(tmp_path / "tasks").create_task()

    def fail(*args, **kwargs):
        raise OSError("storage unavailable")

    monkeypatch.setattr(annotation_app, "run_chat_agent", fail)
    monkeypatch.setattr(annotation_app, "TaskStore", fail)
    updates = list(annotation_app.run_chat_agent_stream("input.png", "find target", [], task))

    assert "storage unavailable" in updates[-1][0][-1]["content"]
    assert updates[-1][9]["visible"] is False


def test_restored_active_run_hides_human_actions(tmp_path, monkeypatch):
    monkeypatch.setattr(annotation_app, "TASK_ROOT", tmp_path / "tasks")
    store = TaskStore(tmp_path / "tasks")
    task = store.create_task()
    state = run_state(task["id"], annotated_image_path="partial.png")
    store.save_run_state(task["id"], state)

    restored = annotation_app.resume_chat_task(task["id"])

    assert restored[10]["visible"] is False
    with pytest.raises(Exception, match="任务仍在运行"):
        annotation_app.handle_result_action("accept", None, [], task, state)


def test_accept_without_result_does_not_resume_human_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(annotation_app, "TASK_ROOT", tmp_path / "tasks")
    store = TaskStore(tmp_path / "tasks")
    task = store.create_task()
    state = run_state(task["id"], run_status="awaiting_feedback", interrupt=[{}])
    latest = store.save_run_state(task["id"], state)
    monkeypatch.setattr(annotation_app, "resume_agent_graph", lambda *a, **k: pytest.fail("empty result accepted"))

    response = annotation_app.handle_result_action("accept", None, [], task, state)

    assert "没有保存完整" in response[0][-1]["content"]
    assert store.load_task(task["id"]) == latest


def test_stopped_result_uses_shared_human_review_entry(tmp_path, monkeypatch):
    store = TaskStore(tmp_path / "tasks")
    task = store.create_task()
    state = run_state(task["id"], run_status="stopped", stop_reason="execution_budget_exhausted",
                      annotated_image_path="result.png")
    calls = []

    def resume(thread_id, response):
        calls.append((thread_id, response))
        return {**state, "agent_status": "accepted", "run_status": "completed", "state_version": 1}

    monkeypatch.setattr(annotation_app, "resume_agent_graph", resume)
    updated = annotation_app._resume_human_review(store, task["id"], state, "accept")

    assert calls == [("run_first", {"action": "accept"})]
    assert updated["run_status"] == "completed"
