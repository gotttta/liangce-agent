"""Uncaught entry failures must preserve an honest, recoverable run status."""
from pathlib import Path
import sqlite3

from PIL import Image
import pytest

from core.agent_graph import run_agent_graph, resume_agent_graph
from core.memory.checkpoints import get_checkpointer
from core.orchestration_runtime import ActionStore
from core.request_control import RequestCancelled
from core.task_store import TaskStore


class NeedsInputProvider:
    def __init__(self):
        self.calls = 0

    def propose_action(self, *args, **kwargs):
        self.calls += 1
        return {"kind": "needs_input", "reason": "Need the target boundary"}


@pytest.fixture
def entry(tmp_path, monkeypatch):
    target = tmp_path / "input.png"
    image = Image.new("L", (24, 24), 0)
    image.paste(255, (8, 8, 16, 16))
    image.save(target)
    monkeypatch.setattr("core.sandbox.check_sandbox_available", lambda: {"image_id": "test-image"})
    store = TaskStore(tmp_path / "tasks")
    task_id = store.create_task()["id"]
    provider = NeedsInputProvider()

    def run():
        return run_agent_graph(target, "Find bright regions", provider=provider,
                               output_root=tmp_path / "outputs", task_store=store,
                               task_id=task_id, thread_id="failure_run")

    return store, task_id, provider, run


def checkpoint_state():
    saved = get_checkpointer().get_tuple({"configurable": {"thread_id": "failure_run"}})
    channels = saved.checkpoint["channel_values"]
    return channels.get("__root__") or channels


@pytest.mark.parametrize("failure, expected_status", [
    (ValueError("receipt identity corrupted"), "failed"),
    (RequestCancelled("cancelled before effect receipt"), "cancelled"),
])
def test_unhandled_receipt_failure_commits_terminal_state(entry, monkeypatch, failure, expected_status):
    store, task_id, provider, run = entry

    def broken_receipt(*args, **kwargs):
        raise failure

    monkeypatch.setattr(ActionStore, "load", broken_receipt)
    with pytest.raises(type(failure), match=str(failure)):
        run()

    persisted = store.load_latest_state(task_id)
    assert persisted["run_status"] == expected_status
    assert store.load_task(task_id)["status"] == expected_status
    assert checkpoint_state()["run_status"] == expected_status
    assert provider.calls == 0
    repeated = run()
    assert repeated["run_status"] == expected_status
    assert repeated["budget"] == persisted["budget"]
    assert provider.calls == 0


def test_transient_projection_write_failure_does_not_leave_running(entry, monkeypatch):
    store, task_id, provider, run = entry
    original = TaskStore.save_run_state
    failures = []

    def fail_once(self, current_id, state):
        if state.get("run_status") == "running" and not failures:
            failures.append(1)
            raise OSError("task projection unavailable")
        return original(self, current_id, state)

    monkeypatch.setattr(TaskStore, "save_run_state", fail_once)
    with pytest.raises(OSError, match="task projection unavailable"):
        run()

    assert store.load_latest_state(task_id)["run_status"] == "failed"
    assert checkpoint_state()["run_status"] == "failed"
    assert provider.calls == 0


@pytest.mark.parametrize("derived_write", ["memory", "node", "trajectory"])
def test_derived_record_failure_preserves_checkpointed_outcome(entry, monkeypatch, derived_write):
    from core import agent_graph
    from core.memory.service import MemoryService

    store, task_id, provider, run = entry

    def fail_write(*args, **kwargs):
        raise OSError("derived record unavailable")

    if derived_write == "memory":
        monkeypatch.setattr(MemoryService, "record_result", fail_write)
    elif derived_write == "node":
        monkeypatch.setattr(TaskStore, "save_node_result", fail_write)
    else:
        original = agent_graph.write_trajectory

        def fail_final_trajectory(state):
            if state.get("__interrupt__"):
                fail_write()
            return original(state)

        monkeypatch.setattr(agent_graph, "write_trajectory", fail_final_trajectory)

    with pytest.raises(OSError, match="derived record unavailable"):
        run()

    persisted = store.load_latest_state(task_id)
    checkpoint = checkpoint_state()
    assert persisted["run_status"] == checkpoint["run_status"] == "awaiting_feedback"
    assert persisted["state_version"] == checkpoint["state_version"]
    assert persisted["stop_reason"] == checkpoint["stop_reason"] == "needs_input"
    assert store.load_task(task_id)["status"] == "waiting_for_feedback"
    assert persisted["interrupt"]
    restored = run()
    assert restored["run_status"] == "awaiting_feedback"
    assert restored["budget"] == persisted["budget"]
    assert provider.calls == 1


def test_unavailable_checkpoint_does_not_fabricate_persisted_terminal_state(entry, monkeypatch, caplog):
    from langgraph.checkpoint.sqlite import SqliteSaver

    store, task_id, provider, run = entry
    original = SqliteSaver.put

    def fail_terminal_checkpoint(self, config, checkpoint, metadata, new_versions):
        channels = checkpoint.get("channel_values", {})
        state = channels.get("__root__") or channels
        if state.get("run_status") == "failed":
            raise sqlite3.OperationalError("checkpoint unavailable")
        return original(self, config, checkpoint, metadata, new_versions)

    def broken_receipt(*args, **kwargs):
        raise ValueError("receipt identity corrupted")

    monkeypatch.setattr(SqliteSaver, "put", fail_terminal_checkpoint)
    monkeypatch.setattr(ActionStore, "load", broken_receipt)
    with pytest.raises(ValueError, match="receipt identity corrupted"):
        run()

    assert checkpoint_state()["run_status"] == "running"
    assert store.load_latest_state(task_id)["run_status"] == "running"
    assert "checkpoint unavailable" in caplog.text
    assert provider.calls == 0


@pytest.mark.parametrize("automatic_decision", ["present", "revise"])
def test_human_completion_advances_both_checkpoint_and_projection(tmp_path, monkeypatch, automatic_decision):
    from agent_types import normalize_strategy
    from core.pipelines.dsl import strategy_to_pipeline

    monkeypatch.setattr("core.sandbox.check_sandbox_available", lambda: {"image_id": "test-image"})
    monkeypatch.setenv("LIANGCE_RUN_MAX_EXECUTIONS", "1")
    target = tmp_path / "input.png"
    image = Image.new("L", (24, 24), 0)
    image.paste(255, (8, 8, 16, 16))
    image.save(target)
    strategy = normalize_strategy({"segmentation": {
        "method": "bright_threshold", "sensitivity": 1, "min_area_px": 2, "morphology": "none"}})

    class Provider:
        def propose_action(self, *args, **kwargs):
            return {"kind": "propose", "understanding": {
                "task_summary": "Find bright region", "recommended_strategy": strategy,
                "target_constraints": {}, "rendering": {}},
                "pipeline": strategy_to_pipeline(strategy), "change_reason": "Threshold bright pixels"}

        def review_action(self, target, description, candidates, **kwargs):
            return {"kind": "review", "review": {"decision": automatic_decision,
                "selected_candidate": candidates[0]["name"], "reason": "Boundary checked",
                "observed_issues": [] if automatic_decision == "present" else ["Boundary needs correction"]}}

    store = TaskStore(tmp_path / "tasks")
    task_id = store.create_task()["id"]
    waiting = run_agent_graph(target, "Find bright region", provider=Provider(),
                              output_root=tmp_path / "outputs", task_store=store, task_id=task_id,
                              thread_id="human_completion")
    assert Path(waiting["annotated_image_path"]).is_file()
    if automatic_decision == "revise":
        assert waiting["run_status"] == "stopped"
        assert waiting["stop_reason"] == "execution_budget_exhausted"
        assert not waiting.get("interrupt")
    completed = resume_agent_graph("human_completion", {"action": "accept"})

    persisted = store.load_latest_state(task_id)
    assert persisted["run_status"] == "completed"
    assert persisted["state_version"] > waiting["state_version"]
    assert persisted["state_version"] == completed["state_version"]
    assert store.load_task(task_id)["status"] == "accepted"
    saved = get_checkpointer().get_tuple({"configurable": {"thread_id": "human_completion"}})
    checkpoint = saved.checkpoint["channel_values"]["__root__"]
    assert checkpoint["run_status"] == "completed"
    assert checkpoint["budget"] == waiting["budget"]
    experiment = next(item for item in persisted["experiment_records"]
                      if item["experiment_id"] == persisted["selected_experiment_id"])
    assert experiment["acceptance_status"] == "accepted"
