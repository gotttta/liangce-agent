import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from core.agent_workflow import (
    IterationPolicy,
    WorkflowRuntime,
    build_workflow_graph,
)
from core.iteration_tracker import IterationTracker
from core.reference_store import ReferenceStore
from core.scoring import ImageScore, RunScore, score_run
from core.workflow_state import AlgorithmSpec, WorkflowState, restore_state


class FakeProvider:
    """脚本化的视觉模型：按序返回预设回复，不做真实推理。"""

    def __init__(self, replies):
        self.replies = list(replies)
        self.model = "fake-model"
        self.calls = 0

    def _complete_action(self, messages):
        self.calls += 1
        return self.replies.pop(0)


PIPELINE_REPLY = json.dumps({"pipeline": [
    {"op": "normalize"},
    {"op": "global_threshold", "polarity": "bright", "sensitivity": 2.0},
], "notes": "初始版本"})


def _write_sample(tmp_path: Path, name: str = "img_a") -> tuple[Path, np.ndarray]:
    image = np.zeros((32, 32), dtype=np.uint8)
    image[8:16, 8:16] = 220
    path = tmp_path / "images" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    return path, image.astype(bool).copy()


def _reference_mask() -> np.ndarray:
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:16, 8:16] = True
    return mask


def _half_mask() -> np.ndarray:
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:16, 8:12] = True
    return mask


def _empty_mask() -> np.ndarray:
    return np.zeros((32, 32), dtype=bool)


def _store_reference(tmp_path: Path, image_path: Path) -> ReferenceStore:
    store = ReferenceStore(tmp_path / "refs")
    store.save(image_path.stem, _reference_mask(), {
        "image_id": image_path.stem, "image_path": str(image_path), "task": "找出亮块",
        "target_type": "defect", "sam_iou_score": 0.99,
        "confirmed_at": "2026-09-28T00:00:00Z", "confirmed_by": "user",
    })
    return store


def _runtime(tmp_path: Path, provider=None, policy=None) -> WorkflowRuntime:
    return WorkflowRuntime(provider, references_root=tmp_path / "refs",
                           policy=policy or IterationPolicy(patience=5, target=0.85, max_iterations=100))


def _prepared_state(tmp_path: Path, image_path: Path, provider=None, policy=None) -> WorkflowState:
    runtime = _runtime(tmp_path, provider=provider, policy=policy)
    state = WorkflowState(image_paths=[str(image_path)], task="找出亮块", run_id="run-1",
                          output_root=str(tmp_path / "outputs"))
    return runtime.prepare(state)


def test_prepare_routes_to_iterate_when_reference_exists(tmp_path):
    image_path, _ = _write_sample(tmp_path)
    _store_reference(tmp_path, image_path)
    state = _prepared_state(tmp_path, image_path)
    assert state.next_node == "iterate"


def test_prepare_routes_to_gen_reference_without_reference(tmp_path):
    image_path, _ = _write_sample(tmp_path)
    ReferenceStore(tmp_path / "refs")
    state = _prepared_state(tmp_path, image_path)
    assert state.next_node == "gen_reference"


def test_score_routes_to_promote_when_improved(tmp_path, monkeypatch):
    image_path, _ = _write_sample(tmp_path)
    _store_reference(tmp_path, image_path)
    monkeypatch.setattr(WorkflowRuntime, "_predicted_instances",
                        lambda self, image, pipeline: [_half_mask()])
    runtime = _runtime(tmp_path)
    state = _prepared_state(tmp_path, image_path)
    state = replace(state, current_spec=AlgorithmSpec(pipeline=[{"op": "normalize"}]),
                    best_score=0.3)
    state = runtime.score(state)
    assert state.next_node == "promote"
    assert state.last_run_score is not None
    assert state.last_run_score.composite_mean > 0.3


def test_score_routes_to_iterate_when_not_improved_and_not_stopping(tmp_path, monkeypatch):
    image_path, _ = _write_sample(tmp_path)
    _store_reference(tmp_path, image_path)
    monkeypatch.setattr(WorkflowRuntime, "_predicted_instances",
                        lambda self, image, pipeline: [_half_mask()])
    runtime = _runtime(tmp_path)
    state = _prepared_state(tmp_path, image_path)
    state = replace(state, current_spec=AlgorithmSpec(pipeline=[{"op": "normalize"}]),
                    best_score=0.9)
    state = runtime.score(state)
    assert state.next_node == "iterate"
    assert state.stop_reason == ""


def test_score_routes_to_human_gate_when_stopping(tmp_path, monkeypatch):
    image_path, _ = _write_sample(tmp_path)
    _store_reference(tmp_path, image_path)
    monkeypatch.setattr(WorkflowRuntime, "_predicted_instances",
                        lambda self, image, pipeline: [_half_mask()])
    policy = IterationPolicy(patience=2, target=0.99, max_iterations=100)
    runtime = _runtime(tmp_path, policy=policy)
    state = _prepared_state(tmp_path, image_path, policy=policy)
    tracker = IterationTracker(Path(state.run_dir))
    for score in (0.5, 0.5):
        tracker.record(_run_score(score), {"pipeline": []})
    state = replace(state, current_spec=AlgorithmSpec(pipeline=[{"op": "normalize"}]),
                    best_score=0.5)
    state = runtime.score(state)
    assert state.next_node == "human_gate"
    assert state.stop_reason == "no_improvement"
    assert "composite_mean" in state.human_message


def _run_score(composite: float) -> RunScore:
    return score_run([ImageScore("img_a", composite, 0, 0, 1, composite)])


def test_promote_updates_best_spec_and_score(tmp_path, monkeypatch):
    image_path, _ = _write_sample(tmp_path)
    _store_reference(tmp_path, image_path)
    monkeypatch.setattr(WorkflowRuntime, "_predicted_instances",
                        lambda self, image, pipeline: [_half_mask()])
    runtime = _runtime(tmp_path)
    state = _prepared_state(tmp_path, image_path)
    spec = AlgorithmSpec(pipeline=[{"op": "normalize"}], notes="v2")
    run_score = _run_score(0.6)
    state = replace(state, current_spec=spec, last_run_score=run_score, best_score=0.2)
    state = runtime.promote(state)
    assert state.best_spec == spec
    assert state.best_score == 0.6
    assert state.best_run_score == run_score
    assert state.next_node == "iterate"


def test_promote_routes_to_human_gate_when_stopping(tmp_path, monkeypatch):
    image_path, _ = _write_sample(tmp_path)
    _store_reference(tmp_path, image_path)
    monkeypatch.setattr(WorkflowRuntime, "_predicted_instances",
                        lambda self, image, pipeline: [_half_mask()])
    policy = IterationPolicy(patience=2, target=0.4, max_iterations=100)
    runtime = _runtime(tmp_path, policy=policy)
    state = _prepared_state(tmp_path, image_path, policy=policy)
    tracker = IterationTracker(Path(state.run_dir))
    tracker.record(_run_score(0.9), {"pipeline": []})
    spec = AlgorithmSpec(pipeline=[{"op": "normalize"}])
    state = replace(state, current_spec=spec, last_run_score=_run_score(0.9), best_score=0.0)
    state = runtime.promote(state)
    assert state.next_node == "human_gate"
    assert state.stop_reason == "target_reached"


def _invoke_to_interrupt(tmp_path, monkeypatch, provider, policy, image_path, thread_id):
    monkeypatch.setattr(WorkflowRuntime, "_predicted_instances",
                        lambda self, image, pipeline: [_half_mask()])
    graph = build_workflow_graph(provider=provider, checkpointer=MemorySaver(),
                                 references_root=tmp_path / "refs", policy=policy)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 200}
    state = WorkflowState(image_paths=[str(image_path)], task="找出亮块",
                          run_id=thread_id, output_root=str(tmp_path / "outputs"))
    graph.invoke(state, config=config, durability="sync")
    return graph, config


def _interrupts(graph, config):
    snapshot = graph.get_state(config)
    return [item for task in (snapshot.tasks or []) for item in (task.interrupts or [])]


def test_graph_stops_on_target_reached_then_accept_finishes(tmp_path, monkeypatch):
    image_path, _ = _write_sample(tmp_path)
    _store_reference(tmp_path, image_path)
    provider = FakeProvider([PIPELINE_REPLY])
    policy = IterationPolicy(patience=2, target=0.4, max_iterations=10)
    graph, config = _invoke_to_interrupt(tmp_path, monkeypatch, provider, policy,
                                         image_path, "target-run")
    interrupts = _interrupts(graph, config)
    assert interrupts and interrupts[0].value["stop_reason"] == "target_reached"

    graph.invoke(Command(resume={"action": "accept"}), config=config, durability="sync")
    values = restore_state(graph.get_state(config).values)
    assert values.next_node == ""
    run_dir = Path(values.run_dir)
    algorithm = json.loads((run_dir / "algorithm.json").read_text(encoding="utf-8"))
    assert algorithm["pipeline"][0]["op"] == "normalize"
    score = json.loads((run_dir / "score.json").read_text(encoding="utf-8"))
    assert score["composite_mean"] > 0.4


def test_graph_stops_on_no_improvement(tmp_path, monkeypatch):
    image_path, _ = _write_sample(tmp_path)
    _store_reference(tmp_path, image_path)
    provider = FakeProvider([PIPELINE_REPLY, PIPELINE_REPLY])
    policy = IterationPolicy(patience=2, target=0.99, max_iterations=10)
    graph, config = _invoke_to_interrupt(tmp_path, monkeypatch, provider, policy,
                                         image_path, "plateau-run")
    interrupts = _interrupts(graph, config)
    assert interrupts and interrupts[0].value["stop_reason"] == "no_improvement"
    values = restore_state(graph.get_state(config).values)
    assert values.best_score > 0.0


def test_graph_stops_on_max_iterations(tmp_path, monkeypatch):
    image_path, _ = _write_sample(tmp_path)
    _store_reference(tmp_path, image_path)
    provider = FakeProvider([PIPELINE_REPLY, PIPELINE_REPLY])
    policy = IterationPolicy(patience=2, target=0.99, max_iterations=2)
    graph, config = _invoke_to_interrupt(tmp_path, monkeypatch, provider, policy,
                                         image_path, "cap-run")
    interrupts = _interrupts(graph, config)
    assert interrupts and interrupts[0].value["stop_reason"] == "max_iterations"


def test_graph_continue_after_stop_extends_iterations(tmp_path, monkeypatch):
    image_path, _ = _write_sample(tmp_path)
    _store_reference(tmp_path, image_path)
    provider = FakeProvider([PIPELINE_REPLY, PIPELINE_REPLY, PIPELINE_REPLY, PIPELINE_REPLY])
    policy = IterationPolicy(patience=2, target=0.4, max_iterations=10)
    graph, config = _invoke_to_interrupt(tmp_path, monkeypatch, provider, policy,
                                         image_path, "continue-run")
    assert _interrupts(graph, config)[0].value["stop_reason"] == "target_reached"

    graph.invoke(Command(resume={"action": "continue"}), config=config, durability="sync")
    # 宽限轮耗尽后分数仍达 target，会再次停在 human_gate
    values = restore_state(graph.get_state(config).values)
    second = _interrupts(graph, config)
    assert second and second[0].value["stop_reason"] == "target_reached"
    assert values.grace_iterations == 0


def test_iterate_retries_after_invalid_model_output(tmp_path):
    runtime = _runtime(tmp_path, provider=FakeProvider([
        json.dumps({"pipeline": []}),
        PIPELINE_REPLY,
    ]))
    state = WorkflowState(image_paths=[], task="找出亮块")
    state = runtime.iterate(state)
    assert state.current_spec is not None
    assert state.current_spec.pipeline[0]["op"] == "normalize"
    assert state.next_node == "score"


def test_iterate_rejects_unknown_operator(tmp_path):
    runtime = _runtime(tmp_path, provider=FakeProvider([
        json.dumps({"pipeline": [{"op": "no_such_op"}]}),
        json.dumps({"pipeline": [{"op": "no_such_op"}]}),
    ]))
    import pytest
    with pytest.raises(ValueError, match="no_such_op"):
        runtime.iterate(WorkflowState(image_paths=[], task="找出亮块"))


def test_gen_reference_generates_then_confirm_saves(tmp_path, monkeypatch):
    image_path, _ = _write_sample(tmp_path)
    store = ReferenceStore(tmp_path / "refs")
    assert store.list_scoreable() == []

    fake_session = MagicMock()
    monkeypatch.setattr("core.agent_workflow.SamSession", lambda model_name: fake_session)
    select = MagicMock(return_value=[(0.92, _reference_mask())])
    monkeypatch.setattr("core.agent_workflow._select_defect_masks", select)

    provider = FakeProvider([PIPELINE_REPLY])  # 确认参考后继续迭代直到触发停止
    policy = IterationPolicy(patience=2, target=0.4, max_iterations=5)
    graph = build_workflow_graph(provider=provider, checkpointer=MemorySaver(),
                                 references_root=tmp_path / "refs", policy=policy)
    config = {"configurable": {"thread_id": "ref-run"}, "recursion_limit": 200}
    state = WorkflowState(image_paths=[str(image_path)], task="找出亮块",
                          run_id="ref-run", output_root=str(tmp_path / "outputs"))
    graph.invoke(state, config=config, durability="sync")

    values = restore_state(graph.get_state(config).values)
    interrupts = _interrupts(graph, config)
    assert interrupts and interrupts[0].value["stage"] == "reference"
    assert values.pending_reference_image_id == "img_a"
    assert Path(values.pending_reference_overlay_path).is_file()
    assert store.list_scoreable() == []  # 确认前不写入

    graph.invoke(Command(resume={"action": "continue"}), config=config, durability="sync")
    assert store.list_scoreable() == ["img_a"]
    _, meta = store.load("img_a")
    assert meta["confirmed_by"] == "user"
    assert meta["sam_iou_score"] == 0.92


def test_gen_reference_reject_feedback_regenerates(tmp_path, monkeypatch):
    image_path, _ = _write_sample(tmp_path)
    ReferenceStore(tmp_path / "refs")

    fake_session = MagicMock()
    monkeypatch.setattr("core.agent_workflow.SamSession", lambda model_name: fake_session)
    select = MagicMock(return_value=[(0.92, _reference_mask())])
    monkeypatch.setattr("core.agent_workflow._select_defect_masks", select)

    provider = FakeProvider([])
    graph = build_workflow_graph(
        provider=provider, checkpointer=MemorySaver(),
        references_root=tmp_path / "refs",
        policy=IterationPolicy(patience=2, target=0.4, max_iterations=5))
    config = {"configurable": {"thread_id": "reject-run"}, "recursion_limit": 200}
    state = WorkflowState(image_paths=[str(image_path)], task="找出亮块",
                          run_id="reject-run", output_root=str(tmp_path / "outputs"))
    graph.invoke(state, config=config, durability="sync")
    assert _interrupts(graph, config)
    assert select.call_count == 1

    graph.invoke(Command(resume={"action": "continue", "feedback": "右下角漏了一个目标"}),
                 config=config, durability="sync")
    assert select.call_count == 2
    task_text = select.call_args[0][4]
    assert "右下角漏了一个目标" in task_text
    assert _interrupts(graph, config)  # 重新生成后再次等待确认


def test_gen_reference_marks_low_score_skip(tmp_path, monkeypatch):
    image_path, _ = _write_sample(tmp_path)
    store = ReferenceStore(tmp_path / "refs")

    fake_session = MagicMock()
    monkeypatch.setattr("core.agent_workflow.SamSession", lambda model_name: fake_session)
    monkeypatch.setattr("core.agent_workflow._select_defect_masks",
                        MagicMock(return_value=[(0.55, _reference_mask())]))

    provider = FakeProvider([])
    graph = build_workflow_graph(
        provider=provider, checkpointer=MemorySaver(),
        references_root=tmp_path / "refs",
        policy=IterationPolicy(patience=2, target=0.4, max_iterations=5))
    config = {"configurable": {"thread_id": "low-score-run"}, "recursion_limit": 200}
    state = WorkflowState(image_paths=[str(image_path)], task="找出亮块",
                          run_id="low-score-run", output_root=str(tmp_path / "outputs"))
    graph.invoke(state, config=config, durability="sync")
    # 唯一的样本图被标记 skip 后没有可打分的参考，确认后流程明确失败而不是空转
    import pytest
    with pytest.raises(ValueError, match="可打分"):
        graph.invoke(Command(resume={"action": "continue"}), config=config, durability="sync")

    assert store.list_scoreable() == []  # 低分参考不参与打分
    _, meta = store.load("img_a")
    assert meta["skip_scoring"] is True
