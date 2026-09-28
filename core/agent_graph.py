"""Canonical Agent orchestration for the visual pipeline developer.

LangGraph owns task state, the human confirmation interrupt and run
bookkeeping; pixel processing stays in the deterministic pipeline executor
and scoring lives in core.scoring against user-confirmed reference masks.
"""

from core.runtime_logging import logger, logged_operation, bind_context
from core.request_control import RequestCancelled

from dataclasses import asdict, replace
import functools
import inspect
import time
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

from core.memory.checkpoints import get_checkpointer
from langgraph.types import Command

from core.agent_events import (
    register_event_listener,
    unregister_event_listener,
)
from core.graph_nodes import write_trajectory
from core.task_store import TaskStore
from agent_types import normalize_strategy
from core.workflow_state import WorkflowState, restore_state


_CHECKPOINTER = None  # Optional override for integrations/tests; default is durable SQLite.

OLD_CHECKPOINT_MESSAGE = '该运行来自已停用的旧版工作流，无法继续；请重新开始任务'

# 每 100 轮迭代约 300+ 个图步进，再加人工确认轮次
WORKFLOW_STEP_LIMIT = 512


def build_agent_graph(provider=None, algorithm_registry=None, checkpointer=None):
    from core.agent_workflow import build_workflow_graph
    return build_workflow_graph(
        provider, None if checkpointer is False else checkpointer or _CHECKPOINTER or get_checkpointer(),
        algorithm_registry=algorithm_registry)


def _require_current_checkpoint(saved):
    """Pre-reference-scoring checkpoints hold dict state; never guess a migration."""
    if not saved:
        return
    root = saved.checkpoint.get('channel_values', {}).get('__root__')
    if not isinstance(root, WorkflowState):
        raise ValueError(OLD_CHECKPOINT_MESSAGE)


def _public_state(state, *, interrupted=False) -> dict:
    """dict 视图供 CLI/UI/TaskStore 消费；entry 字段名保持与旧运行记录兼容。"""
    state = restore_state(state)
    data = asdict(state)
    spec = state.best_spec or state.current_spec
    data.update({
        "graph_thread_id": state.run_id,
        "status": "ok",
        "run_status": "awaiting_feedback" if interrupted else "completed",
        "agent_status": "waiting_for_feedback" if interrupted else "accepted",
        "pipeline": {"pipeline": spec.pipeline, "notes": spec.notes} if spec else {},
        "trajectory": [],
        "conversation": ([{"role": "assistant", "content": state.human_message}]
                         if state.human_message else []),
        "strategy": normalize_strategy({}),
        "measurements": {"summary": {"count": 0, "unit": "pixel", "total_area": 0}, "results": []},
        "quality_report": {},
        "budget": None,
    })
    return data


def _serialize_task_run(function):
    @functools.wraps(function)
    def serialized(*args, **kwargs):
        from core.orchestration_runtime import task_lock
        bound = inspect.signature(function).bind(*args, **kwargs)
        values = bound.arguments
        context = values.get('memory_context') or {}
        if values.get('thread_id'):
            saved = (_CHECKPOINTER or get_checkpointer()).get_tuple({'configurable': {'thread_id': values['thread_id']}})
            if saved:
                channels = saved.checkpoint.get('channel_values', {})
                persisted = channels.get('__root__') or channels
                if isinstance(persisted, WorkflowState):
                    context = persisted.memory_context or context
                    values['task_id'] = persisted.task_id or values.get('task_id')
                else:
                    context = (persisted if isinstance(persisted, dict) else {}).get('memory_context') or context
        store = values.get('task_store') or TaskStore(
            context.get('task_root') or Path(values.get('output_root', 'outputs')).parent / 'workspace' / 'tasks')
        task_id = context.get('task_id') or values.get('task_id') or (values.get('previous_state') or {}).get('task_id')
        if not task_id:
            task_id = store.create_task()['id']
        values.update(task_store=store, task_id=task_id)
        with task_lock(store.root, task_id):
            return function(*bound.args, **bound.kwargs)
    return serialized


def _pending_interrupts(graph, config):
    snapshot = graph.get_state(config)
    tasks = snapshot.tasks or []
    return [item for task in tasks for item in (task.interrupts or [])]


@logged_operation("run_agent_graph")
@_serialize_task_run
def run_agent_graph(
    target_image_path,
    description,
    output_root="outputs",
    reference_annotation_path=None,
    reference_examples=None,
    ground_truth_mask_path=None,
    ground_truth_annotation_path=None,
    unit="pixel",
    max_auto_revisions=2,
    provider=None,
    algorithm_registry=None,
    retrieved_algorithms=None,
    previous_state=None,
    thread_id=None,
    event_callback=None,
    memory_context=None,
    task_store=None,
    task_id=None,
):
    run_started = time.monotonic()
    graph_thread_id = thread_id or f"agent_{uuid4().hex}"
    if memory_context is None:
        task_store = task_store or TaskStore(Path(output_root).parent / "workspace" / "tasks")
        algorithm_registry = algorithm_registry or task_store.algorithm_registry
    graph = build_agent_graph(provider=provider, algorithm_registry=algorithm_registry)
    config = {"configurable": {"thread_id": graph_thread_id}}
    _require_current_checkpoint(graph.checkpointer.get_tuple(config) if graph.checkpointer else None)
    snapshot = graph.get_state(config)
    restoring = bool(snapshot.values)
    if restoring:
        saved = restore_state(snapshot.values)
        if (list(saved.image_paths or []) != [str(target_image_path)]
                or saved.task != description):
            raise ValueError("Existing graph thread belongs to a different input")
        interrupts = _pending_interrupts(graph, config)
        if interrupts:
            _project_checkpoint(saved)
            return {**_public_state(saved, interrupted=True),
                    "interrupt": _serialize_interrupts(interrupts)}
        if not snapshot.next:
            _project_checkpoint(saved)
            return _public_state(saved)
        memory_context = saved.memory_context or {}
        task_id = saved.task_id or task_id
        if memory_context.get("task_root"):
            task_store = TaskStore(memory_context["task_root"])

    if memory_context is None and not restoring:
        task_store = task_store or TaskStore(Path(output_root).parent / "workspace" / "tasks")
        task_id = task_id or (previous_state or {}).get("task_id") or task_store.create_task()["id"]
        task_store.load_task(task_id)  # Validate an explicit task ID before writing memory.
        previous_state = previous_state or task_store.load_latest_state(task_id)
        previous_state, memory_context = task_store.memory_service.prepare(
            task_id, description, target_image_path, previous_state)
        task_store.append_message(task_id, "user", description)
    bind_context(task_id=memory_context.get("task_id") or task_id or "-")
    initial_state = WorkflowState(
        image_paths=[str(target_image_path)],
        task=description,
        run_id=graph_thread_id,
        output_root=str(output_root),
        task_id=task_id or "",
        memory_context=memory_context or {},
    )
    if event_callback:
        register_event_listener(event_callback)
    try:
        graph.invoke(
            None if restoring else initial_state,
            config={"configurable": {"thread_id": graph_thread_id},
                    'recursion_limit': WORKFLOW_STEP_LIMIT},
            durability='sync',
        )
    except (Exception, RequestCancelled) as exc:
        _record_run_failure(graph, config, exc)
        raise
    finally:
        if event_callback:
            unregister_event_listener(event_callback)
    result_state = graph.get_state(config).values
    interrupts = _pending_interrupts(graph, config)
    result = _public_state(result_state, interrupted=bool(interrupts))
    result.setdefault('run_id', graph_thread_id)
    result.setdefault('run_started_at', datetime.now(timezone.utc).isoformat())
    task_store.save_run_state(task_id, result)
    if interrupts:
        result["interrupt"] = _serialize_interrupts(interrupts)
        write_trajectory(result)
    result["memory_summary"] = task_store.memory_service.record_result(
        task_id, description, result.get("understanding") or {}, result, previous_state)
    task_store.save_node_result(task_id, "execute_candidate",
                                {"description": description},
                                {key: value for key, value in result.items()
                                 if key not in {"__interrupt__", "previous_state"}},
                                time.monotonic() - run_started)
    return result


def _project_checkpoint(state):
    state = restore_state(state)
    if (state.run_id and state.run_dir
            and (state.memory_context or {}).get('task_root') and state.task_id):
        TaskStore(state.memory_context['task_root']).save_run_state(state.task_id, _public_state(state))


def _record_run_failure(graph, config, error):
    """Persist an explicit stop if storage still works; never mask the original error."""
    try:
        state = restore_state(graph.get_state(config).values)
        if state.next_node == "":
            return
        status = 'cancelled' if isinstance(error, RequestCancelled) else 'failed'
        state = replace(state, next_node="", stop_reason=state.stop_reason or
                        ('cancelled' if status == 'cancelled' else 'orchestration_failed'))
        graph.update_state(config, state, as_node='finish')
        _project_checkpoint(state)
    except Exception:
        logger.error('Could not persist run failure; recover from the last durable checkpoint', exc_info=True)


@logged_operation("resume_agent_graph")
def resume_agent_graph(thread_id, response, event_callback=None):
    """Resume a paused human-gate node without re-running earlier nodes."""
    if not thread_id:
        raise ValueError("缺少 Agent thread ID，无法恢复工作流")
    graph = build_agent_graph()
    _require_current_checkpoint(graph.checkpointer.get_tuple({'configurable': {'thread_id': thread_id}}))
    snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
    if not snapshot.values:
        raise ValueError("未找到持久化工作流检查点")
    saved = restore_state(snapshot.values)
    if event_callback:
        register_event_listener(event_callback)
    try:
        from core.orchestration_runtime import task_lock
        context = saved.memory_context or {}
        task_id = saved.task_id or thread_id
        root = context.get('task_root') or Path('outputs').parent / 'workspace' / 'tasks'
        with task_lock(root, task_id):
            config = {"configurable": {"thread_id": thread_id},
                      'recursion_limit': WORKFLOW_STEP_LIMIT}
            snapshot = graph.get_state(config)
            store = TaskStore(root)
            if not snapshot.next:
                _project_checkpoint(snapshot.values)
                return _public_state(snapshot.values)
            if not any(task.interrupts for task in snapshot.tasks or []):
                raise ValueError('工作流尚未到人工确认阶段，请先恢复运行')
            graph.invoke(Command(resume=response), config=config, durability='sync')
            snapshot = graph.get_state(config)
            result = _public_state(snapshot.values,
                                   interrupted=bool(_pending_interrupts(graph, config)))
            store.save_run_state(task_id, result)
    finally:
        if event_callback:
            unregister_event_listener(event_callback)
    write_trajectory(result)
    return result


def _serialize_interrupts(interrupts):
    return [
        {
            "id": item.id,
            "value": item.value,
        }
        for item in interrupts
    ]
