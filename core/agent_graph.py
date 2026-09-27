"""Canonical Agent orchestration for the visual pipeline developer.

The graph deliberately keeps pixel processing in the deterministic pipeline
executor. LangGraph owns task state, planning boundaries and the human decision
boundary; it does not replace the CV execution layer.
"""

from core.runtime_logging import logger, logged_operation, bind_context
from core.request_control import RequestCancelled

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
from core.graph_nodes import (
    apply_human_response,
    write_trajectory,
)
from core.task_store import TaskStore
from agent_types import normalize_strategy
from providers.vision import normalize_reference_examples


_CHECKPOINTER = None  # Optional override for integrations/tests; default is durable SQLite.

OLD_CHECKPOINT_MESSAGE = '该运行来自已停用的旧版工作流，无法继续；请重新开始任务'


def build_agent_graph(provider=None, algorithm_registry=None, checkpointer=None):
    from core.agent_workflow import build_tool_agent_graph
    return build_tool_agent_graph(
        provider, None if checkpointer is False else checkpointer or _CHECKPOINTER or get_checkpointer(),
        algorithm_registry=algorithm_registry)


def _require_current_checkpoint(saved):
    """Pre-v2 checkpoints have incompatible state and node names; never guess a migration."""
    if not saved:
        return
    root = saved.checkpoint.get('channel_values', {}).get('__root__')
    # v1 used per-key channels; the controller stored version 1. A v2 input
    # checkpoint written before initialize_run has no version yet.
    if not isinstance(root, dict) or root.get('orchestration_version', 2) != 2:
        raise ValueError(OLD_CHECKPOINT_MESSAGE)


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
                context = persisted.get('memory_context') or context
                values['task_id'] = persisted.get('task_id') or values.get('task_id')
        store = values.get('task_store') or TaskStore(
            context.get('task_root') or Path(values.get('output_root', 'outputs')).parent / 'workspace' / 'tasks')
        task_id = context.get('task_id') or values.get('task_id') or (values.get('previous_state') or {}).get('task_id')
        if not task_id:
            task_id = store.create_task()['id']
        values.update(task_store=store, task_id=task_id)
        with task_lock(store.root, task_id):
            return function(*bound.args, **bound.kwargs)
    return serialized


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
    if snapshot.values:
        from core.memory.service import image_hash
        if (snapshot.values.get("target_image_path") != str(target_image_path)
                or snapshot.values.get("input_sha256") != image_hash(target_image_path)
                or snapshot.values.get("description") != description):
            raise ValueError("Existing graph thread belongs to a different input")
        if snapshot.tasks and any(task.interrupts for task in snapshot.tasks):
            _project_checkpoint(snapshot.values)
            return {**snapshot.values, "interrupt": _serialize_interrupts(
                [item for task in snapshot.tasks for item in task.interrupts])}
        # Pending node writes can outlive a failed checkpoint commit. LangGraph
        # reports no next task until invoke(None) applies those durable writes.
        if not snapshot.next and snapshot.values.get('run_status') != 'running':
            _project_checkpoint(snapshot.values)
            return dict(snapshot.values)
        memory_context = snapshot.values.get("memory_context") or {}
        task_id = snapshot.values.get("task_id")
        if memory_context.get("task_root"):
            task_store = TaskStore(memory_context["task_root"])
    restoring = bool(snapshot.values)

    owns_memory = True
    if memory_context is None and not restoring:
        task_store = task_store or TaskStore(Path(output_root).parent / "workspace" / "tasks")
        task_id = task_id or (previous_state or {}).get("task_id") or task_store.create_task()["id"]
        task_store.load_task(task_id)  # Validate an explicit task ID before writing memory.
        previous_state = previous_state or task_store.load_latest_state(task_id)
        previous_state, memory_context = task_store.memory_service.prepare(
            task_id, description, target_image_path, previous_state)
        task_store.append_message(task_id, "user", description)
    bind_context(task_id=memory_context.get("task_id") or task_id or "-")
    initial_state = {
        'run_id': graph_thread_id,
        'run_started_at': datetime.now(timezone.utc).isoformat(),
        'state_version': 0,
        "memory_context": memory_context,
        "task_id": memory_context.get("task_id"),
        "input_sha256": memory_context.get("input_sha256"),
        "coordinate_version": "stored-pixels-v1",
        "target_image_path": str(target_image_path),
        "description": description,
        "reference_examples": normalize_reference_examples(
            [*(reference_examples or []), *([reference_annotation_path] if reference_annotation_path else [])]
        ),
        "ground_truth_mask_path": str(ground_truth_mask_path) if ground_truth_mask_path else None,
        "ground_truth_annotation_path": str(ground_truth_annotation_path) if ground_truth_annotation_path else None,
        "output_root": str(output_root),
        "unit": unit,
        "max_auto_revisions": max(0, int(max_auto_revisions)),
        "revision_count": 0,
        "experiment_history": [],
        "experiment_records": (previous_state or {}).get("experiment_records", []),
        "verified_baseline": (previous_state or {}).get("verified_baseline"),
        "previous_state": previous_state,
        "original_task_goal": (previous_state or {}).get("original_task_goal") or description,
        "task_contract": (previous_state or {}).get("task_contract") or {},
        "retrieved_algorithms": retrieved_algorithms,
        "trajectory": [],
        "status": "pending",
        "graph_thread_id": graph_thread_id,
    }
    if event_callback:
        register_event_listener(event_callback)
    try:
        result = graph.invoke(
            None if restoring else initial_state,
            config={"configurable": {"thread_id": graph_thread_id}, 'recursion_limit': _step_limit(snapshot.values)},
            durability='sync',
        )
    except (Exception, RequestCancelled) as exc:
        _record_run_failure(graph, config, exc)
        raise
    finally:
        if event_callback:
            unregister_event_listener(event_callback)
    if result.get("__interrupt__"):
        result["interrupt"] = _serialize_interrupts(result["__interrupt__"])
    if owns_memory:
        result.setdefault('run_id', graph_thread_id)
        result.setdefault('run_started_at', initial_state['run_started_at'])
        result.setdefault('run_status', 'awaiting_feedback' if result.get('interrupt') else
                          'completed' if result.get('agent_status') == 'accepted' else 'stopped')
        # Publish the durable graph outcome before any derived record can fail.
        task_store.save_run_state(task_id, result)
    if result.get("__interrupt__"):
        write_trajectory(result)
    if owns_memory:
        result["memory_summary"] = task_store.memory_service.record_result(
            task_id, description, result.get("understanding") or {}, result, previous_state)
        task_store.save_node_result(task_id, "execute_candidate",
                                    {"description": description},
                                    {key: value for key, value in result.items() if key not in {"__interrupt__", "previous_state"}},
                                    time.monotonic() - run_started)
    return result


def _project_checkpoint(state):
    context = state.get('memory_context') or {}
    if state.get('run_id') and state.get('run_status') and context.get('task_root') and state.get('task_id'):
        TaskStore(context['task_root']).save_run_state(state['task_id'], state)


def _record_run_failure(graph, config, error):
    """Persist an explicit stop if storage still works; never mask the original error."""
    try:
        state = dict(graph.get_state(config).values)
        if not state or state.get('run_status') in {'completed', 'cancelled', 'failed'}:
            return
        status = 'cancelled' if isinstance(error, RequestCancelled) else 'failed'
        state.update(run_status=status, agent_status=status, status=status, phase='finished',
                     pending_action=None, stop_reason='cancelled' if status == 'cancelled' else 'orchestration_failed',
                     state_version=int(state.get('state_version', 0)) + 1,
                     decision={'next_action': 'stop', 'reason': str(error), 'automatic_review_passed': False})
        state.pop('action_outcome', None)
        state.setdefault('strategy', normalize_strategy({}))
        state.setdefault('measurements', {'summary': {'count': 0, 'unit': state.get('unit', 'pixel'), 'total_area': 0}, 'results': []})
        state.setdefault('quality_report', {})
        state['conversation'] = [*state.get('conversation', []), {'role': 'assistant', 'content': str(error)}]
        graph.update_state(config, state, as_node='finish')
        _project_checkpoint(state)
    except Exception:
        logger.error('Could not persist run failure; recover from the last durable checkpoint', exc_info=True)


def _step_limit(state):
    from core.orchestration_runtime import RunLimits
    limits = (state.get('budget') or {}).get('limits') or RunLimits.from_env().as_dict()
    return max(64, 8 * (limits['max_model_calls'] + limits['max_executions']) + 32)


@logged_operation("resume_agent_graph")
def resume_agent_graph(thread_id, response, event_callback=None):
    """Resume a paused human-review node without re-running earlier nodes."""
    if not thread_id:
        raise ValueError("缺少 Agent thread ID，无法恢复工作流")
    graph = build_agent_graph()
    _require_current_checkpoint(graph.checkpointer.get_tuple({'configurable': {'thread_id': thread_id}}))
    snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
    if not snapshot.values:
        raise ValueError("未找到持久化工作流检查点")
    if event_callback:
        register_event_listener(event_callback)
    try:
        from core.orchestration_runtime import task_lock
        context = snapshot.values.get('memory_context') or {}
        task_id = snapshot.values.get('task_id') or thread_id
        root = context.get('task_root') or Path(snapshot.values.get('output_root', 'outputs')).parent / 'workspace' / 'tasks'
        with task_lock(root, task_id):
            config = {"configurable": {"thread_id": thread_id}, 'recursion_limit': _step_limit(snapshot.values)}
            snapshot = graph.get_state(config)
            store = TaskStore(root)
            task = store.load_task(task_id)
            if task.get('latest_run_id') not in {None, snapshot.values.get('run_id') or snapshot.values.get('graph_thread_id')}:
                raise ValueError('此结果已被更新的运行替代，请打开最新结果')
            if snapshot.values.get('run_status') == 'running':
                raise ValueError('工作流尚未到人工确认阶段，请先恢复运行')
            if not snapshot.next:
                if snapshot.values.get('agent_status') in {'accepted', 'exited'}:
                    _project_checkpoint(snapshot.values)
                    return dict(snapshot.values)
                result = apply_human_response(snapshot.values, response)
                graph.update_state(config, result, as_node='finish')
            else:
                result = graph.invoke(Command(resume=response), config=config, durability='sync')
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
