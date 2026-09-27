"""Canonical Agent orchestration for the visual pipeline developer.

The graph deliberately keeps pixel processing in the deterministic pipeline
executor. LangGraph owns task state, planning boundaries and the human decision
boundary; it does not replace the CV execution layer.
"""

from core.task_contract import establish_contract, apply_contract
from core.runtime_logging import logger, logged_operation, bind_context
from core.request_control import RequestCancelled

import functools
import inspect
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, TypedDict
from uuid import uuid4

from core.memory.checkpoints import get_checkpointer
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from core.agent_events import (
    emit_node_complete,
    emit_node_start,
    register_event_listener,
    unregister_event_listener,
)
from core.agent_loop import (
    pipeline_fingerprint,
    plan_candidate_definitions,
    promote_candidate_result,
    run_planned_agent,
)
from core.graph_nodes import (
    apply_human_response,
    make_review_candidates_node,
    prepare_inputs,
    wait_for_human,
    with_event,
    write_trajectory,
)
from core.experiments.lifecycle import compatible_baseline, revision_evidence
from core.task_store import TaskStore
from agent_types import normalize_strategy
from providers.vision import normalize_acceptance_criteria, normalize_reference_examples


_CHECKPOINTER = None  # Optional override for integrations/tests; default is durable SQLite.

class AgentGraphState(TypedDict, total=False):
    run_id: str
    run_status: str
    run_started_at: str
    state_version: int
    stop_reason: str
    memory_context: dict
    task_id: str
    coordinate_version: str
    contours_path: str
    input_sha256: str
    target_image_path: str
    description: str
    original_task_goal: str
    reference_examples: list[dict]
    ground_truth_mask_path: Optional[str]
    ground_truth_annotation_path: Optional[str]
    output_root: str
    unit: str
    max_candidates: int
    max_calibration_candidates: int
    max_auto_revisions: int
    revision_count: int
    experiment_history: list[dict]
    experiment_records: list[dict]
    verified_baseline: Optional[dict]
    previous_state: Optional[dict]
    understanding: dict
    acceptance_criteria: dict
    task_contract: dict
    retrieved_algorithms: list[dict]
    planned_candidates: list[dict]
    trajectory: list[dict]
    decision: dict
    review: dict
    agent_status: str
    status: str
    run_dir: str
    iteration: int
    parent_iteration: Optional[int]
    parent_result_image_path: Optional[str]
    strategy: dict
    retained_experiment_id: str
    selected_experiment_id: str
    selected_candidate: str
    rendering: dict
    quality_report: dict
    evaluation_report: Optional[dict]
    candidate_attempts: list[dict]
    pipeline: dict
    pipeline_diff: dict
    feedback: dict
    measurements: dict
    annotated_image_path: str
    predicted_mask_path: str
    conversation: list[dict]
    segmentation: dict
    errors: list[str]
    graph_thread_id: str
    human_response: dict
    human_feedback: dict
    reference_masks: list[dict]
    interrupt: list[dict]


def build_agent_graph(
    provider=None,
    algorithm_registry=None,
    max_candidates=1,
    checkpointer=None,
    controller=None,
    orchestration_version=None,
):
    if orchestration_version == 2 or (controller is not False and orchestration_version is None
                                     and callable(getattr(provider, 'agent_action', None))):
        from core.agent_workflow import build_tool_agent_graph
        return build_tool_agent_graph(provider,
            None if checkpointer is False else checkpointer or _CHECKPOINTER or get_checkpointer(),
            algorithm_registry=algorithm_registry)
    if controller is True or (controller is None and callable(getattr(provider, 'propose_action', None))):
        from core.orchestration import build_controller_graph
        return build_controller_graph(provider, None if checkpointer is False else checkpointer or _CHECKPOINTER or get_checkpointer(),
                                      algorithm_registry=algorithm_registry)
    workflow = StateGraph(AgentGraphState)
    workflow.add_node("prepare_inputs", prepare_inputs)
    workflow.add_node("understand_task", _make_understand_task_node(provider))
    workflow.add_node("retrieve_algorithms", _make_retrieve_algorithms_node(algorithm_registry))
    workflow.add_node("plan_candidates", _make_plan_candidates_node(max_candidates))
    workflow.add_node("execute_candidates", _execute_candidates)
    workflow.add_node("review_candidates", make_review_candidates_node(provider))
    workflow.add_node("revise_candidates", _make_revise_candidates_node(provider, max_candidates))
    workflow.add_node("report_failure", _report_failure)
    workflow.add_node("decide_next_action", _decide_next_action)
    workflow.add_node("wait_for_human", wait_for_human)

    workflow.set_entry_point("prepare_inputs")
    workflow.add_edge("prepare_inputs", "understand_task")
    workflow.add_edge("understand_task", "retrieve_algorithms")
    workflow.add_edge("retrieve_algorithms", "plan_candidates")
    workflow.add_edge("plan_candidates", "execute_candidates")
    workflow.add_edge("execute_candidates", "review_candidates")
    workflow.add_conditional_edges(
        "review_candidates",
        _route_after_review,
        {"revise": "revise_candidates", "present": "decide_next_action", "fail": "report_failure"},
    )
    workflow.add_conditional_edges(
        "revise_candidates",
        _route_after_revision,
        {"execute": "execute_candidates", "fail": "report_failure"},
    )
    # A bounded automatic retry failure is still a human-review outcome.  It
    # must create the same interrupt as an ordinary rendered candidate so the
    # UI can accept, continue, or exit consistently.
    workflow.add_edge("report_failure", "wait_for_human")
    workflow.add_edge("decide_next_action", "wait_for_human")
    workflow.add_edge("wait_for_human", END)
    if checkpointer is False:
        return workflow.compile()
    return workflow.compile(checkpointer=checkpointer or _CHECKPOINTER or get_checkpointer())


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
    max_candidates=1,
    max_auto_revisions=2,
    provider=None,
    algorithm_registry=None,
    understanding=None,
    retrieved_algorithms=None,
    previous_state=None,
    thread_id=None,
    event_callback=None,
    max_calibration_candidates=0,
    memory_context=None,
    task_store=None,
    task_id=None,
):
    run_started = time.monotonic()
    graph_thread_id = thread_id or f"agent_{uuid4().hex}"
    if memory_context is None:
        task_store = task_store or TaskStore(Path(output_root).parent / "workspace" / "tasks")
        algorithm_registry = algorithm_registry or task_store.algorithm_registry
    graph = build_agent_graph(
        provider=provider,
        algorithm_registry=algorithm_registry,
        max_candidates=max_candidates,
        controller=False if isinstance(understanding, dict) else None,
    )
    config = {"configurable": {"thread_id": graph_thread_id}}
    saved = graph.checkpointer.get_tuple(config) if graph.checkpointer else None
    if saved:
        use_controller = '__root__' in saved.checkpoint.get('channel_values', {})
        saved_version = (saved.checkpoint.get('channel_values', {}).get('__root__') or {}).get('orchestration_version', 1)
        graph = build_agent_graph(provider=provider, algorithm_registry=algorithm_registry,
                                  max_candidates=max_candidates, controller=use_controller,
                                  orchestration_version=saved_version if use_controller else 0)
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
    initial_state: AgentGraphState = {
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
        "max_candidates": max_candidates,
        "max_calibration_candidates": max(0, int(max_calibration_candidates)),
        "max_auto_revisions": max(0, int(max_auto_revisions)),
        "revision_count": 0,
        "experiment_history": [],
        "experiment_records": (previous_state or {}).get("experiment_records", []),
        "verified_baseline": (previous_state or {}).get("verified_baseline"),
        "previous_state": previous_state,
        "original_task_goal": (previous_state or {}).get("original_task_goal") or description,
        "task_contract": (previous_state or {}).get("task_contract") or {},
        "understanding": understanding,
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
            config={"configurable": {"thread_id": graph_thread_id}, 'recursion_limit': _controller_step_limit(snapshot.values)},
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
        if not result.get('orchestration_version'):
            result['state_version'] = int(result.get('state_version', 0)) + 1
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
        graph.update_state(config, state, as_node=_terminal_node(state))
        _project_checkpoint(state)
    except Exception:
        logger.error('Could not persist run failure; recover from the last durable checkpoint', exc_info=True)


def _controller_step_limit(state):
    from core.orchestration_runtime import RunLimits
    limits = (state.get('budget') or {}).get('limits') or RunLimits.from_env().as_dict()
    return max(64, 8 * (limits['max_model_calls'] + limits['max_executions']) + 32)


def _terminal_node(state):
    return ('finish' if state.get('orchestration_version') == 2 else
            'controller' if state.get('orchestration_version') else 'wait_for_human')


@logged_operation("resume_agent_graph")
def resume_agent_graph(thread_id, response, event_callback=None):
    """Resume a paused human-review node without re-running earlier nodes."""
    if not thread_id:
        raise ValueError("缺少 Agent thread ID，无法恢复工作流")
    graph = build_agent_graph()
    saved = graph.checkpointer.get_tuple({'configurable': {'thread_id': thread_id}})
    if saved and '__root__' in saved.checkpoint.get('channel_values', {}):
        version = saved.checkpoint['channel_values']['__root__'].get('orchestration_version', 1)
        graph = build_agent_graph(controller=True, orchestration_version=version)
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
            config = {"configurable": {"thread_id": thread_id}, 'recursion_limit': _controller_step_limit(snapshot.values)}
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
                graph.update_state(config, result, as_node=_terminal_node(result))
            else:
                result = graph.invoke(Command(resume=response), config=config, durability='sync')
            if not result.get('orchestration_version'):
                result['state_version'] = int(result.get('state_version', 0)) + 1
            store.save_run_state(task_id, result)
    finally:
        if event_callback:
            unregister_event_listener(event_callback)
    write_trajectory(result)
    return result


def _make_understand_task_node(provider):
    def understand_task(state):
        emit_node_start("understand_task", "分析图像和任务描述，理解检测目标")
        started = time.monotonic()
        understanding = state.get("understanding")
        if not isinstance(understanding, dict):
            if provider is None:
                raise ValueError("缺少任务理解结果或 Vision Provider")
            try:
                understanding = _call_understand_task(
                    provider,
                    state["target_image_path"],
                    state["description"],
                    previous_context={
                        **(state.get("memory_context") or {}),
                        **{key: (state.get("previous_state") or {}).get(key) for key in (
                            "false_positive_mask_path", "false_negative_mask_path", "feedback",
                            "include_mask_path", "exclude_mask_path", "selected_experiment_id")},
                        "experiment_output_root": state.get("output_root", "outputs"),
                        "original_task_goal": (state.get("previous_state") or {}).get("original_task_goal")
                        or state.get("original_task_goal")
                        or state.get("description"),
                        "task_contract": state.get("task_contract"),
                        "previous_pipeline": (state.get("previous_state") or {}).get("pipeline"),
                        "previous_quality": (state.get("previous_state") or {}).get("quality_report"),
                        "previous_result_image_path": (state.get("previous_state") or {}).get("annotated_image_path"),
                        "review": (state.get("previous_state") or {}).get("review"),
                        "execution_feedback": {"attempts": revision_evidence({**state, **{
                            key: (state.get("previous_state") or {}).get(key) for key in (
                                "human_feedback", "include_mask_path", "exclude_mask_path")
                        }})},
                        "reference_examples": state.get("reference_examples", []),
                        "reference_masks": state.get("reference_masks", []),
                        "ground_truth_mask_path": state.get("ground_truth_mask_path"),
                        "ground_truth_annotation_path": state.get("ground_truth_annotation_path"),
                        "human_feedback": (state.get("previous_state") or {}).get("human_feedback", {}),
                    },
                    reference_examples=state.get("reference_examples", []),
                )
            except Exception as exc:
                logger.warning("Graph operation recovered from exception", exc_info=True)
                # Provider/schema failures are recoverable. Preserve the error
                # as context while letting the deterministic planner continue.
                description = state["description"]
                periodic = any(token in description.lower() for token in ("periodic", "周期", "条纹", "重复"))
                understanding = {
                    "task_summary": description,
                    "target_defect": "用户描述的视觉异常",
                    "normal_context": "周期背景" if periodic else "未知背景",
                    "ambiguities": [f"provider_invalid_json_or_call: {type(exc).__name__}: {exc}"],
                    "questions": [],
                    "candidate_pipelines": [],
                    "recommended_strategy": normalize_strategy({
                        "visual_observation": {
                            "background_pattern": "periodic_lines" if periodic else "unknown",
                        },
                    }),
                    "target_constraints": {},
                    "rendering": {},
                }
        memory_context = dict(state.get("memory_context") or {})
        if memory_context.get("task_id") and memory_context.get("task_root"):
            service = TaskStore(memory_context["task_root"]).memory_service
            memory_context["task_memory"] = service.apply_updates(
                memory_context["task_id"], state["description"], understanding,
                memory_context["memory_source_id"])
            service.context_manifest(memory_context["task_id"], memory_context)
        criteria = normalize_acceptance_criteria(
            understanding.get("acceptance_criteria"),
            task_summary=understanding.get("task_summary") or state.get("description"),
            output_requirements=understanding.get("output_requirements"),
        )
        understanding = {**understanding, "acceptance_criteria": criteria}
        contract = establish_contract(state, understanding)
        understanding = apply_contract(understanding, contract)
        criteria = contract['acceptance_criteria']
        duration = time.monotonic() - started
        emit_node_complete(
            "understand_task",
            duration,
            {
                "provider": type(provider).__name__ if provider else "precomputed",
                "strategy": understanding.get("recommended_strategy", {}).get("name"),
            }
        )
        return with_event(
            {**state, "task_contract": contract, "memory_context": memory_context, "understanding": understanding, "acceptance_criteria": criteria},
            "understand_task",
            started,
            {"provider": type(provider).__name__ if provider else "precomputed"},
        )

    return understand_task


def _make_retrieve_algorithms_node(algorithm_registry):
    def retrieve_algorithms(state):
        emit_node_start("retrieve_algorithms", "检索历史成功的算法")
        started = time.monotonic()
        matches = state.get("retrieved_algorithms")
        if not isinstance(matches, list):
            matches = []
            if algorithm_registry is not None and not (state.get("previous_state") or {}).get("pipeline"):
                matches = algorithm_registry.search(state["understanding"], limit=2, min_score=0.2)
        summary = [
            {
                "algorithm_id": item.get("algorithm_id"),
                "name": item.get("name"),
                "score": item.get("score"),
                "match_reasons": item.get("match_reasons", []),
            }
            for item in matches
            if isinstance(item, dict)
        ]
        duration = time.monotonic() - started
        emit_node_complete("retrieve_algorithms", duration, {"match_count": len(summary)})
        return with_event(
            {**state, "retrieved_algorithms": matches,
             "memory_context": {**(state.get("memory_context") or {}), "procedural_memory": summary}},
            "retrieve_algorithms",
            started,
            {"match_count": len(summary), "matches": summary},
        )

    return retrieve_algorithms


def _make_plan_candidates_node(max_candidates):
    def plan_candidates(state):
        emit_node_start("plan_candidates", "准备下一次实验")
        started = time.monotonic()
        candidates = plan_candidate_definitions(
            state["understanding"],
            previous_state=state.get("previous_state"),
            retrieved_algorithms=state.get("retrieved_algorithms"),
            max_candidates=state.get("max_candidates", max_candidates),
        )
        summary = [
            {
                "name": item.get("name"),
                "source": item.get("source", {"type": "qwen"}),
            }
            for item in candidates
        ]
        duration = time.monotonic() - started
        emit_node_complete("plan_candidates", duration, {"candidate_count": len(candidates)})
        return with_event(
            {**state, "planned_candidates": candidates},
            "plan_candidates",
            started,
            {"generated_pipeline": summary[0] if summary else None},
        )

    return plan_candidates


def _execute_candidates(state):
    emit_node_start("execute_candidates", "执行实验并保存中间产物")
    started = time.monotonic()
    result = run_planned_agent(
        target_image_path=state["target_image_path"],
        description=state["description"],
        understanding=state["understanding"],
        output_root=state["output_root"],
        unit=state.get("unit", "pixel"),
        max_candidates=state.get("max_candidates", 1),
        previous_state=state.get("previous_state"),
        retrieved_algorithms=state.get("retrieved_algorithms"),
        planned_candidates=state.get("planned_candidates"),
        run_dir=state.get("run_dir"),
        return_failure_state=True,
        ground_truth_mask_path=state.get("ground_truth_mask_path"),
        max_calibration_candidates=state.get("max_calibration_candidates", 0),
        task_contract=state.get("task_contract"),
    )
    experiment = {
        "iteration": result.get("iteration"),
        "selected_candidate": result.get("selected_candidate"),
        "quality_report": result.get("quality_report", {}),
        "pipeline": result.get("pipeline", {}),
        "candidate_attempts": result.get("candidate_attempts", []),
        "pipeline_diff": result.get("pipeline_diff", ),
    }
    merged = {
        **state,
        **result,
        "experiment_history": [*(state.get("experiment_history") or []), experiment],
    }
    duration = time.monotonic() - started
    emit_node_complete("execute_candidates", duration, {
        "attempt_count": len(result.get("candidate_attempts", [])),
        "selected_candidate": result.get("selected_candidate"),
    })
    return with_event(
        merged,
        "execute_candidates",
        started,
        {
            "attempt_count": len(result.get("candidate_attempts", [])),
            "selected_candidate": result.get("selected_candidate"),
            "quality": result.get("quality_report", {}),
        },
    )


def _decide_next_action(state):
    emit_node_start("decide_next_action", "准备人工验收")
    started = time.monotonic()
    review = state.get("review") if isinstance(state.get("review"), dict) else {}
    decision = {
        "next_action": "wait_for_acceptance",
        "reason": review.get("reason") or "标注结果已生成，等待用户确认标注是否准确。",
    }
    result = {
        **state,
        "decision": decision,
        "agent_status": "waiting_for_acceptance",
    }
    result = with_event(result, "decide_next_action", started, decision)
    emit_node_complete("decide_next_action", time.monotonic() - started, decision)
    write_trajectory(result)
    return result


def _route_after_review(state):
    review = state.get("review") if isinstance(state.get("review"), dict) else {}
    revision_count = int(state.get("revision_count", 0))
    max_revisions = int(state.get("max_auto_revisions", 1))
    if review.get("decision") == "review_unavailable":
        return "present"
    if review.get("decision") == "revise":
        return "revise" if revision_count < max_revisions else "fail"
    if review.get("decision") == "failed":
        return "fail"
    return "present"


def _route_after_revision(state):
    if state.get("planned_candidates"):
        return "execute"
    return "fail"


def _make_revise_candidates_node(provider, max_candidates):
    def revise_candidates(state):
        emit_node_start("revise_candidates", "根据证据修改下一版算法")
        started = time.monotonic()
        if provider is None or not hasattr(provider, "understand_task"):
            emit_node_complete("revise_candidates", time.monotonic() - started, {
                "skipped": True,
                "reason": "缺少 Vision Provider",
            })
            return {**state, 'planned_candidates': []}
        previous_state = {
            "selected_experiment_id": state.get("selected_experiment_id"),
            "iteration": state.get("iteration", 0),
            "original_task_goal": state.get("original_task_goal")
            or (state.get("previous_state") or {}).get("original_task_goal")
            or state.get("description"),
            "annotated_image_path": state.get("annotated_image_path"),
            "predicted_mask_path": state.get("predicted_mask_path"),
            "pipeline": state.get("pipeline", {}),
            "task_contract": state.get("task_contract"),
            "quality_report": state.get("quality_report", {}),
            "evaluation_report": state.get("evaluation_report"),
            "ground_truth_mask_path": state.get("ground_truth_mask_path"),
            "ground_truth_annotation_path": state.get("ground_truth_annotation_path"),
            "feedback_image_path": state.get("feedback", {}).get("feedback_image_path"),
            "false_positive_mask_path": state.get("feedback", {}).get("false_positive_mask_path"),
            "false_negative_mask_path": state.get("feedback", {}).get("false_negative_mask_path"),
            "human_feedback": dict(state.get("human_feedback") or {}),
            "include_mask_path": state.get("include_mask_path")
            or (state.get("human_feedback") or {}).get("include_mask_path"),
            "exclude_mask_path": state.get("exclude_mask_path")
            or (state.get("human_feedback") or {}).get("exclude_mask_path"),
        }
        has_completed_result = any(
            item.get("status") == "selected_for_review"
            for item in state.get("candidate_attempts", [])
        )
        execution_feedback = {
            "status": "needs_visual_revision" if has_completed_result else "no_usable_annotation",
            "instruction": (
                "先检查复查指出的问题和相关中间产物，定位原因，再对当前方案作有依据的修改；说明修改和预期变化。"
                if has_completed_result
                else "本次实验未生成可用结果。先分析执行错误和逐步记录，修复具体问题；只有证据表明方法不适用时才换方法。"
            ),
            "attempts": revision_evidence(state),
            "review": state.get("review", {}),
        }
        if int(state.get('revision_count', 0)) >= 1:
            execution_feedback['instruction'] += " 连续修改后仍未通过，请重新检查错误假设；必要时提出另一种有依据的方法并按需比较。"
        try:
            understanding = _call_understand_task(
                provider,
                state["target_image_path"],
                state["description"],
                previous_context={
                    **previous_state,
                    **(state.get("memory_context") or {}),
                    "experiment_output_root": state.get("output_root", "outputs"),
                    "task_contract": state.get("task_contract"),
                    "previous_pipeline": state.get("pipeline", {}),
                    "previous_quality": state.get("quality_report", {}),
                    "previous_evaluation": state.get("evaluation_report"),
                    "previous_result_image_path": state.get("annotated_image_path"),
                    "review": state.get("review", {}),
                    "execution_feedback": execution_feedback,
                    "original_task_goal": previous_state.get("original_task_goal"),
                    "reference_examples": state.get("reference_examples", []),
                    "reference_masks": state.get("reference_masks", []),
                    "ground_truth_mask_path": state.get("ground_truth_mask_path"),
                    "ground_truth_annotation_path": state.get("ground_truth_annotation_path"),
                },
                reference_examples=state.get("reference_examples", []),
            )
        except Exception as exc:
            logger.warning("Graph operation recovered from exception", exc_info=True)
            understanding = _fallback_understanding(state["description"], exc)
        # Automatic revision may change algorithms only, including after a failed model call.
        contract = state.get("task_contract") or establish_contract(state, state.get("understanding") or {})
        understanding = apply_contract(understanding, contract)
        criteria = contract["acceptance_criteria"]
        candidates = plan_candidate_definitions(
            understanding,
            previous_state=previous_state,
            retrieved_algorithms=[],
            max_candidates=state.get("max_candidates", max_candidates),
        )
        failed_fingerprints = {
            pipeline_fingerprint(attempt.get("pipeline"))
            for experiment in state.get("experiment_history", [])
            for attempt in experiment.get("candidate_attempts", [])
            if (
                isinstance(attempt, dict)
                and attempt.get("pipeline")
                and (attempt.get("acceptance_status") == "rejected" or attempt.get("status") in {
                    "failed", "no_annotation", "health_failed", "duplicate_pipeline",
                })
            )
        }
        candidates = [
            candidate for candidate in candidates
            if pipeline_fingerprint(candidate.get("pipeline")) not in failed_fingerprints
        ]
        result = {
            **state,
            "understanding": understanding,
            "acceptance_criteria": contract.get("acceptance_criteria", criteria),
            "task_contract": contract,
            "planned_candidates": candidates,
            "previous_state": previous_state,
            "retrieved_algorithms": [],
            "revision_count": int(state.get("revision_count", 0)) + 1,
        }
        emit_node_complete("revise_candidates", time.monotonic() - started, {
            "revision_count": result["revision_count"],
            "candidate_count": len(candidates),
        })
        return with_event(
            result,
            "revise_candidates",
            started,
            {
                "revision_count": result["revision_count"],
                "reason": (state.get("review") or {}).get("reason"),
                "candidate_count": len(candidates),
            },
        )

    return revise_candidates


def _report_failure(state):
    """Hand the last rendered result to the user after automatic retries.

    A visual review can reject an otherwise renderable candidate. That is not
    an execution error: the user must still be able to inspect and decide on
    the final candidate instead of losing its image in an exception.
    """
    emit_node_start("report_failure", "自动复查未通过，交由用户判断")
    started = time.monotonic()
    attempts = sum(len(item.get('candidate_attempts', [])) for item in state.get('experiment_history', []))
    baseline = compatible_baseline(state)
    if baseline:
        latest_attempts = state.get('candidate_attempts', [])
        state = promote_candidate_result({**state, 'candidate_attempts': [baseline]}, baseline['name'])
        state.update(candidate_attempts=latest_attempts, retained_experiment_id=baseline['experiment_id'])
    # A failed revision must not erase a previously renderable candidate.
    # Without Ground Truth this is a fallback, not a claim of higher accuracy.
    if not baseline and not any(item.get("status") == "selected_for_review" for item in state.get("candidate_attempts", [])):
        for experiment in reversed(state.get("experiment_history", [])):
            usable = [item for item in experiment.get("candidate_attempts", [])
                      if item.get("status") == "selected_for_review"
                      and (Path(item.get("directory", "")) / "result_annotation.png").exists()]
            if not usable:
                continue
            selected = next((item for item in usable if item.get("name") == experiment.get("selected_candidate")), usable[0])
            promoted = promote_candidate_result({**state, "candidate_attempts": [selected]}, selected["name"])
            state = {**promoted, "candidate_attempts": state.get("candidate_attempts", []),
                     "retained_experiment_id": selected.get("experiment_id")}
            break
    review = state.get("review") if isinstance(state.get("review"), dict) else {}
    had_result = any(
        item.get("status") == "selected_for_review"
        for item in state.get("candidate_attempts", [])
        if isinstance(item, dict)
    )
    reason = "结果没有满足当前任务的验收条件" if had_result else "没有生成可用结果"
    message = (
        f"自动复查没有通过：{reason}。本次自动迭代已结束。"
        + ("已回退到相同输入和验收条件下通过验证的版本。" if baseline else "展示保留的实验结果，仍需人工验收。")
    )
    decision = {
        "next_action": "wait_for_acceptance",
        "reason": message,
        "automatic_review_passed": False,
    }
    result = with_event(
        {
            **state,
            "status": "needs_human_review",
            "agent_status": "waiting_for_feedback",
            "decision": decision,
            "review": {**review, "reason": review.get("reason") or message},
        },
        "report_failure",
        started,
        {"attempt_count": attempts, "message": message, "had_result": had_result},
    )
    emit_node_complete("report_failure", time.monotonic() - started, {
        "attempt_count": attempts,
        "message": message,
    })
    write_trajectory(result)
    return result


def _call_understand_task(provider, target_image_path, description, previous_context, reference_examples):
    try:
        return provider.understand_task(
            target_image_path,
            description,
            previous_context=previous_context,
            reference_examples=reference_examples,
        )
    except TypeError as exc:
        if "reference_examples" not in str(exc):
            raise
        return provider.understand_task(
            target_image_path,
            description,
            previous_context=previous_context,
        )


def _fallback_understanding(description, error):
    periodic = any(token in str(description).lower() for token in ("periodic", "周期", "条纹", "重复"))
    return {
        "task_summary": str(description),
        "target_defect": "用户描述的视觉异常",
        "normal_context": "周期背景" if periodic else "未知背景",
        "ambiguities": [f"provider_invalid_json_or_call: {type(error).__name__}: {error}"],
        "questions": [],
        "candidate_pipelines": [],
        "recommended_strategy": normalize_strategy({
            "visual_observation": {
                "background_pattern": "periodic_lines" if periodic else "unknown",
            },
        }),
        "target_constraints": {},
        "rendering": {},
    }


def _serialize_interrupts(interrupts):
    return [
        {
            "id": item.id,
            "value": item.value,
        }
        for item in interrupts
    ]
