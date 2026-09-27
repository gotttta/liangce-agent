import asyncio
from copy import deepcopy
import json
import threading
import time

from PIL import Image
import pytest

from core.model_context import ContextBudgetError
from core.input_contract import input_metadata
from core.request_control import RequestCancelled, RequestControl, control
from providers.vision import AliyunVisionProvider, build_action_messages, normalize_model_action


@pytest.fixture
def target(tmp_path):
    path = tmp_path / "target.png"
    Image.new("RGB", (8, 8), "white").save(path)
    return path


def install_client(monkeypatch, response, *, error=None, stall=None, finish_reason="stop"):
    state = {"requests": [], "clients": [], "client_closed": False, "cancelled": False}

    class Client:
        def __init__(self, **kwargs):
            state["clients"].append(kwargs)
            self.chat = self
            self.completions = self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            state["client_closed"] = True

        async def create(self, **kwargs):
            state["requests"].append(kwargs)
            if error:
                raise error
            if stall == "create":
                try:
                    await asyncio.sleep(5)
                except asyncio.CancelledError:
                    state["cancelled"] = True
                    raise
            text = response if isinstance(response, str) else json.dumps(response)
            return {"choices": [{"message": {"content": text}, "finish_reason": finish_reason}],
                    "usage": {"prompt_tokens": 128, "completion_tokens": 64, "total_tokens": 192}}

    monkeypatch.setattr("openai.AsyncOpenAI", Client)
    return state


def proposal():
    return {
        "kind": "propose", "change_reason": "Measure visible centers", "expected_change": "Points at each center",
        "understanding": {"task_summary": "Locate visible objects", "output_requirements": ["points"]},
        "pipeline": {
            "schema_version": 3, "name": "centers", "input_types": {"$rgb": "ImageArtifact"},
            "nodes": [{"id": "measure", "operator": "centers", "inputs": {"artifact": "$rgb"}, "params": {}}],
            "outputs": {"points": "measure"},
            "generated_operators": [{"name": "centers", "input_artifact": "ImageArtifact",
                                     "output_artifact": "MetadataArtifact", "atomic": False,
                                     "source": "def apply(data, params):\n    return {'points': [[2, 2]]}"}],
        },
    }


def test_proposal_uses_one_request_preserves_custom_v3_and_has_no_mutation_tools(monkeypatch, target):
    raw = proposal()
    state = install_client(monkeypatch, raw)
    provider = AliyunVisionProvider(api_key="test", max_retries=9)

    action = provider.propose_action(target, "Locate visible objects", context={"task_contract": {}})

    assert len(state["requests"]) == 1
    assert state["clients"][0]["max_retries"] == 0
    request = state["requests"][0]
    assert request["stream"] is False
    assert "stream_options" not in request
    assert request["max_tokens"] == 8192
    assert request["response_format"] == {"type": "json_object"}
    assert "tools" not in request
    assert action["pipeline"] == raw["pipeline"]
    assert action["understanding"]["output_requirements"] == ["points"]
    assert state["client_closed"]
    assert "save_task" not in request["messages"][0]["content"]
    assert "execute_pipeline" not in request["messages"][0]["content"]


def test_read_action_returns_to_controller_without_executing_or_followup(monkeypatch, target):
    raw = {"kind": "read", "requests": [
        {"tool": "query_operators", "arguments": {"names": ["global_threshold", "filter_components"]}},
        {"tool": "load_skill", "arguments": {"name": "area"}},
    ]}
    state = install_client(monkeypatch, raw)
    assert AliyunVisionProvider(api_key="test").propose_action(target, "Find objects") == raw
    assert len(state["requests"]) == 1


def test_json_completion_emits_usage_and_one_complete_action(monkeypatch, target):
    action = {"kind": "needs_input", "reason": "Missing calibration reference"}
    state = install_client(monkeypatch, action)
    chunks, responses = [], []
    monkeypatch.setattr("providers.vision.emit_llm_chunk", lambda text, **kwargs: chunks.append(text))
    monkeypatch.setattr("providers.vision.emit_llm_response", lambda *args, **kwargs: responses.append(kwargs))
    assert AliyunVisionProvider(api_key="test").propose_action(target, "Measure spacing") == action
    assert chunks == [json.dumps(action)]
    assert responses[0]["usage"] == {"prompt_tokens": 128, "completion_tokens": 64, "total_tokens": 192}
    assert len(state["requests"]) == 1


def test_action_timeout_uses_global_setting_without_changing_legacy_default(monkeypatch):
    monkeypatch.delenv("LIANGCE_MODEL_CALL_TIMEOUT_SECONDS", raising=False)
    provider = AliyunVisionProvider(api_key="test")
    assert provider.action_timeout_seconds == 120
    assert provider.timeout_seconds == 90
    monkeypatch.setenv("LIANGCE_MODEL_CALL_TIMEOUT_SECONDS", "75")
    configured = AliyunVisionProvider(api_key="test")
    assert configured.action_timeout_seconds == 75
    assert configured.timeout_seconds == 90
    explicit = AliyunVisionProvider(api_key="test", timeout_seconds=42)
    assert explicit.action_timeout_seconds == 42
    assert explicit.timeout_seconds == 42


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_action_timeout_rejects_nonpositive_or_nonfinite_setting(monkeypatch, value):
    monkeypatch.setenv("LIANGCE_MODEL_CALL_TIMEOUT_SECONDS", value)
    with pytest.raises(ValueError, match="finite and positive"):
        AliyunVisionProvider(api_key="test")


@pytest.mark.parametrize("tool", ["execute_pipeline", "create_draft", "edit_draft", "save_task", "submit_experiment"])
def test_action_parser_rejects_mutation_tools(tool):
    with pytest.raises(ValueError, match="read-only whitelist"):
        normalize_model_action({"kind": "read", "requests": [{"tool": tool, "arguments": {}}]}, description="target")


def test_revision_cannot_replace_established_contract():
    raw = proposal()
    context = {"task_contract": {"acceptance_criteria": {"visual_checks": ["All objects"]}}}
    before = deepcopy(context)
    with pytest.raises(ValueError, match="immutable_task_contract"):
        normalize_model_action(raw, description="target", context=context)
    assert context == before


def test_new_user_run_accepts_only_source_backed_top_level_updates():
    raw = proposal()
    raw.pop("understanding")
    raw.update(contract_updates=[{"field": "rendering", "value": {"contour_color": "#39FF14"},
                                 "source_quote": "change red to green"}],
               memory_updates=[{"op": "set", "key": "constraint:color", "value": "green",
                                "scope": "task", "source_quote": "change red to green"}])
    context = {"task_contract": {"rendering": {"contour_color": "#ff0000"}}, "allow_contract_updates": True}
    before = deepcopy(context)
    assert normalize_model_action(raw, description="Please change red to green", context=context) == raw
    assert context == before
    with pytest.raises(ValueError, match="immutable_task_contract"):
        normalize_model_action(raw, description="Please change red to green",
                               context={**context, "allow_contract_updates": False})
    with pytest.raises(ValueError, match="source_quote"):
        normalize_model_action(raw, description="Keep all objects", context=context)


def test_new_user_boundary_does_not_allow_full_understanding_override():
    with pytest.raises(ValueError, match="immutable_task_contract"):
        normalize_model_action(proposal(), description="New target", context={
            "task_contract": {"task_summary": "Existing target"}, "allow_contract_updates": True,
        })


def test_edit_can_carry_initial_user_updates_without_changing_edit_schema():
    raw = {"kind": "edit", "draft_id": "draft_1", "base_revision": 2,
           "edits": [{"path": "/name", "old": "old", "new": "new"}], "change_reason": "Requested change",
           "memory_updates": [{"op": "set", "key": "current_goal", "value": "All objects",
                               "source_quote": "All objects"}]}
    assert normalize_model_action(raw, description="All objects", context={"allow_contract_updates": True}) == raw


def test_initial_normalization_does_not_mutate_model_payload():
    raw = proposal()
    raw["understanding"]["target_constraints"] = {"expected_count": 2}
    before = deepcopy(raw)
    action = normalize_model_action(raw, description="All objects")
    assert raw == before
    assert action["understanding"]["target_constraints"]["count_source"] == "model_observed"
    assert "expected_count" not in action["understanding"]["target_constraints"]


def test_edit_keeps_invalid_source_for_controller_draft_validation():
    raw = {"kind": "edit", "draft_id": "draft_1", "base_revision": 2,
           "edits": [{"path": "/generated_operators/0/source", "old": "pass", "new": "invalid("}],
           "change_reason": "Repair measurement"}
    assert normalize_model_action(raw, description="target") == raw
    raw["base_revision"] = 0
    with pytest.raises(ValueError, match="out of range"):
        normalize_model_action(raw, description="target")


def test_snapshot_preserves_exact_contract_versions_errors_source_and_read_evidence(target):
    snapshot = {
        "task_contract": {"task_goal": "All objects", "visual_checks": ["Keep exact boundaries"]},
        "current_draft": {"draft_id": "draft_7", "revision": 3, "pipeline": proposal()["pipeline"]},
        "latest_experiment": {"experiment_id": "exp_8", "status": "completed", "acceptance_status": "rejected"},
        "experiment_summaries": [{"experiment_id": "exp_5", "review": {"reason": "Left object omitted"}}],
        "review": {"reason": "Wrong edges", "observed_issues": ["Missing upper edge"]},
        "last_error": {"code": "syntax_error", "line": 7, "message": "missing colon"},
        "budget": {"remaining_experiments": 1, "remaining_calls": 2},
        "task_memory": {"active_constraints": {"color": "green"}},
        "read_results": [{"tool": "inspect_experiment", "data": {"partial": True, "count": 50}, "images": [str(target)]}],
    }
    original = deepcopy(snapshot)
    messages = build_action_messages(target, "All objects", context=snapshot)
    state_part = next(p["text"] for p in messages[1]["content"] if p.get("text", "").startswith("canonical_state:"))
    assert json.loads(state_part.split("\n", 1)[1]) == {**snapshot, "input_metadata": input_metadata(target)}
    assert snapshot == original
    assert sum(p.get("type") == "image_url" for p in messages[1]["content"]) == 2


def test_original_dimensions_dtype_coordinates_and_remaining_time_reach_prompt(tmp_path):
    target = tmp_path / "sixteen_bit.png"
    Image.new("I;16", (27, 19), 1000).save(target)
    messages = build_action_messages(target, "Measure the target", context={"remaining_seconds": 417.5})
    state_part = next(p["text"] for p in messages[1]["content"] if p.get("text", "").startswith("canonical_state:"))
    snapshot = json.loads(state_part.split("\n", 1)[1])
    assert snapshot["remaining_seconds"] == 417.5
    assert snapshot["input_metadata"]["shape"] == [19, 27]
    assert snapshot["input_metadata"]["dtype"] == "uint16"
    assert snapshot["input_metadata"]["coordinate_version"] == "stored-pixels-v1"
    assert snapshot["input_metadata"]["coordinate_transform"] == {
        "orientation": "stored", "scale_xy": [1, 1], "offset_xy": [0, 0],
    }


def test_action_prompt_distinguishes_builtin_grayscale_from_optional_rgb(target):
    from core.pipelines.dsl import validate_pipeline
    system = build_action_messages(target, 'Find the bright regions')[0]['content']
    assert '不要在input_types声明$image' in system
    assert '只删除input_types中的$image项，保留节点对$image的引用，不要改为$rgb' in system
    assert '$rgb是原始RGB三通道数组，并非$image的替代别名' in system
    example, _ = json.JSONDecoder().raw_decode(system.split('Pipeline格式示例：', 1)[1])
    validate_pipeline(example)
    assert example['nodes'][0]['inputs'] == {'image': '$image'}
    assert 'input_types' not in example


def test_oversized_essential_context_fails_before_model_request(monkeypatch, target):
    monkeypatch.setenv("LIANGCE_LLM_MAX_TEXT_CHARS", "20000")
    state = install_client(monkeypatch, proposal())
    context = {"task_contract": {"visual_checks": ["X" * 30000]}, "latest_experiment": {"experiment_id": "exp_3"}}
    before = deepcopy(context)
    with pytest.raises(ContextBudgetError, match="context_budget_exceeded"):
        AliyunVisionProvider(api_key="test").propose_action(target, "All objects", context=context)
    assert state["requests"] == [] and state["clients"] == []
    assert context == before


def test_revision_includes_latest_overlay_and_mask_with_previous_pipeline(target, tmp_path):
    for filename in ("result_annotation.png", "mask.png"):
        Image.new("RGB", (8, 8), "white").save(tmp_path / filename)
    context = {"previous_pipeline": proposal()["pipeline"],
               "latest_experiment": {"experiment_id": "exp_8", "directory": str(tmp_path)}}
    messages = build_action_messages(target, "All objects", context=context)
    content = messages[1]["content"]
    assert sum(p.get("type") == "image_url" for p in content) == 3
    state_part = next(p["text"] for p in content if p.get("text", "").startswith("canonical_state:"))
    assert json.loads(state_part.split("\n", 1)[1])["previous_pipeline"] == proposal()["pipeline"]


@pytest.mark.parametrize("error", [TypeError("SDK mismatch"), RuntimeError("network failed")])
def test_failed_request_is_not_retried_or_downgraded(monkeypatch, target, error):
    state = install_client(monkeypatch, None, error=error)
    with pytest.raises(type(error), match=str(error)):
        AliyunVisionProvider(api_key="test", max_retries=3).propose_action(target, "All objects")
    assert len(state["requests"]) == 1
    assert state["client_closed"]


@pytest.mark.parametrize("text,finish", [("not JSON", "stop"), (json.dumps(proposal()), "length")])
def test_invalid_or_truncated_response_does_not_start_followup(monkeypatch, target, text, finish):
    state = install_client(monkeypatch, text, finish_reason=finish)
    with pytest.raises(ValueError):
        AliyunVisionProvider(api_key="test").propose_action(target, "All objects")
    assert len(state["requests"]) == 1


def test_wall_timeout_cancels_hanging_json_request(monkeypatch, target):
    state = install_client(monkeypatch, None, stall="create")
    provider = AliyunVisionProvider(api_key="test")
    provider.action_timeout_seconds = 0.03
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        provider.propose_action(target, "All objects")
    assert time.monotonic() - started < 1
    assert state["cancelled"] and state["client_closed"]
    assert len(state["requests"]) == 1


def test_parent_cancellation_interrupts_stalled_network_request(monkeypatch, target):
    state = install_client(monkeypatch, None, stall="create")
    request_control = RequestControl(timeout=30)
    token = control.set(request_control)
    timer = threading.Timer(0.02, request_control.cancelled.set)
    timer.start()
    try:
        with pytest.raises(RequestCancelled):
            AliyunVisionProvider(api_key="test").propose_action(target, "All objects")
    finally:
        control.reset(token)
        timer.join()
    assert state["cancelled"] and state["client_closed"]
    assert state["clients"][0]["timeout"] <= 30


def test_review_is_one_request_and_unresolved_issues_cannot_pass(monkeypatch, target, tmp_path):
    state = install_client(monkeypatch, {"kind": "review", "review": {
        "decision": "present", "selected_candidate": "candidate", "reason": "Looks right",
        "observed_issues": ["One missing region"], "revision_plan": ["Recover missing region"],
    }})
    action = AliyunVisionProvider(api_key="test").review_action(
        target, "All objects", [{"name": "candidate", "status": "completed", "directory": str(tmp_path)}],
        acceptance_criteria={"visual_checks": ["Every visible object"]},
    )
    assert action["review"]["decision"] == "revise"
    assert len(state["requests"]) == 1
    assert "Every visible object" in state["requests"][0]["messages"][0]["content"]


def test_review_cannot_accept_unknown_or_unexecuted_candidate(monkeypatch, target):
    state = install_client(monkeypatch, {"kind": "review", "review": {
        "decision": "present", "selected_candidate": "failed", "reason": "Looks right", "observed_issues": [],
    }})
    action = AliyunVisionProvider(api_key="test").review_action(target, "All objects", [{"name": "failed", "status": "failed"}])
    assert action["review"]["decision"] == "revise"
    assert action["review"]["selected_candidate"] is None
    assert len(state["requests"]) == 1
