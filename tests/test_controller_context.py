"""Exercise durable requirements and visual evidence through the real action parser."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from agent_types import normalize_strategy
from core.agent_graph import run_agent_graph
from core.pipelines.dsl import strategy_to_pipeline
from core.task_store import TaskStore
from providers.vision import AliyunVisionProvider, image_content


def snapshot(messages):
    part = next(part["text"] for message in messages for part in (
        message["content"] if isinstance(message["content"], list) else []
    ) if part.get("text", "").startswith("canonical_state:\n"))
    return json.loads(part.split("\n", 1)[1])


def proposal(*, initial=True, sensitivity=1):
    strategy = normalize_strategy({"segmentation": {"method": "bright_threshold", "sensitivity": sensitivity,
                                                    "min_area_px": 2, "morphology": "none"}})
    action = {"kind": "propose", "pipeline": strategy_to_pipeline(strategy),
              "change_reason": "Threshold observed bright pixels", "expected_change": "Recover bright region"}
    if initial:
        action["understanding"] = {
            "task_summary": "Find bright regions", "recommended_strategy": strategy,
            "target_constraints": {}, "output_requirements": ["mask"],
            "rendering": {"annotation_mode": "mask", "contour_color": "#ff0000"},
        }
    return action


def review(decision="present"):
    def respond(messages):
        state = snapshot(messages)
        return {"kind": "review", "review": {
            "decision": decision, "selected_candidate": state["latest_experiment"]["name"],
            "reason": "Checked the actual image and evidence",
            "observed_issues": [] if decision == "present" else ["Boundary needs adjustment"],
            "revision_plan": [] if decision == "present" else ["Adjust the response threshold"],
        }}
    return respond


class ScriptedProvider(AliyunVisionProvider):
    # These fixtures exercise the version-one checkpoint compatibility protocol.
    agent_action = None

    def __init__(self, responses):
        super().__init__(api_key="test-no-network")
        self.responses = list(responses)
        self.messages = []

    def _complete_action(self, messages):
        self.messages.append(deepcopy(messages))
        assert self.responses, "Unexpected extra model request"
        response = self.responses.pop(0)
        return json.dumps(response(messages) if callable(response) else response)


@pytest.fixture
def target(tmp_path, monkeypatch):
    monkeypatch.setattr("core.sandbox.check_sandbox_available", lambda: {"image_id": "test-image"})
    pixels = np.zeros((32, 32), dtype=np.uint8)
    pixels[10:18, 10:18] = 255
    path = tmp_path / "input.png"
    Image.fromarray(pixels).save(path)
    return path


def run(target, provider, description="Find bright regions", **kwargs):
    return run_agent_graph(target, description, provider=provider, output_root=target.parent / "outputs", **kwargs)


def test_explicit_new_user_change_is_applied_once_and_frozen_through_revision(target):
    first = run(target, ScriptedProvider([proposal(), review()]))
    initial_contract = deepcopy(first["task_contract"])
    description = "Change contour color to green"
    green_rendering = {**initial_contract["rendering"], "contour_color": "#39FF14"}
    changed = {**proposal(initial=False),
               "contract_updates": [{"field": "rendering", "value": green_rendering, "source_quote": description}],
               "memory_updates": [{"op": "set", "key": "constraint:color", "value": "green", "scope": "task",
                                   "source_quote": description}]}
    forbidden = {**proposal(initial=False, sensitivity=.5), "contract_updates": [
        {"field": "rendering", "value": initial_contract["rendering"], "source_quote": description},
    ]}
    provider = ScriptedProvider([changed, review("revise"), forbidden, proposal(initial=False, sensitivity=.5), review()])

    result = run(target, provider, description, task_id=first["task_id"])

    assert result["stop_reason"] == "review_passed"
    assert result["task_contract"]["rendering"] == green_rendering
    assert result["task_contract"]["version"] == initial_contract["version"] + 1
    assert result["task_contract"]["changes"] == [{"field": "rendering", "source_quote": description}]
    assert result["budget"]["usage"] == {"model_calls": 5, "executions": 2}
    contexts = [snapshot(messages) for messages in provider.messages]
    assert contexts[0]["allow_contract_updates"] is True
    assert all(context["allow_contract_updates"] is False for context in contexts[1:])
    assert "immutable_task_contract" in contexts[3]["last_error"]["error"]
    assert all(context["task_contract"]["rendering"] == green_rendering for context in contexts[1:])
    store = TaskStore(result["memory_context"]["task_root"])
    assert store.memory_service.snapshot(result["task_id"], result["input_sha256"])["active_constraints"]["color"] == "green"
    assert first["task_contract"] == initial_contract


def test_unquoted_new_user_update_is_rejected_before_draft_and_contract_stays_exact(target):
    first = run(target, ScriptedProvider([proposal(), review()]))
    invalid = {**proposal(initial=False), "contract_updates": [{
        "field": "rendering", "value": {"contour_color": "#39FF14"}, "source_quote": "Change contour color to green",
    }]}
    provider = ScriptedProvider([invalid, {"kind": "needs_input", "reason": "Need a clearer target description"}])
    result = run(target, provider, "Keep the original contour color", task_id=first["task_id"])
    assert result["task_contract"] == first["task_contract"]
    assert result["budget"]["usage"]["executions"] == 0
    assert not result.get("current_draft")
    assert "source_quote" in snapshot(provider.messages[1])["last_error"]["error"]


def test_read_evidence_accumulates_exact_skill_definitions_and_review_crop_images(target):
    skill = {"kind": "read", "requests": [{"tool": "load_skill", "arguments": {"name": "area_measurement"}}]}
    definitions = {"kind": "read", "requests": [{"tool": "query_operators", "arguments": {"names": ["global_threshold"]}}]}

    def crop(messages):
        return {"kind": "read", "requests": [{"tool": "inspect_experiment", "arguments": {
            "experiment_id": snapshot(messages)["latest_experiment"]["experiment_id"], "region": [0, 0, 24, 24],
        }}]}

    provider = ScriptedProvider([skill, definitions, proposal(), crop, definitions, review()])
    result = run(target, provider)

    assert result["stop_reason"] == "review_passed"
    before_proposal = snapshot(provider.messages[2])
    assert [item["tool"] for item in before_proposal["read_results"]] == ["load_skill", "query_operators"]
    assert before_proposal["read_results"][0] == snapshot(provider.messages[1])["read_results"][0]
    assert before_proposal["read_results"][0]["data"]["skill"]
    assert before_proposal["read_results"][1]["data"]["operators"][0]["name"] == "global_threshold"
    before_review = snapshot(provider.messages[-1])
    assert {item["tool"] for item in before_review["read_results"]} == {
        "load_skill", "inspect_experiment", "query_operators",
    }
    crop_result = next(item for item in before_review["read_results"] if item["tool"] == "inspect_experiment")
    previous_crop = next(item for item in snapshot(provider.messages[-2])["read_results"]
                         if item["tool"] == "inspect_experiment")
    assert crop_result == previous_crop
    assert crop_result["images"]
    encoded = [part["image_url"]["url"] for message in provider.messages[-1] for part in (
        message["content"] if isinstance(message["content"], list) else []
    ) if part.get("type") == "image_url"]
    assert all(image_content(path, "crop")["image_url"]["url"] in encoded for path in crop_result["images"])
    assert result["budget"]["usage"] == {"model_calls": 6, "executions": 1}


def test_static_definitions_survive_review_revision_and_patch_conflict(target):
    definitions = {"kind": "read", "requests": [
        {"tool": "load_skill", "arguments": {"name": "area_measurement"}},
        {"tool": "query_operators", "arguments": {"names": ["global_threshold", "filter_components"]}},
    ]}

    def crop(messages):
        return {"kind": "read", "requests": [{"tool": "inspect_experiment", "arguments": {
            "experiment_id": snapshot(messages)["latest_experiment"]["experiment_id"], "region": [0, 0, 24, 24],
        }}]}

    def conflicting_edit(messages):
        draft = snapshot(messages)["current_draft"]
        return {"kind": "edit", "draft_id": draft["draft_id"], "base_revision": draft["revision"],
                "change_reason": "Fix boundary based on the inspected region",
                "edits": [{"path": "/nonexistent_parameter", "old": 1, "new": 2}]}

    provider = ScriptedProvider([
        definitions, proposal(), review("revise"), crop, conflicting_edit,
        proposal(initial=False, sensitivity=.5), review(),
    ])

    result = run(target, provider)

    assert result["stop_reason"] == "review_passed"
    assert result["budget"]["usage"] == {"model_calls": 7, "executions": 2}
    contexts = [snapshot(messages) for messages in provider.messages]
    known_definitions = contexts[1]["read_results"]
    assert {item["tool"] for item in known_definitions} == {"load_skill", "query_operators"}
    for context in contexts[2:]:
        retained = [item for item in context["read_results"] if item["tool"] in {"load_skill", "query_operators"}]
        assert retained == known_definitions
    before_edit = next(item for item in contexts[4]["read_results"] if item["tool"] == "inspect_experiment")
    after_conflict = next(item for item in contexts[5]["read_results"] if item["tool"] == "inspect_experiment")
    assert before_edit == after_conflict
    assert contexts[5]["last_error"]["code"] == "patch_conflict"
    assert contexts[5]["current_draft"]["revision"] == contexts[4]["current_draft"]["revision"]
    assert all(item["tool"] != "inspect_experiment" for item in contexts[6]["read_results"])


def test_reference_pixels_descriptions_and_scope_reach_proposal_and_review(target, tmp_path):
    examples = []
    for index, color in enumerate(((11, 21, 31), (41, 51, 61), (71, 81, 91))):
        path = tmp_path / f"reference_{index}.png"
        Image.new("RGB", (20 + index, 18 + index), color).save(path)
        examples.append({"image_path": str(path), "description": f"Reference semantic example {index}"})
    original = deepcopy(examples)
    provider = ScriptedProvider([proposal(), review()])

    result = run(target, provider, reference_examples=examples)

    assert result["stop_reason"] == "review_passed"
    for messages in provider.messages:
        parts = [part for message in messages for part in (
            message["content"] if isinstance(message["content"], list) else []
        )]
        urls = [part["image_url"]["url"] for part in parts if part.get("type") == "image_url"]
        text = "\n".join(part.get("text", "") for part in parts)
        for example in examples:
            assert urls.count(image_content(example["image_path"], "reference")["image_url"]["url"]) == 1
            assert example["description"] in text
    expected_scope = [{"sha256": sha256(Path(item["image_path"]).read_bytes()).hexdigest(),
                       "description": item["description"]} for item in examples]
    assert result["input_scope"]["references"] == expected_scope
    assert result["experiment_records"][0]["scope"]["references"] == expected_scope
    assert examples == original
