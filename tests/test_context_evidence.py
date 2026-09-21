import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from core.experiments.context import candidate_for_model, quality_for_model
from core.model_context import ContextBudgetError, check_request_budget
from core.tools.evidence import inspect_experiment
from providers.vision import AliyunVisionProvider, build_candidate_review_messages, image_content


@pytest.fixture
def evidence(tmp_path):
    target = tmp_path / "source.png"
    pixels = np.arange(80 * 60, dtype=np.uint8).reshape(60, 80)
    Image.fromarray(pixels).save(target)
    directory = tmp_path / "candidate"
    directory.mkdir()
    Image.fromarray(pixels).save(directory / "result_annotation.png")
    mask = np.zeros((60, 80), dtype=bool)
    mask[0:3, 0:3] = True  # Edge target.
    mask[20, 20] = True  # Single-pixel target.
    mask[30:40, 30:40] = True  # Connected/merged target.
    Image.fromarray(mask.astype("uint8") * 255).save(directory / "mask.png")
    outputs = {
        "mask": {"kind": "mask", "data": mask.tolist()},
        "contours": {"kind": "contours", "data": [[[1, 2], [3, 4]]] * 1000, "shape": [60, 80]},
        "measurements": {"kind": "metadata", "data": {
            "unit": "pixel", "invalid_count": 1,
            "components": [{"label": i, "width": i / 10, "unit": "pixel"} for i in range(100)]}},
    }
    measurements = {"summary": {"count": 3, "total_area": 110, "unit": "pixel"},
                    "results": outputs["measurements"]["data"]["components"],
                    "structured_outputs": outputs}
    quality = {"coverage": 110 / 4800, "component_count": 3, "health": {
        "usable_for_review": True, "issues": ["border_target"]},
        "user_constraints": {"excluded_pixels": 5}, "outputs": outputs}
    for name, value in (("outputs", outputs), ("measurements", measurements),
                        ("quality_report", quality), ("pipeline", {"nodes": []})):
        (directory / f"{name}.json").write_text(json.dumps(value))
    candidate = {"experiment_id": "exp", "name": "candidate", "status": "selected_for_review",
                 "directory": str(directory), "quality": quality, "measurements": measurements}
    return target, candidate


def test_views_preserve_facts_reports_and_images_without_raw_arrays(evidence):
    target, candidate = evidence
    before = copy.deepcopy(candidate)
    criteria = {"task_goal": "保留小目标与边缘目标", "visual_checks": ["不得漏检", "检查粘连"],
                "memory_contract": {"active_constraints": {"unit": "pixel", "exclude": "信息栏"}}}
    messages = build_candidate_review_messages(target, "提取所有目标", [candidate], acceptance_criteria=criteria)
    content = messages[0]["content"]
    assert [p for p in content if p["type"] == "image_url"] == [
        image_content(target, ""), image_content(Path(candidate["directory"]) / "result_annotation.png", "")]
    assert "不得漏检" in content[0]["text"] and "信息栏" in content[0]["text"]
    view = json.loads(content[2]["text"])
    assert view["experiment_id"] == "exp"
    facts = view["facts"]
    assert facts["health"] == candidate["quality"]["health"]
    assert facts["user_constraints"] == candidate["quality"]["user_constraints"]
    assert facts["outputs"]["mask"]["foreground_pixels"] == 110
    assert facts["outputs"]["mask"]["shape"] == [60, 80]
    assert "data" not in facts["outputs"]["mask"]
    assert "data" not in facts["outputs"]["contours"]
    assert facts["outputs"]["contours"]["point_count"] == 2000
    assert view["measurement_summary"] == candidate["measurements"]["summary"]
    assert len(json.dumps(view)) < 10000
    assert candidate == before
    persisted = json.loads((Path(candidate["directory"]) / "outputs.json").read_text())
    assert persisted == candidate["quality"]["outputs"]


def test_large_metadata_marks_partial_and_exact_last_page_is_retrievable(evidence, tmp_path):
    target, candidate = evidence
    view = candidate_for_model(candidate)
    summary = view["facts"]["outputs"]["measurements"]["data"]
    assert summary["partial"]
    assert summary["fields"]["unit"] == "pixel"
    assert summary["fields"]["invalid_count"] == 1
    result, images = inspect_experiment({
        "experiment_id": "exp", "report": "outputs",
        "selector": "/measurements/data/components", "offset": 90, "limit": 10,
    }, {"exp": candidate}, target, tmp_path)
    assert not images
    assert result["total_items"] == 100 and result["next_offset"] is None
    assert [entry["value"] for entry in result["items"]] == candidate["measurements"]["results"][90:]
    with pytest.raises(ValueError, match="pixel/contour"):
        inspect_experiment({"experiment_id": "exp", "report": "outputs", "selector": "/mask/data"},
                           {"exp": candidate}, target, tmp_path)
    with pytest.raises(ValueError, match="outside"):
        inspect_experiment({"experiment_id": "../other"}, {"exp": candidate}, target, tmp_path)


@pytest.mark.parametrize("region", [[0, 0, 5, 5], [19, 19, 22, 22], [29, 29, 41, 41]])
def test_native_crops_preserve_small_edge_and_merged_targets(evidence, tmp_path, region):
    target, candidate = evidence
    result, images = inspect_experiment({"experiment_id": "exp", "region": region},
                                       {"exp": candidate}, target, tmp_path)
    assert result["scale"] == 1 and result["region_xyxy"] == region
    assert len(images) == 3
    sources = [target, Path(candidate["directory"]) / "result_annotation.png",
               Path(candidate["directory"]) / "mask.png"]
    for path, source in zip(images, sources):
        with Image.open(path) as actual, Image.open(source) as original:
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(original.crop(region)))
    with pytest.raises(ValueError, match="outside"):
        inspect_experiment({"experiment_id": "exp", "region": [0, 0, 81, 60]},
                           {"exp": candidate}, target, tmp_path)


def test_empty_and_failed_results_are_not_hidden():
    failure = {"issues": ["execution_failed"], "error": "operator broke", "evaluation": {"recall": 0}}
    assert quality_for_model(failure) == failure
    empty = quality_for_model({"coverage": 0, "component_count": 0,
                               "health": {"usable_for_review": False, "issues": ["empty_mask"]},
                               "outputs": {"mask": {"kind": "mask", "data": [[False] * 50] * 50}}})
    assert empty["outputs"]["mask"]["foreground_pixels"] == 0
    assert empty["health"]["issues"] == ["empty_mask"]


def test_report_inspection_never_exposes_nested_small_pixel_arrays(evidence, tmp_path):
    target, candidate = evidence
    path = Path(candidate["directory"]) / "quality_report.json"
    path.write_text(json.dumps({"outputs": {"small": {"kind": "mask", "data": [[True]]}}}))
    report, _ = inspect_experiment({"experiment_id": "exp", "report": "quality_report"},
                                  {"exp": candidate}, target, tmp_path)
    mask = report["items"][0]["value"]["small"]
    assert mask["foreground_pixels"] == 1 and "data" not in mask
    (path.parent / "experiment.json").write_text(json.dumps({"input_sha256": "different"}))
    with pytest.raises(ValueError, match="different input"):
        inspect_experiment({"experiment_id": "exp", "region": [0, 0, 5, 5]},
                           {"exp": candidate}, target, tmp_path)


def test_request_budget_counts_tools_excludes_image_encoding_and_never_truncates(monkeypatch):
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "用户要求和验收条件"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 1000000}}]},
        {"role": "assistant", "tool_calls": [{"id": "x", "function": {"arguments": "x" * 50}}]}]
    before = copy.deepcopy(messages)
    stats = check_request_budget(messages, [{"function": {"name": "test"}}])
    assert stats["total_text_chars"] < 1000
    assert stats["tool_call_chars"] > 50 and stats["tool_schema_chars"] > 0
    assert stats["image_count"] == 1 and stats["image_url_chars"] > 1000000
    monkeypatch.setenv("LIANGCE_LLM_MAX_TEXT_CHARS", "10")
    with pytest.raises(ContextBudgetError, match="not sent"):
        check_request_budget(messages)
    assert messages == before


def test_oversized_request_is_rejected_before_api(monkeypatch):
    monkeypatch.setenv("LIANGCE_LLM_MAX_TEXT_CHARS", "100")
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: pytest.fail("Oversized request reached API"))))
    provider = AliyunVisionProvider(api_key="test")
    with pytest.raises(ContextBudgetError):
        provider._complete_streaming(client, [{"role": "user", "content": "x" * 101}])


@pytest.mark.parametrize("mode", ["native", "text"])
def test_review_can_inspect_evidence_but_cannot_execute(evidence, monkeypatch, mode):
    target, candidate = evidence
    provider = AliyunVisionProvider(api_key="test", tool_mode=mode)
    monkeypatch.setattr("providers.vision.ROOT", target.parent)
    seen = []
    from core.planning import ModelReply
    def complete(client, messages, progress_callback=None, **kwargs):
        seen.append(copy.deepcopy(messages))
        assert "execute_pipeline" not in json.dumps(kwargs.get("tools", []))
        if len(seen) == 1:
            args = {"experiment_id": "exp", "region": [0, 0, 5, 5]}
            if mode == "native":
                return ModelReply(calls=[{"id": "crop", "name": "inspect_experiment", "arguments": args}])
            return json.dumps({"type": "call_tool", "tool": "inspect_experiment", "arguments": args})
        assert len([p for p in messages[-1]["content"] if p["type"] == "image_url"]) == 3
        return json.dumps({"decision": "revise", "selected_candidate": "candidate",
                           "observed_issues": ["edge defect"], "reason": "边缘不完整"})
    monkeypatch.setattr(provider, "_complete_streaming", complete)
    result = provider.review_candidates(target, "检测所有目标", [candidate])
    assert result["decision"] == "revise" and len(seen) == 2
    assert result["tool_session"]["executions"] == 0


def test_execute_and_compare_use_summaries_while_retaining_full_outputs(tmp_path, monkeypatch):
    from core.pipelines.dsl import execute_pipeline
    from core.tools.experiments import ExperimentTools
    # Exercise the full runner and serialization with trusted built-ins in process.
    monkeypatch.setattr("core.experiments.runner.execute_pipeline_sandbox", execute_pipeline)
    target = tmp_path / "input.png"
    pixels = np.zeros((200, 300), dtype=np.uint8)
    pixels[20:30, 20:30] = 255
    Image.fromarray(pixels).save(target)
    session = ExperimentTools(target, "bright", output_root=tmp_path)
    ids = []
    for polarity in ("bright", "dark"):
        result, images = session.dispatch({"tool": "execute_pipeline", "arguments": {"pipeline": {
            "schema_version": 3, "name": polarity,
            "nodes": [{"id": "mask", "operator": "global_threshold", "inputs": {"image": "$image"},
                       "params": {"polarity": polarity, "sensitivity": 1}}],
            "outputs": {"mask": "mask"}}}}, None)
        assert result["status"] == "success" and images
        assert len(json.dumps(result)) < 16000
        data = result["data"]
        assert "data" not in data["facts"]["outputs"]["mask"]
        ids.append(data["experiment_id"])
        full = json.loads((Path(session.attempts[ids[-1]]["directory"]) / "outputs.json").read_text())
        assert len(full["mask"]["data"]) == 200
    comparison, _ = session.dispatch({"tool": "compare_candidates", "arguments": {"experiment_ids": ids}}, None)
    assert comparison["status"] == "success"
    assert len(json.dumps(comparison)) < 20000
    assert comparison["data"]["differences"][0]["changed_pixels"] > 0
