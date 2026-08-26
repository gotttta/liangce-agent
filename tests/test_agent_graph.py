import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from agent_types import normalize_strategy
from core.agent_graph import resume_agent_graph, run_agent_graph
from core.algorithm_registry import AlgorithmRegistry
from core.pipelines.dsl import strategy_to_pipeline


def _understanding(expected_shape=None):
    strategy = normalize_strategy({
        "defect_type": "particle",
        "measurement_type": "area_count",
        "visual_observation": {
            "background_pattern": "periodic_lines",
            "polarity": "bright_on_dark",
        },
        "segmentation": {
            "method": "bright_threshold",
            "sensitivity": 1.0,
            "min_area_px": 2,
            "morphology": "none",
        },
    })
    constraints = {"expected_shape": expected_shape} if expected_shape else {}
    return {
        "task_summary": "提取周期背景中的亮色颗粒",
        "target_defect": "particle",
        "normal_context": "periodic_lines",
        "recommended_strategy": strategy,
        "candidate_pipelines": [{
            "name": "qwen_candidate",
            "hypothesis": "bright anomaly",
            "pipeline": strategy_to_pipeline(strategy),
        }],
        "target_constraints": constraints,
        "rendering": {},
    }


def _sample(path):
    image = np.zeros((32, 32), dtype=np.uint8)
    image[10:18, 10:18] = 255
    Image.fromarray(image, mode="L").save(path)


def test_agent_graph_records_nodes_and_waits_for_acceptance(tmp_path):
    source = tmp_path / "sample.png"
    _sample(source)

    state = run_agent_graph(
        source,
        "提取颗粒",
        output_root=tmp_path / "outputs",
        understanding=_understanding(),
    )

    assert [event["node"] for event in state["trajectory"]] == [
        "prepare_inputs",
        "understand_task",
        "retrieve_algorithms",
        "plan_candidates",
        "execute_candidates",
        "review_candidates",
        "decide_next_action",
    ]
    assert state["decision"]["next_action"] == "wait_for_acceptance"
    assert state["agent_status"] == "waiting_for_acceptance"
    assert state["interrupt"][0]["value"]["kind"] == "human_review"
    iteration_dir = Path(state["run_dir"]) / "iteration_0"
    persisted = json.loads((iteration_dir / "graph_state.json").read_text(encoding="utf-8"))
    assert persisted["decision"] == state["decision"]
    assert (iteration_dir / "agent_trajectory.json").exists()


def test_agent_graph_resumes_human_review_without_reexecuting_candidates(tmp_path):
    source = tmp_path / "sample.png"
    _sample(source)
    state = run_agent_graph(
        source,
        "提取颗粒",
        output_root=tmp_path / "outputs",
        understanding=_understanding(),
        thread_id="test-human-review",
    )

    resumed = resume_agent_graph("test-human-review", {"action": "accept"})

    assert resumed["agent_status"] == "accepted"
    assert resumed["human_response"] == {"action": "accept"}
    assert [event["node"] for event in resumed["trajectory"]][-1] == "resume_after_human"
    assert len(resumed["candidate_attempts"]) == len(state["candidate_attempts"])


def test_agent_graph_persists_incremental_human_feedback(tmp_path):
    source = tmp_path / "sample.png"
    _sample(source)
    include = np.zeros((32, 32), dtype=np.uint8)
    include[2:5, 3:7] = 255
    include_path = tmp_path / "include.png"
    Image.fromarray(include).save(include_path)
    run_agent_graph(
        source,
        "提取颗粒",
        output_root=tmp_path / "outputs",
        understanding=_understanding(),
        thread_id="test-human-feedback",
    )
    resumed = resume_agent_graph("test-human-feedback", {
        "action": "continue",
        "incremental_description": "左上角漏了一个颗粒",
        "include_mask_path": str(include_path),
    })
    assert resumed["human_feedback"] == {
        "incremental_description": "左上角漏了一个颗粒",
        "include_mask_path": str(include_path),
    }
    assert resumed["trajectory"][-1]["details"]["has_feedback"] is True


def test_agent_graph_does_not_infer_visual_quality_from_shape_statistics(tmp_path):
    source = tmp_path / "sample.png"
    _sample(source)

    state = run_agent_graph(
        source,
        "提取细长缺陷",
        output_root=tmp_path / "outputs",
        understanding=_understanding(expected_shape="elongated"),
    )

    assert "status" not in state["quality_report"]
    assert state["decision"]["next_action"] == "wait_for_acceptance"
    assert state["agent_status"] == "waiting_for_acceptance"


def test_agent_graph_promotes_candidate_selected_by_visual_review(tmp_path):
    source = tmp_path / "sample.png"
    _sample(source)
    understanding = _understanding()
    second_pipeline = strategy_to_pipeline(
        {
            "segmentation": {
                "method": "bright_threshold",
                "sensitivity": 0.5,
                "min_area_px": 2,
                "morphology": "none",
            }
        },
        name="review_selected_pipeline",
    )
    understanding["candidate_pipelines"].append({
        "name": "review_selected",
        "hypothesis": "broader bright region",
        "pipeline": second_pipeline,
    })

    class ReviewingProvider:
        def review_candidates(self, target_image_path, description, candidates):
            return {
                "decision": "present",
                "selected_candidate": "review_selected",
                "reason": "第二个候选更符合测试要求",
            }

    state = run_agent_graph(
        source,
        "提取颗粒",
        output_root=tmp_path / "outputs",
        understanding=understanding,
        provider=ReviewingProvider(),
    )

    assert state["selected_candidate"] == "review_selected"
    assert state["pipeline"]["name"] == "review_selected_pipeline"
    assert state["review"]["reason"] == "第二个候选更符合测试要求"


def test_agent_graph_runs_one_bounded_revision_when_review_requests_it(tmp_path):
    source = tmp_path / "sample.png"
    _sample(source)
    understanding = _understanding()

    class RevisingProvider:
        def __init__(self):
            self.review_count = 0

        def understand_task(self, target_image_path, description, previous_context=None):
            assert previous_context["review"]["decision"] == "revise"
            return understanding

        def review_candidates(self, target_image_path, description, candidates):
            self.review_count += 1
            if self.review_count == 1:
                return {
                    "decision": "revise",
                    "selected_candidate": candidates[0]["name"],
                    "reason": "第一轮需要修订",
                    "revision_plan": ["调整候选"],
                }
            return {
                "decision": "present",
                "selected_candidate": candidates[0]["name"],
                "reason": "修订后可展示",
            }

    provider = RevisingProvider()
    state = run_agent_graph(
        source,
        "提取颗粒",
        output_root=tmp_path / "outputs",
        understanding=understanding,
        provider=provider,
        max_auto_revisions=1,
    )

    nodes = [event["node"] for event in state["trajectory"]]
    assert nodes.count("execute_candidates") == 2
    assert nodes.count("review_candidates") == 2
    assert "revise_candidates" in nodes
    assert state["revision_count"] == 1
    assert state["iteration"] == 1
    assert state["review"]["reason"] == "修订后可展示"


def test_agent_graph_reviews_single_candidate_and_revises_nonempty_result(tmp_path):
    source = tmp_path / "sample.png"
    _sample(source)
    initial = _understanding()
    revised = _understanding()
    revised["candidate_pipelines"][0]["pipeline"] = strategy_to_pipeline(
        {
            "segmentation": {
                "method": "bright_threshold",
                "sensitivity": 0.5,
                "min_area_px": 2,
                "morphology": "none",
            }
        },
        name="revised_pipeline",
    )

    class VisualReviewer:
        def __init__(self):
            self.review_count = 0
            self.criteria = []

        def understand_task(self, target_image_path, description, previous_context=None):
            return revised

        def review_candidates(
            self,
            target_image_path,
            description,
            candidates,
            reference_examples=None,
            acceptance_criteria=None,
        ):
            self.review_count += 1
            self.criteria.append(acceptance_criteria)
            return {
                "decision": "revise" if self.review_count == 1 else "present",
                "selected_candidate": candidates[0]["name"],
                "reason": "第一轮结果漏标，第二轮符合要求"
                if self.review_count == 1
                else "第二轮结果符合要求",
                "observed_issues": ["第一轮只标出了一部分目标"]
                if self.review_count == 1
                else [],
                "revision_plan": ["改进覆盖范围"] if self.review_count == 1 else [],
            }

    provider = VisualReviewer()
    state = run_agent_graph(
        source,
        "提取颗粒",
        output_root=tmp_path / "outputs",
        understanding=initial,
        provider=provider,
        max_candidates=1,
        max_auto_revisions=1,
    )

    assert provider.review_count == 2
    assert all(criteria["task_goal"] for criteria in provider.criteria)
    assert [event["node"] for event in state["trajectory"]].count("review_candidates") == 2
    assert state["revision_count"] == 1


def test_agent_graph_does_not_rerun_duplicate_revision_pipeline(tmp_path):
    source = tmp_path / "sample.png"
    _sample(source)
    understanding = _understanding()

    class DuplicateProvider:
        def __init__(self):
            self.contexts = []

        def understand_task(self, target_image_path, description, previous_context=None):
            self.contexts.append(previous_context)
            return understanding

        def review_candidates(self, target_image_path, description, candidates, **kwargs):
            return {
                "decision": "revise",
                "selected_candidate": candidates[0]["name"],
                "reason": "结果需要重做",
            }

    state = run_agent_graph(
        source,
        "提取颗粒",
        output_root=tmp_path / "outputs",
        understanding=understanding,
        provider=DuplicateProvider(),
        max_candidates=1,
        max_auto_revisions=1,
    )

    assert state["status"] == "needs_human_review"
    assert state["agent_status"] == "waiting_for_feedback"
    assert Path(state["annotated_image_path"]).exists()
    assert Path(state["predicted_mask_path"]).exists()


def test_agent_graph_automatically_replans_after_empty_annotation(tmp_path):
    source = tmp_path / "sample.png"
    _sample(source)
    initial = _understanding()
    initial["candidate_pipelines"] = [{
        "name": "empty",
        "pipeline": {
            "name": "empty",
            "steps": [{
                "id": "final_mask",
                "op": "global_threshold",
                "input": "image",
                "params": {"polarity": "bright", "sensitivity": 10.0, "max_coverage": 0.9},
            }],
        },
    }]

    class RetryProvider:
        def __init__(self):
            self.contexts = []

        def understand_task(self, target_image_path, description, previous_context=None):
            self.contexts.append(previous_context)
            return _understanding()

    provider = RetryProvider()
    state = run_agent_graph(
        source,
        "提取颗粒",
        output_root=tmp_path / "outputs",
        understanding=initial,
        provider=provider,
        max_candidates=1,
        max_auto_revisions=2,
    )

    assert state["revision_count"] == 1
    assert state["iteration"] == 1
    assert state["measurements"]["summary"]["count"] == 1
    assert provider.contexts[0]["execution_feedback"]["status"] == "no_usable_annotation"
    failed_attempt = provider.contexts[0]["execution_feedback"]["attempts"][0]
    assert failed_attempt["status"] == "no_annotation"
    assert failed_attempt["operator_trace"][0]["operator"] == "global_threshold"
    nodes = [event["node"] for event in state["trajectory"]]
    assert nodes.count("execute_candidates") == 2
    assert "revise_candidates" in nodes


def test_agent_graph_reports_plain_message_after_retry_limit(tmp_path):
    source = tmp_path / "sample.png"
    _sample(source)
    empty = _understanding()
    empty["candidate_pipelines"] = [{
        "name": "empty",
        "pipeline": {
            "name": "empty",
            "steps": [{
                "id": "final_mask",
                "op": "global_threshold",
                "input": "image",
                "params": {"polarity": "bright", "sensitivity": 10.0, "max_coverage": 0.9},
            }],
        },
    }]

    class AlwaysEmptyProvider:
        def understand_task(self, target_image_path, description, previous_context=None):
            return empty

    state = run_agent_graph(
        source,
        "提取颗粒",
        output_root=tmp_path / "outputs",
        understanding=empty,
        provider=AlwaysEmptyProvider(),
        max_candidates=1,
        max_auto_revisions=2,
    )

    assert state["status"] == "needs_human_review"
    assert state["agent_status"] == "waiting_for_feedback"
    assert state["measurements"]["summary"]["count"] == 0
    assert Path(state["annotated_image_path"]).exists()
    assert Path(state["predicted_mask_path"]).exists()


def test_ground_truth_revision_does_not_union_the_rejected_previous_mask(tmp_path):
    source = tmp_path / "sample.png"
    image = np.zeros((32, 32), dtype=np.uint8)
    image[10:18, 10:18] = 255
    Image.fromarray(image).save(source)
    ground_truth = tmp_path / "ground_truth.png"
    Image.fromarray(image).save(ground_truth)

    initial = _understanding()
    initial["candidate_pipelines"] = [{
        "name": "wrong_background",
        "pipeline": {
            "name": "wrong_background",
            "steps": [{
                "id": "final_mask",
                "op": "global_threshold",
                "input": "image",
                "params": {"polarity": "dark", "sensitivity": 0.0, "max_coverage": None},
            }],
        },
    }]
    revised = _understanding()

    class GroundTruthRetryProvider:
        def understand_task(self, target_image_path, description, previous_context=None):
            return revised

    state = run_agent_graph(
        source,
        "提取亮色区域",
        output_root=tmp_path / "outputs",
        understanding=initial,
        provider=GroundTruthRetryProvider(),
        max_candidates=1,
        max_auto_revisions=1,
        ground_truth_mask_path=ground_truth,
    )

    assert state["revision_count"] == 1
    assert state["evaluation_report"]["dice"] == 1.0
    assert state["evaluation_report"]["precision"] == 1.0
    assert state["evaluation_report"]["recall"] == 1.0
    assert state["quality_report"]["user_constraints"] == {}
    assert state["agent_status"] == "waiting_for_acceptance"


def test_ground_truth_retry_limit_preserves_best_metrics_in_final_state(tmp_path):
    source = tmp_path / "sample.png"
    image = np.zeros((32, 32), dtype=np.uint8)
    image[10:18, 10:18] = 255
    Image.fromarray(image).save(source)
    ground_truth_image = image.copy()
    ground_truth_image[20:28, 20:28] = 255
    ground_truth = tmp_path / "ground_truth.png"
    Image.fromarray(ground_truth_image).save(ground_truth)

    initial = _understanding()
    initial["candidate_pipelines"] = [{
        "name": "wrong_background",
        "pipeline": {
            "name": "wrong_background",
            "steps": [{
                "id": "final_mask",
                "op": "global_threshold",
                "input": "image",
                "params": {"polarity": "dark", "sensitivity": 0.0, "max_coverage": None},
            }],
        },
    }]
    revised = _understanding()

    class GroundTruthRetryProvider:
        def understand_task(self, target_image_path, description, previous_context=None):
            return revised

    state = run_agent_graph(
        source,
        "提取亮色区域",
        output_root=tmp_path / "outputs",
        understanding=initial,
        provider=GroundTruthRetryProvider(),
        max_candidates=1,
        max_auto_revisions=1,
        ground_truth_mask_path=ground_truth,
    )

    assert state["status"] == "needs_human_review"
    assert state["evaluation_report"]["dice"] == round(2 / 3, 6)
    assert state["evaluation_report"] == state["quality_report"]["evaluation"]
    assert "Ground Truth 指标最好的候选" in state["conversation"][0]["content"]


def test_agent_graph_retrieves_accepted_algorithm_as_candidate(tmp_path):
    source = tmp_path / "sample.png"
    _sample(source)
    understanding = _understanding()
    registry = AlgorithmRegistry(tmp_path / "algorithms")
    registry.publish("task_old", {
        "selected_candidate": "accepted_particle",
        "description": "周期背景亮色颗粒",
        "strategy": understanding["recommended_strategy"],
        "pipeline": understanding["candidate_pipelines"][0]["pipeline"],
        "quality_report": {},
        "measurements": {"summary": {"count": 1, "unit": "pixel"}},
    })

    state = run_agent_graph(
        source,
        "提取周期背景中的亮色颗粒",
        output_root=tmp_path / "outputs",
        understanding=understanding,
        algorithm_registry=registry,
    )

    retrieval = next(event for event in state["trajectory"] if event["node"] == "retrieve_algorithms")
    assert retrieval["details"]["match_count"] == 1
    # The current task's generated candidate is executed first; accepted
    # algorithms remain available as later replay candidates.
    assert state["candidate_attempts"][0]["source"]["type"] == "qwen"
    # Identical historical pipelines are deduplicated rather than executed twice.
    assert len(state["candidate_attempts"]) == 1


def test_agent_graph_recovers_from_provider_failure_with_local_fallback(tmp_path):
    source = tmp_path / "sample.png"
    _sample(source)

    class BrokenProvider:
        def understand_task(self, *args, **kwargs):
            raise ValueError("invalid provider JSON")

    state = run_agent_graph(
        source,
        "提取周期背景中的颗粒",
        output_root=tmp_path / "outputs",
        provider=BrokenProvider(),
        max_candidates=1,
    )

    assert state["candidate_attempts"]
    assert state["candidate_attempts"][0]["source"]["type"] == "deterministic_fallback"
    assert "provider_invalid_json_or_call" in state["understanding"]["ambiguities"][0]
