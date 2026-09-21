import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from agent_types import normalize_strategy
from core.agent_loop import (
    _fit_candidate_budget,
    _apply_feedback_quality,
    _ground_truth_calibration_candidates,
    _pipeline_diff,
    apply_user_constraints,
    run_planned_agent,
    plan_candidate_definitions,
)
from core.measurement.evaluation import evaluate_prediction
from core.pipelines.dsl import execute_pipeline, strategy_to_pipeline
from core.reference_extraction import extract_ground_truth_mask


def test_apply_user_constraints_forces_include_and_exclude_masks(tmp_path):
    include = np.zeros((5, 5), dtype=np.uint8)
    exclude = np.zeros((5, 5), dtype=np.uint8)
    include[0, 0] = 255
    exclude[2, 2] = 255
    include_path = tmp_path / "include.png"
    exclude_path = tmp_path / "exclude.png"
    Image.fromarray(include).save(include_path)
    Image.fromarray(exclude).save(exclude_path)
    predicted = np.zeros((5, 5), dtype=bool)
    predicted[2, 2] = True

    constrained, report = apply_user_constraints(predicted, {
        "human_feedback": {
            "include_mask_path": str(include_path),
            "exclude_mask_path": str(exclude_path),
        },
    })

    assert constrained[0, 0]
    assert not constrained[2, 2]
    assert report == {"included_pixels": 1, "excluded_pixels": 1}


def test_candidate_budget_reserves_a_slot_for_accepted_algorithm():
    pipeline = lambda name, sensitivity: {
        "name": name,
        "steps": [{"id": "final_mask", "op": "global_threshold", "input": "image", "params": {"polarity": "bright", "sensitivity": sensitivity}}],
    }
    understanding = {
        "recommended_strategy": {},
        "candidate_pipelines": [
            {"name": f"generated_{index}", "pipeline": pipeline(f"generated_{index}", index + 1)}
            for index in range(3)
        ],
    }
    retrieved = [{"algorithm_id": "accepted_1", "name": "accepted", "pipeline": pipeline("accepted", 4)}]

    candidates = plan_candidate_definitions(understanding, retrieved_algorithms=retrieved, max_candidates=3)

    assert len(candidates) == 3
    assert candidates[0]["source"]["type"] == "accepted_algorithm"


@pytest.mark.parametrize("limit", [1, 2, 3])
def test_candidate_budget_keeps_new_revision_and_respects_limit(limit):
    candidates = [
        {"name": f"revision_{index}", "source": {"type": "qwen"}}
        for index in range(3)
    ] + [{"name": "baseline", "source": {"type": "previous_iteration"}}]
    selected = _fit_candidate_budget(candidates, limit)
    assert len(selected) == limit
    assert selected[0]["name"] == "revision_0"
    if limit > 1:
        assert selected[-1]["name"] == f"revision_{limit - 1}"


@pytest.mark.parametrize("calibration_limit", [0, 1])
def test_executor_enforces_external_plan_and_calibration_budgets(tmp_path, calibration_limit):
    image = np.zeros((32, 32), dtype=np.uint8)
    image[10:18, 10:18] = 255
    source = tmp_path / "image.png"
    Image.fromarray(image).save(source)
    candidates = [{
        "name": f"candidate_{index}",
        "pipeline": {"steps": [
            {"id": "residual", "op": "local_background_residual", "input": "image", "params": {"sigma": index + 2}},
            {"id": "mask", "op": "global_threshold", "input": "residual", "params": {}},
            {"id": "filtered", "op": "filter_components", "input": "mask", "params": {"min_area": 1}},
        ]},
    } for index in range(3)]
    state = run_planned_agent(
        source, "bright defect", {"recommended_strategy": {}},
        output_root=tmp_path / "outputs", planned_candidates=candidates,
        max_candidates=1, max_calibration_candidates=calibration_limit,
        ground_truth_mask_path=source,
    )
    assert len(state["candidate_attempts"]) == 1 + calibration_limit
    budget = json.loads((Path(state["run_dir"]) / "iteration_0/candidate_budget.json").read_text())
    assert budget["planned_count"] == 1 + calibration_limit


def test_incompatible_replay_does_not_abort_other_candidates(tmp_path):
    from core.pipelines.dsl import pin_pipeline_operator_versions

    image = np.zeros((32, 32), dtype=np.uint8)
    image[10:18, 10:18] = 255
    source = tmp_path / "image.png"
    Image.fromarray(image).save(source)
    valid = strategy_to_pipeline({"segmentation": {"min_area_px": 1}})
    incompatible = pin_pipeline_operator_versions(valid)
    incompatible["operator_versions"]["normalize"] = "999.0.0"
    state = run_planned_agent(
        source, "bright defect", {"recommended_strategy": {}},
        output_root=tmp_path / "outputs", max_candidates=2,
        planned_candidates=[{"name": "old", "pipeline": incompatible}, {"name": "new", "pipeline": valid}],
    )
    assert state["selected_candidate"] == "new"
    assert "version mismatch" in state["candidate_attempts"][0]["quality"]["error"]


def test_apply_user_constraints_does_not_union_an_unreviewed_previous_prediction(tmp_path):
    previous = np.zeros((5, 5), dtype=np.uint8)
    previous[1, 1] = 255
    previous_path = tmp_path / "previous.png"
    Image.fromarray(previous).save(previous_path)

    predicted = np.zeros((5, 5), dtype=bool)
    predicted[3, 3] = True
    constrained, report = apply_user_constraints(
        predicted,
        {"predicted_mask_path": str(previous_path)},
    )

    assert not constrained[1, 1]
    assert constrained[3, 3]
    assert report == {}


def test_agent_loop_executes_candidates_selects_best_and_saves_replay_artifacts(tmp_path):
    source = tmp_path / "sample.png"
    image = np.full((60, 60), 20, dtype=np.uint8)
    image[10:50, 20:28] = 220
    Image.fromarray(image, mode="L").save(source)
    strategy = normalize_strategy({
        "segmentation": {
            "method": "bright_threshold",
            "sensitivity": 1.0,
            "min_area_px": 10,
            "morphology": "close",
        }
    })
    invalid = {"name": "invalid", "steps": [{"id": "bad", "op": "python", "input": "image"}]}
    understanding = {
        "recommended_strategy": strategy,
        "candidate_pipelines": [
            {"name": "invalid", "hypothesis": "must be rejected", "pipeline": invalid},
            {"name": "elongated", "hypothesis": "bright elongated target", "pipeline": strategy_to_pipeline(strategy)},
        ],
        "target_constraints": {"expected_shape": "elongated"},
        "rendering": {"contour_color": "#39FF14", "contour_thickness": 2},
    }

    state = run_planned_agent(source, "提取绿色长条轮廓", understanding, tmp_path / "outputs", max_candidates=2)

    assert state["selected_candidate"] == "elongated"
    assert state["candidate_attempts"][0]["status"] == "failed"
    assert state["quality_report"]["component_count"] == 1
    assert "status" not in state["quality_report"]
    assert "score" not in state["quality_report"]
    assert all("score" not in attempt["quality"] for attempt in state["candidate_attempts"])
    iteration = Path(state["run_dir"]) / "iteration_0"
    for filename in (
        "pipeline.json",
        "operator_trace.json",
        "quality_report.json",
        "contours.json",
        "candidate_summary.json",
        "mask.png",
        "result_annotated.png",
        "graph_state.json",
        "runtime_environment.json",
        "candidate_budget.json",
    ):
        assert (iteration / filename).exists()
    saved = json.loads((iteration / "graph_state.json").read_text(encoding="utf-8"))
    assert saved["rendering"]["contour_color"] == "#39FF14"
    assert "score" not in saved["quality_report"]
    contours = json.loads((iteration / "contours.json").read_text(encoding="utf-8"))
    assert contours["count"] == 1
    assert contours["image_shape"] == [60, 60]
    assert state["contours_path"] == str(iteration / "contours.json")


def test_agent_loop_uses_deterministic_fallback_when_candidates_are_missing(tmp_path):
    source = tmp_path / "sample.png"
    Image.fromarray(np.zeros((20, 20), dtype=np.uint8), mode="L").save(source)
    strategy = normalize_strategy({})

    state = run_planned_agent(
        source,
        "提取缺陷轮廓",
        {"recommended_strategy": strategy, "candidate_pipelines": [], "rendering": {}},
        tmp_path / "outputs",
        return_failure_state=True,
    )

    assert state["candidate_attempts"]
    assert state["candidate_attempts"][0]["source"]["type"] == "deterministic_fallback"


def test_agent_loop_tracks_parent_iteration_and_previous_pipeline_baseline(tmp_path):
    source = tmp_path / "sample.png"
    image = np.zeros((32, 32), dtype=np.uint8)
    image[10:18, 10:18] = 255
    Image.fromarray(image, mode="L").save(source)
    strategy = normalize_strategy({"segmentation": {"method": "bright_threshold", "min_area_px": 2}})
    understanding = {
        "recommended_strategy": strategy,
        "candidate_pipelines": [{
            "name": "candidate",
            "hypothesis": "test",
            "pipeline": strategy_to_pipeline(strategy),
        }],
        "target_constraints": {},
        "rendering": {},
    }
    first = run_planned_agent(source, "first", understanding, tmp_path / "first")
    second = run_planned_agent(
        source,
        "second",
        understanding,
        tmp_path / "second",
        previous_state=first,
    )

    assert second["iteration"] == 1
    assert second["parent_iteration"] == 0
    assert second["parent_result_image_path"] == first["annotated_image_path"]
    assert second["candidate_attempts"][0]["name"] == "candidate"
    assert len(second["candidate_attempts"]) == 1


def test_agent_loop_keeps_empty_candidate_for_task_review(tmp_path):
    source = tmp_path / "sample.png"
    image = np.zeros((32, 32), dtype=np.uint8)
    image[10:18, 10:18] = 255
    Image.fromarray(image, mode="L").save(source)
    strategy = normalize_strategy({
        "segmentation": {"method": "bright_threshold", "sensitivity": 1.0, "min_area_px": 2}
    })
    empty_pipeline = {
        "name": "empty",
        "steps": [{
            "id": "final_mask",
            "op": "global_threshold",
            "input": "image",
            "params": {"polarity": "bright", "sensitivity": 10.0, "max_coverage": 0.9},
        }],
    }
    understanding = {
        "recommended_strategy": strategy,
        "candidate_pipelines": [
            {"name": "empty", "pipeline": empty_pipeline},
            {"name": "nonempty", "pipeline": strategy_to_pipeline(strategy)},
        ],
        "rendering": {},
    }

    state = run_planned_agent(source, "标注亮色区域", understanding, tmp_path / "outputs", max_candidates=2)

    assert state["candidate_attempts"][0]["status"] == "selected_for_review"
    assert state["candidate_attempts"][0]["measurements"]["summary"]["count"] == 0
    assert any(item["measurements"]["summary"]["count"] == 1 for item in state["candidate_attempts"])


def test_agent_loop_returns_empty_result_for_review(tmp_path):
    source = tmp_path / "sample.png"
    image = np.zeros((32, 32), dtype=np.uint8)
    image[10:18, 10:18] = 255
    Image.fromarray(image, mode="L").save(source)
    strategy = normalize_strategy({
        "segmentation": {"method": "bright_threshold", "sensitivity": 1.0, "min_area_px": 2}
    })
    empty_pipeline = {
        "name": "empty",
        "steps": [{
            "id": "final_mask",
            "op": "global_threshold",
            "input": "image",
            "params": {"polarity": "bright", "sensitivity": 10.0, "max_coverage": 0.9},
        }],
    }
    understanding = {
        "recommended_strategy": strategy,
        "candidate_pipelines": [{"name": "empty", "pipeline": empty_pipeline}],
        "rendering": {},
    }

    state = run_planned_agent(
        source,
        "标注亮色区域",
        understanding,
        tmp_path / "outputs",
        max_candidates=1,
        return_failure_state=True,
    )

    assert state["status"] == "ok"
    assert state["agent_status"] == "waiting_for_acceptance"
    assert state["candidate_attempts"][0]["status"] == "selected_for_review"
    assert state["candidate_attempts"][0]["operator_trace"][0]["operator"] == "global_threshold"
    assert "empty_mask" in state["quality_report"]["health"]["diagnostics"]


def test_agent_loop_retains_large_mask_for_visual_review(tmp_path):
    source = tmp_path / "sample.png"
    image = np.zeros((32, 32), dtype=np.uint8)
    image[0:8, 0:8] = 255
    Image.fromarray(image, mode="L").save(source)
    strategy = normalize_strategy({"segmentation": {"method": "dark_threshold", "min_area_px": 1}})
    pipeline = {
        "name": "all_pixels",
        "steps": [{
            "id": "final_mask", "op": "global_threshold", "input": "image",
            "params": {"polarity": "dark", "sensitivity": 0.0, "max_coverage": None},
        }],
    }

    state = run_planned_agent(
        source,
        "标注缺陷",
        {"recommended_strategy": strategy, "candidate_pipelines": [{"name": "all", "pipeline": pipeline}], "rendering": {}},
        tmp_path / "outputs",
        max_candidates=1,
        return_failure_state=True,
    )

    attempt = state["candidate_attempts"][0]
    assert attempt["status"] == "selected_for_review"
    assert "coverage_too_large" in attempt["quality"]["health"]["issues"]
    assert state["quality_report"]["health"]["usable_for_review"]


def test_agent_loop_replays_retrieved_algorithm_before_qwen_candidates(tmp_path):
    source = tmp_path / "sample.png"
    image = np.zeros((32, 32), dtype=np.uint8)
    image[10:18, 10:18] = 255
    Image.fromarray(image, mode="L").save(source)
    generated_strategy = normalize_strategy({
        "segmentation": {"method": "bright_threshold", "sensitivity": 1.0, "min_area_px": 2}
    })
    historical_strategy = normalize_strategy({
        "segmentation": {"method": "bright_threshold", "sensitivity": 2.0, "min_area_px": 2}
    })
    understanding = {
        "recommended_strategy": generated_strategy,
        "candidate_pipelines": [{
            "name": "qwen_candidate",
            "hypothesis": "generated",
            "pipeline": strategy_to_pipeline(generated_strategy),
        }],
        "target_constraints": {},
        "rendering": {},
    }
    retrieved = [{
        "algorithm_id": "algorithm_1",
        "name": "accepted_particle",
        "score": 0.9,
        "match_reasons": ["defect_type", "background_type"],
        "source_task_id": "task_old",
        "path": "algorithms/algorithm_1/algorithm.json",
        "pipeline": strategy_to_pipeline(historical_strategy),
    }]

    state = run_planned_agent(
        source,
        "particle",
        understanding,
        tmp_path / "outputs",
        retrieved_algorithms=retrieved,
        max_candidates=2,
    )

    assert state["candidate_attempts"][0]["name"] == "accepted::accepted_particle"
    assert state["candidate_attempts"][0]["source"]["type"] == "accepted_algorithm"
    assert state["candidate_attempts"][1]["name"] == "qwen_candidate"
    assert state["candidate_attempts"][1]["source"]["type"] == "qwen"
    assert state["retrieved_algorithms"][0]["algorithm_id"] == "algorithm_1"


def test_feedback_quality_records_whether_user_corrections_were_satisfied(tmp_path):
    red = np.zeros((8, 8), dtype=np.uint8)
    red[1:3, 1:3] = 255
    green = np.zeros((8, 8), dtype=np.uint8)
    green[5:7, 5:7] = 255
    red_path = tmp_path / "red.png"
    green_path = tmp_path / "green.png"
    Image.fromarray(red, mode="L").save(red_path)
    Image.fromarray(green, mode="L").save(green_path)
    predicted = np.zeros((8, 8), dtype=bool)
    predicted[5:7, 5:7] = True

    quality = _apply_feedback_quality(
        {},
        predicted,
        {
            "false_positive_mask_path": str(red_path),
            "false_negative_mask_path": str(green_path),
        },
    )

    assert quality["false_positive_remaining"] == 0.0
    assert quality["false_negative_recovered"] == 1.0
    assert "status" not in quality
    assert "score" not in quality


def test_pipeline_diff_records_parameter_changes():
    before = {"name": "before", "steps": [{"id": "m", "op": "morphology", "params": {"radius": 1}}]}
    after = {"name": "after", "steps": [{"id": "m", "op": "morphology", "params": {"radius": 3}}]}

    diff = _pipeline_diff(before, after)

    assert diff["status"] == "changed"
    assert diff["changes"] == [{"step": "m", "parameter": "radius", "before": 1, "after": 3}]


def test_ground_truth_calibration_fixes_elliptical_nanohole_sample(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = root / "data/samples/outline/01_elliptical_nanohole_array_sem_b.jpg"
    annotation = root / "data/samples/outline/01_elliptical_nanohole_array_sem_b_fluorescent_contours.jpg"
    image = np.asarray(Image.open(source).convert("L"), dtype=np.float32)
    ground_truth, metadata = extract_ground_truth_mask(annotation, expected_shape=image.shape)
    candidate = {
        "name": "residual",
        "pipeline": {
            "name": "residual",
            "steps": [
                {"id": "residual", "op": "local_background_residual", "input": "image", "params": {"sigma": 10.0, "polarity": "dark"}},
                {"id": "threshold", "op": "statistical_threshold", "input": "residual", "params": {"method": "otsu", "polarity": "bright"}},
                {"id": "fill", "op": "fill_holes", "input": "threshold", "params": {}},
                {"id": "final_mask", "op": "filter_components", "input": "fill", "params": {"min_area": 200, "max_aspect_ratio": 5.0}},
                {"id": "contours", "op": "extract_contours", "input": "final_mask", "params": {}},
            ],
        },
    }

    variants = _ground_truth_calibration_candidates([candidate], ground_truth)
    evaluations = [
        evaluate_prediction(execute_pipeline(image, item["pipeline"]).mask.data, ground_truth)
        for item in variants
    ]
    best = max(evaluations, key=lambda item: item["dice"])

    assert metadata["contour_count"] == 9
    assert best["predicted_count"] == 9
    assert best["dice"] >= 0.85
    assert best["recall"] >= 0.90
    assert best["precision"] >= 0.85
    assert best["boundary_f1"] >= 0.80

    ground_truth_path = tmp_path / "ground_truth.png"
    Image.fromarray(ground_truth.astype(np.uint8) * 255).save(ground_truth_path)
    state = run_planned_agent(
        source,
        "提取所有黑灰色椭圆",
        {
            "recommended_strategy": {},
            "candidate_pipelines": [candidate],
            "target_constraints": {},
            "rendering": {"contour_color": "#39FF14"},
        },
        output_root=tmp_path / "outputs",
        planned_candidates=[candidate],
        ground_truth_mask_path=ground_truth_path,
        max_calibration_candidates=6,
    )

    assert state["measurements"]["summary"]["count"] == 9
    assert state["evaluation_report"]["dice"] >= 0.85
    assert state["evaluation_report"]["boundary_f1"] >= 0.80
