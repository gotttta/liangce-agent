import json
from pathlib import Path

import numpy as np
from PIL import Image

from core.agent_loop import run_planned_agent, _ground_truth_calibration_candidates, select_best_candidate
from core.measurement.evaluation import meets_ground_truth_gate
from core.pipelines.dsl import execute_pipeline


def test_passing_replay_stops_before_other_candidates_and_calibration(tmp_path, monkeypatch):
    image = np.zeros((32, 32), dtype=np.uint8)
    image[10:18, 10:18] = 255
    path = tmp_path / "image.png"
    Image.fromarray(image).save(path)
    pipeline = {"steps": [{"id": "mask", "op": "global_threshold", "input": "image",
                           "params": {"polarity": "bright", "sensitivity": 1}}]}
    def unexpected_calibration(*args, **kwargs):
        raise AssertionError("Passing results must not trigger calibration")
    monkeypatch.setattr("core.agent_loop._ground_truth_calibration_candidates", unexpected_calibration)
    state = run_planned_agent(path, "particle", {}, output_root=tmp_path / "out",
        planned_candidates=[
            {"name": "invalid", "pipeline": {"steps": []}},
            {"name": "accepted", "pipeline": pipeline, "source": {"type": "accepted_algorithm"}},
        ], ground_truth_mask_path=path)
    assert len(state["candidate_attempts"]) == 1
    assert state["selected_candidate"] == "accepted"
    assert meets_ground_truth_gate(state["evaluation_report"])
    budget = json.loads((Path(state["run_dir"]) / "iteration_0/candidate_budget.json").read_text())
    assert budget["stopped_on_gate"] is True
    assert budget["attempted_count"] == 1


def test_recalibration_reuses_dilation_step_without_duplicate_ids():
    image = np.zeros((32, 32), dtype=np.float32)
    image[10:18, 10:18] = 255
    base = {"name": "calibrated", "pipeline": {"steps": [
        {"id": "residual", "op": "local_background_residual", "input": "image", "params": {"sigma": 3}},
        {"id": "mask", "op": "global_threshold", "input": "residual", "params": {}},
        {"id": "gt_calibration_dilate", "op": "morphology", "input": "mask", "params": {"method": "dilate", "radius": 2}},
        {"id": "filtered", "op": "filter_components", "input": "gt_calibration_dilate", "params": {"min_area": 1}},
    ]}}
    variants = _ground_truth_calibration_candidates([base], image > 0)
    assert variants
    for variant in variants:
        steps = variant["pipeline"]["steps"]
        assert len({s["id"] for s in steps}) == len(steps)
        assert len([s for s in steps if s["op"] == "morphology"]) == 1
        execute_pipeline(image, variant["pipeline"])
    assert base["pipeline"]["steps"][2]["params"]["radius"] == 2


def test_selection_prefers_all_metrics_passing_over_higher_dice():
    good = {"quality": {"evaluation": {"status": "ok", "dice": .9, "recall": .95, "precision": .9, "boundary_f1": .9}}}
    bad = {"quality": {"evaluation": {"status": "ok", "dice": .95, "recall": .89, "precision": .99, "boundary_f1": .99}}}
    assert select_best_candidate([bad, good], prefer_evaluation=True) is good
