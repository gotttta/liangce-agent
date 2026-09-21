import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from core.agent_loop import run_planned_agent
from core.experiments.artifacts import export_artifacts, record_experiments
from core.experiments.runner import _serializable_attempt, _write_json
from core.operators import ContourArtifact, ImageArtifact, MaskArtifact, MetadataArtifact
from core.pipelines.dsl import execute_pipeline
from core.tools.experiments import ExperimentTools


@pytest.fixture
def source(tmp_path):
    pixels = np.zeros((30, 30), dtype=np.uint8)
    pixels[10:18, 10:18] = 255
    path = tmp_path / "source.png"
    Image.fromarray(pixels).save(path)
    return path


def output_pipeline(kind="mask", sensitivity=1.0):
    node = {"id": "result", "operator": "global_threshold",
            "inputs": {"image": "$image"}, "params": {"sensitivity": sensitivity}}
    if kind == "image":
        node.update(operator="normalize", params={})
    return {"schema_version": 3, "name": kind, "nodes": [node],
            "outputs": {kind: "result"}}


def test_record_experiments_preserves_numpy_outputs_and_scalar_types(source, tmp_path):
    mask = MaskArtifact(np.eye(3, dtype=bool), {"count": np.int64(3)})
    image = ImageArtifact(np.arange(9).reshape(3, 3), {"scale": np.float32(0.5)})
    contours = ContourArtifact((np.array([[0, 0], [1, 1]]),), (3, 3),
                              {"closed": np.bool_(True)})
    outputs = export_artifacts({"mask": mask, "image": image, "contours": contours,
                                "stats": MetadataArtifact({"count": 3})}, final=True)
    attempts = []
    for index, values in enumerate(({}, outputs, outputs)):
        directory = tmp_path / f"candidate_{index}"
        directory.mkdir()
        attempts.append({"index": index, "name": f"candidate_{index}", "hypothesis": "test",
                         "pipeline": {}, "status": "selected_for_review", "directory": str(directory),
                         "quality": {"outputs": values},
                         "measurements": {"structured_outputs": values},
                         "execution": SimpleNamespace(trace=({"count": np.int64(3)},))})

    record_experiments(attempts, source, {}, 0)

    for attempt in attempts:
        path = Path(attempt["directory"]) / "experiment.json"
        record = json.loads(path.read_text())
        assert record["execution_status"] == "completed"
        assert record["experiment_id"] == attempt["experiment_id"]
        assert record["operator_trace"][0]["count"] == 3
        assert "execution" not in record
        assert not path.with_suffix(".json.tmp").exists()
        public = json.loads(json.dumps(_serializable_attempt(attempt)))
        assert public["quality"] == record["quality"]
        report = path.parent / "quality_report.json"
        _write_json(report, attempt["quality"])
        assert json.loads(report.read_text()) == record["quality"]
        if attempt["index"]:
            saved = record["quality"]["outputs"]
            assert saved["mask"]["data"] == mask.data.tolist()
            assert saved["image"]["data"] == image.data.tolist()
            assert saved["mask"]["metadata"]["count"] == 3
            assert saved["image"]["metadata"]["scale"] == 0.5
            assert saved["contours"]["metadata"]["closed"] is True
            assert record["measurements"]["structured_outputs"] == saved
    # Persistence must leave live artifacts available to numerical consumers.
    assert outputs["mask"]["data"] is mask.data


def test_mixed_candidates_reach_review_with_serializable_state(source, tmp_path, monkeypatch):
    # Execute only these trusted built-in fixtures locally; production uses Docker.
    monkeypatch.setattr("core.experiments.runner.execute_pipeline_sandbox", execute_pipeline)
    legacy = {"name": "legacy", "steps": [{"id": "mask", "op": "global_threshold",
                                           "input": "image", "params": {"sensitivity": 1.0}}]}
    candidates = [{"name": "legacy", "pipeline": legacy},
                  {"name": "first", "pipeline": output_pipeline(sensitivity=0.8)},
                  {"name": "second", "pipeline": output_pipeline(sensitivity=1.2)}]
    state = run_planned_agent(source, "bright particles", {}, output_root=tmp_path / "outputs",
                              planned_candidates=candidates, max_candidates=3)

    assert state["agent_status"] == "waiting_for_acceptance"
    assert len(state["candidate_attempts"]) == 3
    serialized = json.loads(json.dumps(state))
    iteration = Path(state["run_dir"]) / "iteration_0"
    assert json.loads((iteration / "graph_state.json").read_text()) == serialized
    for attempt in state["candidate_attempts"]:
        directory = Path(attempt["directory"])
        record = json.loads((directory / "experiment.json").read_text())
        assert record["quality"] == attempt["quality"]
        assert record["execution_status"] == "completed"
        assert (directory / "mask.png").is_file()
        assert (directory / "result_annotation.png").is_file()
    reference = state["candidate_attempts"][1]["quality"]["outputs"]["mask"]
    assert 'data' not in reference
    assert Path(reference['path']).is_file()
    assert reference['stage'] == 'final' 


@pytest.mark.parametrize("kind", ["mask", "image"])
def test_experiment_tool_outputs_are_json_serializable(source, tmp_path, monkeypatch, kind):
    monkeypatch.setattr("core.experiments.runner.execute_pipeline_sandbox", execute_pipeline)
    session = ExperimentTools(source, "inspect result", output_root=tmp_path / "tools")
    result, images = session.dispatch({"tool": "execute_pipeline", "arguments": {
        "pipeline": output_pipeline(kind)}}, None)

    assert result["status"] == "success", result
    assert Path(images[0]).is_file()
    view = json.loads(json.dumps(result))["data"]["facts"]["outputs"][kind]
    assert view["shape"] == [30, 30]
    assert "data" not in view
    attempt = session.attempts[result["data"]["experiment_id"]]
    data = json.loads((Path(attempt["directory"]) / "outputs.json").read_text())[kind]["data"]
    assert np.asarray(data).shape == (30, 30)
    json.dumps(session.context)
