from core.algorithm_registry import AlgorithmRegistry
from core.pipelines.dsl import strategy_to_pipeline


def _accepted_state(sensitivity=1.8):
    strategy = {
        "defect_type": "particle",
        "measurement_type": "area_count",
        "visual_observation": {
            "background_pattern": "periodic_lines",
            "polarity": "bright_on_dark",
        },
        "segmentation": {
            "method": "bright_threshold",
            "sensitivity": sensitivity,
            "min_area_px": 20,
            "morphology": "close",
        },
    }
    return {
        "selected_candidate": "periodic_particle",
        "description": "周期线条背景中的亮色颗粒",
        "strategy": strategy,
        "pipeline": strategy_to_pipeline(strategy),
        "quality_report": {},
        "measurements": {"summary": {"count": 1, "total_area": 80, "unit": "pixel"}},
    }


def test_registry_publishes_and_retrieves_matching_accepted_algorithm(tmp_path):
    registry = AlgorithmRegistry(tmp_path / "algorithms")
    published = registry.publish("task_source", _accepted_state())

    matches = registry.search({
        "task_summary": "提取周期背景中的颗粒",
        "target_defect": "particle",
        "normal_context": "periodic_lines",
        "recommended_strategy": {
            "defect_type": "particle",
            "measurement_type": "area_count",
            "visual_observation": {
                "background_pattern": "periodic_lines",
                "polarity": "bright_on_dark",
            },
        },
    })

    assert published["path"].endswith("algorithm.json")
    assert matches[0]["algorithm_id"] == published["id"]
    assert matches[0]["score"] >= 0.9
    assert "defect_type" in matches[0]["match_reasons"]
    assert matches[0]["pipeline"]["steps"]


def test_registry_does_not_return_unrelated_algorithm(tmp_path):
    registry = AlgorithmRegistry(tmp_path / "algorithms")
    registry.publish("task_source", _accepted_state())

    matches = registry.search({
        "task_summary": "检测暗色断线",
        "recommended_strategy": {
            "defect_type": "open_line",
            "measurement_type": "gap_width",
            "visual_observation": {
                "background_pattern": "random_texture",
                "polarity": "dark_on_bright",
            },
        },
    })

    assert matches == []


def test_registry_reuses_accepted_generated_operator_with_pipeline(tmp_path):
    registry = AlgorithmRegistry(tmp_path / "algorithms")
    state = _accepted_state()
    state["pipeline"] = {
        "name": "custom_pipeline",
        "generated_operators": [{
            "name": "texture_mask",
            "input_artifact": "ImageArtifact",
            "output_artifact": "MaskArtifact",
            "source": "def apply(data, params):\n    return data > np.mean(data)",
        }],
        "steps": [{"id": "final_mask", "op": "texture_mask", "input": "image", "params": {}}],
    }
    published = registry.publish("task_source", state)

    matches = registry.search({
        "target_defect": "particle",
        "recommended_strategy": state["strategy"],
    })

    assert matches[0]["pipeline"]["generated_operators"][0]["name"] == "texture_mask"
    assert published["pipeline"]["generated_operators"]
