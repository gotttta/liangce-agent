from pathlib import Path
import json

import numpy as np
import pytest
from PIL import Image
from providers.vision import MockVisionProvider

from ui.annotation_app import (
    SHOW_LATEST_RESULT_JS,
    _candidate_progress_events,
    _format_understanding,
    _resolve_task_image,
    build_annotation_app,
    consume_chat_attachment_ui,
    handle_result_action,
    load_latest_chat_task,
    resume_chat_task,
    reset_chat_task,
    run_chat_agent,
    run_chat_agent_stream,
    save_canvas_feedback,
    store_chat_attachment,
    store_ground_truth_annotation,
)
from ui.utils.formatters import format_task_card


def test_understanding_message_does_not_ask_novice_to_confirm_technical_plan():
    message = _format_understanding(
        {
            "task_summary": "提取黑灰色椭圆的轮廓",
            "candidate_plans": [{"name": "threshold", "operators": ["fill_holes"]}],
            "questions": ["是否严格限制长宽比？"],
            "ambiguities": ["是否填充内部空洞？"],
            "rendering": {"contour_color": "#39FF14", "contour_thickness": 2},
        },
        "Qwen",
        1.2,
    )

    assert "我理解的任务" in message
    assert "自动选择识别方法" in message
    assert "CV 算法" not in message
    assert "候选实验计划" not in message
    assert "需要你确认" not in message
    assert "长宽比" not in message
    assert "fill_holes" not in message


def test_candidate_progress_hides_technical_pipeline_details():
    events = _candidate_progress_events({
        "selected_candidate": None,
        "candidate_attempts": [{
            "index": 0,
            "name": "dark_ellipse_contour_extraction",
            "status": "no_annotation",
            "hypothesis": "adaptive threshold and morphology",
            "pipeline": {"steps": [{"op": "adaptive_threshold"}, {"op": "fill_holes"}]},
        }],
    })

    assert events[0]["label"] == "识别方法 1"
    assert events[0]["detail"] == "这次没有找到目标，系统会自动换一种方法。"
    assert "adaptive_threshold" not in events[0]["detail"]


def test_save_canvas_feedback_persists_composite_marks_and_state(tmp_path, monkeypatch):
    monkeypatch.setattr("ui.annotation_app.TASK_ROOT", tmp_path / "tasks")
    task = reset_chat_task()[6]
    layer = np.zeros((12, 16, 4), dtype=np.uint8)
    layer[3:7, 5:10, 0] = 255
    layer[3:7, 5:10, 3] = 255
    green_layer = np.zeros((12, 16, 4), dtype=np.uint8)
    green_layer[8:10, 2:4, 1] = 255
    green_layer[8:10, 2:4, 3] = 255
    composite = np.zeros((12, 16, 4), dtype=np.uint8)
    composite[:, :, 3] = 255
    composite[3:7, 5:10, 0] = 255

    state = save_canvas_feedback(
        {"layers": [layer, green_layer], "composite": composite},
        task,
        {"annotated_image_path": "previous.png", "pipeline": {"name": "old"}},
    )

    assert state["pipeline"] == {"name": "old"}
    assert state["feedback_pixel_count"] == 24
    assert state["false_positive_pixel_count"] == 20
    assert state["false_negative_pixel_count"] == 4
    assert Path(state["feedback_image_path"]).exists()
    assert Path(state["feedback_layer_path"]).exists()
    assert Path(state["false_positive_mask_path"]).exists()
    assert Path(state["false_negative_mask_path"]).exists()
    events = (tmp_path / "tasks" / task["id"] / "events.jsonl").read_text(encoding="utf-8")
    assert "canvas_feedback_saved" in events


def test_save_canvas_feedback_merges_separate_red_and_green_editors(tmp_path, monkeypatch):
    monkeypatch.setattr("ui.annotation_app.TASK_ROOT", tmp_path / "tasks")
    task = reset_chat_task()[6]
    background = np.zeros((8, 10, 4), dtype=np.uint8)
    background[:, :, 3] = 255
    red = np.zeros_like(background)
    red[1:3, 1:4, 0] = 255
    red[1:3, 1:4, 3] = 255
    green = np.zeros_like(background)
    green[5:7, 6:9, 1] = 255
    green[5:7, 6:9, 3] = 255

    state = save_canvas_feedback(
        {"layers": [red], "background": background},
        task,
        {"annotated_image_path": "missing.png"},
        green_editor_value={"layers": [green], "background": background},
    )

    assert state["false_positive_pixel_count"] == 6
    assert state["false_negative_pixel_count"] == 6
    assert state["feedback_pixel_count"] == 12


def test_save_canvas_feedback_accumulates_green_marks_across_rounds(tmp_path, monkeypatch):
    monkeypatch.setattr("ui.annotation_app.TASK_ROOT", tmp_path / "tasks")
    task = reset_chat_task()[6]
    background_path = tmp_path / "background.png"
    Image.new("RGBA", (10, 10), "white").save(background_path)

    first = np.zeros((10, 10, 4), dtype=np.uint8)
    first[1:3, 1:3, 1] = 255
    first[1:3, 1:3, 3] = 255
    state = save_canvas_feedback(
        {"layers": [first], "background": str(background_path)},
        task,
        {"annotated_image_path": str(background_path)},
    )

    second = np.zeros((10, 10, 4), dtype=np.uint8)
    second[7:9, 7:9, 1] = 255
    second[7:9, 7:9, 3] = 255
    state = save_canvas_feedback(
        {"layers": [second], "background": str(background_path)},
        task,
        {"annotated_image_path": str(background_path)},
    )

    assert state["false_negative_pixel_count"] == 8
    accumulated = np.asarray(Image.open(state["include_mask_path"]).convert("L")) > 0
    assert accumulated[1:3, 1:3].all()
    assert accumulated[7:9, 7:9].all()


def test_build_annotation_app_returns_blocks():
    import gradio as gr

    assert isinstance(build_annotation_app(), gr.Blocks)


def test_same_image_handbook_ground_truth_is_stored_for_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr("ui.annotation_app.TASK_ROOT", tmp_path / "tasks")
    source = tmp_path / "source.png"
    annotation = tmp_path / "source_handbook.png"
    image = np.zeros((32, 32), dtype=np.uint8)
    image[10:18, 12:20] = 255
    Image.fromarray(image).save(source)
    handbook = Image.fromarray(np.dstack([image, image, image]))
    from PIL import ImageDraw

    ImageDraw.Draw(handbook).rectangle((12, 10, 19, 17), outline=(255, 0, 0), width=1)
    handbook.save(annotation)

    task = reset_chat_task()[6]
    attachment, _, task = store_chat_attachment(str(source), task)
    stored = store_ground_truth_annotation(str(annotation), task, attachment)
    ground_truth = stored["ground_truth"]

    assert ground_truth["type"] == "same_image_ground_truth"
    assert Path(ground_truth["mask_path"]).is_file()
    assert ground_truth["extraction"]["contour_count"] == 1


def test_task_card_shows_objective_metrics_only_when_ground_truth_exists():
    without_ground_truth = format_task_card({"measurements": {"summary": {}}})
    with_ground_truth = format_task_card({
        "measurements": {"summary": {"count": 1, "total_area": 64}},
        "evaluation_report": {
            "status": "ok",
            "dice": 0.9,
            "recall": 0.8,
            "precision": 0.95,
            "boundary_f1": 0.85,
        },
    })

    assert "同图 Ground Truth" not in without_ground_truth
    assert "同图 Ground Truth" in with_ground_truth
    assert "0.900" in with_ground_truth


def test_feedback_editors_use_separate_fixed_red_and_green_brushes():
    config = build_annotation_app().get_config_file()
    red_editor = next(
        component
        for component in config["components"]
        if component.get("props", {}).get("elem_id") == "annotation-feedback-editor"
    )
    green_editor = next(
        component
        for component in config["components"]
        if component.get("props", {}).get("elem_id") == "annotation-green-feedback-editor"
    )

    assert red_editor["props"]["brush"] == {
        "default_size": 12,
        "colors": ["#ff3b30"],
        "default_color": "#ff3b30",
        "color_mode": "fixed",
    }
    assert green_editor["props"]["brush"] == {
        "default_size": 12,
        "colors": ["#22c55e"],
        "default_color": "#22c55e",
        "color_mode": "fixed",
    }


def test_handbook_uploads_are_split_into_example_and_same_image_ground_truth():
    config = build_annotation_app().get_config_file()
    components_by_elem_id = {
        component.get("props", {}).get("elem_id"): component["id"]
        for component in config["components"]
        if component.get("props", {}).get("elem_id")
    }

    assert "annotation-handbook-examples" in components_by_elem_id
    assert "annotation-ground-truth" in components_by_elem_id

    def find_layout_node(node, component_id):
        if node["id"] == component_id:
            return node
        for child in node.get("children", []):
            if found := find_layout_node(child, component_id):
                return found
        return None

    composer_row = find_layout_node(
        config["layout"],
        components_by_elem_id["annotation-composer-row"],
    )
    composer_child_ids = {child["id"] for child in composer_row["children"]}

    assert components_by_elem_id["annotation-handbook-examples"] in composer_child_ids
    assert components_by_elem_id["annotation-ground-truth"] in composer_child_ids


def test_feedback_event_js_preserves_gradio_input_order():
    config = build_annotation_app().get_config_file()
    signatures = [
        dependency["js"].strip().splitlines()[0]
        for dependency in config["dependencies"]
        if dependency.get("js") and "feedback-open" in dependency["js"]
    ]

    assert signatures.count(
        "(image, prompt, history, task, state, red, green, examples) => {"
    ) == 1
    assert signatures.count(
        "(action, image, history, task, state, prompt) => {"
    ) == 5
    assert signatures.count("() => {") == 1
    assert all("...args" not in signature for signature in signatures)


def test_result_context_scroll_is_wired_to_result_and_restore_events():
    config = build_annotation_app().get_config_file()
    context_scroll_dependencies = [
        dependency
        for dependency in config["dependencies"]
        if (dependency.get("js") or "").strip() == SHOW_LATEST_RESULT_JS.strip()
    ]

    assert len(context_scroll_dependencies) == 6
    assert "querySelectorAll('.task-card')" in SHOW_LATEST_RESULT_JS
    assert "querySelectorAll('.progress-card')" in SHOW_LATEST_RESULT_JS
    assert "targetTop - 12" in SHOW_LATEST_RESULT_JS
    assert "[120, 350, 800]" in SHOW_LATEST_RESULT_JS
    assert "behavior: 'auto'" in SHOW_LATEST_RESULT_JS


def test_chat_agent_runs_local_pipeline_and_offers_follow_up_actions(tmp_path, monkeypatch):
    source = tmp_path / "sample.png"
    image = np.zeros((64, 64), dtype=np.uint8)
    image[20:28, 20:28] = 255
    Image.fromarray(image, mode="L").save(source)

    monkeypatch.setattr("ui.annotation_app.ROOT", tmp_path)
    monkeypatch.setattr("ui.annotation_app.TASK_ROOT", tmp_path / "tasks")
    class FakeQwenProvider:
        model = "qwen-test"

        def understand_task(self, target_image_path, description, previous_context=None):
            understanding = MockVisionProvider().understand_task(target_image_path, description)
            understanding["recommended_strategy"]["segmentation"]["sensitivity"] = 2.25
            understanding["recommended_strategy"]["notes"] = ["Qwen test strategy."]
            return understanding

    monkeypatch.setattr("ui.annotation_app.build_runtime_provider", FakeQwenProvider)

    task = reset_chat_task()[6]
    attachment, label, task = store_chat_attachment(str(source), task)
    result = run_chat_agent(attachment, "提取亮色缺陷轮廓", [], task)

    assert source.name in label
    assert result[0][0]["role"] == "user"
    assert result[0][0]["content"][0] == attachment
    assert result[0][0]["content"][1] == Path(attachment).name
    assert result[0][1] == {"role": "user", "content": "提取亮色缺陷轮廓"}
    assert Path(result[1]["value"]).exists()
    assert Path(result[2]["value"]).exists()
    assert result[1]["visible"] is False
    assert result[2]["visible"] is False
    assert result[9]["visible"] is True
    assert (tmp_path / "tasks" / task["id"] / "nodes" / "understand_task" / "latest.json").exists()
    execution_path = tmp_path / "tasks" / task["id"] / "nodes" / "execute_candidate" / "latest.json"
    assert execution_path.exists()
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    assert execution["inputs"]["provider"] == "Qwen (qwen-test)"
    assert execution["inputs"]["candidate"] == "qwen_recommended_strategy"
    assert result[5]["strategy"]["segmentation"]["sensitivity"] == 2.25
    assert result[5]["strategy"]["notes"] == ["Qwen test strategy."]
    assert any(
        "只看结果是否符合你的描述" in item["content"]
        for item in result[0]
        if isinstance(item["content"], str)
    )
    assert result[0][-2]["content"] == "这是本轮标注结果。点击图片放大，检查标注是否符合你的描述。"
    assert result[0][-1]["content"][0] == result[5]["annotated_image_path"]
    assert result[0][-1]["content"][1] == "点击放大标注结果"

    viewed = run_chat_agent(
        attachment,
        "查看结果图",
        result[0],
        result[6],
        result[5],
    )
    assert viewed[1]["visible"] is True
    assert viewed[2]["visible"] is False
    assert viewed[9]["visible"] is True
    assert viewed[0][-1]["content"].startswith("已按你的要求打开")

    accepted = handle_result_action(
        "accept",
        attachment,
        viewed[0],
        viewed[6],
        None,
    )
    assert accepted[5]["agent_status"] == "accepted"
    assert accepted[6]["status"] == "accepted"
    assert accepted[9]["visible"] is False
    assert (tmp_path / "tasks" / task["id"] / "acceptance" / "latest.json").exists()
    algorithm_path = tmp_path / "tasks" / task["id"] / "acceptance" / "algorithm.json"
    assert algorithm_path.exists()
    assert json.loads(algorithm_path.read_text(encoding="utf-8"))["pipeline"]
    registry_algorithms = list((tmp_path / "algorithms").glob("algorithm_*/algorithm.json"))
    assert len(registry_algorithms) == 1
    assert json.loads(registry_algorithms[0].read_text(encoding="utf-8"))["source_task_id"] == task["id"]
    acceptance = json.loads(
        (tmp_path / "tasks" / task["id"] / "acceptance" / "latest.json").read_text(encoding="utf-8")
    )
    assert "accepted_ground_truth_path" not in acceptance
    assert "accepted_ground_truth_path" not in accepted[5]


def test_result_continue_and_exit_actions_update_conversation_and_task(tmp_path, monkeypatch):
    monkeypatch.setattr("ui.annotation_app.TASK_ROOT", tmp_path / "tasks")
    task = reset_chat_task()[6]
    state = {
        "annotated_image_path": "result.png",
        "predicted_mask_path": "mask.png",
        "measurements": {"results": []},
    }

    continued = handle_result_action("continue", "sample.png", [], task, state)

    assert continued[5]["agent_status"] == "waiting_for_feedback"
    assert continued[9]["visible"] is False
    assert continued[10]["visible"] is True
    assert continued[11]["value"] == {
        "background": "result.png",
        "layers": [],
        "composite": "result.png",
    }
    assert continued[0] == []
    conversation_path = tmp_path / "tasks" / task["id"] / "conversation.jsonl"
    assert not conversation_path.exists()
    assert continued[7]["placeholder"].startswith("描述需要修改")

    exited = handle_result_action("exit", "sample.png", continued[0], task, state)

    assert exited[5]["agent_status"] == "exited"
    assert exited[6]["status"] == "exited"
    assert "已退出当前任务" in exited[0][-1]["content"]


def test_continue_action_works_when_gradio_agent_state_is_temporarily_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("ui.annotation_app.TASK_ROOT", tmp_path / "tasks")
    task = reset_chat_task()[6]

    continued = handle_result_action("continue", "sample.png", [], task, None)

    assert continued[5]["agent_status"] == "waiting_for_feedback"
    assert continued[9]["visible"] is False
    assert continued[0] == []


def test_stream_restores_task_image_after_attachment_state_is_cleared(tmp_path, monkeypatch):
    sample = tmp_path / "task-sample.png"
    sample.write_bytes(b"sample")
    final_result = (
        [{"role": "assistant", "content": "完成"}],
        *tuple(range(1, 12)),
    )
    received = {}

    def fake_run(image_path, *args, **kwargs):
        received["image_path"] = image_path
        kwargs["progress_callback"]({
            "stage": "prepare",
            "label": "准备输入",
            "status": "completed",
        })
        return final_result

    monkeypatch.setattr("ui.annotation_app.run_chat_agent", fake_run)

    updates = list(run_chat_agent_stream(
        None,
        "请根据画布标记修正标注",
        [],
        {"id": "task", "samples": [{"path": str(sample)}]},
    ))

    assert received["image_path"] == str(sample)
    assert len(updates) >= 3
    assert "正在处理图片" in updates[0][0][-1]["content"]
    assert "准备输入" in updates[-1][0][1]["content"]


def test_stream_places_processing_state_in_assistant_message(monkeypatch):
    final_result = (
        [{"role": "assistant", "content": "完成"}],
        *tuple(range(1, 12)),
    )

    def fake_run(*args, **kwargs):
        progress = kwargs["progress_callback"]
        progress({"stage": "prepare", "label": "准备输入", "status": "completed"})
        progress({"stage": "understand_task", "label": "视觉理解", "status": "running"})
        progress({
            "stage": "understand_task",
            "label": "视觉理解",
            "status": "completed",
            "duration_seconds": 1.25,
        })
        return final_result

    monkeypatch.setattr("ui.annotation_app.run_chat_agent", fake_run)

    updates = list(run_chat_agent_stream("sample.png", "提取缺陷", [], {"id": "task"}))

    assert updates[0][0][0] == {"role": "user", "content": "提取缺陷"}
    assert "Agent 执行过程" in updates[0][0][1]["content"]
    assert "准备输入" in updates[1][0][-1]["content"]
    assert "视觉理解" in updates[-2][0][-1]["content"]
    assert "1.25s" in updates[-1][0][1]["content"]
    assert updates[-1][0][0] == {"role": "assistant", "content": "完成"}


def test_stream_converts_failures_to_assistant_message(monkeypatch):
    def fail(*args):
        raise RuntimeError("candidate failed")

    monkeypatch.setattr("ui.annotation_app.run_chat_agent", fail)

    updates = list(run_chat_agent_stream("sample.png", "提取缺陷", [], None))

    assert "这次没有处理成功" in updates[-1][0][-1]["content"]
    assert "candidate failed" not in updates[-1][0][-1]["content"]


def test_chat_agent_reports_qwen_failure_without_mock_fallback(tmp_path, monkeypatch):
    source = tmp_path / "sample.png"
    Image.fromarray(np.zeros((16, 16), dtype=np.uint8), mode="L").save(source)
    monkeypatch.setattr("ui.annotation_app.TASK_ROOT", tmp_path / "tasks")
    monkeypatch.setattr(
        "ui.annotation_app.build_runtime_provider",
        lambda: (_ for _ in ()).throw(ConnectionError("offline")),
    )

    task = reset_chat_task()[6]
    attachment, _, task = store_chat_attachment(str(source), task)

    with pytest.raises(Exception, match="Qwen视觉理解失败"):
        run_chat_agent(attachment, "提取亮色轮廓", [], task)

    record = json.loads(
        (tmp_path / "tasks" / task["id"] / "nodes" / "understand_task" / "latest.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["status"] == "failed"
    assert record["inputs"]["provider"] == "qwen"
    assert "offline" in record["error"]


def test_reset_chat_task_clears_conversation_attachment_and_results():
    result = reset_chat_task()

    assert result[0] == [
        {
            "role": "assistant",
            "content": "新任务已创建。请点击 + 上传图片，然后描述希望 Agent 标注的目标或效果。",
        }
    ]
    assert result[1]["visible"] is False
    assert result[2]["visible"] is False
    assert result[3]["visible"] is False
    assert result[5] is None
    assert result[6]["id"].startswith("task_")
    assert result[7] is None
    assert result[8] == ""
    assert result[9] == ""
    assert result[10]["visible"] is False


def test_consuming_attachment_clears_only_the_composer(tmp_path):
    attachment = tmp_path / "sample.png"
    attachment.write_bytes(b"sample")

    result = consume_chat_attachment_ui()

    assert result[0] is None
    assert result[1] == ""
    assert result[2]["visible"] is False
    assert result[3]["visible"] is False
    assert result[4]["visible"] is False
    assert attachment.exists()


def test_follow_up_resolves_original_image_after_composer_is_cleared(tmp_path):
    sample = tmp_path / "task-sample.png"
    sample.write_bytes(b"sample")

    resolved_from_state = _resolve_task_image(
        None,
        {"samples": []},
        {"target_image_path": str(sample)},
    )
    resolved_from_task = _resolve_task_image(
        None,
        {"samples": [{"path": str(sample)}]},
        None,
    )

    assert resolved_from_state == str(sample)
    assert resolved_from_task == str(sample)


def test_resume_chat_task_restores_messages_sample_and_latest_state(tmp_path, monkeypatch):
    monkeypatch.setattr("ui.annotation_app.TASK_ROOT", tmp_path / "tasks")
    task = reset_chat_task()[6]
    source = tmp_path / "sample.png"
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L").save(source)
    attachment, _, task = store_chat_attachment(str(source), task)
    result_dir = tmp_path / "result" / "iteration_2"
    result_dir.mkdir(parents=True)
    annotated = result_dir / "result_annotated.png"
    mask = result_dir / "mask.png"
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L").save(annotated)
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L").save(mask)
    state = {
        "iteration": 2,
        "annotated_image_path": str(annotated),
        "predicted_mask_path": str(mask),
        "measurements": {"results": []},
        "pipeline": {"name": "restored", "steps": []},
    }
    (result_dir / "graph_state.json").write_text(json.dumps(state), encoding="utf-8")
    from core.task_store import TaskStore
    store = TaskStore(tmp_path / "tasks")
    store.append_message(task["id"], "user", "继续优化")
    store.save_node_result(
        task["id"],
        "execute_candidate",
        {},
        {"annotated_image_path": str(annotated), "predicted_mask_path": str(mask)},
        0.1,
    )

    restored = resume_chat_task(task["id"])

    assert restored[0][-1]["content"] == "继续优化"
    assert restored[5]["iteration"] == 2
    assert restored[7] is None
    assert restored[9] == ""
    assert restored[10]["visible"] is True
    assert restored[15]["value"] == task["id"]


def test_resume_draft_restores_an_unsent_attachment(tmp_path, monkeypatch):
    monkeypatch.setattr("ui.annotation_app.TASK_ROOT", tmp_path / "tasks")
    task = reset_chat_task()[6]
    source = tmp_path / "sample.png"
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L").save(source)
    attachment, _, task = store_chat_attachment(str(source), task)

    restored = resume_chat_task(task["id"])

    assert restored[7] == attachment
    assert source.name in restored[9]


def test_load_latest_chat_task_skips_newer_empty_drafts(tmp_path, monkeypatch):
    monkeypatch.setattr("ui.annotation_app.TASK_ROOT", tmp_path / "tasks")
    meaningful = reset_chat_task()[6]
    source = tmp_path / "sample.png"
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8), mode="L").save(source)
    _, _, meaningful = store_chat_attachment(str(source), meaningful)
    empty = reset_chat_task()[6]

    restored = load_latest_chat_task()

    assert restored[6]["id"] == meaningful["id"]
    assert restored[6]["id"] != empty["id"]
