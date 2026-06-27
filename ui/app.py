from pathlib import Path
import os

import gradio as gr

from graph_workflow import run_graph
from providers.vision import MockVisionProvider
from ui.gradio_adapters import (
    measurement_rows,
    output_files,
    save_feedback_layer,
    state_to_chat_messages,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs"


def run_initial(target_image, description, reference_annotation, unit):
    if not target_image:
        raise gr.Error("请先上传原图。")
    if not description:
        raise gr.Error("请写一句缺陷描述。")

    state = run_graph(
        target_image_path=target_image,
        description=description,
        reference_annotation_path=reference_annotation,
        output_root=OUTPUT_ROOT,
        provider=MockVisionProvider(),
        unit=unit or "pixel",
    )
    return gradio_outputs(state)


def run_feedback(app_state, feedback_editor, feedback_text):
    if not app_state:
        raise gr.Error("请先完成一轮初始量测。")
    if not feedback_text:
        raise gr.Error("请写一句反馈，例如“这里漏了”。")

    feedback_brush_path = save_feedback_layer(feedback_editor, app_state["run_dir"])
    next_iteration = app_state.get("iteration", 0) + 1
    state = run_graph(
        target_image_path=app_state["target_image_path"],
        description=app_state["description"],
        reference_annotation_path=app_state.get("reference_annotation_path"),
        output_root=OUTPUT_ROOT,
        existing_run_dir=app_state["run_dir"],
        initial_iteration=next_iteration,
        feedback_brush_path=feedback_brush_path,
        feedback_text=feedback_text,
        provider=MockVisionProvider(),
        unit=app_state.get("unit", "pixel"),
        max_iterations=next_iteration + 1,
    )
    return gradio_outputs(state)


def gradio_outputs(state):
    return (
        state_to_chat_messages(state),
        state.get("annotated_image_path"),
        state.get("predicted_mask_path"),
        state.get("annotated_image_path"),
        measurement_rows(state),
        state.get("strategy", {}),
        state.get("metrics", {}),
        output_files(state),
        state,
    )


def build_app():
    with gr.Blocks(title="DRAM 缺陷量测 Agent", theme=gr.themes.Soft()) as app:
        gr.Markdown("# DRAM 缺陷量测 Agent")
        gr.Markdown("像和 Codex 协作一样描述缺陷，Agent 会生成策略、标注缺陷，并支持画笔反馈重跑。")

        app_state = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=4):
                chat = gr.Chatbot(
                    label="Agent 对话",
                    type="messages",
                    height=520,
                )
                description = gr.Textbox(
                    label="缺陷描述 / 反馈描述",
                    placeholder="例如：找亮色残留，量面积和数量",
                    lines=3,
                )
                with gr.Row():
                    run_button = gr.Button("开始量测", variant="primary")
                    feedback_button = gr.Button("应用画笔反馈并重跑")

            with gr.Column(scale=5):
                target_image = gr.Image(
                    label="原图 target（必填）",
                    type="filepath",
                )
                reference_annotation = gr.Image(
                    label="参考标注图（可选，同图人工标注）",
                    type="filepath",
                )
                annotated_image = gr.Image(label="标注结果", type="filepath")
                mask_image = gr.Image(label="预测 mask", type="filepath")
                feedback_editor = gr.ImageEditor(
                    label="画笔反馈：在标注结果上圈出要修改的位置",
                    type="numpy",
                )

            with gr.Column(scale=4):
                unit = gr.Textbox(label="单位", value="pixel")
                measurements = gr.Dataframe(
                    headers=["ID", "面积", "宽", "高", "长宽比", "外接框"],
                    label="面积 / 数量结果",
                    interactive=False,
                )
                strategy_json = gr.JSON(label="当前 strategy")
                metrics_json = gr.JSON(label="参考标注评估")
                files = gr.File(label="输出文件", file_count="multiple")

        run_outputs = [
            chat,
            annotated_image,
            mask_image,
            feedback_editor,
            measurements,
            strategy_json,
            metrics_json,
            files,
            app_state,
        ]
        run_button.click(
            fn=run_initial,
            inputs=[target_image, description, reference_annotation, unit],
            outputs=run_outputs,
        )
        feedback_button.click(
            fn=run_feedback,
            inputs=[app_state, feedback_editor, description],
            outputs=run_outputs,
        )

    return app


def launch_kwargs():
    kwargs = {"server_name": "127.0.0.1"}
    port = os.getenv("GRADIO_SERVER_PORT")
    if port:
        kwargs["server_port"] = int(port)
    return kwargs


def main():
    build_app().launch(**launch_kwargs())


if __name__ == "__main__":
    main()
