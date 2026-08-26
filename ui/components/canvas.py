"""Floating annotation feedback panel builder."""

import gradio as gr


def build_canvas_component():
    with gr.Group(visible=False, elem_id="annotation-feedback-canvas"):
        gr.HTML('<div class="feedback-canvas-note">标记需要修改的区域</div>')
        editor = gr.ImageEditor(type="filepath", elem_id="annotation-feedback-editor")
        cancel = gr.Button("取消", elem_id="annotation-cancel-feedback")
        submit = gr.Button("提交修改", variant="primary", elem_id="annotation-submit-feedback")
    return {"editor": editor, "cancel": cancel, "submit": submit}
