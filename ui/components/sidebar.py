"""Sidebar builder for embedding the workspace in another Gradio app."""

import gradio as gr


def build_sidebar_component():
    with gr.Column(elem_id="annotation-sidebar"):
        gr.HTML('<div id="annotation-brand"><div class="brand-mark">CV</div><h1>Vision Agent</h1></div>')
        new_task = gr.Button("＋ 新建标注任务", variant="secondary", elem_id="annotation-new-task")
    return {"new_task": new_task}
