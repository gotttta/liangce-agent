"""Message composer builder."""

import gradio as gr


def build_composer_component():
    with gr.Row(elem_id="annotation-composer-row"):
        attach = gr.UploadButton("+", elem_id="annotation-attach", type="filepath", file_types=["image"])
        prompt = gr.Textbox(show_label=False, lines=1, elem_id="annotation-prompt")
        send = gr.Button("↑", elem_id="annotation-send")
    return {"attach": attach, "prompt": prompt, "send": send}
