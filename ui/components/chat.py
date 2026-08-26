"""Chat surface builder kept independent from task callbacks."""

import gradio as gr


def build_chat_component():
    """Build the chat display primitives used by integrations and tests."""
    with gr.Column(elem_id="annotation-chat"):
        gr.HTML('<div id="annotation-welcome"><h2>今天想处理什么？</h2></div>')
        chatbot = gr.Chatbot(type="messages", show_label=False, elem_id="annotation-chatbot")
    return {"chatbot": chatbot}
