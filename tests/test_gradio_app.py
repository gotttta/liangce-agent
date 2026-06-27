import gradio as gr

from ui.app import build_app


def test_build_app_returns_gradio_blocks():
    app = build_app()

    assert isinstance(app, gr.Blocks)
