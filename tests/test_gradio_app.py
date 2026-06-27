import gradio as gr

from ui.app import build_app, launch_kwargs


def test_build_app_returns_gradio_blocks():
    app = build_app()

    assert isinstance(app, gr.Blocks)


def test_launch_kwargs_lets_gradio_find_port_by_default(monkeypatch):
    monkeypatch.delenv("GRADIO_SERVER_PORT", raising=False)

    kwargs = launch_kwargs()

    assert kwargs == {"server_name": "127.0.0.1"}


def test_launch_kwargs_uses_env_port_when_set(monkeypatch):
    monkeypatch.setenv("GRADIO_SERVER_PORT", "7867")

    kwargs = launch_kwargs()

    assert kwargs == {"server_name": "127.0.0.1", "server_port": 7867}
