import os

import gradio as gr

from ui.app import build_app, configure_gradio_environment, launch_kwargs


def test_build_app_returns_gradio_blocks():
    app = build_app()

    assert isinstance(app, gr.Blocks)


def test_build_app_hides_events_from_gradio_api_info():
    app = build_app()

    assert app.get_api_info() == {"named_endpoints": {}, "unnamed_endpoints": {}}


def test_configure_gradio_environment_disables_network_probe_env(monkeypatch):
    monkeypatch.delenv("GRADIO_ANALYTICS_ENABLED", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    configure_gradio_environment()

    assert os.environ["GRADIO_ANALYTICS_ENABLED"] == "False"
    assert os.environ["NO_PROXY"] == "127.0.0.1,localhost"
    assert os.environ["no_proxy"] == "127.0.0.1,localhost"


def test_launch_kwargs_lets_gradio_find_port_by_default(monkeypatch):
    monkeypatch.delenv("GRADIO_SERVER_PORT", raising=False)

    kwargs = launch_kwargs()

    assert kwargs == {
        "server_name": "127.0.0.1",
        "show_api": False,
        "share": False,
        "show_error": True,
    }


def test_launch_kwargs_uses_env_port_when_set(monkeypatch):
    monkeypatch.setenv("GRADIO_SERVER_PORT", "7867")

    kwargs = launch_kwargs()

    assert kwargs == {
        "server_name": "127.0.0.1",
        "server_port": 7867,
        "show_api": False,
        "share": False,
        "show_error": True,
    }
