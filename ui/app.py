"""Canonical web entry point for the current v2 Agent workspace."""

import os

from ui.annotation_app import build_annotation_app, configure_gradio_environment


def build_app():
    return build_annotation_app()


def launch_kwargs():
    kwargs = {
        "server_name": "127.0.0.1",
        "show_api": False,
        "share": False,
        "show_error": True,
    }
    port = os.getenv("GRADIO_SERVER_PORT")
    if port:
        kwargs["server_port"] = int(port)
    return kwargs


def main():
    configure_gradio_environment()
    build_app().launch(**launch_kwargs())


if __name__ == "__main__":
    main()
