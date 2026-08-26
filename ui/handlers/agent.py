"""Agent callback exports."""


def run_chat_agent(*args, **kwargs):
    from ui.annotation_app import run_chat_agent as callback
    return callback(*args, **kwargs)


def run_chat_agent_stream(*args, **kwargs):
    from ui.annotation_app import run_chat_agent_stream as callback
    return callback(*args, **kwargs)


def submit_canvas_feedback(*args, **kwargs):
    from ui.annotation_app import submit_canvas_feedback as callback
    return callback(*args, **kwargs)
