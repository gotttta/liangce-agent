"""Feedback callback exports."""


def save_canvas_feedback(*args, **kwargs):
    from ui.annotation_app import save_canvas_feedback as callback
    return callback(*args, **kwargs)


def handle_result_action(*args, **kwargs):
    from ui.annotation_app import handle_result_action as callback
    return callback(*args, **kwargs)
