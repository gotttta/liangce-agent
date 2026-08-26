"""Task lifecycle callback exports."""


def reset_chat_task(*args, **kwargs):
    from ui.annotation_app import reset_chat_task as callback
    return callback(*args, **kwargs)


def resume_chat_task(*args, **kwargs):
    from ui.annotation_app import resume_chat_task as callback
    return callback(*args, **kwargs)
