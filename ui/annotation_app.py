from pathlib import Path
import json
import os
import queue
import threading
import time

import gradio as gr
import numpy as np
from PIL import Image

from core.agent_graph import resume_agent_graph, run_agent_graph
from core.task_store import TaskStore, save_rejection_record
from ui.gradio_adapters import measurement_rows, state_to_chat_messages
from ui.styles import load_styles
from ui.utils.formatters import format_progress_card, format_task_card

try:
    from providers.vision import build_runtime_provider
except ModuleNotFoundError:
    build_runtime_provider = None


ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "workspace" / "tasks"

# Backwards-compatible export for integrations that used the former inline CSS constant.
ANNOTATION_CSS = load_styles()


def configure_gradio_environment():
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    for key in ("NO_PROXY", "no_proxy"):
        entries = [item.strip() for item in os.getenv(key, "").split(",") if item.strip()]
        for host in ("127.0.0.1", "localhost"):
            if host not in entries:
                entries.append(host)
        os.environ[key] = ",".join(entries)


SCROLL_CHAT_JS = """
() => {
    window.setTimeout(() => {
        const messages = document.querySelector('#annotation-chatbot .bubble-wrap');
        const target = messages || document.querySelector('#annotation-chat');
        if (target) {
            target.scrollTo({ top: target.scrollHeight, behavior: 'smooth' });
        }
    }, 120);
}
"""

SHOW_LATEST_RESULT_JS = """
() => {
    const focusLatestResult = () => {
        const messages = document.querySelector('#annotation-chatbot .bubble-wrap');
        if (!messages) return false;
        const taskCards = messages.querySelectorAll('.task-card');
        const progressCards = messages.querySelectorAll('.progress-card');
        const target = taskCards[taskCards.length - 1]
            || progressCards[progressCards.length - 1];
        if (!target) return false;
        const targetTop = target.getBoundingClientRect().top
            - messages.getBoundingClientRect().top
            + messages.scrollTop;
        messages.scrollTo({ top: Math.max(0, targetTop - 12), behavior: 'auto' });
        return true;
    };
    [120, 350, 800].forEach((delay, index, delays) => {
        window.setTimeout(() => {
            const focused = focusLatestResult();
            if (!focused && index === delays.length - 1) {
                const messages = document.querySelector('#annotation-chatbot .bubble-wrap');
                messages?.scrollTo({ top: messages.scrollHeight, behavior: 'auto' });
            }
        }, delay);
    });
}
"""

OPEN_FEEDBACK_ACTION_JS = """
(action, image, history, task, state, prompt) => {
    document.querySelector('#annotation-feedback-canvas')?.classList.add('feedback-open');
    return [action, image, history, task, state, prompt];
}
"""

CLOSE_FEEDBACK_ACTION_JS = """
(action, image, history, task, state, prompt) => {
    document.querySelector('#annotation-feedback-canvas')?.classList.remove('feedback-open');
    return [action, image, history, task, state, prompt];
}
"""

CLOSE_FEEDBACK_SUBMIT_JS = """
(image, prompt, history, task, state, red, green, examples) => {
    document.querySelector('#annotation-feedback-canvas')?.classList.remove('feedback-open');
    return [image, prompt, history, task, state, red, green, examples];
}
"""

CLOSE_FEEDBACK_CANCEL_JS = """
() => {
    document.querySelector('#annotation-feedback-canvas')?.classList.remove('feedback-open');
    return [];
}
"""


def _layer_array(layer):
    if isinstance(layer, str):
        return np.asarray(Image.open(layer))
    return np.asarray(layer)


def load_editor_background(source_path):
    return source_path or None


def _painted_feedback_masks(editor_value, forced_color=None):
    if not isinstance(editor_value, dict):
        return None, None
    layers = editor_value.get("layers") or []
    if not layers:
        return None, None
    false_positive = None
    false_negative = None
    for layer in layers:
        array = _layer_array(layer)
        if array.ndim == 2:
            painted = array > 0
            red = painted
            green = np.zeros_like(painted)
        elif array.shape[2] == 4:
            painted = array[:, :, 3] > 0
            if forced_color == "red":
                red = painted
                green = np.zeros_like(painted)
            elif forced_color == "green":
                red = np.zeros_like(painted)
                green = painted
            else:
                red = painted & (array[:, :, 0] >= array[:, :, 1])
                green = painted & (array[:, :, 1] > array[:, :, 0])
        else:
            painted = np.any(array[:, :, :3] > 0, axis=2)
            if forced_color == "red":
                red = painted
                green = np.zeros_like(painted)
            elif forced_color == "green":
                red = np.zeros_like(painted)
                green = painted
            else:
                red = painted & (array[:, :, 0] >= array[:, :, 1])
                green = painted & (array[:, :, 1] > array[:, :, 0])
        if false_positive is None:
            false_positive = np.zeros(painted.shape, dtype=bool)
            false_negative = np.zeros(painted.shape, dtype=bool)
        if painted.shape != false_positive.shape:
            raise ValueError("画布图层尺寸不一致，请清空画布后重新标记。")
        false_positive |= red
        false_negative |= green
    if not np.any(false_positive) and not np.any(false_negative):
        return None, None
    return false_positive, false_negative


def _save_editor_image(value, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        with Image.open(value) as image:
            image.save(output_path)
    elif isinstance(value, Image.Image):
        value.save(output_path)
    else:
        Image.fromarray(np.asarray(value)).save(output_path)
    return str(output_path)


def _editor_value(background):
    """Return the structured value required by Gradio ImageEditor updates."""
    if not background:
        return None
    return {
        "background": str(background),
        "layers": [],
        "composite": str(background),
    }


def save_canvas_feedback(editor_value, task, previous_state, green_editor_value=None):
    false_positive, false_negative = _painted_feedback_masks(
        editor_value,
        forced_color="red" if green_editor_value is not None else None,
    )
    green_false_positive, green_false_negative = _painted_feedback_masks(
        green_editor_value,
        forced_color="green",
    )
    if green_false_negative is not None:
        if false_negative is None:
            false_negative = green_false_negative.copy()
        elif false_negative.shape == green_false_negative.shape:
            false_negative |= green_false_negative
        else:
            raise ValueError("红色和绿色画板尺寸不一致，请清空画板后重新标记。")
    if green_false_positive is not None and np.any(green_false_positive):
        if false_positive is None:
            false_positive = green_false_positive.copy()
        elif false_positive.shape == green_false_positive.shape:
            false_positive |= green_false_positive
    if false_positive is None and false_negative is None:
        return previous_state
    if not isinstance(task, dict) or not task.get("id"):
        raise ValueError("当前任务状态已丢失，请新建任务后重试。")

    store = TaskStore(TASK_ROOT)
    feedback_dir = store.task_dir(task["id"]) / "feedback" / f"edit_{time.time_ns()}"
    feedback_dir.mkdir(parents=True, exist_ok=False)

    false_positive = false_positive if false_positive is not None else np.zeros_like(false_negative)
    false_negative = false_negative if false_negative is not None else np.zeros_like(false_positive)

    # Feedback is cumulative across correction rounds. A later green mark is
    # an additional missing target, not a replacement for earlier marks.
    previous_feedback = (previous_state or {}).get("human_feedback") or {}
    legacy_feedback = (previous_state or {}).get("feedback") or {}
    previous_paths = {
        "false_positive": [
            previous_state.get("false_positive_mask_path"),
            previous_feedback.get("exclude_mask_path"),
            legacy_feedback.get("false_positive_mask_path"),
            legacy_feedback.get("exclude_mask_path"),
        ],
        "false_negative": [
            previous_state.get("false_negative_mask_path"),
            previous_feedback.get("include_mask_path"),
            legacy_feedback.get("false_negative_mask_path"),
            legacy_feedback.get("include_mask_path"),
        ],
    }
    # Recover feedback from earlier UI submissions when the latest graph
    # state came from a separate run directory and only has the last mask.
    events_path = store.task_dir(task["id"]) / "events.jsonl"
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except (ValueError, TypeError):
                continue
            if event.get("type") != "canvas_feedback_saved":
                continue
            payload = event.get("payload") or {}
            previous_paths["false_positive"].append(payload.get("false_positive_mask_path"))
            previous_paths["false_negative"].append(payload.get("false_negative_mask_path"))
    for kind, paths in previous_paths.items():
        for path in paths:
            if not path or not Path(path).exists():
                continue
            historical = np.asarray(Image.open(path).convert("L")) > 0
            current = false_positive if kind == "false_positive" else false_negative
            if historical.shape != current.shape:
                raise ValueError("历史画布反馈与当前图片尺寸不一致，请清空画布后重新标记。")
            current |= historical

    combined = false_positive | false_negative
    marks = np.zeros((*combined.shape, 4), dtype=np.uint8)
    marks[false_positive, 0] = 255
    marks[false_positive, 3] = 210
    marks[false_negative, 1] = 220
    marks[false_negative, 3] = 210
    marks_path = feedback_dir / "feedback_marks.png"
    Image.fromarray(marks, mode="RGBA").save(marks_path)
    false_positive_path = feedback_dir / "false_positive_mask.png"
    false_negative_path = feedback_dir / "false_negative_mask.png"
    Image.fromarray(false_positive.astype(np.uint8) * 255, mode="L").save(false_positive_path)
    Image.fromarray(false_negative.astype(np.uint8) * 255, mode="L").save(false_negative_path)

    composite_path = feedback_dir / "feedback_composite.png"
    background_value = (previous_state or {}).get("annotated_image_path")
    if isinstance(background_value, str) and not Path(background_value).exists():
        background_value = None
    if background_value is None and isinstance(editor_value, dict):
        background_value = editor_value.get("background")
        if background_value is None:
            background_value = editor_value.get("composite")
    if background_value is None and isinstance(green_editor_value, dict):
        background_value = green_editor_value.get("background")
    if background_value is None:
        raise ValueError("画板缺少背景图，请重新打开“继续修改”。")
    if isinstance(background_value, str):
        background = Image.open(background_value).convert("RGBA")
    elif isinstance(background_value, Image.Image):
        background = background_value.convert("RGBA")
    else:
        background = Image.fromarray(np.asarray(background_value)).convert("RGBA")
    with background:
        overlay = Image.fromarray(marks, mode="RGBA")
        background.alpha_composite(overlay)
        background.save(composite_path)

    updated = dict(previous_state or {})
    updated.update({
        "feedback_image_path": str(composite_path),
        "feedback_layer_path": str(marks_path),
        "false_positive_mask_path": str(false_positive_path),
        "false_negative_mask_path": str(false_negative_path),
        "exclude_mask_path": str(false_positive_path),
        "include_mask_path": str(false_negative_path),
        "false_positive_pixel_count": int(np.count_nonzero(false_positive)),
        "false_negative_pixel_count": int(np.count_nonzero(false_negative)),
        "feedback_pixel_count": int(np.count_nonzero(combined)),
    })
    store.append_event(task["id"], "canvas_feedback_saved", {
        "feedback_image_path": str(composite_path),
        "feedback_layer_path": str(marks_path),
        "false_positive_mask_path": str(false_positive_path),
        "false_negative_mask_path": str(false_negative_path),
        "false_positive_pixels": updated["false_positive_pixel_count"],
        "false_negative_pixels": updated["false_negative_pixel_count"],
        "painted_pixels": updated["feedback_pixel_count"],
    })
    return updated


def create_chat_task():
    return TaskStore(TASK_ROOT).create_task()


def _task_choices(include_task_id=None):
    store = TaskStore(TASK_ROOT)
    choices = []
    for task in store.list_tasks(limit=40):
        meaningful = bool(
            task.get("samples")
            or task.get("current_node")
            or task.get("status") not in {"draft"}
        )
        if not meaningful and task.get("id") != include_task_id:
            continue
        updated = str(task.get("updated_at", ""))[:16].replace("T", " ")
        status = task.get("status", "draft")
        title = task.get("title") or "未命名任务"
        display_title = title if len(title) <= 28 else f"{title[:28]}…"
        choices.append((f"{display_title} · {status} · {updated}", task["id"]))
    return choices


def resume_chat_task(task_id):
    if not task_id:
        raise gr.Error("请先选择一个历史任务。")
    store = TaskStore(TASK_ROOT)
    try:
        task = store.load_task(task_id)
        messages = store.load_messages(task_id)
        state = store.load_latest_state(task_id)
    except (FileNotFoundError, OSError, ValueError, TypeError) as exc:
        raise gr.Error(f"无法恢复任务：{exc}") from exc
    sample = (task.get("samples") or [])[-1] if task.get("samples") else None
    has_user_message = any(item.get("role") == "user" for item in messages)
    attachment = sample.get("path") if sample and not has_user_message else None
    attachment_label = f"已恢复：`{sample['source_name']}`" if attachment else ""
    show_actions = bool(
        state
        and state.get("annotated_image_path")
        and task.get("status") not in {"accepted", "exited"}
    )
    if not messages:
        messages = [{
            "role": "assistant",
            "content": "任务已恢复。请继续描述修改要求。",
        }]
    return (
        messages,
        gr.update(value=(state or {}).get("annotated_image_path"), visible=False),
        gr.update(value=(state or {}).get("predicted_mask_path"), visible=False),
        gr.update(visible=False),
        measurement_rows(state or {}),
        state,
        task,
        attachment,
        gr.update(value="", placeholder="描述需要修改的地方，或继续询问 Agent…"),
        attachment_label,
        gr.update(visible=show_actions),
        gr.update(visible=True),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=[item.get("image_path") or item.get("path") for item in task.get("reference_examples", [])]),
        gr.update(choices=_task_choices(task_id), value=task_id),
        gr.update(value=(task.get("ground_truth") or {}).get("annotation_path")),
    )


def load_latest_chat_task():
    tasks = TaskStore(TASK_ROOT).list_tasks(limit=40)
    resumable = next(
        (
            task for task in tasks
            if task.get("status") not in {"exited"}
            and (task.get("samples") or task.get("current_node"))
        ),
        None,
    )
    if resumable:
        return resume_chat_task(resumable["id"])
    return reset_chat_task()


def _task_memory_context(store, task_id):
    try:
        return store.load_memory(task_id)
    except (OSError, ValueError, TypeError, FileNotFoundError):
        return {}


def _save_task_memory(store, task_id, message, understanding, state, previous_state):
    existing = _task_memory_context(store, task_id)
    corrections = list(existing.get("corrections") or [])
    if previous_state and (
        previous_state.get("feedback_pixel_count")
        or previous_state.get("agent_status") == "waiting_for_feedback"
    ):
        corrections.append({
            "iteration": state.get("iteration"),
            "request": message,
            "false_positive_pixels": previous_state.get("false_positive_pixel_count", 0),
            "false_negative_pixels": previous_state.get("false_negative_pixel_count", 0),
        })
    memory = {
        "task_goal": existing.get("task_goal") or understanding.get("task_summary") or message,
        "latest_user_request": message,
        "latest_iteration": state.get("iteration"),
        "active_constraints": understanding.get("target_constraints") or {},
        "selected_pipeline": state.get("pipeline") or {},
        "pipeline_diff": state.get("pipeline_diff") or {},
        "strategy": state.get("strategy") or {},
        "quality_report": state.get("quality_report") or {},
        "measurement_summary": state.get("measurements", {}).get("summary", {}),
        "latest_result_image_path": state.get("annotated_image_path"),
        "latest_mask_path": state.get("predicted_mask_path"),
        "corrections": corrections[-20:],
    }
    return store.save_memory(task_id, memory)


def store_chat_attachment(path, task):
    if not path:
        return None, "", task
    store = TaskStore(TASK_ROOT)
    current = task if isinstance(task, dict) and task.get("id") else store.create_task()
    sample = store.add_sample(current["id"], path)
    current = store.load_task(current["id"])
    return sample["path"], f"已附加：`{sample['source_name']}`", current


def store_reference_examples(paths, task):
    store = TaskStore(TASK_ROOT)
    current = task if isinstance(task, dict) and task.get("id") else store.create_task()
    raw_paths = paths if isinstance(paths, list) else ([paths] if paths else [])
    for item in raw_paths:
        path = item.get("path") if isinstance(item, dict) else getattr(item, "name", item)
        if path and Path(path).is_file():
            store.add_reference_example(current["id"], path)
    return store.load_task(current["id"])


def store_ground_truth_annotation(path, task, target_image_path):
    """Store a pixel-aligned Handbook annotation for objective evaluation."""
    store = TaskStore(TASK_ROOT)
    current = task if isinstance(task, dict) and task.get("id") else store.create_task()
    source_path = path.get("path") if isinstance(path, dict) else getattr(path, "name", path)
    if not source_path:
        return store.load_task(current["id"])
    if not Path(source_path).is_file():
        raise gr.Error("无法读取同图 Ground Truth 文件，请重新上传。")
    if not target_image_path or not Path(target_image_path).is_file():
        raise gr.Error("请先上传原图，再添加同图 Ground Truth。")
    try:
        store.set_ground_truth_annotation(current["id"], source_path, target_image_path)
    except (OSError, ValueError) as exc:
        raise gr.Error(f"同图 Ground Truth 无法使用：{exc}") from exc
    return store.load_task(current["id"])


def store_chat_attachment_ui(path, task):
    attachment, label, current = store_chat_attachment(path, task)
    return (
        attachment,
        label,
        current,
        gr.update(visible=bool(attachment)),
        gr.update(visible=bool(attachment)),
        gr.update(value=None, visible=False),
    )


def clear_chat_attachment(path, task):
    current = task
    if isinstance(task, dict) and task.get("id") and path:
        current = TaskStore(TASK_ROOT).remove_sample(task["id"], path)
    return (
        None,
        "",
        current,
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(value=None, visible=False),
    )


def show_chat_attachment(path):
    if not path or not Path(path).exists():
        return gr.update(value=None, visible=False)
    return gr.update(value=path, visible=True)


def sync_chat_attachment_ui(path):
    return (
        gr.update(visible=bool(path)),
        gr.update(visible=bool(path)),
        gr.update(value=None, visible=False),
    )


def consume_chat_attachment_ui():
    """Clear the composer after its image has been committed to chat history."""
    return (
        None,
        "",
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(value=None, visible=False),
    )


def _resolve_task_image(image_path, task, previous_state):
    if image_path and Path(image_path).exists():
        return str(image_path)
    previous_target = (previous_state or {}).get("target_image_path")
    if previous_target and Path(previous_target).exists():
        return str(previous_target)
    if isinstance(task, dict):
        for sample in reversed(task.get("samples") or []):
            sample_path = sample.get("path")
            if sample_path and Path(sample_path).exists():
                return str(sample_path)
    return None


def _user_submission_messages(message, attachment_path=None):
    messages = []
    if attachment_path and Path(attachment_path).exists():
        messages.append({
            "role": "user",
            "content": (str(attachment_path), Path(attachment_path).name),
        })
    messages.append({"role": "user", "content": message})
    return messages


def _format_understanding(
    understanding,
    provider_name,
    duration_seconds,
    retrieved_algorithms=None,
):
    """Summarize the goal without asking the user to design the CV pipeline."""
    lines = [
        f"视觉理解已完成（{provider_name}，{duration_seconds:.1f}s）。",
        f"\n\n**我理解的任务**：{understanding.get('task_summary') or understanding.get('target_defect') or '按描述标注目标'}",
        "\n\n我会自动选择识别方法，再把标注结果展示给你。你只需要看结果对不对。",
    ]
    if retrieved_algorithms:
        lines.append(
            f"\n\n我还找到了 {len(retrieved_algorithms)} 个相似任务的处理记录，会一起尝试并重新检查效果。"
        )
    return "".join(lines)


def _format_agent_process(events, running=False):
    """Render auditable Agent progress without exposing private chain-of-thought."""
    return format_progress_card(events, running=running)


def _progress_event(stage, label, detail=None, status="running", duration_seconds=None):
    event = {
        "stage": stage,
        "label": label,
        "status": status,
    }
    if detail:
        event["detail"] = detail
    if duration_seconds is not None:
        event["duration_seconds"] = round(float(duration_seconds), 3)
    return event


def _merge_progress_event(events, event):
    stage = event.get("stage")
    event_type = event.get("type")
    if event_type == "llm_chunk":
        existing = next((item for item in reversed(events) if item.get("type") == "llm_chunk"), None)
        if existing is not None:
            existing["content"] = f"{existing.get('content', '')}{event.get('content', '')}"
            return
        events.append(dict(event))
        return
    if event_type and not stage:
        events.append(dict(event))
        return
    for index in range(len(events) - 1, -1, -1):
        if events[index].get("stage") == stage:
            events[index] = event
            return
    events.append(event)


def _candidate_progress_events(state):
    events = []
    selected = state.get("selected_candidate")
    for index, attempt in enumerate(state.get("candidate_attempts") or [], start=1):
        status = attempt.get("status")
        if attempt.get("name") == selected:
            detail = "已经找到目标并生成标注。"
        elif status == "no_annotation":
            detail = "这次没有找到目标，系统会自动换一种方法。"
        elif status == "failed":
            detail = "这次方法没有运行成功，系统会自动换一种方法。"
        else:
            detail = "这次检查已完成。"
        events.append(_progress_event(
            f"pipeline_{attempt.get('index', len(events))}",
            f"识别方法 {index}",
            detail,
            "completed" if status == "completed" else "failed",
        ))
    return events


def _state_process_events(state):
    events = []
    for event in state.get("trajectory") or []:
        node = event.get("node")
        label = {
            "prepare_inputs": "准备输入",
            "understand_task": "视觉理解",
            "retrieve_algorithms": "查找相似经验",
            "plan_candidates": "准备识别方法",
            "execute_candidates": "识别目标",
            "review_candidates": "检查识别结果",
            "revise_candidates": "自动换一种方法",
            "report_failure": "识别未成功",
            "decide_next_action": "准备人工验收",
            "evaluate_quality": "检查标注输出",
            "refine_candidates": "自动调整候选算法",
        }.get(node, node or "执行步骤")
        events.append(_progress_event(
            node or f"step_{len(events)}",
            label,
            _trajectory_progress_detail(event),
            event.get("status", "completed"),
            event.get("duration_seconds"),
        ))
    events.extend(_candidate_progress_events(state))
    return events


def _trajectory_progress_detail(event):
    node = event.get("node")
    details = event.get("details") or {}
    if node == "plan_candidates":
        return "识别方法已经准备好。"
    if node == "execute_candidates":
        if not details.get("selected_candidate"):
            return "这次没有找到可用目标，系统将自动换一种方法。"
        return (
            "已经找到目标并生成标注。"
        )
    if node == "review_candidates":
        return str(details.get("reason") or "正在检查这次识别结果。")
    if node == "decide_next_action":
        return str(details.get("reason") or "标注结果已生成，等待人工验收。")
    if node == "revise_candidates":
        return f"上一种方法没有找到目标，正在尝试第 {int(details.get('revision_count', 0)) + 1} 种方法。"
    if node == "report_failure":
        return str(details.get("message") or "自动尝试后仍未识别成功。")
    if node == "evaluate_quality":
        return "标注结果已生成，等待用户检查标注。"
    if node == "refine_candidates":
        return "根据上一轮反馈自动调整候选算法。"
    if "previous_quality" in details or "score" in details:
        return "执行记录已完成。"
    return json.dumps(details, ensure_ascii=False, default=str)[:500]


def _artifact_command(message):
    text = (message or "").strip().lower()
    if any(word in text for word in ("隐藏结果", "收起结果", "关闭结果", "hide results")):
        return "hide"
    if any(word in text for word in ("查看mask", "查看 mask", "显示mask", "显示 mask", "查看掩膜")):
        return "mask"
    if any(word in text for word in ("查看量测", "量测明细", "查看参数", "measurement")):
        return "measurements"
    if any(word in text for word in ("查看结果图", "显示结果图", "打开结果图", "查看轮廓", "show result")):
        return "result"
    return None


def _handle_artifact_command(command, message, history, task, previous_state):
    messages = list(history or [])
    messages.append({"role": "user", "content": message})
    process_events = _state_process_events(previous_state or {})
    if process_events:
        messages.append({
            "role": "assistant",
            "content": _format_agent_process(process_events, running=False),
        })
    has_result = bool(
        isinstance(previous_state, dict) and previous_state.get("annotated_image_path")
    )
    if not has_result and command != "hide":
        reply = "当前任务还没有可查看的算法产物。请先上传图片并描述检测目标。"
        result_update = gr.update(visible=False)
        mask_update = gr.update(visible=False)
        panel_update = gr.update(visible=False)
    elif command == "result":
        reply = "已按你的要求打开本轮标注结果。看完后可以继续描述哪里标错或漏标。"
        result_update = gr.update(value=previous_state["annotated_image_path"], visible=True)
        mask_update = gr.update(visible=False)
        panel_update = gr.update(visible=False)
    elif command == "mask":
        reply = "已按你的要求打开本轮二值 Mask。"
        result_update = gr.update(visible=False)
        mask_update = gr.update(value=previous_state["predicted_mask_path"], visible=True)
        panel_update = gr.update(visible=False)
    elif command == "measurements":
        reply = "已展开本轮量测明细。"
        result_update = gr.update(visible=False)
        mask_update = gr.update(visible=False)
        panel_update = gr.update(visible=True)
    else:
        reply = "已收起本轮图片、Mask 和量测明细。"
        result_update = gr.update(visible=False)
        mask_update = gr.update(visible=False)
        panel_update = gr.update(visible=False)
    messages.append({"role": "assistant", "content": reply})
    if isinstance(task, dict) and task.get("id"):
        store = TaskStore(TASK_ROOT)
        store.append_message(task["id"], "user", message)
        store.append_message(task["id"], "assistant", reply)
    return (
        messages,
        result_update,
        mask_update,
        panel_update,
        measurement_rows(previous_state or {}),
        previous_state,
        task,
        "",
        "",
        gr.update(visible=has_result),
        gr.update(visible=True),
        gr.update(value=None),
        gr.update(value=None),
    )


def _inline_result_messages(state):
    messages = []
    annotated = state.get("annotated_image_path")
    if annotated:
        messages.append({
            "role": "assistant",
        "content": "这是本轮标注结果。点击图片放大，检查标注是否符合你的描述。",
        })
        messages.append({
            "role": "assistant",
            "content": (annotated, "点击放大标注结果"),
        })
    return messages


def _recover_agent_state(task, previous_state):
    if isinstance(previous_state, dict) and previous_state.get("annotated_image_path"):
        return previous_state
    if not isinstance(task, dict) or not task.get("id"):
        return previous_state

    latest_path = TASK_ROOT / task["id"] / "nodes" / "execute_candidate" / "latest.json"
    if not latest_path.exists():
        return previous_state
    try:
        record = json.loads(latest_path.read_text(encoding="utf-8"))
        outputs = record.get("outputs") if isinstance(record.get("outputs"), dict) else {}
        annotated = outputs.get("annotated_image_path")
        if annotated:
            graph_state_path = Path(annotated).parent / "graph_state.json"
            if graph_state_path.exists():
                return json.loads(graph_state_path.read_text(encoding="utf-8"))
        return outputs or previous_state
    except (OSError, ValueError, TypeError):
        return previous_state


def run_chat_agent(
    image_path,
    message,
    history,
    task,
    previous_state=None,
    editor_value=None,
    green_editor_value=None,
    progress_callback=None,
    reference_example_paths=None,
    ground_truth_annotation_path=None,
):
    from core.agent_events import register_event_listener, unregister_event_listener

    def report(event):
        if progress_callback:
            progress_callback(dict(event))

    def agent_event_listener(event):
        """捕获 Agent 内部事件并转换为进度报告"""
        event_type = event.get("type")
        if event_type == "node_start":
            report(_progress_event(
                event.get("node"),
                event.get("description", event.get("node")),
                "执行中...",
                "running",
            ))
        elif event_type == "node_complete":
            report(_progress_event(
                event.get("node"),
                event.get("node"),
                json.dumps(event.get("metadata", {}), ensure_ascii=False)[:200],
                "completed",
                event.get("duration"),
            ))
        elif event_type == "thinking":
            report({
                "type": "thinking",
                "message": event.get("message"),
                "context": event.get("context"),
                "timestamp": event.get("timestamp"),
            })
        elif event_type == "tool_call":
            report({
                "type": "tool_call",
                "tool": event.get("tool"),
                "args": event.get("args") or event.get("parameters"),
                "timestamp": event.get("timestamp"),
            })
        elif event_type == "tool_result":
            report({
                "type": "tool_result",
                "tool": event.get("tool"),
                "result": event.get("result"),
                "success": event.get("success", True),
                "timestamp": event.get("timestamp"),
            })
        elif event_type == "llm_request":
            report({
                "type": "llm_request",
                "provider": event.get("provider"),
                "model": event.get("model"),
                "message_count": event.get("message_count"),
                "timestamp": event.get("timestamp"),
            })
        elif event_type == "llm_response":
            report({
                "type": "llm_response",
                "provider": event.get("provider"),
                "content_preview": event.get("content_preview"),
                "timestamp": event.get("timestamp"),
            })
        elif event_type == "llm_chunk":
            report({
                "type": "llm_chunk",
                "provider": event.get("provider"),
                "model": event.get("model"),
                "content": event.get("content", ""),
                "timestamp": event.get("timestamp"),
            })
        elif event_type == "error":
            report({
                "type": "error",
                "message": event.get("message"),
                "node": event.get("node"),
                "timestamp": event.get("timestamp"),
            })

    register_event_listener(agent_event_listener)

    message = (message or "").strip()
    submitted_attachment_path = image_path if image_path and Path(image_path).exists() else None
    command = _artifact_command(message)
    if command:
        return _handle_artifact_command(command, message, history, task, previous_state)
    image_path = _resolve_task_image(image_path, task, previous_state)
    if not image_path:
        raise gr.Error("请先点击 + 上传一张图片。")
    if not message:
        raise gr.Error("请描述希望 Agent 标注的目标或效果。")

    store = TaskStore(TASK_ROOT)
    current = task if isinstance(task, dict) and task.get("id") else store.create_task()
    task_id = current["id"]
    current = store_reference_examples(reference_example_paths, current)
    current = store_ground_truth_annotation(
        ground_truth_annotation_path,
        current,
        image_path,
    )
    reference_examples = current.get("reference_examples") or []
    ground_truth = current.get("ground_truth") or {}
    previous_state = _recover_agent_state(current, previous_state)
    task_memory = _task_memory_context(store, task_id)
    original_task_goal = (
        (previous_state or {}).get("original_task_goal")
        or task_memory.get("task_goal")
        or message
    )
    if previous_state is not None:
        previous_state["original_task_goal"] = original_task_goal
    report(_progress_event("prepare", "准备输入", f"已载入图片：{Path(image_path).name}", "completed"))
    try:
        previous_state = save_canvas_feedback(
            editor_value,
            current,
            previous_state,
            green_editor_value=green_editor_value,
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    if previous_state:
        feedback = dict(previous_state.get("human_feedback") or {})
        if message:
            feedback["incremental_description"] = message
        for key in ("include_mask_path", "exclude_mask_path"):
            if previous_state.get(key):
                feedback[key] = previous_state[key]
        if feedback:
            previous_state["human_feedback"] = feedback
    if submitted_attachment_path:
        store.append_message(
            task_id,
            "user",
            (str(submitted_attachment_path), Path(submitted_attachment_path).name),
        )
    store.append_message(task_id, "user", message)
    understanding_started = time.monotonic()
    report(_progress_event("understand_task", "视觉理解", "正在分析图片并确定需要标注的区域。"))
    provider_name = "Qwen"
    try:
        provider = build_runtime_provider()
        provider_name = f"Qwen ({provider.model})"
        provider_context = {
            "original_task_goal": original_task_goal,
            "conversation": list(history or [])[-12:],
            "task_memory": task_memory,
            "previous_strategy": (previous_state or {}).get("strategy"),
            "previous_pipeline": (previous_state or {}).get("pipeline"),
            "previous_measurements": (previous_state or {}).get("measurements", {}).get("summary"),
            "previous_quality": (previous_state or {}).get("quality_report"),
            "previous_result_image_path": (previous_state or {}).get("annotated_image_path"),
            "previous_mask_path": (previous_state or {}).get("predicted_mask_path"),
            "feedback_image_path": (previous_state or {}).get("feedback_image_path"),
            "feedback_layer_path": (previous_state or {}).get("feedback_layer_path"),
            "feedback_pixel_count": (previous_state or {}).get("feedback_pixel_count"),
            "human_feedback": (previous_state or {}).get("human_feedback", {}),
            "reference_examples": reference_examples,
            "ground_truth": ground_truth,
        }
        try:
            understanding = provider.understand_task(
                image_path,
                message,
                previous_context=provider_context,
                reference_examples=reference_examples,
            )
        except TypeError as exc:
            if "reference_examples" not in str(exc):
                raise
            understanding = provider.understand_task(
                image_path,
                message,
                previous_context=provider_context,
            )
    except Exception as exc:
        understanding_duration = time.monotonic() - understanding_started
        error_message = f"Qwen视觉理解失败：{type(exc).__name__}: {exc}"
        report(_progress_event(
            "understand_task", "视觉理解", error_message, "failed", understanding_duration,
        ))
        store.save_node_result(
            task_id,
            "understand_task",
            {"image_path": image_path, "description": message, "provider": "qwen"},
            {},
            understanding_duration,
            status="failed",
            error=error_message,
        )
        store.append_message(task_id, "assistant", error_message)
        raise gr.Error(error_message) from exc
    understanding_duration = time.monotonic() - understanding_started
    report(_progress_event(
        "understand_task", "视觉理解", f"已生成任务理解（{provider_name}）。", "completed", understanding_duration,
    ))
    retrieved_algorithms = []
    if not (previous_state or {}).get("pipeline"):
        report(_progress_event("retrieve_algorithms", "查找相似经验", "正在查找以前处理过的相似图片。"))
        retrieved_algorithms = store.search_algorithms(
            understanding,
            limit=2,
            min_score=0.2,
        )
        store.append_event(task_id, "algorithm_search_completed", {
            "match_count": len(retrieved_algorithms),
            "matches": [
                {
                    "algorithm_id": item.get("algorithm_id"),
                    "name": item.get("name"),
                    "score": item.get("score"),
                    "match_reasons": item.get("match_reasons", []),
                    "source_task_id": item.get("source_task_id"),
                }
                for item in retrieved_algorithms
            ],
        })
        report(_progress_event(
            "retrieve_algorithms", "查找相似经验", f"找到了 {len(retrieved_algorithms)} 个相似记录。", "completed",
        ))
    else:
        report(_progress_event("retrieve_algorithms", "查找相似经验", "这次会参考上一轮的结果。", "completed"))
    store.save_node_result(
        task_id,
        "understand_task",
        {"image_path": image_path, "description": message},
        understanding,
        understanding_duration,
    )
    understanding_message = _format_understanding(
        understanding,
        provider_name,
        understanding_duration,
        retrieved_algorithms=retrieved_algorithms,
    )
    if not _task_memory_context(store, task_id).get("task_goal"):
        current = store.set_title(
            task_id,
            understanding.get("task_summary") or message,
        )
    store.append_message(task_id, "assistant", understanding_message)

    execution_started = time.monotonic()
    report(_progress_event("plan_candidates", "准备识别方法", "正在选择适合这张图片的识别方法。"))
    report(_progress_event("execute_candidates", "识别目标", "正在查找并标出目标。"))
    selected_strategy = understanding["recommended_strategy"]
    state = run_agent_graph(
        target_image_path=image_path,
        description=message,
        understanding=understanding,
        output_root=ROOT / "outputs",
        unit="pixel",
        max_candidates=3,
        previous_state=previous_state,
        retrieved_algorithms=retrieved_algorithms,
        reference_examples=reference_examples,
        ground_truth_mask_path=ground_truth.get("mask_path"),
        ground_truth_annotation_path=ground_truth.get("annotation_path"),
        algorithm_registry=store.algorithm_registry,
        provider=provider,
    )
    state["memory_summary"] = _save_task_memory(
        store,
        task_id,
        message,
        understanding,
        state,
        previous_state,
    )
    execution_duration = time.monotonic() - execution_started
    trajectory = state.get("trajectory") or []
    for event in trajectory:
        if event.get("node") in {"plan_candidates", "execute_candidates", "decide_next_action"}:
            report(_progress_event(
                event.get("node"),
                {
                    "plan_candidates": "准备识别方法",
                    "execute_candidates": "识别目标",
                    "decide_next_action": "准备人工验收",
                }.get(event.get("node"), event.get("node")),
                _trajectory_progress_detail(event),
                "completed",
                event.get("duration_seconds"),
            ))
    for event in _candidate_progress_events(state):
        report(event)
    store.save_node_result(
        task_id,
        "execute_candidate",
        {
            "candidate": "qwen_recommended_strategy",
            "provider": provider_name,
            "strategy": selected_strategy,
            "selected_pipeline": state.get("pipeline"),
            "candidate_attempts": state.get("candidate_attempts"),
            "retrieved_algorithms": state.get("retrieved_algorithms"),
            "reference_examples": reference_examples,
            "ground_truth": ground_truth or None,
        },
        {
            "annotated_image_path": state.get("annotated_image_path"),
            "predicted_mask_path": state.get("predicted_mask_path"),
            "measurements": state.get("measurements"),
            "quality_report": state.get("quality_report"),
            "iteration": state.get("iteration"),
            "parent_iteration": state.get("parent_iteration"),
            "pipeline": state.get("pipeline"),
            "pipeline_diff": state.get("pipeline_diff"),
            "decision": state.get("decision"),
            "trajectory": state.get("trajectory"),
            "evaluation_report": state.get("evaluation_report"),
            "ground_truth_mask_path": state.get("ground_truth_mask_path"),
        },
        execution_duration,
    )
    messages = list(history or [])
    messages.extend(_user_submission_messages(message, submitted_attachment_path))
    messages.append({"role": "assistant", "content": understanding_message})
    result_messages = state_to_chat_messages(state)
    result_messages[-1]["content"] += (
        f"\n\n本次识别用时 {execution_duration:.1f}s。"
        "我把标注结果作为下一条图片消息发给你。请只看结果是否符合你的描述："
        "符合就选“标注准确”，否则选“有漏标或错标”。"
    )
    pipeline_changes = state.get("pipeline_diff", {}).get("changes") or []
    if state.get("parent_iteration") is not None:
        result_messages[-1]["content"] += (
            f"\n\n系统根据上一次结果自动调整了识别方法，共修改了 {len(pipeline_changes)} 处。"
        )
    result_messages[-1]["content"] += "\n\n" + format_task_card(state)
    inline_messages = _inline_result_messages(state)
    messages.extend(result_messages)
    messages.extend(inline_messages)
    for item in [*result_messages, *inline_messages]:
        store.append_message(task_id, item["role"], item["content"])
    current = store.load_task(task_id)
    unregister_event_listener(agent_event_listener)
    return (
        messages,
        gr.update(value=state.get("annotated_image_path"), visible=False),
        gr.update(value=state.get("predicted_mask_path"), visible=False),
        gr.update(visible=False),
        measurement_rows(state),
        state,
        current,
        "",
        "",
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(value=None),
        gr.update(value=None),
    )


def run_chat_agent_stream(
    image_path,
    message,
    history,
    task,
    previous_state=None,
    editor_value=None,
    green_editor_value=None,
    reference_example_paths=None,
    ground_truth_annotation_path=None,
):
    message = (message or "").strip()
    submitted_attachment_path = image_path if image_path and Path(image_path).exists() else None
    stream_image_path = _resolve_task_image(image_path, task, previous_state) or image_path
    if _artifact_command(message) or not stream_image_path or not message:
        yield run_chat_agent(
            stream_image_path, message, history, task, previous_state, editor_value,
            green_editor_value=green_editor_value,
            reference_example_paths=reference_example_paths,
            ground_truth_annotation_path=ground_truth_annotation_path,
        )
        return
    image_path = stream_image_path

    working_messages = list(history or [])
    process_events = []

    def process_message(running=True):
        return _format_agent_process(process_events, running=running)

    working_messages.extend(_user_submission_messages(message, submitted_attachment_path))
    working_messages.append({
            "role": "assistant",
            "content": (
                "⏳ 正在处理图片…\n\n"
                "正在理解检测目标、生成并执行算法，请稍候。\n\n"
                + process_message(True)
            ),
        })
    has_previous_result = bool(
        isinstance(previous_state, dict) and previous_state.get("annotated_image_path")
    )
    yield (
        working_messages,
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        measurement_rows(previous_state or {}),
        previous_state,
        task,
        "",
        "",
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(value=editor_value),
        gr.update(value=green_editor_value),
    )

    updates = queue.Queue()
    result_holder = {}

    def report(event):
        updates.put(event)

    def worker():
        try:
            try:
                result_holder["value"] = run_chat_agent(
                    image_path,
                    message,
                    history,
                    task,
                    previous_state,
                    editor_value,
                    green_editor_value=green_editor_value,
                    reference_example_paths=reference_example_paths,
                    ground_truth_annotation_path=ground_truth_annotation_path,
                    progress_callback=report,
                )
            except TypeError as exc:
                if "unexpected keyword argument" not in str(exc):
                    raise
                result_holder["value"] = run_chat_agent(
                    image_path, message, history, task, previous_state, editor_value,
                )
        except Exception as exc:
            result_holder["error"] = exc
        finally:
            updates.put(None)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    while True:
        event = updates.get()
        if event is None:
            break
        _merge_progress_event(process_events, event)
        live_messages = list(working_messages)
        live_messages[-1] = {"role": "assistant", "content": process_message(True)}
        yield (
            live_messages,
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
            measurement_rows(previous_state or {}), previous_state, task, "", "",
            gr.update(visible=False), gr.update(visible=True), gr.update(value=editor_value),
            gr.update(value=green_editor_value),
        )

    if "error" not in result_holder:
        final_result = result_holder["value"]
        if process_events:
            final_messages = list(final_result[0])
            process_content = _format_agent_process(process_events, running=False)
            inline_count = len(_inline_result_messages(final_result[5])) if isinstance(final_result[5], dict) else 0
            insert_at = len(final_messages) - inline_count
            final_messages.insert(insert_at, {"role": "assistant", "content": process_content})
            if isinstance(final_result[6], dict) and final_result[6].get("id"):
                TaskStore(TASK_ROOT).append_message(final_result[6]["id"], "assistant", process_content)
            final_result = (final_messages, *final_result[1:])
        yield final_result
        return

    try:
        raise result_holder["error"]
    except Exception as exc:
        detail = _friendly_processing_error(exc)
        failed_messages = list(history or [])
        failed_messages.extend(_user_submission_messages(message, submitted_attachment_path))
        failed_messages.append({
                "role": "assistant",
                "content": f"{process_message(False)}\n\n{detail}",
            })
        if isinstance(task, dict) and task.get("id"):
            store = TaskStore(TASK_ROOT)
            store.append_message(task["id"], "assistant", detail)
        yield (
            failed_messages,
            gr.update(
                value=(previous_state or {}).get("annotated_image_path"),
                visible=False,
            ),
            gr.update(
                value=(previous_state or {}).get("predicted_mask_path"),
                visible=False,
            ),
            gr.update(visible=False),
            measurement_rows(previous_state or {}),
            previous_state,
            task,
            "",
            "",
            gr.update(visible=has_previous_result),
            gr.update(visible=True),
            gr.update(value=editor_value),
            gr.update(value=green_editor_value),
        )


def _friendly_processing_error(exc):
    detail = getattr(exc, "message", None) or str(exc) or ""
    if detail.startswith("这次没有成功标出目标。"):
        return detail
    if "empty annotations" in detail or "empty_annotation" in detail:
        return "这次没有成功标出目标。系统已经自动换过识别方法，但仍然没有得到可用结果。"
    return "这次没有处理成功。系统已保留执行记录，请稍后重试。"


def submit_canvas_feedback(
    image_path,
    message,
    history,
    task,
    previous_state=None,
    editor_value=None,
    green_editor_value=None,
    reference_example_paths=None,
    ground_truth_annotation_path=None,
):
    """Submit marks from the floating panel even when no prose was entered."""
    feedback_message = (message or "").strip() or "请根据画布上的红色和绿色标记修正标注。"
    yield from run_chat_agent_stream(
        image_path,
        feedback_message,
        history,
        task,
        previous_state,
        editor_value,
        green_editor_value,
        reference_example_paths,
        ground_truth_annotation_path,
    )


def handle_result_action(action, image_path, history, task, previous_state, feedback_text=""):
    previous_state = _recover_agent_state(task, previous_state)
    artifact_prompts = {
        "result": "查看结果图",
        "mask": "查看 Mask",
        "measurements": "查看量测",
    }
    if action in artifact_prompts:
        return run_chat_agent(
            image_path,
            artifact_prompts[action],
            history,
            task,
            previous_state,
        )

    store = TaskStore(TASK_ROOT)
    if not isinstance(task, dict) or not task.get("id"):
        task = store.create_task()
    updated_state = dict(previous_state or {})
    if action != "continue":
        _resume_human_review(
            store, task["id"], updated_state, action,
            rejection_reason=feedback_text if action == "exit" else None,
        )
    if action == "accept":
        if not updated_state.get("annotated_image_path"):
            failed_messages = list(history or [])
            failed_messages.append({
                "role": "assistant",
                "content": "当前结果状态没有保存完整，请重新生成一版结果后再确认算法。",
            })
            return (
                failed_messages,
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                measurement_rows(updated_state),
                updated_state or None,
                task,
                gr.update(value=""),
                "",
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(value=None),
                gr.update(value=None),
            )
        user_message = "标注准确"
        acceptance = store.accept_result(task["id"], updated_state)
        assistant_message = (
            "已确认当前结果，并保存完整算法。Handbook示例图只作为few-shot参照，不参与当前图片的准确率计算。"
            f"\n\n任务内配置：`{acceptance['algorithm_path']}`"
            f"\n\n算法库版本：`{acceptance['registry_algorithm_path']}`"
        )
        updated_state["agent_status"] = "accepted"
        current = store.load_task(task["id"])
        prompt_update = gr.update(value="")
    elif action == "continue":
        updated_state["agent_status"] = "waiting_for_feedback"
        current = store.load_task(task["id"])
        prompt_update = gr.update(
            value="",
            placeholder="描述需要修改的地方，例如：右上角有漏检…",
        )
    elif action == "exit":
        user_message = "结束任务"
        assistant_message = "已退出当前任务。结果仍保留在本地；需要继续时请点击左侧“新建标注任务”。"
        updated_state["agent_status"] = "exited"
        save_rejection_record(
            task_id=task["id"],
            pipeline=updated_state.get("pipeline"),
            rejection_reason=feedback_text,
            quality_report=updated_state.get("quality_report"),
            task_root=TASK_ROOT,
        )
        current = store.exit_task(task["id"])
        prompt_update = gr.update(value="")
    else:
        raise gr.Error("未知的后续操作。")

    show_canvas = action == "continue"
    canvas_background = updated_state.get("annotated_image_path") or image_path

    messages = list(history or [])
    if action != "continue":
        messages.extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ])
        store.append_message(task["id"], "user", user_message)
        store.append_message(task["id"], "assistant", assistant_message)
    return (
        messages,
        gr.update(value=updated_state.get("annotated_image_path"), visible=False),
        gr.update(value=updated_state.get("predicted_mask_path"), visible=False),
        gr.update(visible=False),
        measurement_rows(updated_state),
        updated_state,
        current,
        prompt_update,
        "",
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(value=_editor_value(canvas_background) if show_canvas else None),
        gr.update(value=_editor_value(canvas_background) if show_canvas else None),
    )


def _resume_human_review(store, task_id, state, action, **response_fields):
    thread_id = state.get("graph_thread_id")
    if not thread_id or not state.get("interrupt"):
        return state
    try:
        response = {"action": action, **{key: value for key, value in response_fields.items() if value}}
        resumed = resume_agent_graph(thread_id, response)
    except Exception as exc:
        # File-backed state remains usable after an app restart, when the
        # in-memory LangGraph checkpoint is no longer available.
        store.append_event(task_id, "human_review_resume_unavailable", {
            "thread_id": thread_id,
            "action": action,
            "error": f"{type(exc).__name__}: {exc}",
        })
        return state
    state.update({
        "agent_status": resumed.get("agent_status", state.get("agent_status")),
        "human_response": resumed.get("human_response"),
        "human_feedback": resumed.get("human_feedback", state.get("human_feedback", {})),
        "trajectory": resumed.get("trajectory", state.get("trajectory", [])),
        "interrupt": resumed.get("interrupt", []),
    })
    store.append_event(task_id, "human_review_resumed", {
        "thread_id": thread_id,
        "action": action,
        "agent_status": state.get("agent_status"),
    })
    return state


def reset_feedback_canvas(previous_state, image_path):
    state = _recover_agent_state(None, previous_state)
    background = (state or {}).get("annotated_image_path") or image_path
    return gr.update(value=_editor_value(background)), gr.update(value=_editor_value(background))


def cancel_feedback_canvas():
    """Close the feedback panel without mutating the current annotation state."""
    return (
        gr.update(visible=True),
        gr.update(value=None),
        gr.update(value=None),
    )


def reset_chat_task():
    task = create_chat_task()
    return (
        [
            {
                "role": "assistant",
                "content": "新任务已创建。请点击 + 上传图片，然后描述希望 Agent 标注的目标或效果。",
            }
        ],
        gr.update(value=None, visible=False),
        gr.update(value=None, visible=False),
        gr.update(visible=False),
        [],
        None,
        task,
        None,
        "",
        "",
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(choices=_task_choices(task["id"]), value=task["id"]),
        gr.update(value=None),
    )


def build_annotation_app():
    theme = gr.themes.Soft(
        primary_hue="gray",
        secondary_hue="gray",
        neutral_hue="gray",
        radius_size="sm",
        font=["Inter", "sans-serif"],
    )
    with gr.Blocks(title="DRAM 视觉算法开发 Agent", theme=theme, css=load_styles()) as app:
        task_state = gr.State(value=None)
        attachment_state = gr.State(value=None)
        agent_state = gr.State(value=None)
        with gr.Row(elem_id="annotation-shell"):
            with gr.Column(elem_id="annotation-sidebar"):
                gr.HTML(
                    """
                    <div id="annotation-brand">
                      <div class="brand-mark">CV</div>
                      <div><h1>Vision Agent</h1><p>DRAM 视觉算法开发工作区</p></div>
                    </div>
                    """
                )
                new_task_button = gr.Button(
                    "＋ 新建标注任务",
                    variant="secondary",
                    elem_id="annotation-new-task",
                )
                task_history = gr.Dropdown(
                    label="历史任务",
                    choices=_task_choices(),
                    interactive=True,
                    elem_id="annotation-task-history",
                )
                resume_task_button = gr.Button(
                    "继续所选任务",
                    variant="secondary",
                    elem_id="annotation-resume-task",
                )
                gr.HTML(
                    """
                    <div class="sidebar-group-label">工作区</div>
                    <div class="sidebar-nav active"><span class="nav-glyph">□</span>算法开发</div>
                    <div class="sidebar-group-label">项目</div>
                    <div class="sidebar-project"><span class="project-dot"></span>DRAM 缺陷量测</div>
                    """
                )
                gr.HTML('<div class="sidebar-spacer"></div>')
                gr.HTML('<div class="sidebar-footer"><span class="footer-dot"></span>本地工作区</div>')

            with gr.Column(elem_id="annotation-main"):
                gr.HTML(
                    """
                    <div id="annotation-topbar">
                      <div class="title">DRAM 缺陷量测 Agent</div>
                      <div class="status"><span class="dot"></span>仅保存在本地</div>
                    </div>
                    """
                )
                with gr.Column(elem_id="annotation-chat"):
                    gr.HTML(
                        """
                        <div id="annotation-welcome">
                          <h2>今天想处理什么？</h2>
                          <p>上传一张图片，用自然语言描述希望 Agent 标注的目标或效果。</p>
                        </div>
                        """
                    )
                    chatbot = gr.Chatbot(
                        value=[],
                        type="messages",
                        height="100%",
                        show_label=False,
                        show_copy_button=False,
                        elem_id="annotation-chatbot",
                    )
                    with gr.Row(elem_id="annotation-results"):
                        result_image = gr.Image(
                            label="算法标注结果",
                            type="filepath",
                            height=300,
                            show_download_button=True,
                            visible=False,
                        )
                        mask_image = gr.Image(
                            label="二值 mask",
                            type="filepath",
                            height=300,
                            show_download_button=True,
                            visible=False,
                        )
                    with gr.Accordion(
                        "量测明细",
                        open=False,
                        visible=False,
                        elem_id="annotation-measurements",
                    ) as measurements_panel:
                        measurements = gr.Dataframe(
                            headers=["ID", "面积", "宽", "高", "长宽比", "外接框"],
                            interactive=False,
                        )
                with gr.Group(
                    visible=True,
                    elem_id="annotation-feedback-canvas",
                ) as feedback_canvas:
                    gr.HTML(
                        '<div class="feedback-canvas-note">'
                        '两个画板分别固定为红色误检（需要删除）和绿色漏检（需要补充）；'
                        '标记完成后请同时写一句修改说明。'
                        '</div>'
                    )
                    # ImageEditor does not reliably preserve two tabbed canvas
                    # instances in Gradio. Keeping both canvases mounted makes
                    # their paint state and the action buttons deterministic.
                    feedback_editor = gr.ImageEditor(
                        label="误检删除（红色）",
                        type="filepath",
                        height=280,
                        sources=[],
                        format="png",
                        brush=gr.Brush(
                            default_size=12,
                            colors=["#ff3b30"],
                            default_color="#ff3b30",
                            color_mode="fixed",
                        ),
                        elem_id="annotation-feedback-editor",
                    )
                    green_feedback_editor = gr.ImageEditor(
                        label="漏检补充（绿色）",
                        type="filepath",
                        height=280,
                        sources=[],
                        format="png",
                        brush=gr.Brush(
                            default_size=12,
                            colors=["#22c55e"],
                            default_color="#22c55e",
                            color_mode="fixed",
                        ),
                        elem_id="annotation-green-feedback-editor",
                    )
                    with gr.Row(elem_id="annotation-feedback-controls"):
                        clear_feedback_button = gr.Button(
                            "清空标记",
                            variant="secondary",
                            elem_id="annotation-clear-feedback",
                        )
                        cancel_feedback_button = gr.Button(
                            "取消",
                            variant="secondary",
                            elem_id="annotation-cancel-feedback",
                        )
                        submit_feedback_button = gr.Button(
                            "提交修改",
                            variant="primary",
                            elem_id="annotation-submit-feedback",
                        )
                with gr.Group(
                    visible=False,
                    elem_id="annotation-next-actions",
                ) as next_actions:
                    gr.HTML(
                        """
                        <div class="next-action-title">这个标注准确吗？</div>
                        <div class="next-action-note">只需检查算法标注是否符合你的描述，不需要判断处理方法。</div>
                        """
                    )
                    with gr.Row(elem_classes=["next-action-row", "primary-action-row"]):
                        accept_button = gr.Button(
                            "标注准确",
                            elem_id="annotation-accept",
                        )
                        continue_button = gr.Button(
                            "有漏标或错标",
                            elem_id="annotation-continue",
                        )
                        exit_button = gr.Button(
                            "结束任务",
                            elem_id="annotation-exit",
                        )
                    with gr.Row(elem_classes=["next-action-row", "artifact-action-row"]):
                        show_mask_button = gr.Button("查看 Mask")
                        show_measurements_button = gr.Button("量测明细")
                with gr.Column(elem_id="annotation-composer"):
                    with gr.Row(elem_id="annotation-attachment-row"):
                        attachment_label = gr.Markdown("", elem_id="annotation-attachment-label")
                        view_attachment_button = gr.Button(
                            "查看缩略图",
                            elem_id="annotation-attachment-view",
                            visible=False,
                        )
                        clear_attachment_button = gr.Button(
                            "删除",
                            elem_id="annotation-attachment-clear",
                            visible=False,
                        )
                    attachment_preview = gr.Image(
                        value=None,
                        show_label=False,
                        show_download_button=False,
                        interactive=False,
                        visible=False,
                        elem_id="annotation-attachment-preview",
                    )
                    with gr.Row(elem_id="annotation-composer-row"):
                        attach_button = gr.UploadButton(
                            "+",
                            elem_id="annotation-attach",
                            type="filepath",
                            file_count="single",
                            file_types=["image"],
                        )
                        with gr.Accordion(
                            "Handbook 示例",
                            open=False,
                            elem_id="annotation-handbook-examples",
                        ):
                            handbook_examples = gr.File(
                                label="不同图，可选（不计算准确率）",
                                file_count="multiple",
                                file_types=["image"],
                                type="filepath",
                            )
                        with gr.Accordion(
                            "Ground Truth",
                            open=False,
                            elem_id="annotation-ground-truth",
                        ):
                            ground_truth_annotation = gr.File(
                                label="同一张图，可选（计算准确率）",
                                file_count="single",
                                file_types=["image"],
                                type="filepath",
                            )
                        prompt = gr.Textbox(
                            placeholder="描述你要标注的区域，或询问下一步怎么做...",
                            show_label=False,
                            lines=1,
                            elem_id="annotation-prompt",
                        )
                        send_button = gr.Button("↑", elem_id="annotation-send")

        agent_outputs = [
            chatbot,
            result_image,
            mask_image,
            measurements_panel,
            measurements,
            agent_state,
            task_state,
            prompt,
            attachment_label,
            next_actions,
            feedback_canvas,
            feedback_editor,
            green_feedback_editor,
        ]
        resume_outputs = [
            *agent_outputs[:7],
            attachment_state,
            prompt,
            attachment_label,
            next_actions,
            feedback_canvas,
            feedback_editor,
            green_feedback_editor,
            handbook_examples,
            task_history,
            ground_truth_annotation,
        ]

        attach_button.upload(
            fn=store_chat_attachment_ui,
            inputs=[attach_button, task_state],
            outputs=[
                attachment_state,
                attachment_label,
                task_state,
                view_attachment_button,
                clear_attachment_button,
                attachment_preview,
            ],
            show_progress="hidden",
            show_api=False,
        )
        view_attachment_button.click(
            fn=show_chat_attachment,
            inputs=[attachment_state],
            outputs=[attachment_preview],
            show_progress="hidden",
            show_api=False,
        )
        clear_attachment_button.click(
            fn=clear_chat_attachment,
            inputs=[attachment_state, task_state],
            outputs=[
                attachment_state,
                attachment_label,
                task_state,
                view_attachment_button,
                clear_attachment_button,
                attachment_preview,
            ],
            show_progress="hidden",
            show_api=False,
        )
        prompt_event = prompt.submit(
            fn=run_chat_agent_stream,
            inputs=[
                attachment_state, prompt, chatbot, task_state, agent_state,
                feedback_editor, green_feedback_editor, handbook_examples,
                ground_truth_annotation,
            ],
            outputs=agent_outputs,
            show_progress="hidden",
            show_api=False,
        )
        prompt_event.then(
            fn=consume_chat_attachment_ui,
            inputs=[],
            outputs=[
                attachment_state,
                attachment_label,
                view_attachment_button,
                clear_attachment_button,
                attachment_preview,
            ],
            show_progress="hidden",
            show_api=False,
        ).then(
            fn=None,
            js=SHOW_LATEST_RESULT_JS,
            show_progress="hidden",
            show_api=False,
        )
        send_event = send_button.click(
            fn=run_chat_agent_stream,
            inputs=[
                attachment_state, prompt, chatbot, task_state, agent_state,
                feedback_editor, green_feedback_editor, handbook_examples,
                ground_truth_annotation,
            ],
            outputs=agent_outputs,
            show_progress="hidden",
            show_api=False,
        )
        send_event.then(
            fn=consume_chat_attachment_ui,
            inputs=[],
            outputs=[
                attachment_state,
                attachment_label,
                view_attachment_button,
                clear_attachment_button,
                attachment_preview,
            ],
            show_progress="hidden",
            show_api=False,
        ).then(
            fn=None,
            js=SHOW_LATEST_RESULT_JS,
            show_progress="hidden",
            show_api=False,
        )
        submit_feedback_event = submit_feedback_button.click(
            fn=submit_canvas_feedback,
            inputs=[
                attachment_state, prompt, chatbot, task_state, agent_state,
                feedback_editor, green_feedback_editor, handbook_examples,
                ground_truth_annotation,
            ],
            outputs=agent_outputs,
            js=CLOSE_FEEDBACK_SUBMIT_JS,
            show_progress="hidden",
            trigger_mode="once",
            show_api=False,
        )
        submit_feedback_event.then(
            fn=None,
            js=SHOW_LATEST_RESULT_JS,
            show_progress="hidden",
            show_api=False,
        )
        new_task_event = new_task_button.click(
            fn=reset_chat_task,
            inputs=[],
            outputs=(
                agent_outputs[:7]
                + [
                    attachment_state,
                    prompt,
                    attachment_label,
                    next_actions,
                    feedback_canvas,
                    feedback_editor,
                    green_feedback_editor,
                    handbook_examples,
                    task_history,
                    ground_truth_annotation,
                ]
            ),
            show_progress="hidden",
            show_api=False,
        ).then(
            fn=sync_chat_attachment_ui,
            inputs=[attachment_state],
            outputs=[view_attachment_button, clear_attachment_button, attachment_preview],
            show_progress="hidden",
            show_api=False,
        )
        resume_task_button.click(
            fn=resume_chat_task,
            inputs=[task_history],
            outputs=resume_outputs,
            show_progress="hidden",
            show_api=False,
        ).then(
            fn=sync_chat_attachment_ui,
            inputs=[attachment_state],
            outputs=[view_attachment_button, clear_attachment_button, attachment_preview],
            show_progress="hidden",
            show_api=False,
        ).then(
            fn=None,
            js=SHOW_LATEST_RESULT_JS,
            show_progress="hidden",
            show_api=False,
        )
        app.load(
            fn=load_latest_chat_task,
            inputs=[],
            outputs=resume_outputs,
            show_progress="hidden",
            show_api=False,
        ).then(
            fn=sync_chat_attachment_ui,
            inputs=[attachment_state],
            outputs=[view_attachment_button, clear_attachment_button, attachment_preview],
            show_progress="hidden",
            show_api=False,
        ).then(
            fn=None,
            js=SHOW_LATEST_RESULT_JS,
            show_progress="hidden",
            show_api=False,
        )
        for button, action in (
            (accept_button, "accept"),
            (continue_button, "continue"),
            (exit_button, "exit"),
            (show_mask_button, "mask"),
            (show_measurements_button, "measurements"),
        ):
            action_state = gr.State(value=action)
            action_event = button.click(
                fn=handle_result_action,
                inputs=[action_state, attachment_state, chatbot, task_state, agent_state, prompt],
                outputs=agent_outputs,
                js=(
                    OPEN_FEEDBACK_ACTION_JS
                    if action == "continue"
                    else CLOSE_FEEDBACK_ACTION_JS
                ),
                show_progress="hidden",
                show_api=False,
            )
            action_event.then(
                fn=None,
                js=SHOW_LATEST_RESULT_JS if action == "continue" else SCROLL_CHAT_JS,
                show_progress="hidden",
                show_api=False,
            )
        clear_feedback_button.click(
            fn=reset_feedback_canvas,
            inputs=[agent_state, attachment_state],
            outputs=[feedback_editor, green_feedback_editor],
            show_progress="hidden",
            queue=False,
            show_api=False,
        )
        cancel_feedback_button.click(
            fn=cancel_feedback_canvas,
            inputs=[],
            outputs=[feedback_canvas, feedback_editor, green_feedback_editor],
            js=CLOSE_FEEDBACK_CANCEL_JS,
            show_progress="hidden",
            queue=False,
            cancels=[submit_feedback_event],
            show_api=False,
        )
    return app
