"""Structured, user-facing chat card formatters."""

from html import escape


def format_task_card(state: dict) -> str:
    """Render the completed annotation summary as a compact result card."""
    status = state.get("agent_status")
    labels = {
        "waiting_for_acceptance": ("任务完成", "#216e4e", "#e8f5e9"),
        "waiting_for_feedback": ("等待反馈", "#8a5a00", "#fef3c7"),
        "accepted": ("已确认", "#216e4e", "#e8f5e9"),
    }
    label, color, background = labels.get(status, ("任务完成", "#216e4e", "#e8f5e9"))
    summary = (state.get("measurements") or {}).get("summary") or {}
    count = summary.get("count", 0)
    area = summary.get("total_area", 0)
    try:
        area_label = f"{float(area):.0f} px2"
    except (TypeError, ValueError):
        area_label = "0 px2"
    evaluation = state.get("evaluation_report") or {}
    evaluation_html = ""
    if evaluation.get("status") == "ok":
        metrics = []
        for key, label in (
            ("dice", "Dice"),
            ("recall", "Recall"),
            ("precision", "Precision"),
            ("boundary_f1", "Boundary F1"),
        ):
            value = evaluation.get(key)
            if isinstance(value, (int, float)):
                metrics.append(f"<span><strong>{value:.3f}</strong> {label}</span>")
        if metrics:
            evaluation_html = (
                '<div class="task-card-evaluation"><span class="evaluation-label">'
                "同图 Ground Truth</span>"
                + "".join(metrics)
                + "</div>"
            )
    return (
        '<div class="task-card">'
        '<div class="task-card-header"><span>算法执行结果</span>'
        f'<span class="task-card-status" style="background:{background};color:{color};">'
        f'{escape(label)}</span></div>'
        '<div class="task-card-metrics">'
        f'<div><strong>{escape(str(count))}</strong> 个区域</div>'
        f'<div><strong>{escape(area_label.split()[0])}</strong> px2</div>'
        '</div>'
        f'{evaluation_html}'
        '</div>'
    )


def format_progress_card(events: list[dict], running: bool = False) -> str:
    """Render a compact Codex-like activity timeline.

    The timeline deliberately shows observable work (files, commands, model
    calls and pipeline stages) without exposing chain-of-thought text.
    """
    steps = []
    for event in events:
        event_type = event.get("type")

        if event_type is None or event_type == "progress":
            status = event.get("status", "running")
            icon = {"completed": "check", "failed": "error", "running": "pending"}.get(status, "dot")
            label = escape(str(event.get("label") or event.get("stage") or "执行步骤"))
            duration = event.get("duration_seconds")
            suffix = f"{float(duration):.2f}s" if isinstance(duration, (int, float)) else ""
            detail = str(event.get("detail") or "")
            steps.append(_activity_row(icon, label, detail, suffix, status))

        elif event_type == "thinking":
            steps.append(_activity_row("thinking", "正在分析任务", "", "", "running"))

        elif event_type == "tool_call":
            tool = str(event.get("tool", "unknown"))
            args = event.get("args", {})
            args_preview = escape(str(args)[:100])
            if len(str(args)) > 100:
                args_preview += "..."
            action, icon = _tool_label(tool)
            steps.append(_activity_row(icon, action, f"<code>{escape(tool)}</code> · {args_preview}", "", "running", raw_detail=True))

        elif event_type == "tool_result":
            tool = escape(str(event.get("tool", "unknown")))
            success = event.get("success", True)
            result_preview = escape(str(event.get("result", ""))[:80])
            if result_preview:
                steps.append(_activity_row("check" if success else "error", f"已完成 {tool}", result_preview, "", "completed" if success else "failed", raw_detail=True))

        elif event_type == "llm_request":
            provider = escape(str(event.get("provider", "unknown")))
            model = escape(str(event.get("model", "")))
            msg_count = event.get("message_count", 0)
            steps.append(_activity_row("model", f"请求 {provider}", f"{model} · {msg_count} 条消息", "", "running"))

        elif event_type == "llm_response":
            steps.append(_activity_row("check", "已收到模型响应", "", "", "completed"))

        elif event_type == "llm_chunk":
            if event.get("content"):
                steps.append(_activity_row("model", "模型正在输出", "", "", "running"))

        elif event_type == "node_progress":
            node = escape(str(event.get("node") or "执行步骤"))
            message = str(event.get("message") or "")
            progress = event.get("progress")
            suffix = f" · {float(progress):.0f}%" if isinstance(progress, (int, float)) else ""
            steps.append(_activity_row("pending", node, message, suffix, "running"))

        # 处理错误事件
        elif event_type == "error":
            error_msg = escape(str(event.get("message", "未知错误")))
            node = event.get("node")
            node_html = f" (在 {escape(str(node))})" if node else ""
            steps.append(_activity_row("error", f"错误{node_html}", error_msg, "", "failed", raw_detail=True))

    if not steps:
        steps.append(_activity_row("pending", "准备任务", "", "", "running"))
    state = "正在继续执行" if running else "执行记录已完成"
    steps_html = "".join(steps)
    return (
        '<details open class="progress-card"><summary><span class="progress-summary-mark">⌁</span>Agent 执行过程'
        f'<span class="progress-state">{state}</span></summary><div class="progress-steps">{steps_html}</div></details>'
    )


def _tool_label(tool: str) -> tuple[str, str]:
    lowered = tool.lower()
    if any(word in lowered for word in ("read", "load", "open")):
        return "读取文件", "read"
    if any(word in lowered for word in ("edit", "write", "save", "patch")):
        return "编辑文件", "edit"
    if any(word in lowered for word in ("run", "exec", "shell", "command")):
        return "运行命令", "command"
    if any(word in lowered for word in ("search", "find", "grep", "rg")):
        return "搜索代码", "search"
    return "调用工具", "tool"


def _activity_row(icon: str, label: str, detail: str = "", meta: str = "", status: str = "running", *, raw_detail: bool = False) -> str:
    detail_html = f'<span class="activity-detail">{detail if raw_detail else escape(detail)}</span>' if detail else ""
    meta_html = f'<span class="activity-meta">{escape(meta)}</span>' if meta else ""
    return (
        f'<div class="activity-row activity-{escape(status)}">'
        f'<span class="activity-icon icon-{escape(icon)}" aria-hidden="true"></span>'
        f'<span class="activity-label">{label}</span>{meta_html}{detail_html}</div>'
    )
