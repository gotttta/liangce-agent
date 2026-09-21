"""Structured, user-facing chat card formatters."""

from html import escape

from ui.utils.progress import short_candidate_name


def format_task_card(state: dict) -> str:
    """Render the completed annotation summary as a compact result card."""
    status = state.get("agent_status")
    labels = {
        "waiting_for_acceptance": ("待人工确认", "status-pending"),
        "waiting_for_feedback": ("等待反馈", "status-feedback"),
        "accepted": ("已验收", "status-accepted"),
    }
    label, status_class = labels.get(status, ("任务完成", "status-pending"))
    structured = (state.get("measurements") or {}).get("structured_outputs")
    if structured and not state.get("predicted_mask_path"):
        import json
        body = escape(json.dumps(structured, ensure_ascii=False, indent=2))
        return ('<div class="task-card"><div class="task-card-header">算法执行结果 · '
                + escape(label) + '</div><pre style="white-space:pre-wrap">' + body + '</pre></div>')
    summary = (state.get("measurements") or {}).get("summary") or {}
    count = summary.get("count", 0)
    area = summary.get("total_area", 0)
    try:
        area_label = f"{float(area):.0f} px²"
    except (TypeError, ValueError):
        area_label = "0 px²"
    evaluation = state.get("evaluation_report") or {}
    evaluation_html = ""
    if evaluation.get("status") == "ok":
        metrics = []
        for key, metric_label in (
            ("dice", "Dice"),
            ("recall", "Recall"),
            ("precision", "Precision"),
            ("boundary_f1", "Boundary F1"),
        ):
            value = evaluation.get(key)
            if isinstance(value, (int, float)):
                metrics.append(f"<span><strong>{value:.3f}</strong> {metric_label}</span>")
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
        f'<span class="task-card-status {status_class}">'
        f'{escape(label)}</span></div>'
        '<div class="task-card-metrics">'
        f'<div><strong>{escape(str(count))}</strong> 个区域</div>'
        f'<div><strong>{escape(area_label.split()[0])}</strong> px²</div>'
        '</div>'
        f'{evaluation_html}'
        '</div>'
    )


_GROUP_ICON = {"failed": "error", "warning": "dot", "skipped": "dot"}


def format_tokens(count) -> str:
    """Compact token count, e.g. 980 / 34.6K / 1.2M."""
    try:
        count = int(count)
    except (TypeError, ValueError):
        return "0"
    if count >= 1_000_000:
        value = count / 1_000_000
        return f"{value:.1f}M" if value < 10 else f"{value:.0f}M"
    if count >= 1000:
        value = count / 1000
        return f"{value:.1f}K" if value < 100 else f"{value:.0f}K"
    return str(count)


def format_context_usage(usage_state) -> str:
    """Render the topbar context-window chip (Codex-style); '' keeps it hidden.

    ``usage_state`` accumulates per chat turn: ``last`` is the latest
    llm_response usage (its prompt_tokens is the current context footprint),
    ``window`` the model context window, ``turn_tokens`` the running total.
    """
    state = usage_state if isinstance(usage_state, dict) else {}
    usage = state.get("last")
    window = state.get("window")
    used = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    if not isinstance(used, (int, float)) or used < 0:
        return ""
    if not isinstance(window, (int, float)) or window <= 0:
        return ""
    percent = max(0.0, min(100.0, used / window * 100))
    level = "danger" if percent >= 90 else "warning" if percent >= 70 else "ok"
    title = (
        f"模型 {state.get('model') or '未知'} · "
        f"上次请求输入 {int(used):,} / {int(window):,} tokens"
    )
    if usage.get("estimated"):
        title += "（估算值）"
    turn = state.get("turn_tokens")
    if isinstance(turn, (int, float)) and turn > 0:
        title += f" · 本轮累计 {int(turn):,} tokens"
    return (
        f'<div class="context-usage context-{level}" title="{escape(title)}">'
        '<span class="context-bar" aria-hidden="true">'
        f'<span class="context-bar-fill" style="width:{percent:.0f}%"></span></span>'
        f'<span class="context-text">上下文 {percent:.0f}% · '
        f'{format_tokens(used)}/{format_tokens(int(window))}</span></div>'
    )


def format_progress_card(events: list[dict], running: bool = False) -> str:
    """Render a ZCode-style activity tree.

    Top-level rows stay flat; candidate attempts are emitted as `group` events
    and everything reported while a group is open nests underneath it. Rows
    flip in place from running to completed: only the newest running row keeps
    the pulse, everything older reads as done. Groups follow the same rule —
    历史收起、当前展开；整轮结束后全部收起。连续的思考行合并成一个可折叠的
    “思考过程”组：进行中自动展开，结束后收起。
    """
    sections = _merge_thinking_rows(_collect_sections(events))
    last_running_id = _last_running_id(events)
    last_open_group = next(
        (
            section.get("stage")
            for section in reversed(sections)
            if section.get("type") == "group" and section.get("children")
        ),
        None,
    )
    steps = []
    for section in sections:
        if section.get("type") == "thinking_group":
            steps.append(_thinking_group_html(section, running=running, last_running_id=last_running_id))
        elif section.get("type") == "group" and section.get("children"):
            steps.append(_group_html(
                section,
                running=running,
                open_group=running and section.get("stage") == last_open_group,
                last_running_id=last_running_id,
            ))
        else:
            steps.append(_row_html(section, running=running, last_running_id=last_running_id))
    if not steps:
        steps.append(_activity_row("pending", "准备任务", "", "", "running"))
    state = "正在继续执行" if running else "执行记录已完成"
    steps_html = "".join(steps)
    return (
        '<details open class="progress-card"><summary><span class="progress-summary-mark">⌁</span>Agent 执行过程'
        f'<span class="progress-state">{state}</span></summary><div class="progress-steps">{steps_html}</div></details>'
    )


def _collect_sections(events):
    """Split the flat event list into ordered rows and groups with children.

    行对象保持原引用（不拷贝），渲染层的 id() 配对依赖原始对象身份。
    """
    sections = []
    for event in events:
        if event.get("type") == "group":
            sections.append({**event, "children": []})
            continue
        group_id = event.get("group")
        if group_id:
            group = next(
                (s for s in sections if s.get("type") == "group" and s.get("stage") == group_id),
                None,
            )
            if group is not None:
                group["children"].append(event)
                continue
        sections.append(event)
    return sections


def _merge_thinking_rows(events):
    """把连续的 thinking 行合并为一个思考组，同一嵌套层级内的才相邻。"""
    merged = []
    for event in events:
        if event.get("type") == "thinking":
            current = merged[-1] if merged else None
            if (
                current is not None
                and current.get("type") == "thinking_group"
                and current.get("group") == event.get("group")
            ):
                current["items"].append(event)
                continue
            merged.append({"type": "thinking_group", "group": event.get("group"), "items": [event]})
            continue
        merged.append(event)
    return merged


def _last_running_id(events):
    last_id = None
    for event in events:
        if event.get("type") == "group":
            continue
        if event.get("status", "running") == "running":
            last_id = id(event)
    return last_id


def _effective_status(event, running, last_running_id):
    status = event.get("status", "running")
    if status == "running" and (id(event) != last_running_id or not running):
        return "completed"
    return status


def _row_html(event, running, last_running_id):
    event_type = event.get("type")
    if event_type == "group":
        # 没有子步骤的分组（恢复历史重建时）退化为单行摘要。
        status = event.get("status") or "completed"
        icon = _GROUP_ICON.get(status, "check" if status == "completed" else "pending")
        label = escape(str(event.get("label") or "识别方法"))
        name = short_candidate_name(event.get("name"))
        if name:
            label = f"{label} · {escape(name)}"
        return _activity_row(icon, label, str(event.get("detail") or ""), "", status)
    status = _effective_status(event, running, last_running_id)
    if event_type in {"llm_chunk", "llm_response"}:
        return ""
    if event_type is None or event_type == "progress":
        icon = {"completed": "check", "failed": "error", "running": "pending", "warning": "dot", "skipped": "dot"}.get(status, "dot")
        label = escape(str(event.get("label") or event.get("stage") or "执行步骤"))
        duration = event.get("duration_seconds")
        suffix = f"{float(duration):.2f}s" if isinstance(duration, (int, float)) else ""
        detail = str(event.get("detail") or "")
        return _activity_row(icon, label, detail, suffix, status)
    if event_type == "tool_call":
        tool = str(event.get("tool", "unknown"))
        args = event.get("args", {})
        args_preview = escape(str(args)[:100])
        if len(str(args)) > 100:
            args_preview += "..."
        action, icon = _tool_label(tool)
        return _activity_row(icon, action, f"<code>{escape(tool)}</code> · {args_preview}", "", status, raw_detail=True)
    if event_type == "tool_result":
        tool = str(event.get("tool", "unknown"))
        success = event.get("success", True)
        result_preview = escape(str(event.get("result", ""))[:80])
        duration = event.get("duration_seconds")
        suffix = f"{float(duration):.1f}s" if isinstance(duration, (int, float)) else ""
        action, icon = _tool_label(tool)
        if result_preview:
            return _activity_row(
                icon, action, f"<code>{escape(tool)}</code> {result_preview}", suffix,
                "completed" if success else "failed", raw_detail=True,
            )
        return ""
    if event_type == "llm_request":
        provider = escape(str(event.get("provider", "unknown")))
        model = escape(str(event.get("model", "")))
        msg_count = event.get("message_count", 0)
        return _activity_row("model", f"请求 {provider}", f"{model} · {msg_count} 条消息", "", status)
    if event_type == "node_progress":
        node = escape(str(event.get("node") or "执行步骤"))
        message = str(event.get("message") or "")
        progress = event.get("progress")
        suffix = f" · {float(progress):.0f}%" if isinstance(progress, (int, float)) else ""
        return _activity_row("pending", node, message, suffix, status)
    if event_type == "error":
        error_msg = escape(str(event.get("message", "未知错误")))
        node = event.get("node")
        node_html = f" (在 {escape(str(node))})" if node else ""
        return _activity_row("error", f"错误{node_html}", error_msg, "", "failed", raw_detail=True)
    return ""


def _group_html(group, running, open_group, last_running_id):
    children = _merge_thinking_rows(group.get("children") or [])
    child_rows = "".join(
        _thinking_group_html(child, running=running, last_running_id=last_running_id)
        if child.get("type") == "thinking_group"
        else _row_html(child, running=running, last_running_id=last_running_id)
        for child in children
    )
    status = group.get("status") or _derive_group_status(children, running, last_running_id)
    icon = _GROUP_ICON.get(status, "check" if status == "completed" else "pending")
    label = escape(str(group.get("label") or "识别方法"))
    name = short_candidate_name(group.get("name"))
    if name:
        label = f"{label} · {escape(name)}"
    detail = str(group.get("detail") or "")
    total = sum(
        float(child.get("duration_seconds"))
        for child in children
        if isinstance(child.get("duration_seconds"), (int, float))
    )
    meta = f"{total:.1f}s" if total > 0.005 else ""
    header = (
        f'<span class="group-caret" aria-hidden="true"></span>'
        + _activity_row(icon, label, detail, meta, status)
    )
    return (
        f'<details{" open" if open_group else ""} class="attempt-group">'
        f'<summary>{header}</summary>'
        f'<div class="attempt-group-steps">{child_rows}</div></details>'
    )


def _derive_group_status(children, running, last_running_id):
    statuses = []
    for child in children:
        if child.get("type") == "thinking_group":
            statuses.extend(
                _effective_status(item, running, last_running_id)
                for item in child.get("items") or []
            )
        else:
            statuses.append(_effective_status(child, running, last_running_id))
    if "failed" in statuses:
        return "failed"
    if "warning" in statuses:
        return "warning"
    if "running" in statuses:
        return "running"
    return "completed"


def _thinking_group_html(section, running, last_running_id):
    """连续思考行渲染为一个折叠组：进行中展开、完成后收起。"""
    items = section.get("items") or []
    statuses = [_effective_status(item, running, last_running_id) for item in items]
    if "running" in statuses:
        status = "running"
    elif "failed" in statuses:
        status = "failed"
    else:
        status = "completed"
    rows = "".join(_thinking_item_html(item, running, last_running_id) for item in items)
    latest = str(items[-1].get("message") or "").strip()
    preview = escape(latest[:48] + "…") if len(latest) > 48 else escape(latest)
    timestamps = [
        float(item["timestamp"]) for item in items if isinstance(item.get("timestamp"), (int, float))
    ]
    total = timestamps[-1] - timestamps[0] if len(timestamps) >= 2 else 0.0
    meta = f"{total:.1f}s" if total > 0.05 else f"{len(items)} 条"
    header = (
        '<div class="thinking-head">'
        '<span class="group-caret" aria-hidden="true"></span>'
        '<span class="activity-icon icon-thinking" aria-hidden="true"></span>'
        '<span class="activity-label">思考过程</span>'
        f'<span class="thinking-preview">{preview}</span>'
        f'<span class="thinking-meta">{meta}</span>'
        '</div>'
    )
    return (
        f'<details{" open" if running and status == "running" else ""} '
        f'class="thinking-group is-{status}">'
        f'<summary>{header}</summary>'
        f'<div class="thinking-group-items">{rows}</div></details>'
    )


def _thinking_item_html(item, running, last_running_id):
    status = _effective_status(item, running, last_running_id)
    message = str(item.get("message") or "").strip() or "正在分析任务"
    return (
        f'<div class="activity-row activity-{escape(status)} thinking-item">'
        f'<span class="activity-icon icon-dot" aria-hidden="true"></span>'
        f'<span class="activity-label">{escape(message)}</span></div>'
    )


def _tool_label(tool: str) -> tuple[str, str]:
    lowered = tool.lower()
    exact = {
        "execute_pipeline_sandbox": ("执行流水线", "command"),
    }
    if tool in exact:
        return exact[tool]
    if any(word in lowered for word in ("read", "load", "open")):
        return "读取文件", "read"
    if any(word in lowered for word in ("edit", "write", "save", "patch")):
        return "编辑文件", "edit"
    if any(word in lowered for word in ("run", "exec", "shell", "command", "sandbox", "pipeline")):
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
