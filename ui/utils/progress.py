"""Progress event aggregation and candidate presentation, independent of Gradio."""

from html.parser import HTMLParser

from core.measurement.evaluation import meets_ground_truth_gate


class _ProgressDetector(HTMLParser):
    found = False

    def handle_starttag(self, tag, attrs):
        if tag == "details" and "progress-card" in dict(attrs).get("class", "").split():
            self.found = True


def is_progress_message(message):
    if message.get("role") != "assistant" or not isinstance(message.get("content"), str):
        return False
    parser = _ProgressDetector()
    parser.feed(message["content"])
    return parser.found


def progress_event(stage, label, detail=None, status="running", duration_seconds=None):
    event = {"stage": stage, "label": label, "status": status}
    if detail:
        event["detail"] = detail
    if duration_seconds is not None:
        event["duration_seconds"] = round(float(duration_seconds), 3)
    return event


def short_candidate_name(name):
    """accepted::bright_threshold → bright_threshold（去掉内部命名空间前缀）。"""
    text = str(name or "").strip()
    return text.rsplit("::", 1)[-1] or text


def merge_progress_event(events, event):
    stage = event.get("stage")
    event_type = event.get("type")
    if event_type == "llm_chunk":
        # 分块输出只是流式噪音：请求行已经以脉冲状态展示，不落时间线。
        return
    if event_type == "llm_response":
        # 响应让最近一条"请求 …"行原地完成，而不是另起一行。
        for index in range(len(events) - 1, -1, -1):
            if events[index].get("type") == "llm_request":
                events[index] = {**events[index], "status": "completed"}
                return
        return
    if event_type == "group" and event.get("name"):
        # 候选汇总按名称回填到实时分组头上（状态/详情），避免同一尝试出现两行。
        for index, item in enumerate(events):
            if (
                item.get("type") == "group"
                and item.get("stage") != stage
                and item.get("name") == event.get("name")
            ):
                events[index] = {
                    **item,
                    "status": event.get("status") or item.get("status"),
                    "detail": event.get("detail") or item.get("detail"),
                }
                return
    if event_type and not stage:
        events.append(dict(event))
        return
    for index in range(len(events) - 1, -1, -1):
        if events[index].get("stage") == stage:
            events[index] = event
            return
    events.append(event)


def candidate_progress_events(state):
    events = []
    selected = state.get("selected_candidate")
    for index, attempt in enumerate(state.get("candidate_attempts") or [], start=1):
        status = attempt.get("status")
        quality = attempt.get("quality") or {}
        evaluation = quality.get("evaluation") or {}
        display_status = "completed"
        if status == "failed":
            invalid = attempt.get("failure_type") == "pipeline_invalid" or "pipeline_invalid" in quality.get("issues", [])
            detail = "方案校验失败，未执行。" if invalid else "执行失败。"
            display_status = "failed"
        elif status == "duplicate_pipeline":
            detail, display_status = "重复方案，已跳过。", "skipped"
        elif status == "no_annotation" or "empty_mask" in (quality.get("health") or {}).get("issues", []):
            detail, display_status = "运行完成，未检出目标。", "warning"
        elif status == "health_failed":
            detail, display_status = "已生成结果，质量检查未通过，待优化。", "warning"
        elif status in {"completed", "selected_for_review"}:
            if attempt.get("name") == selected and state.get("agent_status") == "accepted":
                detail = "已验收。"
            elif meets_ground_truth_gate(evaluation):
                detail = "指标达标，待人工确认。" if attempt.get("name") == selected else "指标达标。"
            elif evaluation:
                detail, display_status = "已生成结果，指标未达标，待优化。", "warning"
            else:
                detail = "已生成结果，待人工确认。"
        else:
            detail, display_status = "状态待确认。", "skipped"
        source = (attempt.get("source") or {}).get("type")
        kind = {"accepted_algorithm": "复用算法", "ground_truth_calibration": "参数校准"}.get(source, "识别方法")
        events.append({
            "type": "group",
            "stage": f"pipeline_{attempt.get('index', len(events))}",
            "label": f"{kind} {index}",
            "name": attempt.get("name"),
            "detail": detail,
            "status": display_status,
        })
    return events
