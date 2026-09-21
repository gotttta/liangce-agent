from ui.utils.progress import candidate_progress_events, merge_progress_event
from ui.utils.formatters import format_progress_card, format_task_card


def test_candidate_outcomes_distinguish_failures_quality_and_acceptance():
    evaluation = {"status": "ok", "dice": .96, "recall": .98, "precision": .93, "boundary_f1": .94}
    attempts = [
        {"name": "invalid", "status": "failed", "failure_type": "pipeline_invalid"},
        {"name": "error", "status": "failed", "failure_type": "pipeline_execution_failed"},
        {"name": "empty", "status": "no_annotation"},
        {"name": "weak", "status": "selected_for_review", "quality": {"evaluation": {**evaluation, "recall": .5}}},
        {"name": "good", "status": "selected_for_review", "quality": {"evaluation": evaluation}},
        {"name": "duplicate", "status": "duplicate_pipeline"},
    ]
    events = candidate_progress_events({"selected_candidate": "good", "candidate_attempts": attempts})
    assert [e["status"] for e in events] == ["failed", "failed", "warning", "warning", "completed", "skipped"]
    assert "未执行" in events[0]["detail"]
    assert "待人工确认" in events[4]["detail"]
    assert 'icon-error' not in format_progress_card([events[4]])
    accepted = candidate_progress_events({"selected_candidate": "good", "candidate_attempts": attempts, "agent_status": "accepted"})
    assert accepted[4]["detail"] == "已验收。"
    card = format_task_card({"agent_status": "waiting_for_acceptance", "evaluation_report": evaluation})
    assert "待人工确认</span>" in card


def test_progress_message_detection_does_not_match_user_text():
    from ui.utils.progress import is_progress_message
    html = format_progress_card([])
    assert is_progress_message({"role": "assistant", "content": html})
    assert not is_progress_message({"role": "user", "content": html})
    assert not is_progress_message({"role": "assistant", "content": "progress-card"})


def test_executed_candidates_are_completed_while_failed_candidates_remain_failed():
    events = candidate_progress_events({
        "selected_candidate": "selected",
        "candidate_attempts": [
            {"name": "selected", "status": "selected_for_review"},
            {"name": "alternative", "status": "selected_for_review"},
            {"name": "invalid", "status": "failed"},
        ],
    })
    assert [event["status"] for event in events] == ["completed", "completed", "failed"]


def test_progress_updates_replace_stage_without_dropping_tool_events():
    events = []
    merge_progress_event(events, {"stage": "execute", "status": "running"})
    merge_progress_event(events, {"type": "tool_call", "tool": "threshold"})
    merge_progress_event(events, {"stage": "execute", "status": "completed"})
    assert events == [
        {"stage": "execute", "status": "completed"},
        {"type": "tool_call", "tool": "threshold"},
    ]


def test_candidate_summaries_merge_into_live_attempt_groups():
    events = []
    merge_progress_event(events, {
        "type": "group", "stage": "attempt:1", "label": "识别方法 1",
        "name": "accepted::bright_x", "status": "running",
    })
    merge_progress_event(events, {
        "type": "tool_call", "stage": "tool:1:t", "tool": "execute_pipeline_sandbox",
        "group": "attempt:1",
    })
    merge_progress_event(events, {
        "type": "group", "stage": "pipeline_0", "label": "识别方法 1",
        "name": "accepted::bright_x", "status": "completed", "detail": "已生成结果，待人工确认。",
    })

    # 汇总回填到实时分组头上，而不是再追加一行重复的尝试。
    assert [event.get("type") for event in events] == ["group", "tool_call"]
    assert events[0]["status"] == "completed"
    assert events[0]["detail"] == "已生成结果，待人工确认。"


def test_llm_response_completes_request_row_and_stream_noise_is_dropped():
    events = []
    merge_progress_event(events, {"type": "llm_request", "provider": "Aliyun", "model": "qwen", "message_count": 1})
    merge_progress_event(events, {"type": "llm_chunk", "content": "abc"})
    merge_progress_event(events, {"type": "llm_response", "provider": "Aliyun"})

    assert len(events) == 1
    assert events[0]["type"] == "llm_request"
    assert events[0]["status"] == "completed"


def test_progress_card_keeps_only_current_group_open_while_running():
    events = [
        {"type": "group", "stage": "a1", "label": "识别方法 1", "name": "accepted::bright_x", "status": "running"},
        {"type": "thinking", "message": "应用用户约束", "status": "running", "group": "a1"},
        {"type": "group", "stage": "a2", "label": "识别方法 2", "name": "global_y", "status": "running"},
        {"type": "thinking", "message": "测量组件", "status": "running", "group": "a2"},
    ]

    live = format_progress_card(events, running=True)
    assert live.count('<details class="attempt-group">') == 1
    assert live.count('<details open class="attempt-group">') == 1
    # open 的 details 位于当前组（识别方法 2）的标题之前；历史组在前且已收起
    assert live.index('<details open class="attempt-group">') < live.index("识别方法 2")
    assert live.index("识别方法 1") < live.index("识别方法 2")
    # 当前组的子步骤可见
    assert "测量组件" in live

    finished = format_progress_card(events, running=False)
    assert finished.count('<details open class="attempt-group">') == 0


def test_consecutive_thinking_rows_collapse_into_one_fold():
    events = [
        {"type": "progress", "stage": "prepare", "label": "准备输入", "status": "completed"},
        {"type": "tool_call", "tool": "read_image"},
        {"type": "thinking", "message": "正在调用视觉模型理解任务...", "timestamp": 10.0},
        {"type": "thinking", "message": "任务理解完成，生成了候选算法", "timestamp": 12.4},
        {"type": "thinking", "message": "选择最佳候选: bright_x", "timestamp": 12.6},
    ]
    card = format_progress_card(events, running=True)
    # 连续思考合并为一个折叠组，不逐行平铺
    assert card.count('class="thinking-group is-') == 1
    assert "思考过程" in card
    assert card.count("正在调用视觉模型理解任务") == 1
    # 思考仍在进行：折叠组自动展开
    assert '<details open class="thinking-group is-running">' in card
    # 折叠头给出最新一条思考作为预览，折叠体保留完整内容
    assert "选择最佳候选: bright_x" in card

    finished = format_progress_card(events, running=False)
    assert '<details class="thinking-group is-completed">' in finished
    assert "2.6s" in finished


def test_thinking_groups_break_at_other_events():
    events = [
        {"type": "thinking", "message": "第一次思考"},
        {"type": "tool_call", "tool": "read_image"},
        {"type": "thinking", "message": "第二次思考"},
    ]
    card = format_progress_card(events, running=False)
    assert card.count('class="thinking-group is-') == 2


def test_stale_running_rows_flip_to_completed_and_only_latest_pulses():
    events = [
        {"type": "thinking", "message": "应用用户约束", "status": "running"},
        {"type": "thinking", "message": "评估质量", "status": "running"},
    ]
    card = format_progress_card(events, running=True)
    assert card.count("activity-running") == 1
    assert card.count("activity-completed") == 1
    # 折叠头本身不带 activity-* 状态类，脉冲只出现在最新一条思考上
    assert '<details open class="thinking-group is-running">' in card
    assert "评估质量" in card


def test_childless_candidate_group_renders_as_single_row_without_internal_names():
    events = candidate_progress_events({
        "selected_candidate": None,
        "candidate_attempts": [{"name": "accepted::dark_ellipse", "status": "no_annotation"}],
    })
    card = format_progress_card(events, running=False)
    assert "accepted::" not in card
    assert "识别方法 1 · dark_ellipse" in card
    assert "运行完成，未检出目标。" in card
