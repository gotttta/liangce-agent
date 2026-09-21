import os

import pytest
from PIL import Image

from providers.vision import (
    AliyunVisionProvider,
    MockVisionProvider,
    build_revision_context_text,
    build_runtime_provider,
    build_candidate_review_messages,
    build_task_understanding_messages,
    extract_json_object,
    image_content,
    load_env_file,
    normalize_task_understanding,
    summarize_candidate_attempt,
)


def test_mock_provider_returns_normalized_strategy():
    provider = MockVisionProvider()

    strategy = provider.create_strategy(
        target_image_path="target.png",
        description="找亮色残留，量面积和数量",
        reference_annotation_path=None,
        previous_state=None,
    )

    assert strategy["measurement_type"] == "area_count"
    assert strategy["segmentation"]["method"] == "bright_threshold"
    assert strategy["segmentation"]["min_area_px"] == 20


def test_mock_provider_returns_structured_task_understanding():
    understanding = MockVisionProvider().understand_task(
        target_image_path="target.png",
        description="检测用户定义的缺陷",
    )

    assert understanding["task_summary"] == "检测用户定义的缺陷"
    assert understanding["candidate_plans"]
    assert understanding["recommended_strategy"]["measurement_type"] == "area_count"


def test_task_understanding_builds_executable_pipeline_and_rendering_from_request():
    understanding = normalize_task_understanding({
        "task_summary": "用荧光绿色提取白色长条",
        "recommended_strategy": {
            "segmentation": {"method": "bright_threshold", "sensitivity": 1.5}
        },
        "candidate_pipelines": [{
            "name": "generated_threshold",
            "pipeline": {
                "steps": [{"id": "final_mask", "op": "threshold_mask", "input": "image"}],
                "generated_operators": [{
                    "name": "threshold_mask",
                    "source": "def apply(data, params):\n    return data > np.mean(data)",
                }],
            },
        }],
    })

    assert understanding["candidate_pipelines"][0]["pipeline"]["steps"]
    assert understanding["rendering"]["contour_color"] == "#39FF14"


def test_task_understanding_normalizes_task_specific_acceptance_criteria():
    understanding = normalize_task_understanding({
        "task_summary": "找出每个亮色斑点",
        "acceptance_criteria": {
            "task_goal": "每个斑点都要被标出",
            "requested_output": ["points"],
            "visual_checks": ["每个斑点位置都有一个点", "不应在背景纹理上出现点"],
            "failure_examples": ["漏掉斑点", "把背景纹理当成斑点"],
        },
        "candidate_pipelines": [{
            "name": "candidate",
            "pipeline": {"steps": [{"id": "final_mask", "op": "global_threshold", "input": "image"}]},
        }],
    })

    assert understanding["acceptance_criteria"]["task_goal"] == "每个斑点都要被标出"
    assert understanding["acceptance_criteria"]["visual_checks"] == [
        "每个斑点位置都有一个点",
        "不应在背景纹理上出现点",
    ]


def test_count_constraint_is_dynamic_and_only_explicit_user_count_is_hard():
    raw = {
        "task_summary": "图中观察到9个椭圆",
        "target_constraints": {"expected_count": 9},
        "candidate_pipelines": [{
            "name": "candidate",
            "pipeline": {
                "steps": [{"id": "mask", "op": "global_threshold", "input": "image", "params": {}}, {
                    "id": "filter",
                    "op": "filter_components",
                    "input": "mask",
                    "params": {"min_area": 10, "max_components": 9},
                }],
            },
        }],
    }

    observed = normalize_task_understanding(raw)
    assert observed["target_constraints"]["observed_count"] == 9
    assert "expected_count" not in observed["target_constraints"]
    assert "max_components" not in observed["candidate_pipelines"][0]["pipeline"]["steps"][1]["params"]

    explicit = normalize_task_understanding(raw, task_description="提取10个椭圆")
    assert explicit["target_constraints"]["expected_count"] == 10
    assert explicit["target_constraints"]["count_source"] == "user_explicit"
    assert explicit["acceptance_criteria"]["count_policy"] == "exact"


def test_review_prompt_includes_dynamic_acceptance_criteria(tmp_path):
    target = tmp_path / "target.png"
    Image.new("L", (4, 4), 0).save(target)
    messages = build_candidate_review_messages(
        target,
        "找斑点",
        [],
        acceptance_criteria={
            "task_goal": "每个斑点都要被标出",
            "requested_output": ["points"],
            "visual_checks": ["点要落在斑点中心"],
            "failure_examples": ["点落在背景上"],
        },
    )
    prompt = messages[0]["content"][0]["text"]
    assert "每个斑点都要被标出" in prompt
    assert "点要落在斑点中心" in prompt
    assert "填洞" not in prompt


def test_task_understanding_preserves_valid_generated_operator_in_pipeline():
    understanding = normalize_task_understanding({
        "task_summary": "提取特殊纹理",
        "recommended_strategy": {"segmentation": {"method": "bright_threshold"}},
        "candidate_pipelines": [{
            "name": "custom_texture",
            "pipeline": {
                "steps": [{"id": "final_mask", "op": "texture_mask", "input": "image", "params": {}}],
                "generated_operators": [{
                    "name": "texture_mask",
                    "input_artifact": "ImageArtifact",
                    "output_artifact": "MaskArtifact",
                    "source": "def apply(data, params):\n    return data > np.mean(data)",
                }],
            },
        }],
    })

    generated = understanding["candidate_pipelines"][0]["pipeline"]["generated_operators"]
    assert generated[0]["name"] == "texture_mask"


def test_extract_json_object_from_markdown_response():
    text = '观察如下：```json\n{"defect_type": "residue", "confidence": 0.8}\n```'

    parsed = extract_json_object(text)

    assert parsed == {"defect_type": "residue", "confidence": 0.8}


def test_image_content_uses_openai_compatible_multimodal_shape(tmp_path):
    image = tmp_path / "target.jpg"
    Image.new("RGB", (5, 5), "white").save(image)

    content = image_content(image, "target image")

    assert set(content) == {"type", "image_url"}
    assert content["type"] == "image_url"
    assert content["image_url"]["url"].startswith("data:image/png;base64,")
    assert content["image_url"]["detail"] == "high"


def test_task_understanding_includes_previous_result_and_canvas_feedback_images(tmp_path):
    target = tmp_path / "target.png"
    previous = tmp_path / "previous.png"
    feedback = tmp_path / "feedback.png"
    for path in (target, previous, feedback):
        Image.new("RGB", (4, 4), "white").save(path)

    messages = build_task_understanding_messages(
        target,
        "修正红圈区域",
        previous_context={
            "original_task_goal": "提取当前图片中的全部椭圆轮廓",
            "previous_result_image_path": str(previous),
            "feedback_image_path": str(feedback),
            "previous_pipeline": {"name": "baseline"},
        },
    )

    image_parts = [
        item for item in messages[0]["content"]
        if item.get("type") == "image_url"
    ]
    assert len(image_parts) == 3
    text_parts = [item.get("text", "") for item in messages[0]["content"] if item.get("type") == "text"]
    assert any("原始任务目标（供追溯；用户明确修改时以当前目标和本轮请求为准）" in text for text in text_parts)


def test_handbook_examples_are_few_shot_images_without_pixel_alignment(tmp_path):
    target = tmp_path / "target.png"
    handbook = tmp_path / "handbook.png"
    result = tmp_path / "result_annotation.png"
    for path in (target, handbook, result):
        Image.new("RGB", (4, 4), "white").save(path)

    messages = build_task_understanding_messages(
        target,
        "提取黑灰色椭圆轮廓",
        reference_examples=[{
            "image_path": str(handbook),
            "description": "甲方已画好的绿色椭圆轮廓",
        }],
    )
    prompt = messages[0]["content"][0]["text"]
    image_parts = [item for item in messages[0]["content"] if item.get("type") == "image_url"]

    assert len(image_parts) == 2
    assert "few-shot视觉参照" in prompt
    assert "参考图不对应当前图片的像素坐标" in prompt
    assert "不得据此计算准确率" in prompt

    review_messages = build_candidate_review_messages(
        target,
        "提取黑灰色椭圆轮廓",
        [{
            "name": "candidate",
            "status": "completed",
            "directory": str(tmp_path),
            "quality": {},
        }],
        reference_examples=[str(handbook)],
    )
    review_images = [
        item for item in review_messages[0]["content"]
        if item.get("type") == "image_url"
    ]
    assert len(review_images) == 3


def test_task_understanding_prompt_uses_pipeline_operator_catalog(tmp_path):
    target = tmp_path / "target.png"
    Image.new("RGB", (4, 4), "white").save(target)

    messages = build_task_understanding_messages(target, "提取轮廓")
    prompt = messages[0]["content"][0]["text"]

    assert "Pipeline Operator Catalog" in prompt
    assert '"name": "normalize"' in prompt
    assert '"name": "adaptive_threshold"' in prompt
    assert '"name": "extract_contours"' in prompt
    assert "MaskArtifact" in prompt
    assert "直接生成一个可执行的CV Pipeline" in prompt
    assert "默认只实现一个方案" in prompt
    assert "submit_experiment(experiment_id,reason)" in prompt
    assert "生成2到3个语义不同的candidate_pipelines条目" not in prompt
    assert "不得要求用户在执行前确认" in prompt
    assert "questions必须返回空数组" in prompt


def test_task_understanding_rejects_unknown_operator():
    with pytest.raises(ValueError, match="unknown operators"):
        normalize_task_understanding({
            "candidate_pipelines": [{
                "name": "unknown_operator",
                "pipeline": {
                    "steps": [{"id": "final_mask", "op": "not_in_registry", "input": "image"}],
                },
            }],
        })


def test_task_understanding_allows_empty_candidates_for_local_fallback():
    understanding = normalize_task_understanding({"task_summary": "周期背景颗粒"})

    assert understanding["candidate_pipelines"] == []


def test_load_env_file_sets_missing_values(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join([
            "DASHSCOPE_API_KEY=test-key",
            "ALIYUN_BASE_URL=https://example.test/v1",
            "ALIYUN_VISION_MODEL=deepseek-flash",
        ]),
        encoding="utf-8",
    )
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("ALIYUN_BASE_URL", "https://already-set.test/v1")

    load_env_file(env_file)

    assert os.environ["DASHSCOPE_API_KEY"] == "test-key"
    assert os.environ["ALIYUN_BASE_URL"] == "https://already-set.test/v1"
    assert os.environ["ALIYUN_VISION_MODEL"] == "deepseek-flash"


def test_aliyun_provider_defaults_to_deepseek_v41_flash(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("ALIYUN_BASE_URL", "https://example.test/v1")
    monkeypatch.delenv("ALIYUN_VISION_MODEL", raising=False)

    provider = AliyunVisionProvider()

    assert provider.model == "deepseek-v4.1-flash"
    assert provider.timeout_seconds == 90
    assert provider.max_retries == 3


def test_build_runtime_provider_requires_real_aliyun_config(tmp_path, monkeypatch):
    monkeypatch.delenv("ALIYUN_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("ALIYUN_BASE_URL", raising=False)

    with pytest.raises(ValueError, match="Missing Alibaba Cloud vision configuration"):
        build_runtime_provider(env_path=tmp_path / ".env")


def test_build_runtime_provider_uses_aliyun_from_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join([
            "DASHSCOPE_API_KEY=test-key",
            "ALIYUN_BASE_URL=https://example.test/v1",
        ]),
        encoding="utf-8",
    )
    monkeypatch.delenv("ALIYUN_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("ALIYUN_BASE_URL", raising=False)
    monkeypatch.delenv("ALIYUN_VISION_MODEL", raising=False)

    provider = build_runtime_provider(env_path=env_file)

    assert isinstance(provider, AliyunVisionProvider)
    assert provider.api_key == "test-key"
    assert provider.base_url == "https://example.test/v1"
    assert provider.model == "deepseek-v4.1-flash"


def _revision_attempt(index, status="selected_for_review"):
    return {
        "name": f"candidate_{index}",
        "status": status,
        "failure_type": None if status == "selected_for_review" else "pipeline_execution_failed",
        "hypothesis": "h" * 300,
        "source": {"type": "qwen"},
        "quality": (
            {"health": {"issues": []}, "coverage": 0.4}
            if status == "selected_for_review"
            else {"error": "boom"}
        ),
        "measurements": {"summary": {"count": 2, "total_area": 48, "unit": "pixel"}},
        "operator_trace": [
            {
                "step_id": "threshold",
                "operator": "global_threshold",
                "params": {"note": "x" * 800},
                "mask_statistics": {"coverage": 0.5, "component_count": 9},
                "warnings": ["coverage_exceeded"],
            },
        ],
    }


def test_revision_context_replaces_fixed_width_truncation(tmp_path):
    target = tmp_path / "target.png"
    Image.new("RGB", (4, 4), "white").save(target)
    # The raw context JSON would exceed the previous 8000-character dump, which
    # dropped everything after the boundary, including the newest attempts.
    attempts = [_revision_attempt(index) for index in range(8)]
    messages = build_task_understanding_messages(
        target,
        "继续修改",
        previous_context={
            "original_task_goal": "提取椭圆",
            "previous_pipeline": {
                "name": "base",
                "steps": [{"id": "m", "op": "global_threshold", "input": "image", "params": {}}],
            },
            "previous_quality": {"coverage": 0.4, "component_count": 9},
            "review": {
                "decision": "revise",
                "selected_candidate": "candidate_7",
                "reason": "存在误检",
                "observed_issues": ["误检偏多"],
                "revision_plan": ["加强背景抑制"],
            },
            "execution_feedback": {
                "status": "needs_visual_revision",
                "instruction": "结合复查意见生成改进方法",
                "attempts": attempts,
            },
        },
    )
    text = "\n".join(
        item.get("text", "") for item in messages[0]["content"] if item.get("type") == "text"
    )

    assert "已有任务上下文" not in text
    for index in range(8):
        assert f"candidate_{index}" in text
    assert "修改建议：加强背景抑制" in text
    assert "上一轮候选执行记录" in text
    assert "优先在previous_pipeline基础上做参数级修改" in text


def test_candidate_attempt_summary_locates_where_targets_are_lost():
    attempt = {
        "name": "residual_v1",
        "status": "no_annotation",
        "hypothesis": "残差增强后按面积过滤",
        "quality": {"issues": ["empty_mask"]},
        "measurements": {"summary": {"count": 0, "total_area": 0, "unit": "pixel"}},
        "operator_trace": [
            {
                "step_id": "residual",
                "operator": "local_background_residual",
                "params": {"sigma": 10},
                "mask_statistics": {"coverage": 0.08, "component_count": 41},
            },
            {
                "step_id": "filter",
                "operator": "filter_components",
                "params": {"min_area": 20},
                "mask_statistics": {"coverage": 0.0, "component_count": 0},
                "warnings": ["empty_mask", "kept_components=0"],
            },
        ],
    }

    summary = summarize_candidate_attempt(attempt)

    assert "coverage=0.08" in summary and "组件数=41" in summary
    assert "coverage=0.0" in summary and "empty_mask" in summary
    assert "[未检出目标]" in summary
    assert "最终测量：0 个区域" in summary


def test_failed_attempt_without_trace_shows_pipeline_structure():
    summary = summarize_candidate_attempt({
        "name": "broken",
        "status": "failed",
        "failure_type": "pipeline_invalid",
        "quality": {"error": "ValueError: unknown operator"},
        "pipeline": {
            "name": "broken",
            "steps": [
                {"id": "a", "op": "normalize", "input": "image", "params": {}},
                {"id": "b", "op": "global_threshold", "input": "a", "params": {}},
            ],
        },
    })

    assert "Pipeline结构：normalize -> global_threshold" in summary
    assert "ValueError: unknown operator" in summary


def test_revision_context_compacts_oldest_attempts_under_budget():
    attempts = [
        {
            "name": f"c{index}",
            "status": "failed",
            "quality": {"error": "e"},
            "hypothesis": "h" * 500,
            "operator_trace": [
                {
                    "step_id": "s",
                    "operator": "normalize",
                    "params": {"p": index},
                    "mask_statistics": {"coverage": 0.1, "component_count": 2},
                }
            ] * 20,
        }
        for index in range(12)
    ]

    text = build_revision_context_text({
        "execution_feedback": {"status": "no_usable_annotation", "attempts": attempts},
    })

    assert "c11" in text
    assert "逐步执行统计" in text
    assert "篇幅所限" in text
    assert len(text) <= 20000


# ---- 上下文窗口用量：流式 usage 采集与估算兜底 ----


class _UsageStreamCompletions:
    def __init__(self, chunks):
        self._chunks = chunks
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self._chunks)


class _UsageStreamClient:
    def __init__(self, chunks):
        self.chat = type("Chat", (), {})()
        self.chat.completions = _UsageStreamCompletions(chunks)


def _stream_provider(monkeypatch, chunks, model="deepseek-v4.1-flash"):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.delenv("ALIYUN_CONTEXT_WINDOW", raising=False)
    provider = AliyunVisionProvider(model=model)
    return provider, _UsageStreamClient(chunks)


def test_unlimited_request_keeps_model_network_timeout(monkeypatch):
    from core.request_control import RequestControl, control

    provider, client = _stream_provider(monkeypatch, [
        {'choices': [{'delta': {'content': 'done'}}]},
    ])
    token = control.set(RequestControl())
    try:
        assert provider._complete_streaming(client, [{'role': 'user', 'content': '检测'}]) == 'done'
    finally:
        control.reset(token)
    assert client.chat.completions.calls[0]['timeout'] == 60


def test_streaming_captures_usage_chunk(monkeypatch):
    from core.agent_events import clear_event_listeners, register_event_listener, unregister_event_listener

    provider, client = _stream_provider(monkeypatch, [
        {"choices": [{"delta": {"content": "结果如下"}}]},
        {"choices": [{"delta": {"content": "。"}}]},
        {"choices": [], "usage": {"prompt_tokens": 34571, "completion_tokens": 218, "total_tokens": 34789}},
    ])
    events = []
    listener = events.append
    register_event_listener(listener)
    try:
        reply = provider._complete_streaming(client, [{"role": "user", "content": "你好"}])
    finally:
        unregister_event_listener(listener)
        clear_event_listeners()

    assert reply == "结果如下。"
    assert client.chat.completions.calls[0]["stream_options"] == {"include_usage": True}
    response_events = [event for event in events if event["type"] == "llm_response"]
    assert len(response_events) == 1
    event = response_events[0]
    assert event["usage"] == {"prompt_tokens": 34571, "completion_tokens": 218, "total_tokens": 34789}
    assert event["model"] == "deepseek-v4.1-flash"
    assert event["context_window"] == 1048576


def test_streaming_without_usage_chunk_falls_back_to_estimate(monkeypatch):
    from core.agent_events import clear_event_listeners, register_event_listener, unregister_event_listener

    provider, client = _stream_provider(monkeypatch, [
        {"choices": [{"delta": {"content": "done"}}]},
    ])
    events = []
    listener = events.append
    register_event_listener(listener)
    try:
        provider._complete_streaming(client, [
            {"role": "system", "content": "你是量测助手"},
            {"role": "user", "content": "标注颗粒缺陷，测量面积"},
        ])
    finally:
        unregister_event_listener(listener)
        clear_event_listeners()

    event = next(event for event in events if event["type"] == "llm_response")
    usage = event["usage"]
    assert usage["estimated"] is True
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0


def test_context_window_for_model_env_override(monkeypatch):
    from providers.vision import DEFAULT_CONTEXT_WINDOW, context_window_for_model

    monkeypatch.delenv("ALIYUN_CONTEXT_WINDOW", raising=False)
    assert context_window_for_model("deepseek-v4.1-flash") == 1048576
    assert context_window_for_model("unknown-model") == DEFAULT_CONTEXT_WINDOW

    monkeypatch.setenv("ALIYUN_CONTEXT_WINDOW", "262144")
    assert context_window_for_model("unknown-model") == 262144

    monkeypatch.setenv("ALIYUN_CONTEXT_WINDOW", "not-a-number")
    assert context_window_for_model("unknown-model") == DEFAULT_CONTEXT_WINDOW


def test_provider_reads_context_window_from_env(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("ALIYUN_CONTEXT_WINDOW", "1000000")
    provider = AliyunVisionProvider(model="qwen3.7-plus")
    assert provider.context_window == 1000000
