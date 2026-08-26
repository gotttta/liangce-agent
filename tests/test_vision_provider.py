import os

import pytest
from PIL import Image

from providers.vision import (
    AliyunVisionProvider,
    MockVisionProvider,
    build_runtime_provider,
    build_candidate_review_messages,
    build_task_understanding_messages,
    extract_json_object,
    image_content,
    load_env_file,
    normalize_task_understanding,
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
                "steps": [{
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
    assert "max_components" not in observed["candidate_pipelines"][0]["pipeline"]["steps"][0]["params"]

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
    image.write_bytes(b"fake-jpeg")

    content = image_content(image, "target image")

    assert set(content) == {"type", "image_url"}
    assert content["type"] == "image_url"
    assert content["image_url"]["url"].startswith("data:image/jpeg;base64,")
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
    assert any("原始任务目标（不可因本轮修正而缩小或替换）" in text for text in text_parts)


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


def test_task_understanding_prompt_uses_pipeline_tool_catalog(tmp_path):
    target = tmp_path / "target.png"
    Image.new("RGB", (4, 4), "white").save(target)

    messages = build_task_understanding_messages(target, "提取轮廓")
    prompt = messages[0]["content"][0]["text"]

    assert "Pipeline Tool Catalog" in prompt
    assert '"name": "normalize"' in prompt
    assert '"name": "adaptive_threshold"' in prompt
    assert '"name": "extract_contours"' in prompt
    assert "MaskArtifact" in prompt
    assert "直接生成一个可执行的CV Pipeline" in prompt
    assert "生成2到3个语义不同的candidate_pipelines条目" in prompt
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
            "ALIYUN_VISION_MODEL=qwen3.7-plus",
        ]),
        encoding="utf-8",
    )
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("ALIYUN_BASE_URL", "https://already-set.test/v1")

    load_env_file(env_file)

    assert os.environ["DASHSCOPE_API_KEY"] == "test-key"
    assert os.environ["ALIYUN_BASE_URL"] == "https://already-set.test/v1"
    assert os.environ["ALIYUN_VISION_MODEL"] == "qwen3.7-plus"


def test_aliyun_provider_defaults_to_qwen37_plus(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("ALIYUN_BASE_URL", "https://example.test/v1")
    monkeypatch.delenv("ALIYUN_VISION_MODEL", raising=False)

    provider = AliyunVisionProvider()

    assert provider.model == "qwen3.7-plus"
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
    assert provider.model == "qwen3.7-plus"
