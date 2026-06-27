import os

import pytest

from providers.vision import (
    AliyunVisionProvider,
    MockVisionProvider,
    build_runtime_provider,
    extract_json_object,
    load_env_file,
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


def test_extract_json_object_from_markdown_response():
    text = '观察如下：```json\n{"defect_type": "residue", "confidence": 0.8}\n```'

    parsed = extract_json_object(text)

    assert parsed == {"defect_type": "residue", "confidence": 0.8}


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
