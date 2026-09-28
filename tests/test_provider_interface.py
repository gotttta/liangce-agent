"""Workflow↔provider 接口契约：消息格式必须与 providers/vision 的 _complete_action 兼容。

离线测试验证消息 schema 和上下文预算；真实端点验证由
LIANGCE_REQUIRE_PROVIDER_TESTS=1 门控的集成测试完成（需要 .env 里的 API key）。
"""
import base64
import os
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from core.model_context import check_request_budget
from core.sam_session import _chat_json


def _tiny_image(tmp_path):
    image = np.zeros((32, 32), dtype=np.uint8)
    image[8:16, 8:16] = 220
    path = tmp_path / "probe.png"
    cv2.imwrite(str(path), image)
    return str(path)


def _assert_multimodal_contract(messages):
    assert isinstance(messages, list) and messages
    for message in messages:
        assert message["role"] in {"system", "user", "assistant"}
        assert isinstance(message["content"], list)
        for part in message["content"]:
            if part.get("type") == "image_url":
                assert part["image_url"]["url"].startswith("data:image/")
            else:
                assert part.get("type") == "text" and isinstance(part["text"], str)
    stats = check_request_budget(messages)
    assert stats["over_budget"] is False


class _RecordingProvider:
    model = "recording-model"

    def __init__(self, reply='{"count": 1}'):
        self.reply = reply
        self.captured = []

    def _complete_action(self, messages):
        self.captured.append(messages)
        return self.reply


def test_vision_json_messages_match_provider_contract(tmp_path):
    from core.agent_workflow import WorkflowRuntime

    provider = _RecordingProvider()
    runtime = WorkflowRuntime(provider, references_root=tmp_path / "refs")
    image_path = _tiny_image(tmp_path)

    runtime._vision_json("描述图片内容", image_paths=[image_path])

    assert provider.captured
    _assert_multimodal_contract(provider.captured[-1])
    assert provider.captured[-1][0]["role"] == "user"


def test_provider_chat_client_forwards_messages_and_unwraps_reply(tmp_path):
    from core.agent_workflow import _ProviderChatClient

    provider = _RecordingProvider(reply='{"choice": "A"}')
    client = _ProviderChatClient(provider)
    image_path = _tiny_image(tmp_path)
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(
            open(image_path, "rb").read()).decode()}},
        {"type": "text", "text": "选一个"},
    ]}]

    response = client.chat.completions.create(model="ignored", messages=messages)

    assert provider.captured == [messages]  # sam_session 的消息原样到达 provider
    assert response.choices[0].message.content == '{"choice": "A"}'


def test_sam_session_chat_json_messages_pass_budget(tmp_path):
    from core.agent_workflow import _ProviderChatClient

    image = np.zeros((32, 32), dtype=np.uint8)
    image[8:16, 8:16] = 220
    provider = _RecordingProvider(reply='{"selected": [0]}')
    client = _ProviderChatClient(provider)  # 生产组合方式：sam_session 消息经适配器到达 provider

    answer = _chat_json(client, "unused-model", [image], "只输出JSON：{\"selected\":[0]}")

    assert answer == {"selected": [0]}
    _assert_multimodal_contract(provider.captured[0])


@pytest.mark.skipif(os.getenv("LIANGCE_REQUIRE_PROVIDER_TESTS") != "1",
                    reason="set LIANGCE_REQUIRE_PROVIDER_TESTS=1 to hit the real provider endpoint")
def test_real_provider_accepts_workflow_message_shapes(tmp_path):
    pytest.importorskip("openai")
    from providers.vision import build_runtime_provider, extract_json_object, image_content

    try:
        provider = build_runtime_provider()
    except ValueError as exc:
        pytest.skip(f"provider unavailable: {exc}")
    image_path = _tiny_image(tmp_path)
    prompt = '图里有几个亮块？只输出JSON：{"count": <整数>}'

    workflow_messages = [{"role": "user", "content": [
        image_content(image_path, "输入图片"), {"type": "text", "text": prompt}]}]
    with open(image_path, "rb") as handle:
        inline = base64.b64encode(handle.read()).decode()
    sam_messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + inline}},
        {"type": "text", "text": prompt}]}]

    for name, messages in (("workflow", workflow_messages), ("sam_session", sam_messages)):
        parsed = extract_json_object(provider._complete_action(messages))
        assert parsed.get("count") == 1, f"{name} message shape failed: {parsed}"
