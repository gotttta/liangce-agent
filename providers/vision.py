import base64
import json
import os
from pathlib import Path

from agent_types import normalize_strategy


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALIYUN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_ALIYUN_VISION_MODEL = "qwen3.7-plus"


def load_env_file(path=None):
    env_path = Path(path or ROOT / ".env")
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def build_runtime_provider(env_path=None):
    load_env_file(env_path)
    api_key = os.getenv("ALIYUN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("ALIYUN_BASE_URL")
    if not api_key or not base_url:
        raise ValueError(
            "Missing Alibaba Cloud vision configuration. "
            "Please set DASHSCOPE_API_KEY or ALIYUN_API_KEY, and ALIYUN_BASE_URL in .env."
        )
    return AliyunVisionProvider(api_key=api_key, base_url=base_url)


class MockVisionProvider:
    def create_strategy(
        self,
        target_image_path,
        description,
        reference_annotation_path=None,
        previous_state=None,
    ):
        text = (description or "").lower()
        method = "dark_threshold" if any(word in text for word in ["暗", "dark", "black"]) else "bright_threshold"
        raw = {
            "defect_type": "particle_residue",
            "measurement_type": "area_count",
            "visual_observation": {
                "defect_appearance": "small anomaly regions inferred from user description",
                "background_pattern": "unknown",
                "polarity": "dark_on_bright" if method == "dark_threshold" else "bright_on_dark",
            },
            "segmentation": {
                "method": method,
                "sensitivity": 1.8,
                "min_area_px": 20,
                "morphology": "open_then_close",
            },
            "confidence": 0.5,
            "notes": ["Mock provider used; no remote multimodal model was called."],
        }
        return normalize_strategy(raw)


class AliyunVisionProvider:
    def __init__(self, api_key=None, base_url=None, model=None):
        self.api_key = api_key or os.getenv("ALIYUN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = base_url or os.getenv("ALIYUN_BASE_URL", DEFAULT_ALIYUN_BASE_URL)
        self.model = model or os.getenv("ALIYUN_VISION_MODEL", DEFAULT_ALIYUN_VISION_MODEL)
        if not self.api_key:
            raise ValueError("Missing ALIYUN_API_KEY or DASHSCOPE_API_KEY")

    def create_strategy(
        self,
        target_image_path,
        description,
        reference_annotation_path=None,
        previous_state=None,
    ):
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        messages = build_strategy_messages(
            target_image_path=target_image_path,
            description=description,
            reference_annotation_path=reference_annotation_path,
            previous_state=previous_state,
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
        )
        content = response.choices[0].message.content
        return normalize_strategy(extract_json_object(content))


def build_strategy_messages(target_image_path, description, reference_annotation_path=None, previous_state=None):
    content = [
        {
            "type": "text",
            "text": (
                "你是DRAM/SEM缺陷量测agent。只输出JSON，不要输出Markdown。"
                "任务是为本地OpenCV/skimage分割生成结构化strategy。"
                "LLM不直接生成mask。measurement_type固定为area_count。"
                f"用户描述：{description or ''}"
            ),
        },
        image_content(target_image_path, "target image"),
    ]
    if reference_annotation_path:
        content.append(image_content(reference_annotation_path, "same-image reference annotation"))
    if previous_state:
        content.append({
            "type": "text",
            "text": "上一轮状态：" + json.dumps(previous_state, ensure_ascii=False)[:6000],
        })

    return [{"role": "user", "content": content}]


def image_content(path, label):
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{encode_image(path)}",
            "detail": "high",
        },
        "text": label,
    }


def encode_image(path):
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def extract_json_object(text):
    cleaned = (text or "").strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                return json.loads(candidate)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Provider response did not contain a JSON object")
    return json.loads(cleaned[start:end + 1])
