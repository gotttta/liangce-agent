"""API 路径与端口常量；测试通过 create_app(root=...) 覆盖全部路径。"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "workspace" / "tasks"
OUTPUT_ROOT = ROOT / "outputs"
WEB_DIST = ROOT / "web" / "dist"

DEFAULT_PORT = 8765


def api_port() -> int:
    raw = os.environ.get("LIANGCE_API_PORT", "").strip()
    return int(raw) if raw else DEFAULT_PORT
