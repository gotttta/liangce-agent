"""受限的文件读取：只放行 workspace/ outputs/ 下的图片和 JSON 产物（计划 §6.3）。

返回给前端的路径一律通过 file_url() 转成 /api/files?path=... 形式，
前端永远不拼接文件系统路径。
"""
from dataclasses import dataclass
from io import BytesIO
from os.path import relpath
from pathlib import Path
from urllib.parse import quote

from fastapi import Response

ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".json"}

_IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".bmp": "image/bmp",
}


class ForbiddenPath(Exception):
    """请求的路径不在允许的根目录内，或后缀不在白名单里。"""


@dataclass(frozen=True)
class FileRoots:
    root: Path
    allowed: tuple[Path, ...]


def file_roots(root: Path) -> FileRoots:
    root = Path(root).resolve()
    return FileRoots(root=root, allowed=(root / "workspace", root / "outputs"))


def resolve_allowed(raw: str, roots: FileRoots) -> Path:
    candidate = Path(raw)
    path = candidate.resolve() if candidate.is_absolute() else (roots.root / candidate).resolve()
    if not any(path.is_relative_to(base) for base in roots.allowed):
        raise ForbiddenPath(raw)
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ForbiddenPath(raw)
    if not path.is_file():
        raise FileNotFoundError(raw)
    return path


def file_url(path, roots: FileRoots) -> str:
    """把任意存储路径转成相对 root 的 /api/files URL；中文与空格做百分号编码。"""
    relative = relpath(Path(path).resolve(), roots.root)
    return f"/api/files?path={quote(relative)}"


def read_response(path: Path, roots: FileRoots) -> Response:
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        # 浏览器不能直接显示 tif：转 PNG 放内存返回，不落盘。
        from PIL import Image

        buffer = BytesIO()
        with Image.open(path) as image:
            image.save(buffer, format="PNG")
        return Response(buffer.getvalue(), media_type="image/png")
    if suffix == ".json":
        return Response(path.read_bytes(), media_type="application/json")
    return Response(path.read_bytes(), media_type=_IMAGE_MEDIA_TYPES[suffix])
