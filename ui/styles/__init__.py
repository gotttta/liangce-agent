"""CSS loading for the annotation workspace."""

from pathlib import Path


_STYLE_FILES = (
    "base.css",
    "layout.css",
    "chat.css",
)


def load_styles() -> str:
    """Return component styles in deterministic cascade order."""
    styles_dir = Path(__file__).parent
    parts = []
    for filename in _STYLE_FILES:
        path = styles_dir / filename
        if path.exists():
            parts.append(path.read_text(encoding="utf-8"))
    return "\n\n".join(parts)


__all__ = ["load_styles"]
