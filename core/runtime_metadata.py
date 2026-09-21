"""Record replay provenance without collecting environment variables or secrets."""

import hashlib
from importlib.metadata import distributions
from pathlib import Path
import platform


def runtime_metadata():
    root = Path(__file__).resolve().parents[1]
    sources = [path for directory in ("core", "providers", "ui") for path in (root / directory).rglob("*.py")]
    sources.extend(root / name for name in ("agent.py", "agent_types.py", "requirements.lock"))
    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "dependencies": dict(sorted(
            (package.metadata["Name"], package.version)
            for package in distributions() if package.metadata["Name"]
        )),
        "source_sha256": {
            path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(sources) if path.is_file()
        },
    }
