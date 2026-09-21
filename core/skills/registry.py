"""Progressive file-based skills with legacy JSON template compatibility."""

from __future__ import annotations

from copy import deepcopy
import json
import re
import warnings
from pathlib import Path


BUILTIN_ROOT = Path(__file__).resolve().parents[2] / "skills"


def _markdown_metadata(path):
    """Read frontmatter only; body and referenced files are loaded on demand."""
    with path.open(encoding="utf-8") as stream:
        if stream.readline().strip() != "---":
            raise ValueError("SKILL.md requires frontmatter")
        metadata = {}
        for line in stream:
            if line.strip() == "---":
                break
            key, separator, value = line.partition(":")
            if not separator or key.strip() not in {"name", "description", "version"}:
                raise ValueError("frontmatter supports name, description and optional version scalars")
            metadata[key.strip()] = value.strip().strip('"\'')
        else:
            raise ValueError("unterminated frontmatter")
    if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]*", metadata.get("name", "")) or not metadata.get("description"):
        raise ValueError("skill name and description are required")
    metadata.setdefault("version", "1.0.0")
    if not re.fullmatch(r"\d+\.\d+\.\d+", metadata["version"]):
        raise ValueError("invalid skill version")
    return {**metadata, "_path": str(path.resolve())}


def builtin_skills():
    return [_markdown_metadata(path) for path in sorted(BUILTIN_ROOT.glob("*/SKILL.md"))]


def validate_skill(raw):
    """Validate legacy workspace JSON templates and their operator dependencies."""
    from core.pipelines.dsl import normalize_pipeline, validate_pipeline
    if not isinstance(raw, dict):
        raise ValueError("skill must be an object")
    skill = deepcopy(raw)
    if skill.get("schema_version") != 1:
        raise ValueError("unsupported skill schema_version")
    if not isinstance(skill.get("name"), str) or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", skill["name"]):
        raise ValueError("skill name must be an identifier")
    if not isinstance(skill.get("version"), str) or not re.fullmatch(r"\d+\.\d+\.\d+", skill["version"]):
        raise ValueError("skill version must use major.minor.patch")
    if not isinstance(skill.get("description"), str) or not skill["description"]:
        raise ValueError("skill description is required")
    applicability = skill.get("applicability")
    if not isinstance(applicability, dict):
        raise ValueError("skill applicability must be an object")
    for key in ("defect_types", "background_types"):
        labels = applicability.get(key)
        if not isinstance(labels, list) or not labels or not all(isinstance(label, str) and label for label in labels):
            raise ValueError(f"skill applicability.{key} must contain strings")
    instructions = skill.get("instructions")
    if not isinstance(instructions, list) or not instructions or not all(isinstance(item, str) and item for item in instructions):
        raise ValueError("skill instructions must contain strings")
    if "required_tools" in skill:
        legacy = skill.pop("required_tools")
        if "required_operators" in skill and skill["required_operators"] != legacy:
            raise ValueError("conflicting skill operator dependencies")
        skill["required_operators"] = legacy
    required = skill.get("required_operators")
    if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
        raise ValueError("skill required_operators must be a list of names")
    pipeline = normalize_pipeline(skill.get("pipeline_template"))
    validate_pipeline(pipeline)
    operators = {node.get("operator") or node.get("op") for node in pipeline.get("nodes", pipeline.get("steps", []))}
    if set(required) != operators or len(required) != len(set(required)):
        raise ValueError("skill dependencies must exactly match template operators")
    verification = skill.get("verification")
    if not isinstance(verification, dict) or not isinstance(verification.get("requires_mask"), bool):
        raise ValueError("skill verification.requires_mask must be boolean")
    coverage = verification.get("max_coverage")
    if isinstance(coverage, bool) or not isinstance(coverage, (int, float)) or not 0 < coverage <= 1:
        raise ValueError("skill verification.max_coverage must be in (0, 1]")
    checks = verification.get("checks", [])
    if not isinstance(checks, list) or not all(isinstance(item, str) for item in checks):
        raise ValueError("skill verification.checks must be strings")
    skill["pipeline_template"] = pipeline
    return skill


class SkillRegistry:
    """On-demand skill loading with legacy workspace JSON compatibility.

    The model selects skills from the name/description catalog. Markdown bodies
    hold workflow guidance; only legacy JSON skills contain pipeline templates.
    Built-in versions are immutable. Duplicate workspace name/version pairs are
    rejected rather than silently replacing one another. A newer version may
    coexist; get(name) selects the highest semantic version.
    """

    def __init__(self, root=None):
        self.root = Path(root) if root else None
        self._cache = {}
        self._builtins = builtin_skills()
        self.errors = []

    def list_skills(self, *, all_versions=False):
        skills = {(skill["name"], skill["version"]): skill for skill in self._builtins}
        self.errors = []
        seen_paths = set()
        if self.root and self.root.exists():
            for path in sorted([*self.root.glob("skill_*/skill.json"), *self.root.glob("*/SKILL.md")]):
                seen_paths.add(path)
                try:
                    stat = path.stat()
                    signature = (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
                    cached = self._cache.get(path)
                    if cached is None or cached[0] != signature:
                        value = _markdown_metadata(path) if path.name == "SKILL.md" else json.loads(path.read_text(encoding="utf-8"))
                        if not isinstance(value, dict):
                            raise ValueError("skill must be an object")
                        value = value if path.name == "SKILL.md" else (validate_skill(value) if value.get("status") == "accepted" else None)
                        self._cache[path] = signature, value
                    value = self._cache[path][1]
                    if value is None:
                        continue
                    key = value["name"], value["version"]
                    if key in skills:
                        raise ValueError(f"duplicate skill name/version: {key}")
                    skills[key] = value
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    self.errors.append({"path": str(path), "error": str(exc)})
                    warnings.warn(f"Ignoring invalid workspace skill {path.name}: {exc}", RuntimeWarning, stacklevel=2)
        self._cache = {path: value for path, value in self._cache.items() if path in seen_paths}
        ordered = sorted(skills.values(), key=lambda item: (item["name"], _version_key(item["version"])))
        if not all_versions:
            ordered = list({skill["name"]: skill for skill in ordered}.values())
        return deepcopy(ordered)

    def get(self, name, version=None):
        skill = next((skill for skill in self.list_skills(all_versions=version is not None)
                     if skill["name"] == name and (version is None or skill["version"] == version)), None)
        if skill and "_path" in skill:
            path = Path(skill.pop("_path"))
            skill["content"] = path.read_text(encoding="utf-8")
            skill["resources"] = [str(p.relative_to(path.parent)) for p in sorted(path.parent.rglob("*"))
                                  if p.is_file() and p != path and p.suffix in {".md", ".json", ".py", ".txt"}
                                  and p.resolve().is_relative_to(path.parent.resolve())]
        return skill

    def read_resource(self, name, resource, version=None):
        skill = next((item for item in self.list_skills(all_versions=version is not None)
                      if item["name"] == name and (version is None or item["version"] == version)), None)
        if not skill or "_path" not in skill:
            raise ValueError("unknown file-based skill")
        root = Path(skill["_path"]).parent.resolve()
        path = (root / resource).resolve()
        if Path(resource).is_absolute() or not path.is_relative_to(root) or path.suffix not in {".md", ".json", ".py", ".txt"}:
            raise ValueError("resource must be a supported file inside the skill directory")
        if path.stat().st_size > 65536:
            raise ValueError("skill resource exceeds 64 KiB")
        return {"name": name, "version": skill["version"], "resource": resource, "content": path.read_text(encoding="utf-8")}


def _version_key(version):
    return tuple(int(part) for part in version.split("."))


def skill_catalog(root=None):
    return [{"name": skill["name"], "description": skill["description"]}
            for skill in SkillRegistry(root).list_skills()]
