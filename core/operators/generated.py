"""Validation and execution helpers for model-generated CV operators.

Generated code uses ordinary Python. Validation checks the interface only;
Docker isolation, not AST filtering, is the security boundary.
"""

import ast
from dataclasses import dataclass

import numpy as np

from core.operators.types import ImageArtifact, MaskArtifact, MetadataArtifact, OperatorResult


ARTIFACT_TYPES = {
    "ImageArtifact": ImageArtifact,
    "MaskArtifact": MaskArtifact,
    "MetadataArtifact": MetadataArtifact,
}


class GeneratedSourceError(ValueError):
    """A static diagnostic; validating source never executes it."""

    def __init__(self, exc, source, operator):
        lines = source.splitlines()
        line = exc.lineno or 1
        column = exc.offset or 1
        excerpt = '\n'.join(f'{index + 1}: {lines[index]}'
                            for index in range(max(0, line - 2), min(len(lines), line + 1)))
        self.diagnostic = {
            'operator': operator, 'file': f'<generated:{operator}>',
            'line': line, 'column': column, 'message': exc.msg,
            'source_line': lines[line - 1] if line <= len(lines) else '',
            'excerpt': excerpt, 'indicator': ' ' * (column - 1) + '^',
        }
        super().__init__(f'{operator}: SyntaxError at line {line}, column {column}: {exc.msg}\n{excerpt}')

@dataclass(frozen=True)
class GeneratedOperatorSpec:
    name: str
    source: str
    input_artifact: str = "ImageArtifact"
    output_artifact: str = "MaskArtifact"
    description: str = "模型生成的 CV 算子"
    atomic: bool = True
    version: str = "1"
    input_ports: dict | None = None

    def as_dict(self):
        return {
            "name": self.name,
            "source": self.source,
            "input_artifact": self.input_artifact,
            "output_artifact": self.output_artifact,
            "description": self.description,
            "atomic": self.atomic,
            "version": self.version,
            **({"input_ports": self.input_ports} if self.input_ports else {}),
        }


def normalize_generated_operator(raw):
    if not isinstance(raw, dict):
        raise ValueError("generated operator must be an object")
    name = str(raw.get("name") or "").strip()
    if not name or not name.replace("_", "").isalnum() or not name[0].isalpha():
        raise ValueError("generated operator name must be an identifier")
    input_artifact = str(raw.get("input_artifact") or "ImageArtifact")
    output_artifact = str(raw.get("output_artifact") or "MaskArtifact")
    if input_artifact not in ARTIFACT_TYPES or output_artifact not in ARTIFACT_TYPES:
        raise ValueError("generated operator artifact types are not supported")
    ports = raw.get("input_ports")
    if ports is not None:
        if not isinstance(ports, dict) or not ports or any(
            not isinstance(k, str) or not k.isidentifier() or v not in ARTIFACT_TYPES
            for k, v in ports.items()
        ):
            raise ValueError("input_ports must map names to supported artifact types")
        input_artifact = next(iter(ports.values()))
    source = _clean_source(raw.get("source"))
    validate_generated_source(source, name)
    return GeneratedOperatorSpec(
        name=name,
        source=source,
        input_artifact=input_artifact,
        output_artifact=output_artifact,
        description=str(raw.get("description") or "模型生成的 CV 算子"),
        atomic=bool(raw.get("atomic", True)),
        version=str(raw.get("version") or "1"),
        input_ports=ports,
    )


def normalize_generated_operators(raw):
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("generated_operators must be a list")
    specs = []
    seen = set()
    for item in raw:
        spec = normalize_generated_operator(item)
        if spec.name in seen:
            raise ValueError(f"duplicate generated operator: {spec.name}")
        seen.add(spec.name)
        specs.append(spec)
    return specs


def validate_generated_source(source, operator='generated_operator'):
    if not isinstance(source, str) or not source.strip():
        raise ValueError("generated operator source is empty")
    try:
        tree = ast.parse(source, filename=f'<generated:{operator}>', mode="exec")
        compile(tree, f'<generated:{operator}>', 'exec')
    except SyntaxError as exc:
        raise GeneratedSourceError(exc, source, operator) from exc
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == "apply"]
    if len(functions) != 1:
        raise ValueError("generated operator must define one apply(data, params) function")
    function = functions[0]
    args = function.args
    if (len(args.posonlyargs) + len(args.args) != 2 or args.vararg
            or args.kwarg or args.kwonlyargs):
        raise ValueError("generated operator apply(data, params) signature is required")
    return True


def register_generated_operator(registry, raw_spec):
    spec = raw_spec if isinstance(raw_spec, GeneratedOperatorSpec) else normalize_generated_operator(raw_spec)
    if spec.name in registry.names():
        raise ValueError(f"generated operator conflicts with registered operator: {spec.name}")
    input_type = ARTIFACT_TYPES[spec.input_artifact]
    output_type = ARTIFACT_TYPES[spec.output_artifact]
    validate_generated_source(spec.source)

    def generated_operator(artifact=None, **params):
        from core.sandbox import require_docker_worker
        require_docker_worker()
        namespace = {"np": np, "__name__": "generated_operator"}
        exec(compile(spec.source, f"<generated:{spec.name}>", "exec"), namespace, namespace)
        apply = namespace.get("apply")
        if not callable(apply):
            raise ValueError(f"generated operator {spec.name} did not define apply")
        if spec.input_ports:
            inputs = {port: params.pop(port).data for port in spec.input_ports}
            output = apply(inputs, dict(params))
        else:
            output = apply(artifact.data, dict(params))
        if output_type is MetadataArtifact:
            if not isinstance(output, dict):
                raise ValueError("MetadataArtifact output must be a JSON object")
            import json
            json.dumps(output, allow_nan=False)
            return OperatorResult(MetadataArtifact(output), {"operator": spec.name, "generated": True})
        if not isinstance(output, np.ndarray):
            output = np.asarray(output)
        result_artifact = output_type(output, metadata={"operator": spec.name, "version": spec.version})
        metadata = {"operator": spec.name, "version": spec.version, "generated": True}
        return OperatorResult(result_artifact, metadata)

    registry.register(spec.name, generated_operator, input_type, output_type, version=spec.version,
                      input_ports={k: ARTIFACT_TYPES[v] for k, v in spec.input_ports.items()} if spec.input_ports else None)
    return spec


def _clean_source(source):
    text = str(source or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text
