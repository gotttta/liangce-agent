"""Validation and execution helpers for model-generated CV operators.

Generated code is intentionally a small Python subset. It is compiled and
executed only inside the pipeline sandbox with an empty builtins dictionary.
"""

import ast
from dataclasses import dataclass

import numpy as np

from core.operators.types import ImageArtifact, MaskArtifact, OperatorResult


MAX_SOURCE_LENGTH = 20_000
ARTIFACT_TYPES = {
    "ImageArtifact": ImageArtifact,
    "MaskArtifact": MaskArtifact,
}

_SAFE_NUMPY_CALLS = {
    "abs", "clip", "logical_and", "logical_or", "logical_not", "maximum",
    "minimum", "mean", "median", "percentile", "std", "where", "zeros_like",
    "ones_like", "sqrt", "exp", "log1p", "isfinite", "asarray",
}
_SAFE_NUMPY_ATTRIBUTES = _SAFE_NUMPY_CALLS | {"float32", "uint8", "bool_"}
_SAFE_ARRAY_METHODS = {"astype", "copy"}
_SAFE_ARRAY_ATTRIBUTES = {"shape", "ndim", "size"}
_SAFE_NODE_TYPES = {
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return,
    ast.Assign, ast.If, ast.Expr, ast.Name, ast.Load, ast.Store, ast.Constant,
    ast.Call, ast.keyword, ast.Attribute, ast.Subscript, ast.Slice, ast.Tuple, ast.List,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Add, ast.Sub,
    ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd, ast.And, ast.Or,
    ast.Not, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
}


@dataclass(frozen=True)
class GeneratedOperatorSpec:
    name: str
    source: str
    input_artifact: str = "ImageArtifact"
    output_artifact: str = "MaskArtifact"
    description: str = "模型生成的 CV 算子"
    atomic: bool = True
    version: str = "1"

    def as_dict(self):
        return {
            "name": self.name,
            "source": self.source,
            "input_artifact": self.input_artifact,
            "output_artifact": self.output_artifact,
            "description": self.description,
            "atomic": self.atomic,
            "version": self.version,
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
    source = _clean_source(raw.get("source"))
    if raw.get("atomic", True) is not True:
        raise ValueError("generated operators must be atomic pipeline stages")
    validate_generated_source(source)
    return GeneratedOperatorSpec(
        name=name,
        source=source,
        input_artifact=input_artifact,
        output_artifact=output_artifact,
        description=str(raw.get("description") or "模型生成的 CV 算子"),
        atomic=True,
        version=str(raw.get("version") or "1"),
    )


def normalize_generated_operators(raw):
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("generated_operators must be a list")
    specs = []
    seen = set()
    for item in raw[:8]:
        spec = normalize_generated_operator(item)
        if spec.name in seen:
            raise ValueError(f"duplicate generated operator: {spec.name}")
        seen.add(spec.name)
        specs.append(spec)
    return specs


def validate_generated_source(source):
    if not isinstance(source, str) or not source.strip():
        raise ValueError("generated operator source is empty")
    if len(source.encode("utf-8")) > MAX_SOURCE_LENGTH:
        raise ValueError("generated operator source is too large")
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"generated operator source is invalid: {exc.msg}") from exc
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("generated operator source must contain exactly one apply function")
    function = tree.body[0]
    if function.name != "apply" or len(function.args.args) != 2:
        raise ValueError("generated operator apply(data, params) signature is required")
    if function.args.vararg or function.args.kwarg or function.args.kwonlyargs:
        raise ValueError("generated operator may not use variadic arguments")
    assigned = {"data", "params", "np"}
    for node in ast.walk(tree):
        if type(node) not in _SAFE_NODE_TYPES:
            raise ValueError(f"generated operator syntax is not allowed: {type(node).__name__}")
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id not in assigned:
                if isinstance(node.ctx, ast.Store) and node.id.isidentifier():
                    assigned.add(node.id)
                else:
                    raise ValueError(f"generated operator name is not allowed: {node.id}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise ValueError("dunder attributes are not allowed in generated operators")
            if isinstance(node.value, ast.Name) and node.value.id == "np":
                if node.attr not in _SAFE_NUMPY_ATTRIBUTES:
                    raise ValueError(f"NumPy attribute is not allowed: {node.attr}")
            elif isinstance(node.value, ast.Name) and node.value.id == "params":
                if node.attr != "get":
                    raise ValueError("only params.get is allowed in generated operators")
            elif isinstance(node.value, ast.Name) and node.value.id in assigned:
                if node.attr not in _SAFE_ARRAY_METHODS | _SAFE_ARRAY_ATTRIBUTES:
                    raise ValueError(f"array attribute is not allowed: {node.attr}")
            elif node.attr in _SAFE_ARRAY_METHODS | _SAFE_ARRAY_ATTRIBUTES:
                pass
            else:
                raise ValueError("attribute access is restricted in generated operators")
        if isinstance(node, ast.Call):
            function_node = node.func
            if isinstance(function_node, ast.Attribute):
                if isinstance(function_node.value, ast.Name) and function_node.value.id == "np":
                    if function_node.attr not in _SAFE_NUMPY_CALLS:
                        raise ValueError(f"NumPy call is not allowed: {function_node.attr}")
                elif isinstance(function_node.value, ast.Name) and function_node.value.id == "params":
                    if function_node.attr != "get":
                        raise ValueError("only params.get is allowed in generated operators")
                elif isinstance(function_node.value, ast.Name) and function_node.value.id in assigned:
                    if function_node.attr not in _SAFE_ARRAY_METHODS:
                        raise ValueError(f"array method is not allowed: {function_node.attr}")
                elif function_node.attr in _SAFE_ARRAY_METHODS:
                    pass
                else:
                    raise ValueError("function calls are restricted in generated operators")
            else:
                raise ValueError("only approved NumPy and params.get calls are allowed")
    compile(tree, "<generated_operator>", "exec")
    return True


def register_generated_operator(registry, raw_spec):
    spec = raw_spec if isinstance(raw_spec, GeneratedOperatorSpec) else normalize_generated_operator(raw_spec)
    if spec.name in registry.names():
        raise ValueError(f"generated operator conflicts with registered operator: {spec.name}")
    input_type = ARTIFACT_TYPES[spec.input_artifact]
    output_type = ARTIFACT_TYPES[spec.output_artifact]
    code = compile(ast.parse(spec.source, mode="exec"), f"<generated:{spec.name}>", "exec")

    def generated_operator(artifact, **params):
        namespace = {"__builtins__": {}, "np": np}
        exec(code, namespace, namespace)
        apply = namespace.get("apply")
        if not callable(apply):
            raise ValueError(f"generated operator {spec.name} did not define apply")
        output = apply(artifact.data, dict(params))
        if not isinstance(output, np.ndarray):
            output = np.asarray(output)
        result_artifact = output_type(output, metadata={"operator": spec.name, "version": spec.version})
        metadata = {"operator": spec.name, "version": spec.version, "generated": True}
        return OperatorResult(result_artifact, metadata)

    registry.register(spec.name, generated_operator, input_type, output_type)
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
