from dataclasses import dataclass, field
from copy import deepcopy
import inspect
import time
from typing import Optional

import numpy as np

from core.operators import (
    ContourArtifact,
    ImageArtifact,
    MaskArtifact,
    MetadataArtifact,
    build_default_registry,
    normalize_generated_operators,
)
from core.quality import mask_statistics
from core.operator_catalog import model_visible_operator_names


BUILTIN_PIPELINE_NAMES = {"periodic_particle_builtin"}


ALLOWED_PIPELINE_OPERATORS = model_visible_operator_names()

OPERATOR_DESCRIPTIONS = {
    "adaptive_threshold": "按局部邻域阈值分割，适合晶圆图像中的不均匀照明。",
    "bilateral_denoise": "边缘保持去噪，降低 SEM 噪声同时保留缺陷边界。",
    "component_statistics": "计算候选连通域的面积、质心、边界框、实心度和偏心率。",
    "convex_hull": "填补候选区域的凹陷，适合碎裂或不规则颗粒的形状修复。",
    "hysteresis_threshold": "保留强残差及其连接的弱残差，减少断裂边缘。",
    "invert_intensity": "反转稳健归一化后的灰度极性。",
    "local_contrast": "使用 CLAHE 增强局部对比度，突出低对比度缺陷。",
    "normalize": "按图像分位数归一化灰度，降低亮度和对比度差异。",
    "gaussian_denoise": "高斯去噪，抑制高频成像噪声。",
    "global_threshold": "按亮度或暗度阈值生成初始二值 Mask。",
    "morphology": "通过开闭、膨胀或腐蚀清理和连接 Mask。",
    "fill_holes": "填充目标区域中的封闭空洞。",
    "filter_components": "按面积、长宽比和数量限制筛选连通域。",
    "extract_contours": "从最终二值 Mask 提取闭合的 x/y 像素轮廓，不改变 Mask。",
    "local_background_residual": "从平滑局部背景中提取暗缺陷、亮缺陷或绝对残差。",
    "median_denoise": "中值去噪，抑制孤立像素噪声并保留缺陷边缘。",
    "morphological_residual": "使用黑顶帽或白顶帽增强局部暗缺陷或亮缺陷。",
    "percentile_clip": "裁剪极端灰度值，降低亮点和暗点对后续阈值的影响。",
    "remove_border_components": "移除接触图像边界的候选连通域，避免截断目标造成误检。",
    "remove_small_objects": "移除小于最小缺陷面积的孤立候选区域。",
    "statistical_threshold": "使用 Otsu、Yen、Li、Triangle 或均值阈值进行全局分割。",
    "unsharp_enhance": "增强缺陷边缘和局部纹理对比度。",
}


@dataclass
class PipelineExecutionResult:
    pipeline: dict
    mask: Optional[MaskArtifact]
    contours: Optional[ContourArtifact]
    trace: tuple
    artifacts: dict
    outputs: dict = field(default_factory=dict)


def pipeline_operator_catalog(registry=None, generated_operators=None, include_builtin=True):
    """Return the model-visible v3 operator catalog from the execution registry."""
    if not include_builtin:
        return []
    registry = registry or build_default_registry(generated_operators)
    catalog = []
    for name in registry.names():
        definition = registry.definition(name)
        if not definition.model_visible:
            continue
        parameters = []
        for parameter in inspect.signature(definition.function).parameters.values():
            if parameter.name in definition.input_ports:
                continue
            default = None if parameter.default is inspect.Parameter.empty else parameter.default
            parameters.append({"name": parameter.name, "default": default})
        catalog.append({
            "name": name,
            "version": definition.version,
            "description": definition.description or OPERATOR_DESCRIPTIONS.get(name, "可复用的自定义 CV 算子"),
            "input_artifact": definition.input_type.__name__,
            "output_artifact": definition.output_type.__name__,
            "input_ports": {
                port: artifact_type.__name__
                for port, artifact_type in definition.input_ports.items()
            },
            "parameters": parameters,
        })
    return catalog


def _canonical_pipeline_names(pipeline):
    """Read legacy field names, emitting only the Operator vocabulary."""
    if not isinstance(pipeline, dict):
        return pipeline
    pipeline = deepcopy(pipeline)
    if "tool_versions" in pipeline:
        legacy = pipeline.pop("tool_versions")
        if "operator_versions" in pipeline and pipeline["operator_versions"] != legacy:
            raise ValueError("conflicting operator_versions and legacy tool_versions")
        pipeline["operator_versions"] = legacy
    if isinstance(pipeline.get("nodes"), list):
        for node in pipeline["nodes"]:
            if not isinstance(node, dict):
                continue
            for alias in ("tool", "op"):
                if alias in node:
                    legacy = node.pop(alias)
                    if "operator" in node and node["operator"] != legacy:
                        raise ValueError("conflicting pipeline operator names")
                    node["operator"] = legacy
    return pipeline


def pin_pipeline_operator_versions(pipeline, registry=None):
    """Return a replay record with the exact Operator versions used by this run."""
    pipeline = _canonical_pipeline_names(pipeline)
    if not isinstance(pipeline, dict):
        raise ValueError("pipeline must be an object")
    validate_pipeline(pipeline, registry=registry)
    pinned = deepcopy(pipeline)
    if "operator_versions" in pinned:
        return pinned
    pinned["version_provenance"] = "recorded_at_execution"
    if is_builtin_pipeline(pinned):
        pinned["operator_versions"] = {pinned.get("name", "builtin_pipeline"): "legacy-1.0.0"}
        return pinned
    generated_specs = pinned.get("generated_operators") or []
    registry = registry or build_default_registry(generated_specs)
    entries = pinned.get("nodes") if is_v3_pipeline(pinned) else pinned.get("steps", [])
    versions = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("operator") or entry.get("op")
        if name in registry.names():
            versions[name] = registry.definition(name).version
    pinned["operator_versions"] = dict(sorted(versions.items()))
    return pinned


def pin_pipeline_tool_versions(pipeline, registry=None):
    """Compatibility alias; new callers use pin_pipeline_operator_versions."""
    return pin_pipeline_operator_versions(pipeline, registry=registry)


def _validate_operator_versions(pipeline, expected):
    if "operator_versions" not in pipeline:
        return
    recorded = pipeline["operator_versions"]
    if not isinstance(recorded, dict) or set(recorded) != set(expected):
        raise ValueError("pipeline operator_versions must cover exactly the executed operators")
    for name, version in expected.items():
        if recorded[name] != version:
            raise ValueError(
                f"pipeline operator version mismatch for {name}: "
                f"recorded {recorded[name]!r}, installed {version!r}; explicit migration required"
            )


def _preserve_replay_metadata(raw, normalized):
    for key in ("operator_versions", "skill", "version_provenance"):
        if key in raw:
            normalized[key] = deepcopy(raw[key])
    return normalized


def strategy_to_pipeline(strategy, name="qwen_strategy_baseline"):
    segmentation = (strategy or {}).get("segmentation", {})
    method = segmentation.get("method", "auto_bright_dark_threshold")
    polarity = "dark" if method == "dark_threshold" else "bright"
    morphology = segmentation.get("morphology", "open_then_close")
    steps = [
        {"id": "normalized", "op": "normalize", "input": "image", "params": {}},
        {
            "id": "threshold_mask",
            "op": "global_threshold",
            "input": "normalized",
            "params": {
                "polarity": polarity,
                "sensitivity": float(segmentation.get("sensitivity", 1.8)),
                "max_coverage": 0.35,
            },
        },
    ]
    if morphology != "none":
        steps.append({
            "id": "morphed_mask",
            "op": "morphology",
            "input": "threshold_mask",
            "params": {"method": morphology, "radius": 1},
        })
        mask_input = "morphed_mask"
    else:
        mask_input = "threshold_mask"
    steps.extend([
        {"id": "filled_mask", "op": "fill_holes", "input": mask_input, "params": {}},
        {
            "id": "final_mask",
            "op": "filter_components",
            "input": "filled_mask",
            "params": {
                "min_area": int(segmentation.get("min_area_px", 20)),
                "max_area": segmentation.get("max_area_px"),
            },
        },
        {"id": "contours", "op": "extract_contours", "input": "final_mask", "params": {}},
    ])
    return {"name": name, "steps": steps}


def normalize_pipeline(raw_pipeline, fallback_strategy=None, name="candidate"):
    raw_pipeline = _canonical_pipeline_names(raw_pipeline)
    if isinstance(raw_pipeline, dict) and raw_pipeline.get("kind") == "builtin_pipeline":
        builtin_name = str(raw_pipeline.get("name") or name)
        if builtin_name not in BUILTIN_PIPELINE_NAMES:
            raise ValueError(f"unsupported builtin pipeline: {builtin_name}")
        params = raw_pipeline.get("params")
        return _preserve_replay_metadata(raw_pipeline, {
            "name": builtin_name,
            "kind": "builtin_pipeline",
            "params": dict(params) if isinstance(params, dict) else {},
        })
    if isinstance(raw_pipeline, dict) and (
        raw_pipeline.get("schema_version") == 3 or "nodes" in raw_pipeline
    ):
        if not isinstance(raw_pipeline.get("nodes"), list) or not raw_pipeline.get("nodes"):
            raise ValueError("v3 candidate pipeline must contain non-empty nodes")
        normalized = {
            "schema_version": 3,
            "name": str(raw_pipeline.get("name") or name),
            "nodes": [_normalize_pipeline_node(node) for node in raw_pipeline["nodes"] if isinstance(node, dict)],
        }
        if isinstance(raw_pipeline.get("input_types"), dict):
            normalized["input_types"] = dict(raw_pipeline["input_types"])
        if isinstance(raw_pipeline.get("outputs"), dict):
            normalized["outputs"] = dict(raw_pipeline["outputs"])
        if raw_pipeline.get("generated_operators") is not None:
            normalized["generated_operators"] = [
                spec.as_dict() for spec in normalize_generated_operators(raw_pipeline.get("generated_operators"))
            ]
        return _preserve_replay_metadata(raw_pipeline, normalized)
    if not isinstance(raw_pipeline, dict) or not isinstance(raw_pipeline.get("steps"), list) or not raw_pipeline.get("steps"):
        if fallback_strategy is not None:
            return strategy_to_pipeline(fallback_strategy, name=name)
        raise ValueError("candidate pipeline must contain a non-empty steps list")
    normalized = {
        "name": str(raw_pipeline.get("name") or name),
        "steps": [_normalize_pipeline_step(step) for step in raw_pipeline["steps"] if isinstance(step, dict)],
    }
    if raw_pipeline.get("generated_operators") is not None:
        normalized["generated_operators"] = [
            spec.as_dict() for spec in normalize_generated_operators(raw_pipeline.get("generated_operators"))
        ]
    return _preserve_replay_metadata(raw_pipeline, normalized)


def is_builtin_pipeline(pipeline):
    return isinstance(pipeline, dict) and pipeline.get("kind") == "builtin_pipeline"


def validate_builtin_pipeline(pipeline):
    pipeline = _canonical_pipeline_names(pipeline)
    if not is_builtin_pipeline(pipeline):
        raise ValueError("not a builtin pipeline")
    if pipeline.get("name") not in BUILTIN_PIPELINE_NAMES:
        raise ValueError(f"unsupported builtin pipeline: {pipeline.get('name')}")
    _validate_operator_versions(pipeline, {pipeline["name"]: "legacy-1.0.0"})
    params = pipeline.get("params", {})
    if not isinstance(params, dict):
        raise ValueError("builtin pipeline params must be an object")
    allowed = {"percentile", "min_area", "border_px", "max_components", "roi"}
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f"builtin pipeline has unknown params: {sorted(unknown)}")
    if "percentile" in params and not 0 < float(params["percentile"]) < 100:
        raise ValueError("builtin percentile must be between 0 and 100")
    for key, minimum in (("min_area", 1), ("border_px", 0), ("max_components", 1)):
        if key in params and (not isinstance(params[key], int) or params[key] < minimum):
            raise ValueError(f"builtin {key} must be an integer >= {minimum}")
    if params.get("roi") is not None:
        roi = params["roi"]
        if not isinstance(roi, (list, tuple)) or len(roi) != 4:
            raise ValueError("builtin roi must be [x, y, width, height]")
    return pipeline


def _normalize_pipeline_step(raw_step):
    step = dict(raw_step)
    params = step.get("params")
    params = dict(params) if isinstance(params, dict) else {}
    if step.get("op") == "morphology":
        if "method" not in params:
            for alias in ("operation", "op"):
                if alias in params:
                    params["method"] = params[alias]
                    break
        params.pop("operation", None)
        params.pop("op", None)
        method_aliases = {
            "opening": "open",
            "closing": "close",
            "dilation": "dilate",
            "erosion": "erode",
        }
        method = str(params.get("method", "open_then_close")).lower()
        params["method"] = method_aliases.get(method, method)
        if "radius" not in params and "kernel_size" in params:
            try:
                kernel_size = max(1, int(params.pop("kernel_size")))
                params["radius"] = min(50, kernel_size // 2)
            except (TypeError, ValueError):
                params.pop("kernel_size", None)
                params["radius"] = 1
        else:
            params.pop("kernel_size", None)
    step["params"] = params
    return step


def _normalize_pipeline_node(raw_node):
    node = dict(raw_node)
    if "operator" not in node and node.get("op"):
        node["operator"] = node.pop("op")
    node["params"] = dict(node.get("params") or {})
    node["inputs"] = dict(node.get("inputs") or {})
    if node.get("operator") == "morphology":
        node = _normalize_pipeline_step({
            "id": node.get("id"),
            "op": node["operator"],
            "params": node["params"],
        }) | {"inputs": node["inputs"], "operator": node["operator"]}
        node.pop("op", None)
    return node


def is_v3_pipeline(pipeline):
    return isinstance(pipeline, dict) and (
        pipeline.get("schema_version") == 3 or "nodes" in pipeline
    )


def validate_pipeline(pipeline, registry=None):
    pipeline = _canonical_pipeline_names(pipeline)
    if is_builtin_pipeline(pipeline):
        return validate_builtin_pipeline(pipeline)
    generated_specs = pipeline.get("generated_operators") or [] if isinstance(pipeline, dict) else []
    registry = registry or build_default_registry(generated_specs)
    if is_v3_pipeline(pipeline):
        return _validate_v3_pipeline(pipeline, registry, generated_specs)
    if not isinstance(pipeline, dict):
        raise ValueError("pipeline must be an object")
    steps = pipeline.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("pipeline.steps must be a non-empty list")

    artifact_types = {"image": ImageArtifact}
    generated_names = {
        str(item.get("name"))
        for item in generated_specs
        if isinstance(item, dict) and item.get("name")
    }
    seen_ids = set()
    has_mask = False
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"pipeline step {index} must be an object")
        step_id = step.get("id")
        op = step.get("op")
        input_id = step.get("input", "image" if index == 0 else steps[index - 1].get("id"))
        params = step.get("params", {})
        if not isinstance(step_id, str) or not step_id:
            raise ValueError(f"pipeline step {index} requires a non-empty id")
        if step_id in seen_ids or step_id == "image":
            raise ValueError(f"duplicate or reserved pipeline step id: {step_id}")
        if op not in registry.names():
            raise ValueError(f"pipeline operator is not allowed: {op}")
        if input_id not in artifact_types:
            raise ValueError(f"pipeline step {step_id} references unknown input: {input_id}")
        if not isinstance(params, dict):
            raise ValueError(f"pipeline step {step_id} params must be an object")

        definition = registry.definition(op)
        if op not in generated_names and not definition.legacy_allowed:
            raise ValueError(f"pipeline operator requires v3 named inputs: {op}")
        actual_type = artifact_types[input_id]
        if not issubclass(actual_type, definition.input_type):
            raise ValueError(
                f"pipeline step {step_id} expects {definition.input_type.__name__}, "
                f"but input {input_id} is {actual_type.__name__}"
            )
        if op not in generated_names:
            allowed_params = set(inspect.signature(definition.function).parameters) - {"image", "mask"}
            unknown_params = set(params) - allowed_params
            if unknown_params:
                raise ValueError(f"pipeline step {step_id} has unknown params: {sorted(unknown_params)}")

        artifact_types[step_id] = definition.output_type
        seen_ids.add(step_id)
        has_mask = has_mask or definition.output_type is MaskArtifact

    if not has_mask:
        raise ValueError("pipeline must produce a mask")
    _validate_operator_versions(pipeline, {
        step["op"]: registry.definition(step["op"]).version for step in steps
    })
    return pipeline


def _validate_v3_pipeline(pipeline, registry, generated_specs):
    if not isinstance(pipeline, dict):
        raise ValueError("pipeline must be an object")
    nodes = pipeline.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("pipeline.nodes must be a non-empty list")
    generated_names = {
        str(item.get("name")) for item in generated_specs
        if isinstance(item, dict) and item.get("name")
    }
    node_by_id = {}
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ValueError(f"pipeline node {index} must be an object")
        node_id = node.get("id")
        operator = node.get("operator")
        inputs = node.get("inputs")
        params = node.get("params", {})
        if not isinstance(node_id, str) or not node_id or node_id in {"image", "$image"}:
            raise ValueError(f"pipeline node {index} requires a unique non-reserved id")
        if node_id in node_by_id or node_id in pipeline.get("input_types", {}):
            raise ValueError(f"duplicate pipeline node id: {node_id}")
        if operator not in registry.names():
            raise ValueError(f"pipeline operator is not allowed: {operator}")
        definition = registry.definition(operator)
        if operator not in generated_names and not definition.model_visible:
            raise ValueError(f"pipeline operator is not model-visible: {operator}")
        if not isinstance(inputs, dict):
            raise ValueError(f"pipeline node {node_id} inputs must be an object")
        if set(inputs) != set(definition.input_ports):
            raise ValueError(
                f"pipeline node {node_id} expects input ports {sorted(definition.input_ports)}, "
                f"got {sorted(inputs)}"
            )
        if not all(isinstance(reference, str) and reference for reference in inputs.values()):
            raise ValueError(f"pipeline node {node_id} input references must be non-empty strings")
        if not isinstance(params, dict):
            raise ValueError(f"pipeline node {node_id} params must be an object")
        if operator not in generated_names:
            allowed_params = set(inspect.signature(definition.function).parameters) - set(definition.input_ports)
            unknown_params = set(params) - allowed_params
            if unknown_params:
                raise ValueError(f"pipeline node {node_id} has unknown params: {sorted(unknown_params)}")
        node_by_id[node_id] = node

    artifact_types = {"$image": ImageArtifact, "image": ImageArtifact}
    from core.operators.generated import ARTIFACT_TYPES
    declared_inputs = pipeline.get("input_types", {})
    if not isinstance(declared_inputs, dict):
        raise ValueError("input_types must be a mapping")
    for name, kind in declared_inputs.items():
        if not isinstance(name, str) or not name.startswith("$") or name == "$image" or kind not in ARTIFACT_TYPES:
            raise ValueError("invalid external input declaration")
        artifact_types[name] = ARTIFACT_TYPES[kind]
    visiting = set()
    visited = set()

    def resolve(node_id):
        if node_id in artifact_types:
            return artifact_types[node_id]
        if node_id not in node_by_id:
            raise ValueError(f"pipeline references unknown node: {node_id}")
        if node_id in visiting:
            raise ValueError(f"pipeline contains a cycle at node: {node_id}")
        if node_id in visited:
            return artifact_types[node_id]
        visiting.add(node_id)
        node = node_by_id[node_id]
        definition = registry.definition(node["operator"])
        for port, expected_type in definition.input_ports.items():
            actual_type = resolve(node["inputs"][port])
            if not issubclass(actual_type, expected_type):
                raise ValueError(
                    f"pipeline node {node_id} input {port} expects {expected_type.__name__}, "
                    f"but {node['inputs'][port]} is {actual_type.__name__}"
                )
        artifact_types[node_id] = definition.output_type
        visiting.remove(node_id)
        visited.add(node_id)
        return definition.output_type

    for node_id in node_by_id:
        resolve(node_id)
    outputs = pipeline.get("outputs", {})
    if not isinstance(outputs, dict):
        raise ValueError("pipeline outputs must be an object")
    if "outputs" in pipeline and not outputs:
        raise ValueError("explicit outputs must not be empty")
    mask_output = outputs.get("mask") if isinstance(outputs, dict) else None
    if mask_output is not None:
        if mask_output not in artifact_types or not issubclass(artifact_types[mask_output], MaskArtifact):
            raise ValueError("pipeline outputs.mask must reference a MaskArtifact node")
    elif not outputs and not any(issubclass(artifact_type, MaskArtifact) for artifact_type in artifact_types.values()):
        raise ValueError("pipeline must produce a mask")
    for name, reference in outputs.items():
        if not isinstance(reference, str) or reference not in node_by_id:
            raise ValueError(f"pipeline output {name} must reference a node")
    contour_output = outputs.get("contours") if isinstance(outputs, dict) else None
    if contour_output is not None:
        if contour_output not in artifact_types or not issubclass(artifact_types[contour_output], ContourArtifact):
            raise ValueError("pipeline outputs.contours must reference a ContourArtifact node")
    _validate_operator_versions(pipeline, {
        node["operator"]: registry.definition(node["operator"]).version for node in nodes
    })
    return pipeline


def _v3_execution_order(pipeline):
    nodes = {node["id"]: node for node in pipeline["nodes"]}
    ordered = []
    seen = set()

    def visit(node_id):
        if node_id in seen:
            return
        node = nodes[node_id]
        for reference in node.get("inputs", {}).values():
            if reference in nodes:
                visit(reference)
        seen.add(node_id)
        ordered.append(node)

    for node in pipeline["nodes"]:
        visit(node["id"])
    return ordered


def execute_pipeline(image, pipeline, allow_generated=False, inputs=None):
    pipeline = _canonical_pipeline_names(pipeline)
    if is_builtin_pipeline(pipeline):
        validate_builtin_pipeline(pipeline)
        from core.pipelines.periodic_particle import run_periodic_particle_pipeline

        result = run_periodic_particle_pipeline(np.asarray(image, dtype=np.float32), **pipeline.get("params", {}))
        return PipelineExecutionResult(
            pipeline=pipeline,
            mask=result.mask,
            contours=result.contours,
            trace=result.trace,
            artifacts={},
        )
    generated_specs = pipeline.get("generated_operators") or []
    if generated_specs and not allow_generated:
        raise ValueError("generated operators may only execute inside the sandbox")
    if generated_specs:
        from core.sandbox import require_docker_worker
        require_docker_worker()
    registry = build_default_registry(generated_specs)
    validate_pipeline(pipeline, registry=registry)
    if is_v3_pipeline(pipeline):
        return _execute_v3_pipeline(image, pipeline, registry, inputs=inputs)
    source = ImageArtifact(np.asarray(image, dtype=np.float32))
    artifacts = {"image": source}
    trace = []
    final_mask = None
    final_contours = None

    for step in pipeline["steps"]:
        input_id = step.get("input") or next(reversed(artifacts))
        started = time.monotonic()
        result = registry.run(step["op"], artifacts[input_id], **step.get("params", {}))
        duration = time.monotonic() - started
        artifact = result.artifact
        if isinstance(artifact, MaskArtifact) and artifact.data.shape != source.data.shape[:2]:
            raise ValueError(f"pipeline step {step['id']} returned a mask with the wrong shape")
        artifacts[step["id"]] = artifact
        if isinstance(artifact, MaskArtifact):
            final_mask = artifact
        elif isinstance(artifact, ContourArtifact):
            final_contours = artifact
        warnings = list(result.warnings)
        mask_facts = None
        if isinstance(artifact, MaskArtifact):
            mask_facts = mask_statistics(artifact.data)
            if mask_facts["coverage"] == 0:
                warnings.append("empty_mask")
            if mask_facts["coverage"] > 0.35:
                warnings.append("coverage_exceeded")
            if result.metadata.get("kept_components") == 0:
                warnings.append("kept_components=0")
        trace.append({
            "step_id": step["id"],
            "operator": step["op"],
            "input": input_id,
            "params": step.get("params", {}),
            "duration_seconds": round(duration, 6),
            "metadata": result.metadata,
            "warnings": list(dict.fromkeys(warnings)),
            **({"mask_statistics": mask_facts} if mask_facts is not None else {}),
        })

    if final_mask is None:
        raise ValueError("pipeline must produce a mask for visual annotation")
    if final_contours is None:
        # A contour is a rendering derivative, not a required algorithm output.
        # This keeps mask-only tasks valid while preserving contour display.
        final_contours = registry.run("extract_contours", final_mask).artifact
    return PipelineExecutionResult(
        pipeline=pipeline,
        mask=final_mask,
        contours=final_contours,
        trace=tuple(trace),
        artifacts=artifacts,
    )


def _execute_v3_pipeline(image, pipeline, registry, inputs=None):
    source = ImageArtifact(np.asarray(image, dtype=np.float32))
    artifacts = {"$image": source, "image": source}
    from core.operators.generated import ARTIFACT_TYPES
    supplied = inputs or {}
    if set(supplied) != set(pipeline.get("input_types", {})):
        raise ValueError("external inputs must match input_types")
    for name, kind in pipeline.get("input_types", {}).items():
        artifacts[name] = ARTIFACT_TYPES[kind](supplied[name])
    trace = []
    for node in _v3_execution_order(pipeline):
        input_artifacts = {
            port: artifacts[reference]
            for port, reference in node["inputs"].items()
        }
        started = time.monotonic()
        result = registry.run_inputs(node["operator"], input_artifacts, **node.get("params", {}))
        duration = time.monotonic() - started
        artifact = result.artifact
        if isinstance(artifact, MaskArtifact) and artifact.data.shape != source.data.shape[:2]:
            raise ValueError(f"pipeline node {node['id']} returned a mask with the wrong shape")
        artifacts[node["id"]] = artifact
        warnings = list(result.warnings)
        mask_facts = None
        if isinstance(artifact, MaskArtifact):
            mask_facts = mask_statistics(artifact.data)
            if mask_facts["coverage"] == 0:
                warnings.append("empty_mask")
            if mask_facts["coverage"] > 0.35:
                warnings.append("coverage_exceeded")
            if result.metadata.get("kept_components") == 0:
                warnings.append("kept_components=0")
        trace.append({
            "step_id": node["id"],
            "operator": node["operator"],
            "inputs": node["inputs"],
            "params": node.get("params", {}),
            "duration_seconds": round(duration, 6),
            "metadata": result.metadata,
            "warnings": list(dict.fromkeys(warnings)),
            **({"mask_statistics": mask_facts} if mask_facts is not None else {}),
        })
    outputs = pipeline.get("outputs") or {}
    mask_id = next((ref for ref in outputs.values() if isinstance(artifacts[ref], MaskArtifact)), None)
    if "outputs" not in pipeline:
        mask_id = next(
            (node["id"] for node in reversed(_v3_execution_order(pipeline))
             if isinstance(artifacts[node["id"]], MaskArtifact)), None,
        )
    mask = artifacts[mask_id] if mask_id else None
    contour_id = outputs.get("contours")
    contours = artifacts.get(contour_id) if contour_id else None
    if not isinstance(contours, ContourArtifact) and mask is not None:
        contours = registry.run("extract_contours", mask).artifact
    return PipelineExecutionResult(
        pipeline=pipeline,
        mask=mask,
        contours=contours,
        trace=tuple(trace),
        artifacts=artifacts,
        outputs={name: artifacts[reference] for name, reference in outputs.items()},
    )
