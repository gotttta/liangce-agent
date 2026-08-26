from dataclasses import dataclass
import inspect
import time
from typing import Optional

import numpy as np

from core.operators import (
    ContourArtifact,
    ImageArtifact,
    MaskArtifact,
    build_default_registry,
    normalize_generated_operators,
)
from core.quality import mask_statistics


BUILTIN_PIPELINE_NAMES = {"periodic_particle_builtin"}


ALLOWED_PIPELINE_OPERATORS = {
    "adaptive_threshold",
    "bilateral_denoise",
    "component_statistics",
    "convex_hull",
    "hysteresis_threshold",
    "invert_intensity",
    "local_contrast",
    "local_background_residual",
    "median_denoise",
    "morphological_residual",
    "normalize",
    "percentile_clip",
    "gaussian_denoise",
    "global_threshold",
    "morphology",
    "fill_holes",
    "filter_components",
    "extract_contours",
    "remove_border_components",
    "remove_small_objects",
    "statistical_threshold",
    "unsharp_enhance",
}

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
    mask: MaskArtifact
    contours: Optional[ContourArtifact]
    trace: tuple
    artifacts: dict


def pipeline_operator_catalog(registry=None, generated_operators=None, include_builtin=True):
    """Return operator contracts, optionally excluding unapproved built-ins.

    Built-ins remain available to replay existing accepted pipelines, but they
    are not automatically advertised to the model as approved reusable tools.
    """
    if not include_builtin:
        return []
    registry = registry or build_default_registry(generated_operators)
    catalog = []
    for name in sorted(ALLOWED_PIPELINE_OPERATORS):
        definition = registry.definition(name)
        parameters = []
        for parameter in inspect.signature(definition.function).parameters.values():
            if parameter.name in {"image", "mask"}:
                continue
            default = None if parameter.default is inspect.Parameter.empty else parameter.default
            parameters.append({"name": parameter.name, "default": default})
        catalog.append({
            "name": name,
            "description": OPERATOR_DESCRIPTIONS.get(name, "可复用的自定义 CV 算子"),
            "input_artifact": definition.input_type.__name__,
            "output_artifact": definition.output_type.__name__,
            "parameters": parameters,
        })
    return catalog


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
    if isinstance(raw_pipeline, dict) and raw_pipeline.get("kind") == "builtin_pipeline":
        builtin_name = str(raw_pipeline.get("name") or name)
        if builtin_name not in BUILTIN_PIPELINE_NAMES:
            raise ValueError(f"unsupported builtin pipeline: {builtin_name}")
        params = raw_pipeline.get("params")
        return {
            "name": builtin_name,
            "kind": "builtin_pipeline",
            "params": dict(params) if isinstance(params, dict) else {},
        }
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
    return normalized


def is_builtin_pipeline(pipeline):
    return isinstance(pipeline, dict) and pipeline.get("kind") == "builtin_pipeline"


def validate_builtin_pipeline(pipeline):
    if not is_builtin_pipeline(pipeline):
        raise ValueError("not a builtin pipeline")
    if pipeline.get("name") not in BUILTIN_PIPELINE_NAMES:
        raise ValueError(f"unsupported builtin pipeline: {pipeline.get('name')}")
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


def validate_pipeline(pipeline, registry=None):
    if is_builtin_pipeline(pipeline):
        return validate_builtin_pipeline(pipeline)
    generated_specs = pipeline.get("generated_operators") or [] if isinstance(pipeline, dict) else []
    registry = registry or build_default_registry(generated_specs)
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
    return pipeline


def execute_pipeline(image, pipeline, allow_generated=False):
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
    registry = build_default_registry(generated_specs)
    validate_pipeline(pipeline, registry=registry)
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
        if isinstance(artifact, MaskArtifact) and artifact.data.shape != source.data.shape:
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
