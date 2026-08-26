"""Isolated execution for model-generated, allowlisted Pipeline DSL programs."""

from __future__ import annotations

import math
import multiprocessing as mp
import traceback
from dataclasses import dataclass

import numpy as np

from core.operators import ContourArtifact, MaskArtifact
from core.pipelines.dsl import (
    PipelineExecutionResult,
    execute_pipeline,
    is_builtin_pipeline,
    validate_pipeline,
)


class SandboxExecutionError(RuntimeError):
    """Raised when a generated pipeline cannot safely finish in the sandbox."""


@dataclass(frozen=True)
class SandboxLimits:
    timeout_seconds: float = 20.0
    memory_mb: int = 1024
    max_steps: int = 32


def execute_pipeline_sandbox(image, pipeline, limits=None):
    """Execute a validated DSL pipeline in a disposable child process.

    The model never supplies Python source. The parent validates the operator
    graph, then the child receives only a numeric image and JSON-like pipeline.
    """
    limits = limits or SandboxLimits()
    validate_pipeline(pipeline)
    steps = pipeline.get("steps", [])
    if len(steps) > int(limits.max_steps):
        raise SandboxExecutionError(f"pipeline exceeds the {limits.max_steps}-step sandbox limit")
    if limits.timeout_seconds <= 0 or limits.memory_mb <= 0:
        raise ValueError("sandbox limits must be positive")

    context = mp.get_context("spawn")
    parent_conn, child_conn = context.Pipe(duplex=False)
    process = context.Process(
        target=_sandbox_worker,
        args=(child_conn, np.asarray(image, dtype=np.float32), pipeline, limits),
        daemon=True,
    )
    process.start()
    child_conn.close()
    try:
        if not parent_conn.poll(float(limits.timeout_seconds)):
            process.terminate()
            process.join(timeout=2)
            raise SandboxExecutionError(
                f"pipeline exceeded the {limits.timeout_seconds:g}-second sandbox timeout"
            )
        payload = parent_conn.recv()
    finally:
        parent_conn.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)

    if payload.get("ok") is not True:
        raise SandboxExecutionError(payload.get("error") or "sandbox pipeline execution failed")
    return _deserialize_result(payload["result"])


def _sandbox_worker(connection, image, pipeline, limits):
    try:
        _apply_resource_limits(limits)
        result = execute_pipeline(image, pipeline, allow_generated=True)
        connection.send({"ok": True, "result": _serialize_result(result)})
    except BaseException as exc:  # the parent must always receive a bounded error
        connection.send({
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=5),
        })
    finally:
        connection.close()


def _apply_resource_limits(limits):
    try:
        import resource
    except ImportError:
        return
    try:
        cpu_seconds = max(1, int(math.ceil(float(limits.timeout_seconds))) + 1)
        cpu_soft, cpu_hard = resource.getrlimit(resource.RLIMIT_CPU)
        if cpu_hard == resource.RLIM_INFINITY or cpu_seconds <= cpu_hard:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_hard))
    except (OSError, ValueError):
        pass
    try:
        memory_bytes = int(limits.memory_mb) * 1024 * 1024
        if hasattr(resource, "RLIMIT_AS"):
            current_soft, current_hard = resource.getrlimit(resource.RLIMIT_AS)
            if current_hard != resource.RLIM_INFINITY:
                memory_bytes = min(memory_bytes, current_hard)
            if current_soft == resource.RLIM_INFINITY or memory_bytes < current_soft:
                resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, current_hard))
    except (OSError, ValueError):
        pass


def _serialize_result(result):
    contours = result.contours
    return {
        "pipeline": result.pipeline,
        "mask": result.mask.data.astype(np.uint8).tolist(),
        "mask_metadata": result.mask.metadata,
        "contours": None if contours is None else [item.tolist() for item in contours.contours],
        "contour_shape": None if contours is None else list(contours.image_shape),
        "contour_metadata": {} if contours is None else contours.metadata,
        "trace": list(result.trace),
    }


def _deserialize_result(payload):
    contours = None
    if payload.get("contours") is not None:
        contours = ContourArtifact(
            tuple(np.asarray(item, dtype=np.float32) for item in payload["contours"]),
            tuple(payload["contour_shape"]),
            metadata=payload.get("contour_metadata") or {},
        )
    return PipelineExecutionResult(
        pipeline=payload["pipeline"],
        mask=MaskArtifact(np.asarray(payload["mask"], dtype=bool), metadata=payload.get("mask_metadata") or {}),
        contours=contours,
        trace=tuple(payload.get("trace") or ()),
        artifacts={},
    )
