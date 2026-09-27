"""Disposable Docker execution. No generated Python runs on the host."""
from __future__ import annotations

from contextvars import ContextVar
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import selectors
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, fields

import numpy as np

from core.sandbox_transport import encode_frame, decode_frame
from core.experiments.artifacts import export_artifacts, import_artifacts
from core.operators import ContourArtifact, MaskArtifact
from core.pipelines.dsl import PipelineExecutionResult, is_v3_pipeline, validate_pipeline

DEFAULT_IMAGE = "liangce-sandbox:2"
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
ACTION_CONTAINER_LABEL = 'org.liangce.action-container'
container_name = ContextVar('sandbox_container_name', default=None)


class SandboxExecutionError(RuntimeError):
    def __init__(self, message, code="execution_failed"):
        super().__init__(message)
        self.code = code


def action_container_name(run_id, action_id):
    if any(not isinstance(value, str) or not value for value in (run_id, action_id)):
        raise ValueError('run and action IDs must be non-empty strings')
    identity = json.dumps([run_id, action_id], separators=(',', ':')).encode()
    return 'liangce-sandbox-action-' + sha256(identity).hexdigest()[:32]


def _validate_action_container_name(name):
    if not isinstance(name, str) or not re.fullmatch(r'liangce-sandbox-action-[a-f0-9]{32}', name):
        raise ValueError('invalid action container name')
    return name


def cleanup_action_container(name, timeout_seconds=15):
    """Remove only the exact container whose immutable ownership label matches this action."""
    name = _validate_action_container_name(name)
    if not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError('cleanup timeout must be finite and positive')
    docker = _docker_binary()
    deadline = time.monotonic() + timeout_seconds
    def run(args, cap):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SandboxExecutionError('Docker recovery cleanup timed out for ' + name, code='cleanup_failed')
        return subprocess.run([docker, *args], capture_output=True, timeout=min(cap, remaining))
    def absent(result):
        return result.returncode and any(message in result.stderr for message in
            (b'No such object', b'No such container'))
    try:
        probe = run(['inspect', '--type', 'container', name], 5)
        if absent(probe):
            return
        if probe.returncode:
            raise SandboxExecutionError('Docker could not inspect action container ' + name, code='cleanup_failed')
        records = json.loads(probe.stdout)
        if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict):
            raise ValueError('invalid Docker inspection result')
        record = records[0]
        labels = (record.get('Config') or {}).get('Labels') or {}
        identity = record.get('Id')
        if (record.get('Name') != '/' + name or labels.get(ACTION_CONTAINER_LABEL) != name
                or not isinstance(identity, str) or not re.fullmatch(r'[a-f0-9]{64}', identity)):
            raise SandboxExecutionError('Docker container ownership does not match action ' + name, code='cleanup_failed')
        removed = run(['rm', '-f', identity], 10)
        if removed.returncode and not absent(removed):
            probe = run(['inspect', '--type', 'container', identity], 5)
            if not absent(probe):
                raise SandboxExecutionError('Docker recovery cleanup failed for ' + name, code='cleanup_failed')
    except (OSError, subprocess.TimeoutExpired, ValueError, TypeError, AttributeError) as exc:
        raise SandboxExecutionError('Docker recovery cleanup could not complete for ' + name,
                                    code='cleanup_failed') from exc


@dataclass(frozen=True)
class SandboxLimits:
    timeout_seconds: float = 20.0
    memory_mb: int = 1024
    max_steps: int = 256
    cpus: float = 2.0
    pids: int = 64

    @classmethod
    def from_env(cls):
        defaults = cls()
        values = {}
        for item in fields(cls):
            default = getattr(defaults, item.name)
            raw = os.environ.get("LIANGCE_SANDBOX_" + item.name.upper())
            values[item.name] = type(default)(raw) if raw is not None else default
        return cls(**values)


def require_docker_worker():
    # A guard against accidental host execution, not the isolation boundary.
    # Only the trusted image contains this marker; Docker supplies the isolation.
    if not (Path('/.dockerenv').is_file() and
            Path('/opt/liangce-sandbox-worker').is_file()):
        raise ValueError("generated operators may only execute inside the Docker sandbox")


def _docker_binary():
    executable = shutil.which("docker")
    desktop = Path('/Applications/Docker.app/Contents/Resources/bin/docker')
    if executable:
        return executable
    if desktop.is_file():
        return str(desktop)
    raise SandboxExecutionError(
        "Docker is required. Install/start Docker Desktop and build the sandbox image; "
        "see docs/docker-sandbox.md. Host execution is disabled.", code="sandbox_unavailable")


def _run_args(docker, name, limits):
    args = [docker, "create", "--name", name, "--rm", "--pull=never", "-i",
            "--network=none", "--read-only", "--user=65534:65534",
            "--cap-drop=ALL", "--security-opt=no-new-privileges:true",
            "--memory", f"{limits.memory_mb}m", "--memory-swap", f"{limits.memory_mb}m",
            "--cpus", str(limits.cpus), "--pids-limit", str(limits.pids),
            "--ulimit", "nofile=128:128", "--ulimit", "core=0:0",
            "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=128m,mode=1777",
            "--ipc=none", "--log-driver=none", "--workdir=/tmp",
            "--env=OPENBLAS_NUM_THREADS=1", "--env=OMP_NUM_THREADS=1",
            "--env=MKL_NUM_THREADS=1", "--env=HOME=/tmp",
            os.environ.get("LIANGCE_SANDBOX_IMAGE", DEFAULT_IMAGE)]
    if name.startswith('liangce-sandbox-action-'):
        _validate_action_container_name(name)
        args[-1:-1] = ['--label', ACTION_CONTAINER_LABEL + '=' + name]
    return args


def check_sandbox_available():
    """Fail before model work when the daemon/image is unavailable; never pull."""
    docker = _docker_binary()
    image = os.environ.get('LIANGCE_SANDBOX_IMAGE', DEFAULT_IMAGE)
    try:
        probe = subprocess.run([docker, 'image', 'inspect', '--format', '{{.Id}}', image],
                               capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SandboxExecutionError(f'Docker preflight failed: {exc}', code='sandbox_unavailable') from exc
    if probe.returncode:
        raise SandboxExecutionError(
            'Docker daemon or sandbox image unavailable. Start Docker and check ' + image + ': ' +
            probe.stderr.decode(errors='replace')[-1500:], code='sandbox_unavailable')
    return {'image': image, 'image_id': probe.stdout.decode().strip()}


def _bounded_output(process, timeout, max_bytes=MAX_OUTPUT_BYTES):
    """Drain both pipes without unbounded communicate()/disk log accumulation."""
    deadline = time.monotonic() + timeout
    chunks = {"stdout": bytearray(), "stderr": bytearray()}
    used = 0
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        while selector.get_map():
            from core.request_control import check_cancelled
            check_cancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SandboxExecutionError("Docker sandbox exceeded its timeout", code="timeout")
            for key, _ in selector.select(min(remaining, 0.2)):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                used += len(chunk)
                if used > max_bytes:
                    raise SandboxExecutionError("Docker sandbox output limit exceeded", code="resource_limit")
                chunks[key.data].extend(chunk)
        try:
            process.wait(timeout=max(0.001, deadline - time.monotonic()))
        except subprocess.TimeoutExpired as exc:
            raise SandboxExecutionError("Docker sandbox exceeded its timeout", code="timeout") from exc
    return bytes(chunks["stdout"]), bytes(chunks["stderr"])


def execute_pipeline_sandbox(image, pipeline, limits=None, inputs=None):
    limits = limits or SandboxLimits.from_env()
    for value in (limits.timeout_seconds, limits.memory_mb, limits.max_steps, limits.cpus, limits.pids):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("sandbox limits must be finite and positive")
    validate_pipeline(pipeline)
    steps = pipeline.get("nodes", []) if is_v3_pipeline(pipeline) else pipeline.get("steps", [])
    if len(steps) > limits.max_steps:
        raise SandboxExecutionError(f"pipeline exceeds the {limits.max_steps}-step sandbox limit", code="resource_limit")
    array = np.asarray(image, dtype=np.float32)
    if array.ndim not in (2, 3) or not array.size or not np.isfinite(array).all():
        raise ValueError("sandbox input must be a finite non-empty 2D image")
    if array.nbytes > MAX_INPUT_BYTES:
        raise SandboxExecutionError("sandbox input image is too large", code="resource_limit")
    try:
        request = encode_frame({"image": array, "pipeline": pipeline, "inputs": inputs or {}}, MAX_INPUT_BYTES)
    except ValueError as exc:
        raise SandboxExecutionError(str(exc), code="resource_limit") from exc
    requested_name = container_name.get()
    name = _validate_action_container_name(requested_name) if requested_name is not None else 'liangce-sandbox-' + uuid.uuid4().hex
    docker = _docker_binary()
    process = None
    try:
        # Create first so timeout cleanup cannot race a still-starting `docker run`.
        try:
            created = subprocess.run(_run_args(docker, name, limits),
                                     capture_output=True, timeout=15)
        except subprocess.TimeoutExpired as exc:
            raise SandboxExecutionError("Docker container creation timed out", code="sandbox_unavailable") from exc
        if created.returncode:
            raise SandboxExecutionError(
                "Docker container creation failed: " + created.stderr.decode(errors="replace")[-2000:],
                code="sandbox_unavailable")
        with tempfile.TemporaryFile() as source:
            source.write(request)
            source.seek(0)
            process = subprocess.Popen([docker, "start", "--attach", "--interactive", name], stdin=source,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, errors = _bounded_output(process, limits.timeout_seconds)
        if process.returncode:
            code = "resource_limit" if process.returncode == 137 else "worker_terminated"
            raise SandboxExecutionError(
                "Docker sandbox failed: " + errors.decode(errors="replace")[-2000:], code=code)
        try:
            payload = decode_frame(output, MAX_OUTPUT_BYTES)
            if not isinstance(payload, dict):
                raise ValueError("expected a JSON object")
            if payload.get("ok") is not True:
                raise SandboxExecutionError(str(payload.get("error", "sandbox failed"))[:2000],
                                            code="resource_limit" if payload.get("code") == "resource_limit" else "execution_failed")
            result = _deserialize_result(payload["result"])
            if result.mask is not None and result.mask.data.shape != array.shape[:2]:
                raise ValueError("result mask has the wrong shape")
            if result.contours and result.contours.image_shape != array.shape[:2]:
                raise ValueError("result contours have the wrong image shape")
            # Never trust returned source or replay metadata from generated code.
            result.pipeline = pipeline
            return result
        except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
            raise SandboxExecutionError(f"Invalid sandbox result: {exc}", code="invalid_output") from exc
    except OSError as exc:
        raise SandboxExecutionError(f"Docker could not start: {exc}", code="sandbox_unavailable") from exc
    finally:
        primary_error = sys.exc_info()[1]
        # Killing the CLI alone does NOT kill the container. Always remove by name.
        try:
            # Release attach pipes before deleting a flooding container. The
            # container itself is still removed below; killing the CLI is not cleanup.
            if process is not None and process.poll() is None:
                process.kill()
            if requested_name is not None:
                cleanup_action_container(name)
                cleanup = None
            else:
                cleanup = subprocess.run([docker, "rm", "-f", name], stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, timeout=10)
            if cleanup is not None and cleanup.returncode and b"No such container" not in cleanup.stderr:
                # --rm may finish removing the container concurrently with rm -f.
                # Only a confirmed absence is success; daemon/permission errors stay errors.
                probe = subprocess.run([docker, "inspect", "--type", "container", name],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
                absent = probe.returncode and any(message in probe.stderr for message in
                    (b"No such object", b"No such container"))
                if not absent:
                    raise SandboxExecutionError("Docker cleanup failed for " + name + ": " +
                                                cleanup.stderr.decode(errors="replace")[-1000:], code="cleanup_failed")
        except (SandboxExecutionError, OSError, subprocess.TimeoutExpired) as exc:
            cleanup_error = exc if isinstance(exc, SandboxExecutionError) else SandboxExecutionError(
                'Docker cleanup could not complete for ' + name, code='cleanup_failed')
            if primary_error is None:
                raise cleanup_error from exc
            # Preserve the cause of failure, while retaining secondary cleanup evidence.
            primary_error.add_note(str(cleanup_error))
            from core.runtime_logging import logger
            logger.warning('Sandbox cleanup also failed: %s', cleanup_error)
        finally:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.wait()
                process.stdout.close()
                process.stderr.close()


def _serialize_result(result):
    contours = result.contours
    return {
        "mask": None if result.mask is None else result.mask.data.astype(np.uint8),
        "mask_metadata": {} if result.mask is None else result.mask.metadata,
        "contours": None if contours is None else [item.tolist() for item in contours.contours],
        "contour_shape": None if contours is None else list(contours.image_shape),
        "contour_metadata": {} if contours is None else contours.metadata,
        "trace": list(result.trace),
        "artifacts": export_artifacts(result.artifacts),
        "outputs": export_artifacts(result.outputs, final=True),
    }


def _deserialize_result(payload):
    mask = None if payload["mask"] is None else np.asarray(payload["mask"])
    if mask is not None and (mask.ndim != 2 or not np.isin(mask, [0, 1]).all()):
        raise ValueError("mask must be a two-dimensional binary array")
    for key in ("mask_metadata", "contour_metadata", "artifacts"):
        if not isinstance(payload.get(key, {}), dict):
            raise ValueError(f"invalid {key}")
    trace = payload.get("trace", [])
    if not isinstance(trace, list) or not all(isinstance(x, dict) for x in trace):
        raise ValueError("invalid trace")
    artifacts = import_artifacts(payload.get("artifacts", {}))
    outputs = import_artifacts(payload.get("outputs", {}))
    if mask is None and not outputs:
        raise ValueError("sandbox must return an output")
    contours = None
    if payload.get("contours") is not None:
        contours = ContourArtifact(
            tuple(np.asarray(item, dtype=np.float32) for item in payload["contours"]),
            tuple(payload["contour_shape"]), metadata=payload.get("contour_metadata", {}))
        if any(not np.isfinite(item).all() for item in contours.contours):
            raise ValueError("non-finite contours")
    final_mask = None if mask is None else MaskArtifact(mask, metadata=payload.get("mask_metadata", {}))
    for name, value in outputs.items():
        if isinstance(value, MaskArtifact) and final_mask is not None and np.array_equal(value.data, final_mask.data):
            outputs[name] = final_mask
    return PipelineExecutionResult(
        pipeline={}, mask=final_mask,
        contours=contours, trace=tuple(trace), artifacts=artifacts, outputs=outputs)
