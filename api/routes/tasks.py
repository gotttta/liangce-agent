"""任务 CRUD 与样本图上传（计划 §6.1）。"""
from io import BytesIO
from pathlib import Path
import json
import tempfile

from fastapi import APIRouter, File, Request, UploadFile

from api.files import FileRoots, file_roots, file_url
from api.schemas import SamplesOut, TaskCreateRequest, TaskPatchRequest, status_label

router = APIRouter(tags=["tasks"])

SAMPLE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MAX_SAMPLE_BYTES = 50 * 1024 * 1024


class InvalidRequest(Exception):
    """请求参数校验失败，映射为 422 invalid_request（计划 §6 错误码表）。"""


def _store(request: Request):
    return request.app.state.tasks


def _roots(request: Request) -> FileRoots:
    return file_roots(request.app.state.paths.root)


def _is_running(request: Request, task_id: str) -> bool:
    runs = getattr(request.app.state, "runs", None)
    return bool(runs and runs.is_running(task_id))


def _sample_out(sample: dict, roots: FileRoots) -> dict:
    return {
        "name": Path(sample.get("path", "")).name,
        "url": file_url(sample.get("path", ""), roots),
    }


def _task_summary(request: Request, task: dict) -> dict:
    return {
        "id": task["id"],
        "title": task.get("title", ""),
        "status": str(task.get("status", "")),
        "status_label": status_label(task.get("status")),
        "created_at": task.get("created_at", ""),
        "updated_at": task.get("updated_at", ""),
        "running": _is_running(request, task["id"]),
        "sample_count": len(task.get("samples") or []),
    }


def _run_snapshots(request: Request, task_id: str) -> list[dict]:
    """按开始时间升序读 runs/*/latest.json；坏文件跳过，不拖垮任务详情。"""
    runs_dir = _store(request).task_dir(task_id) / "runs"
    snapshots = []
    if runs_dir.is_dir():
        for path in runs_dir.glob("*/latest.json"):
            try:
                snapshots.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
    snapshots.sort(key=lambda item: str(item.get("run_started_at") or ""))
    return snapshots


def _pending_review(snapshot: dict | None, roots: FileRoots) -> dict | None:
    if not snapshot or snapshot.get("run_status") != "awaiting_feedback":
        return None
    if snapshot.get("pending_reference_image_id"):
        stage = "reference"
    elif snapshot.get("stop_reason"):
        stage = "final"
    else:
        stage = "retry"
    conversation = snapshot.get("conversation") or []
    message = str(conversation[0].get("content", "")) if conversation else ""
    overlay = snapshot.get("pending_reference_overlay_path") or ""
    return {
        "stage": stage,
        "message": message,
        "overlay_url": file_url(overlay, roots) if overlay else None,
        "best_score": float(snapshot.get("best_score") or 0.0),
        "stop_reason": snapshot.get("stop_reason") or None,
    }


def _best_result(snapshot: dict | None) -> dict | None:
    if not snapshot:
        return None
    spec = snapshot.get("pipeline") or {}
    score = float(snapshot.get("best_score") or 0.0)
    if not spec and score <= 0:
        return None
    return {
        "score": score,
        "pipeline": spec.get("pipeline") or [],
        "notes": spec.get("notes") or "",
    }


def _task_detail(request: Request, task: dict) -> dict:
    roots = _roots(request)
    snapshots = _run_snapshots(request, task["id"])
    latest_id = task.get("latest_run_id")
    latest = next((item for item in snapshots if item.get("run_id") == latest_id), None)
    return {
        **_task_summary(request, task),
        "samples": [_sample_out(sample, roots) for sample in task.get("samples") or []],
        "latest_run_id": latest_id,
        "runs": [{
            "run_id": item.get("run_id", ""),
            "status": str(item.get("run_status", "")),
            "status_label": status_label(item.get("run_status")),
            "started_at": item.get("run_started_at"),
            "stop_reason": item.get("stop_reason"),
        } for item in snapshots],
        "pending_review": _pending_review(latest, roots),
        "best": _best_result(latest),
    }


def _verify_image(data: bytes, name: str) -> None:
    from PIL import Image

    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
    except Exception as exc:
        raise InvalidRequest(f"文件不是有效图片：{name}") from exc


@router.get("/tasks")
def list_tasks(request: Request, limit: int = 50):
    return [_task_summary(request, task) for task in _store(request).list_tasks(limit)]


@router.post("/tasks")
def create_task(request: Request, payload: TaskCreateRequest):
    task = _store(request).create_task(payload.title or "新缺陷检测任务")
    return _task_detail(request, task)


@router.get("/tasks/{task_id}")
def get_task(request: Request, task_id: str):
    task = _store(request).load_task(task_id)
    return _task_detail(request, task)


@router.patch("/tasks/{task_id}")
def patch_task(request: Request, task_id: str, payload: TaskPatchRequest):
    _store(request).load_task(task_id)
    task = _store(request).set_title(task_id, payload.title)
    return _task_detail(request, task)


@router.delete("/tasks/{task_id}")
def delete_task(request: Request, task_id: str):
    from core.orchestration_runtime import TaskBusyError

    _store(request).load_task(task_id)
    if _is_running(request, task_id):
        raise TaskBusyError(f"task {task_id} already has an active runner")
    _store(request).delete_task(task_id)
    return {"ok": True}


@router.post("/tasks/{task_id}/samples")
async def upload_samples(request: Request, task_id: str,
                         files: list[UploadFile] = File(...)):
    store = _store(request)
    store.load_task(task_id)
    if not files:
        raise InvalidRequest("请至少上传一张样本图。")
    for upload_file in files:
        name = Path(upload_file.filename or "").name
        if Path(name).suffix.lower() not in SAMPLE_SUFFIXES:
            raise InvalidRequest(f"只接受 png/jpg/jpeg/bmp/tif/tiff 样本图：{name}")
        data = await upload_file.read()
        if len(data) > MAX_SAMPLE_BYTES:
            raise InvalidRequest(f"样本图超过 50 MB 上限：{name}")
        _verify_image(data, name)
        # add_sample 按来源文件名落盘，临时文件必须保留原始名。
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / name
            temp_path.write_bytes(data)
            store.add_sample(task_id, temp_path)
    task = store.load_task(task_id)
    return SamplesOut(samples=[_sample_out(sample, _roots(request))
                               for sample in task.get("samples") or []])


@router.delete("/tasks/{task_id}/samples/{sample_name}")
def remove_sample(request: Request, task_id: str, sample_name: str):
    store = _store(request)
    task = store.load_task(task_id)
    target = next((sample for sample in task.get("samples") or []
                   if Path(sample.get("path", "")).name == sample_name), None)
    if target is None:
        raise FileNotFoundError(sample_name)
    store.remove_sample(task_id, target["path"])
    task = store.load_task(task_id)
    return SamplesOut(samples=[_sample_out(sample, _roots(request))
                               for sample in task.get("samples") or []])
