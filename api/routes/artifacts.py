"""产物与文件接口（计划 §6.3）：/api/files、artifacts 汇总、algorithm.json 下载。"""
from dataclasses import asdict
from pathlib import Path
import json
import re

from fastapi import APIRouter, Request, Response

from api.files import ForbiddenPath, file_roots, file_url, read_response, resolve_allowed
from api.schemas import ArtifactsOut, IterationRecordOut, ReferenceInfoOut

router = APIRouter(tags=["artifacts"])

# 与 TaskStore.save_run_state 的 run_id 白名单一致，防止路径拼接逃逸。
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


def _store(request: Request):
    return request.app.state.tasks


def _run_dir(request: Request, task_id: str, run_id: str) -> Path:
    _store(request).load_task(task_id)
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ForbiddenPath(run_id)
    return request.app.state.paths.output_root / run_id


@router.get("/files")
def read_file(request: Request, path: str):
    roots = file_roots(request.app.state.paths.root)
    return read_response(resolve_allowed(path, roots), roots)


@router.get("/tasks/{task_id}/runs/{run_id}/artifacts")
def get_artifacts(request: Request, task_id: str, run_id: str) -> ArtifactsOut:
    roots = file_roots(request.app.state.paths.root)
    run_dir = _run_dir(request, task_id, run_id)

    def read_json(name: str) -> dict | None:
        path = run_dir / name
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            return None

    from core.iteration_tracker import IterationTracker

    iterations = [IterationRecordOut(iteration=record.iteration,
                                     run_score=asdict(record.run_score),
                                     algorithm_spec=record.algorithm_spec,
                                     timestamp=record.timestamp)
                  for record in IterationTracker(run_dir).history]

    references = []
    refs_dir = _store(request).task_dir(task_id) / "reference_masks"
    if refs_dir.is_dir():
        for meta_path in sorted(refs_dir.glob("*/meta.json")):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            overlay = run_dir / "reference_candidate" / f"{meta.get('image_id')}.png"
            references.append(ReferenceInfoOut(
                image_id=str(meta.get("image_id", "")),
                image_url=file_url(meta.get("image_path", ""), roots),
                overlay_url=file_url(overlay, roots) if overlay.is_file() else None,
                sam_iou_score=float(meta.get("sam_iou_score") or 0.0),
                skip_scoring=bool(meta.get("skip_scoring")),
                skip_reason=str(meta.get("skip_reason") or ""),
                confirmed_at=str(meta.get("confirmed_at") or ""),
            ))

    return ArtifactsOut(algorithm=read_json("algorithm.json"),
                        score=read_json("score.json"),
                        iterations=iterations,
                        references=references)


@router.get("/tasks/{task_id}/runs/{run_id}/algorithm.json")
def download_algorithm(request: Request, task_id: str, run_id: str) -> Response:
    roots = file_roots(request.app.state.paths.root)
    run_dir = _run_dir(request, task_id, run_id)
    path = resolve_allowed(str(run_dir / "algorithm.json"), roots)
    return Response(path.read_bytes(), media_type="application/json",
                    headers={"Content-Disposition":
                             f'attachment; filename="algorithm_{run_id}.json"'})
