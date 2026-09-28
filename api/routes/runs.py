"""运行接口（计划 §6.2）：启动、决策、取消、SSE 订阅、快照。"""
from typing import Literal

from fastapi import APIRouter, Header, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.routes.tasks import InvalidRequest

router = APIRouter(prefix="/tasks/{task_id}/runs", tags=["runs"])

REVIEW_ACTIONS = {"accept", "continue", "exit"}


class RunStartRequest(BaseModel):
    message: str
    target_type: Literal["defect", "array"] = "defect"


class ReviewSubmitRequest(BaseModel):
    action: Literal["accept", "continue", "exit"]
    feedback: str | None = None


def _runs(request: Request):
    return request.app.state.runs


@router.post("")
def start_run(request: Request, task_id: str, payload: RunStartRequest):
    if not payload.message.strip():
        raise InvalidRequest("任务描述不能为空。")
    run_id = _runs(request).start(task_id, payload.message, payload.target_type)
    return {"run_id": run_id}


@router.post("/{run_id}/review")
def submit_review(request: Request, task_id: str, run_id: str,
                  payload: ReviewSubmitRequest):
    if payload.action not in REVIEW_ACTIONS:
        raise InvalidRequest(f"未知的人类决策：{payload.action}")
    _runs(request).resume(task_id, run_id, payload.action, payload.feedback or "")
    return {"ok": True}


@router.post("/{run_id}/cancel")
def cancel_run(request: Request, task_id: str, run_id: str):
    _runs(request).cancel(task_id, run_id)
    return {"ok": True}


@router.get("/{run_id}/events")
def stream_events(request: Request, task_id: str, run_id: str, after: int = 0,
                  last_event_id: str | None = Header(default=None, alias="Last-Event-ID")):
    # Last-Event-ID 优先于查询参数（EventSource 自动重连携带）
    after_seq = int(last_event_id) if last_event_id else after
    generator = _runs(request).subscribe(task_id, run_id, after_seq)
    return StreamingResponse(generator, media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@router.get("/{run_id}")
def run_snapshot(request: Request, task_id: str, run_id: str):
    return _runs(request).snapshot(task_id, run_id)
