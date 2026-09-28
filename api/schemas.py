"""Pydantic 请求/响应模型，与 web/src/api/types.ts 一一对应（计划 §6、§8）。"""
from pydantic import BaseModel

# 任务状态在界面里的中文名（存储值保持英文）。
# 来源：ui/annotation_app.py 的 _TASK_STATUS_LABELS，阶段 7 删除旧 UI 后以此为准。
TASK_STATUS_LABELS = {
    "draft": "草稿",
    "in_progress": "进行中",
    "waiting_for_acceptance": "待确认",
    "waiting_for_feedback": "待反馈",
    "running": "进行中",
    "awaiting_feedback": "待反馈",
    "completed": "已完成",
    "stopped": "已停止",
    "failed": "失败",
    "cancelled": "已取消",
    "interrupted": "已中断",
    "accepted": "已验收",
    "exited": "已结束",
}


def status_label(status) -> str:
    return TASK_STATUS_LABELS.get(str(status or ""), str(status or ""))


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class TaskCreateRequest(BaseModel):
    title: str | None = None


class TaskPatchRequest(BaseModel):
    title: str


class SampleOut(BaseModel):
    name: str
    url: str


class RunSummaryOut(BaseModel):
    run_id: str
    status: str
    status_label: str
    started_at: str | None = None
    stop_reason: str | None = None


class ReviewRequestOut(BaseModel):
    stage: str                 # reference / final / retry
    message: str
    overlay_url: str | None = None
    best_score: float
    stop_reason: str | None = None


class BestResultOut(BaseModel):
    score: float
    pipeline: list
    notes: str


class TaskSummaryOut(BaseModel):
    id: str
    title: str
    status: str
    status_label: str
    created_at: str
    updated_at: str
    running: bool
    active_run_id: str | None = None   # RunManager 里的活动运行（latest_run_id 落盘前的窗口）
    sample_count: int


class TaskDetailOut(TaskSummaryOut):
    samples: list[SampleOut]
    latest_run_id: str | None = None
    runs: list[RunSummaryOut]
    pending_review: ReviewRequestOut | None = None
    best: BestResultOut | None = None


class SamplesOut(BaseModel):
    samples: list[SampleOut]


class ReferenceInfoOut(BaseModel):
    image_id: str
    image_url: str
    overlay_url: str | None = None
    sam_iou_score: float
    skip_scoring: bool
    skip_reason: str
    confirmed_at: str


class IterationRecordOut(BaseModel):
    iteration: int
    run_score: dict
    algorithm_spec: dict
    timestamp: str


class ArtifactsOut(BaseModel):
    algorithm: dict | None = None
    score: dict | None = None
    iterations: list[IterationRecordOut]
    references: list[ReferenceInfoOut]


class HealthOut(BaseModel):
    ok: bool
    model: str | None = None
