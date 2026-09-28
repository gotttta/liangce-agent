"""create_app()：挂路由与统一异常处理（计划 §6 错误码表）。

生产模式的静态前端托管（web/dist + SPA fallback）在阶段 7 接入。
"""
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from core.orchestration_runtime import TaskBusyError
from core.runtime_logging import logger, redact

from api import config
from api.files import ForbiddenPath
from api.routes import artifacts, tasks
from api.routes.tasks import InvalidRequest
from api.schemas import ErrorResponse, HealthOut

_ERROR_DETAIL_MAX_CHARS = 400


def sanitize_error_detail(detail) -> str:
    """与 ui/annotation_app.py 的 _sanitize_error_detail 同一规则：redact + 压平 + 截断。"""
    collapsed = " ".join(redact(str(detail or "")).split())
    if len(collapsed) > _ERROR_DETAIL_MAX_CHARS:
        return collapsed[:_ERROR_DETAIL_MAX_CHARS] + "…"
    return collapsed


def _error(code: str, message: str, status: int) -> JSONResponse:
    body = ErrorResponse(error={"code": code, "message": message})
    return JSONResponse(status_code=status, content=body.model_dump())


def create_app(root: Path | None = None, run_fn=None, resume_fn=None,
               provider_factory=None) -> FastAPI:
    base = Path(root).resolve() if root is not None else config.ROOT
    app = FastAPI(title="liangce-agent", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.paths = SimpleNamespace(
        root=base,
        task_root=base / "workspace" / "tasks",
        output_root=base / "outputs",
        web_dist=base / "web" / "dist",
    )
    from core.task_store import TaskStore

    app.state.tasks = TaskStore(app.state.paths.task_root)

    from api.files import file_roots
    from api.runs import RunManager
    from api.routes import runs as runs_route

    app.state.runs = RunManager(task_store=app.state.tasks,
                                output_root=app.state.paths.output_root,
                                roots=file_roots(base),
                                run_fn=run_fn, resume_fn=resume_fn,
                                provider_factory=provider_factory)
    app.include_router(tasks.router, prefix="/api")
    app.include_router(runs_route.router, prefix="/api")
    app.include_router(artifacts.router, prefix="/api")

    @app.get("/api/health", response_model=HealthOut)
    def health() -> HealthOut:
        # 只读环境推断模型名；不在请求路径里加载 .env（load_env_file 会改写进程环境）。
        import os

        from providers.vision import DEFAULT_ALIYUN_VISION_MODEL

        api_key = os.getenv("ALIYUN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            return HealthOut(ok=True, model=None)
        return HealthOut(ok=True,
                         model=os.getenv("ALIYUN_VISION_MODEL") or DEFAULT_ALIYUN_VISION_MODEL)

    @app.exception_handler(TaskBusyError)
    async def task_busy_handler(request: Request, exc: TaskBusyError):
        return _error("task_busy", "任务仍在运行，请等待本次运行结束后再操作。", 409)

    @app.exception_handler(FileNotFoundError)
    async def not_found_handler(request: Request, exc: FileNotFoundError):
        return _error("task_not_found", sanitize_error_detail(exc), 404)

    @app.exception_handler(ForbiddenPath)
    async def forbidden_path_handler(request: Request, exc: ForbiddenPath):
        return _error("forbidden_path", "拒绝访问该路径。", 403)

    @app.exception_handler(InvalidRequest)
    async def invalid_request_handler(request: Request, exc: InvalidRequest):
        return _error("invalid_request", sanitize_error_detail(exc), 422)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        return _error("invalid_request", sanitize_error_detail(exc), 422)

    @app.exception_handler(ValueError)
    async def bad_request_handler(request: Request, exc: ValueError):
        return _error("bad_request", sanitize_error_detail(exc), 400)

    @app.exception_handler(Exception)
    async def internal_error_handler(request: Request, exc: Exception):
        logger.error("API internal error: %s", request.url.path, exc_info=exc)
        return _error("internal_error", sanitize_error_detail(exc), 500)

    return app
