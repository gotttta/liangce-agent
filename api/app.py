"""create_app()：挂路由与统一异常处理（计划 §6 错误码表）。

生产模式托管 web/dist（静态文件 + SPA fallback）；dist 缺失时返回构建提示页。
"""
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from core.orchestration_runtime import TaskBusyError
from core.runtime_logging import logger, redact

from api import config
from api.files import ForbiddenPath
from api.routes import artifacts, tasks
from api.routes.tasks import InvalidRequest
from api.schemas import ErrorResponse, HealthOut

_ERROR_DETAIL_MAX_CHARS = 400


def sanitize_error_detail(detail) -> str:
    """错误详情统一处理：redact + 压平 + 截断。"""
    collapsed = " ".join(redact(str(detail or "")).split())
    if len(collapsed) > _ERROR_DETAIL_MAX_CHARS:
        return collapsed[:_ERROR_DETAIL_MAX_CHARS] + "…"
    return collapsed


def _error(code: str, message: str, status: int) -> JSONResponse:
    body = ErrorResponse(error={"code": code, "message": message})
    return JSONResponse(status_code=status, content=body.model_dump())


_BUILD_HINT_HTML = (
    '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
    "<title>liangce-agent</title></head><body>"
    "<p>前端尚未构建：请先运行 <code>cd web &amp;&amp; npm install &amp;&amp; npm run build</code>，"
    "然后重启 <code>python -m api</code>。</p></body></html>"
)


def _mount_spa(app: FastAPI, web_dist: Path) -> None:
    """托管 web/dist：命中 dist 内的文件直接返回，其余路径回退 index.html（SPA）。

    未匹配任何 /api 路由的请求返回统一格式的 JSON 404，不能回退成页面。
    必须在所有 API 路由注册之后再挂，否则 catch-all 会抢在它们前面命中。
    """
    index = web_dist / "index.html"
    dist_root = web_dist.resolve()

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path == "api" or full_path.startswith("api/"):
            return _error("not_found", "接口不存在。", 404)
        if not index.is_file():
            return HTMLResponse(_BUILD_HINT_HTML)
        if full_path:
            candidate = (web_dist / full_path).resolve()
            if candidate.is_file() and candidate.is_relative_to(dist_root):
                return FileResponse(candidate)
        return FileResponse(index)


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

    _mount_spa(app, app.state.paths.web_dist)
    return app
