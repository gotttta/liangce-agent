# 前端迁移计划：Gradio → FastAPI + React

> 日期：2026-09-28
> 分支：从 `refactor/reference-scoring` 切出 `feat/react-frontend`
> 执行：GLM；审查：Claude Code（每个阶段结束提交一次，等审查通过再进入下一阶段）

---

## 0. 给执行者的总规则（必读）

1. **按阶段顺序做**，每个阶段结束时：测试全绿 → 单独 commit → 停下等审查。不要一次性把所有阶段做完再提交。
2. **技术选型已定死**（见第 3 节），不要替换成别的库，不要额外引入状态管理、UI 框架或 CSS 方案。确实需要新依赖时，在 commit message 里写明理由。
3. **不改 `core/` 的业务逻辑**。只允许做第 5 节列出的后端前置改动，其他 core 改动一律不做；发现 core 有 bug，记在 `docs/2026-09-28-react-frontend-issues.md`，不要顺手修。
4. **日志沿用现有机制**：后端只用 `core.runtime_logging` 的 `logger`、`bind_context`、`logged_operation`，不另建日志配置（见 `AGENTS.md`）。
5. **不要删除 `ui/`**。旧 Gradio 入口在第 7 阶段之前保持可启动，最后一个阶段才删。
6. **中文文案**：界面文字全部用中文，和现有 Gradio 界面措辞保持一致（状态名见 `ui/annotation_app.py` 的 `_TASK_STATUS_LABELS`）。
7. 每写一个后端模块都要写对应的 pytest；前端至少给事件归并逻辑（reducer）写单元测试。
8. 不确定的地方，**先按本文档最保守的理解实现，并在 issues 文档里记一条**，不要自行扩展需求。

---

## 1. 背景

### 1.1 现状

- 网页入口：`python -m ui.app` → `ui/annotation_app.py`（约 2600 行 Gradio Blocks）。
- 样式：`ui/styles/*.css` 约 1300 行，大部分在覆盖 Gradio 默认样式。
- 流式输出：`run_chat_agent_stream` 在线程里跑 `run_chat_agent`，通过 `queue.Queue` 收事件，每个事件都 `yield` 一整组 15 个组件值，整块重绘。
- **运行生命周期绑定在 HTTP 连接上**：关掉页面，生成器退出，`finally` 里会设置 `request_control.cancelled`，任务随之中断。
- **旧 UI 和新工作流已经对不上**：`annotation_app.py` 还是按 v2 写的（画布红/绿笔反馈、"识别方法 N"候选分组、量测表、`execute_candidate` 节点结果）。当前 `core/agent_workflow.py` 已换成 reference-scoring 工作流（prepare → gen_reference → human_gate → iterate → score → promote → finish），这些 UI 元素大多已经没有对应数据。

### 1.2 新工作流需要前端展示的内容

参考 `docs/2026-09-28-new-agent-design.md` 和 `core/agent_workflow.py`：

| 阶段 | 用户需要看到 / 做的事 |
|---|---|
| 上传 | 上传多张样本图（设计要求 5 张起步），填写任务描述，选择目标类型（`defect` / `array`） |
| gen_reference | 每张图生成 SAM 参考掩膜（每张 30–70 秒），过程中要有进度 |
| human_gate（stage=`reference`） | 看叠加图 `pending_reference_overlay_path`、SAM 分数（< 0.7 时提示"分割质量存疑"），选择：确认 / 文字修正 / 放弃 |
| iterate + score + promote | 自主迭代，可能几十轮。每轮显示：算子序列、`composite_mean`、每张图的 IoU / 误检 / 漏检，是否刷新最优 |
| human_gate（stage=`final`） | 看停止原因（`target_reached` / `no_improvement` / `max_iterations`）、最优分数和最优算法，选择：接受 / 继续迭代（可带文字）/ 放弃 |
| finish | 展示 `algorithm.json`、`score.json` 的内容，可下载 |

设计文档已确认**第一版不做图上标注界面**，用户反馈全部用文字。所以新前端**不需要**画布画笔，也不要移植 `feedback_editor` / `green_feedback_editor`。

### 1.3 目标效果

参照 Claude Code / Codex 网页版：

- 左侧：任务列表（新建、重命名、删除、按更新时间排序、状态徽标）。
- 中间：对话时间线。用户消息、助手消息、**可折叠的步骤卡片**（节点开始/完成、模型调用、工具调用），流式思考文本逐步出现，正在运行的步骤有动效和计时。
- 右侧：产物面板（可收起、可拖动宽度）。包括参考掩膜叠加图、迭代分数曲线、最优算法 JSON、每张图的分数表。
- 底部：输入框，支持拖拽/粘贴图片、Enter 发送、Shift+Enter 换行，运行中显示"停止"按钮。
- 人工确认点在时间线里显示为**操作卡片**（按钮 + 可选文字框），不是让用户手打 `action=continue`。
- **刷新页面或关闭再打开不中断运行**，重新打开后能接上实时事件流。

---

## 2. 目标与非目标

### 目标

- 新建 `api/`（FastAPI）和 `web/`（React），完整覆盖第 1.2 节的流程。
- 运行生命周期和 HTTP 连接解耦：后台线程跑，SSE 只负责订阅；支持断线重连、补发漏掉的事件。
- 生产模式下只起一个进程：`python -m api` 同时提供 API 和打包好的静态前端。
- 历史任务可以完整回放（时间线从持久化事件重建）。

### 非目标（不要做）

- 图上画笔、点选、框选等标注交互。
- 多用户、登录鉴权、远程部署（只监听 `127.0.0.1`）。
- 移植旧 v2 的量测表、候选方法分组、Handbook 示例图、ground truth 上传（新工作流不再使用这些输入）。
- 国际化、深色/浅色以外的主题系统（深色模式可以做，但放在第 6 阶段且是可选）。
- WebSocket。统一用 SSE + REST。

---

## 3. 技术选型（固定）

### 后端

| 用途 | 选择 | 说明 |
|---|---|---|
| Web 框架 | FastAPI（已随 gradio 安装，0.141） | 在 `requirements.txt` 显式加 `fastapi`、`uvicorn[standard]`、`python-multipart` |
| 流式推送 | SSE，用 `StreamingResponse(media_type="text/event-stream")` 手写 | 不引入 `sse-starlette` |
| 运行执行 | `threading.Thread` + `contextvars.copy_context()` | 和现有 `run_chat_agent_stream` 相同的做法，保证 `bind_context`、`request_control.control` 能传进工作线程 |
| 测试 | pytest + `fastapi.testclient.TestClient` | |

### 前端

| 用途 | 选择 |
|---|---|
| 构建 | Vite 6 + React 19 + TypeScript（strict） |
| 包管理 | npm（本机没有 pnpm；Node 24 已装） |
| 样式 | Tailwind CSS v4 + shadcn/ui（按需 `npx shadcn add`，组件源码进仓库） |
| 图标 | lucide-react |
| 服务端数据 | @tanstack/react-query（任务列表、任务详情、产物） |
| 客户端状态 | zustand（当前任务、面板开合、正在运行的事件流） |
| 路由 | react-router v7（只需要 `/` 和 `/tasks/:taskId`） |
| Markdown | react-markdown + remark-gfm + rehype-highlight |
| 图表 | recharts（只画一张迭代分数折线图） |
| 图片查看 | react-zoom-pan-pinch（叠加图缩放/拖动） |
| 可拖动分栏 | react-resizable-panels |
| 测试 | vitest + @testing-library/react |
| 代码规范 | eslint（Vite 模板自带）+ prettier |

**不要引入**：Redux、MobX、axios（用 fetch）、Konva/fabric、Next.js、任何组件库（MUI/AntD/Chakra）。

---

## 4. 目录结构

```
api/
  __init__.py
  __main__.py          # python -m api：configure_logging() 后启动 uvicorn
  app.py               # create_app()：挂路由、静态文件、异常处理
  config.py            # 路径常量（ROOT、TASK_ROOT、OUTPUT_ROOT、WEB_DIST）、端口
  schemas.py           # Pydantic 请求/响应模型
  events.py            # UI 事件协议：把 core 事件转换成前端事件（纯函数）
  runs.py              # RunManager：后台运行、事件缓冲、订阅、取消
  files.py             # 受限的文件读取（图片、JSON 产物）
  routes/
    __init__.py
    tasks.py           # 任务 CRUD、上传样本图
    runs.py            # 启动运行、提交人工决策、取消、SSE 订阅
    artifacts.py       # 产物读取
web/
  package.json
  vite.config.ts       # dev 代理 /api → http://127.0.0.1:8765
  index.html
  src/
    main.tsx
    App.tsx
    api/               # fetch 封装 + 类型
      client.ts
      types.ts         # 和 api/schemas.py、api/events.py 一一对应
      sse.ts           # EventSource 封装，带重连和 Last-Event-ID
    store/
      ui.ts            # zustand：面板状态
      timeline.ts      # 事件 → 时间线条目的 reducer（重点测试）
    components/
      layout/          # AppShell、Sidebar、TopBar
      timeline/        # Timeline、UserMessage、AssistantMessage、StepCard、ThinkingBlock、ReviewCard、ErrorCard
      composer/        # Composer、AttachmentChips
      artifacts/       # ArtifactPanel、ReferenceGallery、ScoreChart、ScoreTable、AlgorithmView
      ui/              # shadcn 生成的基础组件
    pages/
      TaskPage.tsx
      EmptyPage.tsx
    lib/
      format.ts        # 时长、时间、分数格式化，状态中文名
  tests/
tests/
  test_api_tasks.py
  test_api_runs.py
  test_api_events.py
  test_api_files.py
```

`.gitignore` 增加：`web/node_modules/`、`web/dist/`。

---

## 5. 后端前置改动（阶段 1 做，只允许改这些）

### 5.1 参考掩膜按任务隔离

**问题**：`WorkflowRuntime` 默认使用全局 `workspace/references`。任务 A 确认过的参考掩膜会被任务 B 的 `prepare` 读到（`list_scoreable()` 非空时直接跳到 `iterate`），不同任务的参考会串。

**改法**：

- `core/agent_graph.py` 的 `build_agent_graph` 增加参数 `references_root=None`，透传给 `build_workflow_graph(..., references_root=references_root)`。
- `run_agent_graph` 在拿到 `task_store` 和 `task_id` 后，用 `task_store.task_dir(task_id) / "reference_masks"` 作为 `references_root` 构图。**注意**：构图发生在确定 `task_id` 之前，需要把 `build_agent_graph(...)` 调用挪到确定 `task_id` 之后；恢复运行时（`restoring`）要从 `saved.task_id` 推出同一个目录。
- `resume_agent_graph` 同理：从 checkpoint 里的 `saved.task_id` 和 `memory_context["task_root"]` 推出 `references_root`，再重新构图。现在它先 `build_agent_graph()` 再读 state，需要调整顺序：先用不带 references_root 的图读 snapshot（读 state 不跑节点，没问题），再用正确的 root 构图后 `invoke`。
- `ReferenceStore.list_scoreable()` 只列 `task.image_paths` 里的图吗？**不改**，按任务隔离后自然只有本任务的图。
- 加测试：两个任务各自生成参考后，互相 `prepare` 看不到对方的参考（mock SAM，参考 `tests/test_agent_workflow.py` 里已有的 mock 方式）。

### 5.2 `run_agent_graph` 支持多张样本图

**问题**：`run_agent_graph(target_image_path, ...)` 只收一张图，`WorkflowState(image_paths=[str(target_image_path)])`，恢复时也校验 `saved.image_paths == [str(target_image_path)]`。设计要求 5 张起步。

**改法**（保持向后兼容，CLI 和旧测试不改）：

- 增加关键字参数 `target_image_paths: list[str] | None = None`。传了就用它，没传就用 `[target_image_path]`。
- 恢复校验改为和最终列表比较。
- `memory_service.prepare(task_id, description, target_image_path, ...)` 仍传第一张图（不改 memory 模块）。
- 加测试：传 3 张图，`WorkflowState.image_paths` 长度为 3，`gen_reference` 依次对 3 张图要求确认。

### 5.3 迭代分数事件

**问题**：`score` 节点只在 `node_complete` 的 metadata 里带 `composite_mean`，没有每张图的分数和当前算子序列，前端画不出分数表和曲线。

**改法**：在 `core/agent_workflow.py` 的 `score` 节点 `tracker.record(...)` 之后调用一次 `emit_event`：

```python
emit_event({
    "type": "iteration_scored",
    "iteration": state.iteration,
    "composite_mean": run_score.composite_mean,
    "best_score_before": state.best_score,
    "improved": run_score.composite_mean > state.best_score,
    "pipeline": state.current_spec.pipeline,
    "notes": state.current_spec.notes,
    "image_scores": [asdict(item) for item in run_score.image_scores],
    "timestamp": time.time(),
})
```

`gen_reference` 生成候选后也加一个事件：

```python
emit_event({
    "type": "reference_candidate",
    "image_id": image_id,
    "overlay_path": str(overlay_path),
    "sam_score": mean_score,
    "low_quality": mean_score < MIN_MEAN_IOU,
    "timestamp": time.time(),
})
```

这是 core 里**唯一允许**新增的业务相关改动。加测试断言这两个事件在对应节点被发出。

### 5.4 `resume_agent_graph` 返回值补 `interrupt` 字段

**问题**：`resume_agent_graph` 只把 `interrupted=bool(...)` 写进 `_public_state` 的 `run_status`，从不序列化 interrupt 内容；返回值里没有 `interrupt` 字段。而 `run_agent_graph` 有（保存 run state 后 `result["interrupt"] = _serialize_interrupts(interrupts)`）。第 8 节的 `review_requested` 依赖两个入口的返回值都带 `interrupt`——多图参考确认循环（确认第 1 张 → resume → 生成第 2 张 → 再次停在 human_gate）恰好全走 resume 路径，事件流里也没有请求内容（human_gate 只发 node_start）。

**改法**：`resume_agent_graph` 在 `store.save_run_state(task_id, result)` 之后（和 `run_agent_graph` 相同的顺序，避免 interrupt 进 latest.json 快照）：

```python
interrupts = _pending_interrupts(graph, config)
if interrupts:
    result["interrupt"] = _serialize_interrupts(interrupts)
```

同时给 `resume_agent_graph` 增加关键字参数 `provider=None` 并透传给 `build_agent_graph`。**已核实**（2026-09-28 阶段 1 测试暴露）：resume 重建执行图时如果不带 provider，继续迭代进入 `iterate` 节点会直接 `'NoneType' object has no attribute '_complete_action'` 崩溃；旧 Gradio 只走 accept/exit 所以从未触发。API 层（第 7 节 RunManager）调用 resume 时必须传 `provider=build_runtime_provider()`。

加测试：resume 后停在下一个 human_gate 时，返回值包含 `interrupt`，且 `value.stage` 正确（reference / final）。

### 5.5 验证

```bash
.venv/bin/python -m pytest -q tests/test_agent_workflow.py tests/test_reference_store.py tests/test_scoring.py tests/test_iteration_tracker.py tests/test_runtime_logging.py
```

---

## 6. 后端 API 设计

基础路径 `/api`，全部 JSON，错误统一返回：

```json
{ "error": { "code": "task_busy", "message": "任务仍在运行，请等待本次运行结束后再操作。" } }
```

错误码表（`api/app.py` 注册异常处理器）：

| 异常 | HTTP | code |
|---|---|---|
| `TaskBusyError` | 409 | `task_busy` |
| 任务不存在（`FileNotFoundError` from `load_task`） | 404 | `task_not_found` |
| 参数校验失败 | 422 | `invalid_request` |
| 路径越界 | 403 | `forbidden_path` |
| `ValueError`（业务校验） | 400 | `bad_request` |
| 其他 | 500 | `internal_error`（message 用 `_sanitize_error_detail` 同样的规则过滤：`redact` + 压平 + 截断 400 字）|

### 6.1 任务

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/tasks?limit=50` | `TaskStore.list_tasks` |
| POST | `/api/tasks` | `{title?}` → `create_task` |
| GET | `/api/tasks/{task_id}` | 任务详情（见下） |
| PATCH | `/api/tasks/{task_id}` | `{title}` → `set_title` |
| DELETE | `/api/tasks/{task_id}` | `delete_task`；运行中返回 409 |
| POST | `/api/tasks/{task_id}/samples` | `multipart/form-data`，字段 `files`（可多张）→ `add_sample`，返回更新后的 samples |
| DELETE | `/api/tasks/{task_id}/samples/{sample_name}` | `remove_sample` |

`TaskSummary`：

```ts
{
  id: string; title: string; status: string; status_label: string;
  created_at: string; updated_at: string;
  running: boolean;          // RunManager 里是否有活动运行
  sample_count: number;
}
```

`TaskDetail`：

```ts
TaskSummary & {
  samples: { name: string; url: string }[];   // url 指向 /api/files/...
  latest_run_id: string | null;
  runs: RunSummary[];                         // 按开始时间升序
  pending_review: ReviewRequest | null;       // 最近一次运行停在 human_gate 时非空
  best: BestResult | null;                    // 最近一次运行的最优结果
}
```

上传校验：只接受 `.png/.jpg/.jpeg/.bmp/.tif/.tiff`，单文件 ≤ 50 MB，用 `PIL.Image.open(...).verify()` 确认是图片。

### 6.2 运行

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/tasks/{task_id}/runs` | 启动新运行，body `{message, target_type: "defect"\|"array"}`，返回 `{run_id}` |
| POST | `/api/tasks/{task_id}/runs/{run_id}/review` | 提交人工决策，body `{action: "accept"\|"continue"\|"exit", feedback?}` |
| POST | `/api/tasks/{task_id}/runs/{run_id}/cancel` | 请求取消 |
| GET | `/api/tasks/{task_id}/runs/{run_id}/events` | SSE 订阅（支持 `Last-Event-ID` 头和 `?after=<seq>` 参数） |
| GET | `/api/tasks/{task_id}/runs/{run_id}` | 运行快照：状态、全部已持久化事件（用于回放历史） |

**启动运行**时，`message` 作为 `description`，`target_image_paths` 用任务当前全部 samples。没有样本图时返回 400"请先上传样本图"。

**`target_type` 目前没有入口能传进 `WorkflowState`**（`run_agent_graph` 不收这个参数）。第 5.2 节改 `run_agent_graph` 时一并加 `target_type="defect"` 关键字参数，写进 `WorkflowState(target_type=...)`。

**提交决策**：直接调用 `resume_agent_graph(thread_id, {"action": ..., "feedback": ...})`，**不要**经过旧的 `_handle_result_action`（那是 v2 的逻辑）。它同样是一次可能很长的运行（继续迭代可能跑几十轮），所以也交给 RunManager 在后台执行，事件发到**同一个 run_id** 的事件流上，seq 继续递增。

### 6.3 产物与文件

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/files?path=<相对 ROOT 的路径>` | 返回图片或 JSON 文件 |
| GET | `/api/tasks/{task_id}/runs/{run_id}/artifacts` | `{algorithm: {...}\|null, score: {...}\|null, iterations: IterationRecord[], references: ReferenceInfo[]}` |
| GET | `/api/tasks/{task_id}/runs/{run_id}/algorithm.json` | 下载 `algorithm.json`（`Content-Disposition: attachment`） |

`iterations` 读 `run_dir/iteration_history.jsonl`（用 `IterationTracker(run_dir).history`，不要自己解析）。
`references` 读本任务的 `reference_masks`（`ReferenceStore(...).load(image_id)` 的 meta），叠加图如果没有持久化，就读 `run_dir/reference_candidate/<image_id>.png`。

**文件安全（`api/files.py`，必须测试）**：

```python
ALLOWED_ROOTS = (ROOT / "workspace", ROOT / "outputs")

def resolve_allowed(raw: str) -> Path:
    path = (ROOT / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
    if not any(path.is_relative_to(root.resolve()) for root in ALLOWED_ROOTS):
        raise ForbiddenPath(raw)
    if not path.is_file():
        raise FileNotFoundError(raw)
    return path
```

- 只允许后缀 `.png .jpg .jpeg .bmp .tif .tiff .json`。
- `.tif/.tiff` 浏览器不能直接显示：转成 PNG 再返回（`PIL` 转换，结果放内存，不落盘）。
- 返回给前端的所有路径都转成 `/api/files?path=<相对 ROOT 的路径>` 形式的 URL（`api/files.py` 提供 `file_url(path) -> str`），前端**永远不拼接文件系统路径**。
- 测试：`../`、绝对路径 `/etc/passwd`、指向 ROOT 外的符号链接，都要返回 403。

### 6.4 其他

- `GET /api/health` → `{ok: true, model: <provider.model 或 null>}`，前端顶栏显示模型名。
- 生产模式：`web/dist` 存在时挂在 `/`，未匹配 `/api` 的路径都返回 `index.html`（SPA fallback）。
- 只监听 `127.0.0.1`，端口默认 `8765`，环境变量 `LIANGCE_API_PORT` 覆盖。
- 开发模式不需要 CORS（Vite 代理）；**不要**开放 `allow_origins=["*"]`。

---

## 7. 运行管理（`api/runs.py`，核心，重点审查）

### 7.1 职责

```python
class RunManager:
    def start(self, task_id, message, target_type) -> str        # 返回 run_id
    def resume(self, task_id, run_id, action, feedback) -> None
    def cancel(self, task_id, run_id) -> None
    def is_running(self, task_id) -> bool
    def subscribe(self, task_id, run_id, after_seq: int) -> Iterator[UiEvent]
    def snapshot(self, task_id, run_id) -> RunSnapshot
```

全局单例，挂在 `app.state.runs`，测试里可以注入 fake runner。

### 7.2 执行模型

- 每次 `start` / `resume` 开一个后台 daemon 线程，线程内：
  1. `token = control.set(request_control)`（复用 `core.request_control`，**不要**另写取消机制）。
  2. `bind_context(task_id=...)`。
  3. `register_event_listener(listener, run_id=run_id)`：**必须带 `run_id`**。`emit_event` 的过滤规则是：listener 登记了 run_id 就按 run_id 匹配，否则按线程 id 匹配。子线程里的模型调用可能不在同一线程，所以要按 run_id 过滤。
     - 但 `run_id` 由 `run_agent_graph` 内部生成（`graph_thread_id = thread_id or f"agent_{uuid4().hex}"`），所以 **RunManager 要先自己生成 `run_id = f"agent_{uuid4().hex}"`，再通过 `thread_id=run_id` 传进去**，保证 listener 登记的 id 和事件里的 id 一致。**已核实**（2026-09-28 代码评审）：`logged_operation` 只会 `context.setdefault("run_id", uuid4().hex)`——它自己生成随机 hex，和传入的 `thread_id` 无关；所以 RunManager **必须**在 worker 线程里、调用 `run_agent_graph` 之前先 `bind_context(run_id=run_id)`（`setdefault` 才会保留它），这步不是可选的。**这一点写测试验证**：跑一个 fake 工作流，确认事件确实进了 listener。
  4. 调用 `run_agent_graph(..., thread_id=run_id, target_image_paths=..., target_type=..., task_store=store, task_id=task_id, provider=build_runtime_provider(), output_root=ROOT / "outputs")` 或 `resume_agent_graph(run_id, response)`。
  5. `finally`：`unregister_event_listener`、`control.reset(token)`、写入终态事件、标记运行结束。
- 同一任务同时只能有一个活动运行；`run_agent_graph` 内部已经有 `task_lock`，RunManager 在内存里再做一次检查，冲突时直接返回 409，不要等到线程里才报错。
- **HTTP 连接断开不影响运行**。只有 `cancel` 接口会设置 `request_control.cancelled`。
- 进程退出时不需要优雅停止运行（daemon 线程）；重启后未完成的运行在快照里显示为"已中断"（依据：没有终态事件、RunManager 里也没有这个运行）。

### 7.3 事件缓冲与持久化

- 每个事件分配单调递增的 `seq`（从 1 开始，每个 run 独立计数，resume 后接着计）。
- 内存：每个 run 一个 `list[UiEvent]` 加 `threading.Condition`；订阅者按 `after_seq` 取，没有新事件时 `wait(timeout=15)`，超时发 SSE 注释行 `: ping` 保活。
- 持久化：每个事件追加写入 `workspace/tasks/<task_id>/runs/<run_id>/stream.jsonl`（一行一个 JSON）。**不要**写进 `events.jsonl`（那是 TaskStore 的任务级事件）。
- 运行结束后内存缓冲保留 10 分钟再释放；之后 `subscribe` / `snapshot` 从 `stream.jsonl` 读。
- `thinking_delta` 和 `llm_chunk` 在内存里**合并**：同一 context 连续的增量在 150 ms 内合成一条再分配 seq，避免一秒几百条。持久化时只保存合并后的事件。
- 文本字段写盘前过 `core.runtime_logging.redact`。

### 7.4 SSE 格式

```
id: 42
event: ui
data: {"seq":42,"type":"step_update",...}

```

- `subscribe` 先补发 `seq > after_seq` 的历史事件，再推实时事件；运行结束后发一条 `event: end`，然后关闭连接。
- 前端用原生 `EventSource`，自动带 `Last-Event-ID` 重连。后端两种来源都要支持（请求头优先）。

---

## 8. UI 事件协议（`api/events.py` ↔ `web/src/api/types.ts`）

**原则**：后端负责把 core 的原始事件翻译成稳定的 UI 事件，前端只认 UI 事件。`api/events.py` 是纯函数，便于测试：

```python
class EventTranslator:
    def __init__(self, run_id: str): ...
    def translate(self, core_event: dict) -> list[dict]   # 可能 0 条、1 条或多条
```

所有 UI 事件公共字段：`seq`、`run_id`、`ts`（秒，float）、`type`。

| UI 事件 `type` | 来源 core 事件 | 字段 |
|---|---|---|
| `run_started` | RunManager 启动时 | `message`、`target_type`、`image_count`、`resumed: bool`、`action?`、`feedback?` |
| `step_started` | `node_start` | `step_id`（`node:<node>:<n>`，同名节点每次出现 n+1）、`node`、`label`（用 `NODE_LABELS`） |
| `step_finished` | `node_complete` | `step_id`、`duration`、`metadata` |
| `thinking` | `thinking`、合并后的 `thinking_delta` | `step_id?`（归属当前正在运行的 step）、`text`、`delta: bool` |
| `model_call_started` | `llm_request` | `call_id`、`model`、`message_count`、`has_images` |
| `model_call_finished` | `llm_response` | `call_id`（和最近一个未完成的 started 配对）、`usage`、`context_window`、`duration` |
| `model_output` | 合并后的 `llm_chunk` | `call_id`、`text`（增量） |
| `tool_started` / `tool_finished` | `tool_call` / `tool_result` | `tool_id`、`tool`、`args` / `result`、`success`、`duration`（按 FIFO 配对，**不要**像旧代码那样按工具名配对，同名并发调用会配错） |
| `reference_candidate` | 5.3 新增事件 | `image_id`、`overlay_url`、`sam_score`、`low_quality` |
| `iteration_scored` | 5.3 新增事件 | `iteration`、`composite_mean`、`improved`、`pipeline`、`notes`、`image_scores` |
| `review_requested` | 运行返回的 `interrupt[0].value` | `stage`（`reference`/`final`/`retry`）、`message`、`overlay_url?`、`best_score`、`stop_reason?` |
| `error` | `error` 事件或线程异常 | `message`（已过滤）、`node?` |
| `run_finished` | 线程结束 | `status`：`completed` / `awaiting_review` / `failed` / `cancelled`；`best_score?`、`duration` |

注意：

- `review_requested` **不是**从事件监听里来的，是 `run_agent_graph` / `resume_agent_graph` 返回值里的 `interrupt` 字段。RunManager 在线程末尾检查返回值，有 interrupt 就发 `review_requested`，然后发 `run_finished(status="awaiting_review")`。
- `overlay_path` 这类文件路径在 translator 里统一用 `file_url()` 转成 URL。
- `RequestCancelled` 继承自 `BaseException`，线程里要 `except (Exception, RequestCancelled)`，映射到 `status="cancelled"`。
- 未知的 core 事件类型：记 DEBUG 日志，丢弃，不要透传。

测试（`tests/test_api_events.py`）至少覆盖：step 配对、同名节点重复出现、thinking 归属到当前 step、model call 配对、工具 FIFO 配对、未知事件丢弃、路径转 URL、错误文本过滤密钥。

---

## 9. 前端设计

### 9.1 页面布局

```
┌──────────────┬────────────────────────────────────┬─────────────────────┐
│ Sidebar      │ TopBar：任务标题 · 状态 · 模型 · 上下文用量 │                     │
│  + 新建任务   ├────────────────────────────────────┤  ArtifactPanel      │
│  任务列表     │                                    │  [参考] [迭代] [算法] │
│  (状态点)     │  Timeline（居中，max-w-3xl）         │                     │
│              │                                    │  可收起 / 可拖宽      │
│              │                                    │                     │
│              ├────────────────────────────────────┤                     │
│  本地工作区   │  Composer                          │                     │
└──────────────┴────────────────────────────────────┴─────────────────────┘
```

- 用 `react-resizable-panels` 做三栏，右栏默认 380px，可收起，宽度存 `localStorage`。
- 窄屏（< 1024px）：侧栏变抽屉，产物面板变为时间线上方的切换按钮打开的抽屉。
- 视觉风格：参照 Claude 网页版，中性灰底，衬线标题可不做；圆角 `rounded-xl`，边框 `border-border`，不要大面积彩色。状态色只用于小圆点和徽标。

### 9.2 时间线条目

由 `store/timeline.ts` 的 reducer 把 UI 事件归并成条目：

```ts
type TimelineItem =
  | { kind: "user"; id: string; text: string; images?: string[] }
  | { kind: "assistant"; id: string; markdown: string }
  | { kind: "step"; id: string; node: string; label: string; status: "running"|"done"|"failed";
      startedAt: number; duration?: number; thinking: string; children: StepChild[]; metadata?: object }
  | { kind: "reference"; id: string; imageId: string; overlayUrl: string; samScore: number; lowQuality: boolean }
  | { kind: "iteration"; id: string; iteration: number; score: number; improved: boolean; pipeline: object[]; notes: string }
  | { kind: "review"; id: string; stage: "reference"|"final"|"retry"; message: string; overlayUrl?: string;
      bestScore: number; stopReason?: string; resolved?: { action: string; feedback?: string } }
  | { kind: "error"; id: string; message: string }
  | { kind: "run_end"; id: string; status: string; duration: number };

type StepChild =
  | { kind: "model_call"; id: string; model: string; status: string; duration?: number; output: string; usage?: object }
  | { kind: "tool"; id: string; tool: string; args: unknown; result?: unknown; success?: boolean; duration?: number };
```

归并规则：

- `run_started` → 一条 `user` 条目（`resumed` 时显示为"确认参考掩膜" / "继续迭代：<feedback>" / "接受结果"等，文案见 9.5）。
- `step_started` 开一个 step，`step_finished` 关闭；`thinking`、`model_*`、`tool_*` 挂到当前运行中的 step 下；没有运行中的 step 时挂到一个隐式的"准备"step。
- 连续的 `iteration` 条目默认**折叠成一组**："迭代 1–12 · 最优 0.734"，展开才看到每一轮。几十轮平铺会刷屏。
- `review_requested` → `review` 条目；之后该 run 收到 `run_started(resumed=true)` 时，把前一个 review 标为 `resolved`，按钮变灰并显示用户的选择。
- reducer 必须是纯函数，**对同一事件序列重放结果完全一致**，因为历史回放和实时流用的是同一个 reducer。seq 小于等于已处理 seq 的事件直接忽略（重连补发时会重复）。

组件：

| 组件 | 要点 |
|---|---|
| `StepCard` | 标题行：状态图标（运行中转圈）、label、耗时（运行中每秒刷新）。默认折叠；运行中的 step 自动展开，完成后自动折叠（用户手动展开过的不自动折叠）。展开后显示思考文本（等宽小字、灰色、最多 12 行可滚动）和子项 |
| `ThinkingBlock` | 流式文本追加时不要整体重渲染导致闪烁；用 `whitespace-pre-wrap` |
| `ModelCallRow` | 模型名 · token 用量 · 耗时；输出文本折叠显示 |
| `ReferenceCard` | 叠加图缩略图（点击打开大图查看器，支持缩放拖动）、image_id、SAM 分数；`lowQuality` 时显示黄色提示"分割质量存疑，确认后将不参与打分" |
| `IterationGroup` / `IterationRow` | 轮次、分数、相对上一最优的变化（↑ 绿 / — 灰）、算子序列一行摘要（`normalize → clahe → adaptive_threshold`），展开看完整 JSON |
| `ReviewCard` | 见 9.4 |
| `ErrorCard` | 红色边框，显示错误信息和"重试"按钮（重试 = 用同一 message 重新 `POST /runs`） |

### 9.3 Composer

- 多行 `textarea` 自动增高（最多 8 行）；Enter 发送，Shift+Enter 换行，输入法组字中（`isComposing`）不发送。
- 左下角"+"按钮、拖拽到时间线区域、粘贴剪贴板图片都可以添加样本图；添加后**立即上传**到 `/samples`，显示为可删除的缩略图 chip。
- 首次运行前在 Composer 上方显示目标类型切换（缺陷 / 阵列），默认缺陷。
- 运行中：发送按钮变为"停止"（方形图标），点击调用 cancel；输入框可以继续打字但不能发送。
- 停在 human_gate 时：Composer 的 placeholder 变为"输入修正意见，或使用上方按钮确认"；此时直接发送文字等价于 `review(action="continue", feedback=文字)`。
- 没有样本图时发送按钮禁用，并提示"请先添加样本图"。

### 9.4 人工确认卡片（ReviewCard）

| stage | 显示 | 按钮 |
|---|---|---|
| `reference` | 叠加图（大图，可缩放）、SAM 分数、低质量提示、`message` 文本 | **确认** → `continue` 不带 feedback；**需要修正** → 展开文字框，提交 `continue` + feedback；**放弃任务** → `exit`（二次确认） |
| `final` | 停止原因中文说明、最优分数、最优算法摘要、"在右侧查看详情"链接 | **接受结果** → `accept`；**继续迭代** → 展开可选文字框，提交 `continue`；**放弃** → `exit` |
| `retry` | `message` 文本 | **重试** → `continue`；**放弃** → `exit` |

停止原因文案：`target_reached` 达到目标分数；`no_improvement` 连续多轮没有明显提升；`max_iterations` 达到迭代上限；`user_exited` 用户已结束。

提交后按钮立即禁用，防止重复提交；后端返回 409 时显示 toast"任务仍在运行"。

### 9.5 用户侧文案（`run_started.resumed=true` 时的 user 条目）

| 场景 | 文案 |
|---|---|
| reference + continue 无 feedback | 确认参考掩膜 |
| reference + continue 有 feedback | 修正参考掩膜：<feedback> |
| final + accept | 接受当前最优结果 |
| final + continue | 继续迭代（有 feedback 时追加"：<feedback>"） |
| 任意 + exit | 结束任务 |

### 9.6 产物面板（ArtifactPanel）

三个标签页，数据来自 `GET /artifacts`，在收到 `iteration_scored`、`reference_candidate`、`run_finished` 事件时让 react-query 失效重取（节流 2 秒）。

1. **参考**：所有样本图网格；每张显示原图 / 叠加图切换、状态（未生成 / 待确认 / 已确认 / 已跳过打分）、SAM 分数。
2. **迭代**：recharts 折线图，x = 轮次，y = composite_mean，另画一条"历史最优"阶梯线；下面是最优一轮的每图分数表（image_id、IoU、误检、漏检、参考目标数、composite）。
3. **算法**：最优算子序列，按步骤显示成卡片（op 名 + 参数表），可切换成原始 JSON（带复制按钮）；"下载 algorithm.json"按钮。

### 9.7 TopBar

任务标题（点击可编辑，失焦保存）、状态徽标、模型名（`/api/health`）、上下文用量（取最近一次 `model_call_finished` 的 `usage.total_tokens / context_window`，显示为百分比小圆环，逻辑参考 `ui/utils/formatters.py` 的 `format_context_usage`）。

### 9.8 数据流

```
TaskPage 挂载
  ├─ useQuery(["task", id]) → TaskDetail
  ├─ 对每个历史 run：GET /runs/{run_id} → events → reducer 重放 → 时间线
  └─ 如果最新 run 在运行（task.running）：打开 SSE(after = 已有最大 seq)
发送消息 → POST /runs → 拿到 run_id → 打开 SSE(after=0)
提交决策 → POST /review → 继续使用/重开同一 run 的 SSE
```

- 一个任务的所有 run 按顺序拼成一条时间线。
- SSE 封装（`api/sse.ts`）：`onerror` 时由 EventSource 自动重连；收到 `end` 事件主动 `close()`；页面切换任务时关闭旧连接。
- 切换任务、刷新页面后状态必须能完全从后端恢复，**不要**把时间线存 localStorage。

---

## 10. 分阶段任务

每个阶段的"验收"全部满足后再提交。

### 阶段 1：后端前置改动

- 完成第 5 节 5.1、5.2（含 `target_type` 参数）、5.3、5.4。
- 验收：第 5.5 节的测试命令全绿；新增的隔离、多图、事件、resume interrupt 四类测试存在且通过；`git diff core/` 只涉及 `agent_graph.py`、`agent_workflow.py`。
- commit：`feat(core): isolate reference masks per task and support multi-image runs`

### 阶段 2：API 骨架 + 任务与文件接口

- `api/` 目录、`python -m api` 可启动、`/api/health`、第 6.1、6.3 节接口、错误处理、`api/files.py`。
- 测试：`tests/test_api_tasks.py`、`tests/test_api_files.py`（用 `tmp_path` 构造独立的 TASK_ROOT，**不要**读写真实 `workspace/`；`api/config.py` 的路径要能通过依赖注入或 `create_app(root=...)` 覆盖）。
- 验收：`curl` 能创建任务、上传两张图、列出任务、读到图片；越界路径返回 403。
- commit：`feat(api): add FastAPI task and file endpoints`

### 阶段 3：RunManager + 事件协议 + SSE

- `api/events.py`、`api/runs.py`、第 6.2 节接口。
- 测试：
  - `tests/test_api_events.py`：见第 8 节末尾清单。
  - `tests/test_api_runs.py`：用 fake 工作流函数替换 `run_agent_graph` / `resume_agent_graph`（通过 RunManager 构造参数注入），它发出一串 `emit_*` 事件后返回带 `interrupt` 的结果。验证：
    1. SSE 按 seq 顺序收到全部事件，最后是 `review_requested` + `run_finished(awaiting_review)` + `end`；
    2. 用 `after=3` 重连只收到 seq > 3 的事件；
    3. 运行中同一任务再 `POST /runs` 返回 409；
    4. `cancel` 后收到 `run_finished(cancelled)`；
    5. 客户端中途断开 SSE，运行仍然完成，`stream.jsonl` 完整；
    6. `review` 后事件 seq 接着递增；
    7. 线程里抛异常 → `error` + `run_finished(failed)`，错误文本不含测试里注入的假密钥。
- 验收：以上测试通过；手动用真实模型跑一次（`curl -N` 看 SSE），把 `workspace/logs/agent.log` 中该 run_id 的片段贴在 commit message 或 PR 描述里。
- commit：`feat(api): add background run manager with SSE event stream`

### 阶段 4：前端骨架

- `npm create vite@latest web -- --template react-ts`，按第 3 节装依赖，配置 Tailwind v4、shadcn、路径别名 `@/`、Vite 代理。
- AppShell 三栏布局、Sidebar（任务列表、新建、重命名、删除）、TopBar、EmptyPage、路由。
- `api/client.ts` + `api/types.ts`（类型和后端 schema 一一对应）。
- 验收：`npm run build`、`npm run lint`、`npx tsc --noEmit` 无错误；`python -m api` + `npm run dev` 能增删改任务。
- commit：`feat(web): scaffold React app shell with task sidebar`

### 阶段 5：时间线 + Composer + 实时流

- `store/timeline.ts` reducer 和第 9.2 节全部组件、Composer、SSE 接入、ReviewCard。
- 测试：`web/tests/timeline.test.ts`，用一份固定的事件序列 fixture（放 `web/tests/fixtures/run_events.json`，**由阶段 3 的 fake 工作流实际产出后导出**，保证前后端协议一致）覆盖：step 开关与嵌套、thinking 追加、重复 seq 忽略、iteration 折叠分组、review resolved、重放结果一致。
- 验收：
  1. 上传 5 张图 → 发送任务描述 → 看到步骤卡片实时更新 → 出现参考确认卡片 → 点确认 → 下一张 → …… → 进入迭代 → 停在 final → 接受。
  2. 运行中刷新页面：时间线完整恢复，实时事件继续到达。
  3. 运行中点停止：显示已取消。
  4. 附录 B 的截图清单。
- commit：`feat(web): live timeline, composer and review cards`

### 阶段 6：产物面板 + 打磨

- 第 9.6、9.7 节；图片查看器；窄屏适配；键盘快捷键（`Ctrl/Cmd+K` 新建任务、`Esc` 关闭查看器、`Ctrl/Cmd+.` 切换产物面板）；空状态、加载骨架屏、toast 错误提示。
- 可选：深色模式（跟随系统，Tailwind `dark:`）。
- 验收：面板三个标签页数据正确，迭代曲线随运行实时更新；Lighthouse 可访问性 ≥ 90（按钮都有 `aria-label`）。
- commit：`feat(web): artifact panel with score chart and algorithm view`

### 阶段 7：切换入口 + 清理

- `python -m api` 托管 `web/dist`（SPA fallback）。
- `AGENTS.md` 的"启动与查看"改为：
  ```bash
  cd web && npm install && npm run build && cd ..
  python -m api
  ```
  开发模式另写 `npm run dev` 的说明。
- 删除 `ui/` 及其测试（`tests/test_annotation_app.py`、`test_gradio_adapters.py`、`test_gradio_app.py`、`test_ui_result_recovery.py`、`test_progress.py` 中只测 UI 的部分），`requirements.txt` 去掉 `gradio`、`huggingface-hub<1.0`（先 `grep -rn "huggingface_hub\|import gradio" --include=*.py .` 确认没有别处使用）。
- **删除前先列出清单发给审查者确认**，确认后再删。
- 验收：`.venv/bin/python -m pytest -q` 全绿；`grep -rn gradio` 在 `.py` 文件中没有结果；按新的 AGENTS.md 说明从零启动成功。
- commit：`refactor: replace Gradio UI with FastAPI + React frontend`

---

## 11. 审查检查点（审查者会重点看这些，执行者提交前请自查）

**后端**

- [ ] listener 用 `run_id` 登记，且 RunManager 生成的 `run_id` 和事件里的 `run_id` 一致（有测试证明）。
- [ ] `RequestCancelled` 被正确捕获（它是 `BaseException`）。
- [ ] 断开 SSE 不会取消运行；只有 cancel 接口会。
- [ ] SSE 生成器在客户端断开后能退出（不会留下永远 `wait` 的线程）；Condition wait 有超时。
- [ ] 同一 run 的 seq 严格递增、无重复，resume 后接着递增。
- [ ] 所有文件访问经过 `resolve_allowed`；接口返回的都是 URL，没有裸文件系统路径。
- [ ] 错误信息和持久化的事件经过 `redact`。
- [ ] 测试不读写真实 `workspace/`、`outputs/`。
- [ ] 没有新建日志配置；没有 `print`。
- [ ] `core/` 改动只在允许范围内。

**前端**

- [ ] reducer 是纯函数，重复 seq 幂等，有测试。
- [ ] 没有把文件系统路径拼进 URL。
- [ ] SSE 连接在切换任务 / 卸载组件时关闭（没有泄漏，DevTools Network 里只有一条活动连接）。
- [ ] 输入法组字时按 Enter 不发送。
- [ ] 按钮提交后立即禁用，防止重复请求。
- [ ] `tsc --noEmit` 无错误，没有 `any`（确需时用 `unknown` 再收窄）。
- [ ] 没有引入第 3 节以外的库。

---

## 12. 已知风险与待定问题

1. **参考掩膜生成很慢**（每张 30–70 秒，期间 SAM 不发事件）。第一版在 `gen_reference` 的 step 卡片上显示计时即可；如果体验太差，后续再在 `core/sam_session.py` 加进度事件（本计划不做）。
2. **`_public_state` 对 `run_status` 的判断很粗**（有 interrupt 就是 `awaiting_feedback`，否则 `completed`，失败时也可能显示 completed）。前端的运行状态以 RunManager 的 `run_finished.status` 为准，不要用 `run_status` 字段判断。
3. **旧任务的回放**：旧任务没有 `stream.jsonl`，只有 `conversation.jsonl`（里面是 Gradio 时代拼好的 markdown/HTML）。第一版对这类任务只把 `conversation.jsonl` 按 user/assistant 消息显示成纯文本，并在顶部提示"旧版任务，仅显示对话记录"；不要尝试解析里面的 HTML。
4. **旧 checkpoint**：`_require_current_checkpoint` 遇到 v2 的 checkpoint 会抛 `OLD_CHECKPOINT_MESSAGE`，API 映射为 400，前端显示该文案。
5. **一个进程只允许一个 API 实例**（`task_lock` 是 fcntl 文件锁，可以跨进程，但 RunManager 的内存状态不共享）。文档里注明不要同时开两个 `python -m api`。
6. 遇到本文档没有覆盖的情况，记到 `docs/2026-09-28-react-frontend-issues.md`，格式：`- [阶段N] 问题描述 / 我的临时处理 / 需要确认的点`。
7. **`tools/gen_reference.py` 的默认 root 不再被读取**：参考掩膜按任务隔离（5.1）后，离线生成工具必须显式传 `--workspace workspace/tasks/<task_id>/reference_masks`，写到全局 `workspace/references` 的产物不会再被任何运行消费；存量全局目录保留但成为死数据。

---

## 附录 A：开发命令

```bash
# 后端
source .venv/bin/activate
LIANGCE_LOG_LEVEL=DEBUG python -m api          # http://127.0.0.1:8765

# 前端（另一个终端）
cd web && npm install && npm run dev            # http://127.0.0.1:5173，/api 代理到 8765

# 测试
.venv/bin/python -m pytest -q tests/test_api_*.py tests/test_agent_workflow.py tests/test_runtime_logging.py
cd web && npm run test && npx tsc --noEmit && npm run lint

# 排查某次运行
rg -n -F 'run=<run_id>' workspace/logs/agent.log*
cat workspace/tasks/<task_id>/runs/<run_id>/stream.jsonl | head
```

## 附录 B：阶段 5、6 提交时附带的截图

1. 空状态（没有任务）。
2. 运行中：一个展开的 step 卡片，里面有流式思考和模型调用行。
3. 参考确认卡片（包含一张低质量提示的）。
4. 折叠后的迭代分组和展开后的单轮详情。
5. final 确认卡片。
6. 产物面板三个标签页各一张。
7. 窄屏（375px 宽）布局。
