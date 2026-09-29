# 项目调试与日志

## 排查顺序

- 排查卡顿、模型调用失败、工具执行异常时，先查看统一运行日志，再根据任务 ID、运行 ID 和产物路径追查详细记录。
- 默认日志为项目根目录下的 `workspace/logs/agent.log`，同时输出到终端 stderr。先确认文件更新时间；旧日志不代表当前进程。
- 每行包含时间、级别、`task=`、`run=`、`stage=`。同一次请求的嵌套模型和工作流调用沿用运行 ID；CLI 没有网页任务时 `task=-`。
- 节点和工具事件记录执行进度；模型调用记录开始、结束、耗时和异常堆栈；部分可恢复失败也会保留堆栈。根据开始/结束事件判断停在哪一步，不要仅凭页面状态推断失败。

## 启动与查看

以下命令均在项目根目录执行。日志改动或环境变量调整后，需要重启现有服务才能生效。

```bash
source .venv/bin/activate
cd web && npm install && npm run build && cd ..
python -m api          # http://127.0.0.1:8765
```

同一时刻只运行一个 `python -m api` 实例：任务文件锁可以跨进程，但运行状态在各进程内存里，双实例会互相看不到对方的活动运行。

前端开发模式（热更新，另开终端，后端照常启动）：

```bash
cd web && npm install && npm run dev   # http://127.0.0.1:5173，/api 代理到 8765
```

需要工具参数摘要时，以 DEBUG 模式启动：

```bash
LIANGCE_LOG_LEVEL=DEBUG python -m api
```

另开终端实时查看（人工操作）：

```bash
tail -F workspace/logs/agent.log
```

Agent 排查时优先使用有界读取，避免用持续跟随命令阻塞工具调用：

```bash
tail -n 200 workspace/logs/agent.log
rg -n -C 8 'ERROR|WARNING|Traceback' workspace/logs/agent.log
rg -n -F 'run=<实际运行ID>' workspace/logs/agent.log*
```

`tail -F` 会跟随文件轮转；按 Ctrl+C 仅停止查看，不会停止服务。筛选运行 ID 时，堆栈续行不带 ID，需结合原文件上下文查看完整异常。

## 配置

| 环境变量 | 默认值 | 用途 |
|---|---|---|
| `LIANGCE_LOG_LEVEL` | `INFO` | 支持 DEBUG、INFO、WARNING、ERROR、CRITICAL |
| `LIANGCE_LOG_DIR` | 项目根目录下 `workspace/logs` | 日志目录 |
| `LIANGCE_LOG_MAX_BYTES` | `10485760` | 单文件约 10 MiB 时轮转 |
| `LIANGCE_LOG_BACKUP_COUNT` | `5` | 保留 5 个轮转文件 |

网页入口 `python -m api` 和 CLI 入口 `python main.py ...` 自动配置日志。直接调用 Python API 时，先调用 `core.runtime_logging.configure_logging()`。多个应用进程应使用不同的日志目录。

## 详细记录与维护

- `workspace/tasks/<task_id>/events.jsonl`：任务事件。
- `workspace/tasks/<task_id>/nodes/<node>/latest.json`：节点输入、输出、耗时和错误；同目录 `run_*.json` 保存历史记录。
- `outputs/<run_dir>/iteration_<n>/`：算法、质量报告、算子轨迹、运行环境及工作流状态；候选子目录也有相应产物。未完成或失败的运行可能缺少最终文件。
- 统一日志实现位于 `core/runtime_logging.py`，事件桥接位于 `core/agent_events.py`；使用现有 logger 和上下文机制，不另建一套日志配置。
- 终端和文件输出会过滤已知环境密钥、凭据字段和图片 base64；不记录完整模型提示词、完整回复或流式分块。不要为调试关闭过滤，分享前仍需检查业务文本和路径。
- 日志目录已加入 `.gitignore`，不要提交运行日志。
- 修改日志机制后运行 ` .venv/bin/python -m pytest -q tests/test_runtime_logging.py`，并按改动范围验证相关工作流、模型或 UI 测试。
