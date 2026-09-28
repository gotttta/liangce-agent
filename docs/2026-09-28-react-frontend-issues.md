# React 前端迁移：执行中发现的问题

格式：`- [阶段N] 问题描述 / 我的临时处理 / 需要确认的点`

- [阶段3] iterate 提示词把全部算子目录（含 `build_periodic_background` 等周期背景算子）开放给模型，但 `_to_dsl_pipeline` 只能生成线性链式 DSL；模型选用这类算子时 `score` 节点执行抛 `ValueError: pipeline operator requires v3 named inputs`，整个 run 直接 failed（真实模型验收第一次触发，见 agent_21a7f99a… 的 stream.jsonl）。/ 未修（计划 §0.3：core 问题只记录）。引导性任务描述（"只用 normalize 和阈值类算子"）可稳定绕开，第二次真实运行完整走到 target_reached + awaiting_review。/ 需要决策：要么在提示词里收敛可用算子集合，要么让 score 把执行失败当作该轮 0 分继续迭代，要么让 DSL 支持 v3 named inputs——属于 core 行为选择，不在本计划范围内改。

- [阶段5] SSE 的 `event: end` 结束帧不带 `data:` 字段，浏览器按 SSE 规范直接丢弃该事件（data buffer 为空不派发），EventSource 收不到 end 只能靠断线重连，前端永远不知道运行结束。阶段 3 测试逐行解析 SSE 文本所以没发现。/ 已修：`api/runs.py` 结束帧改为 `event: end\ndata: {}\n\n`，`_collect_events` 测试助手跳过无 type 的 data 帧。/ 无需确认，纯 bug 修复。

- [阶段5] `RunManager.resume` 没有把 `RunHandle.finished` 复位：决策提交后前端立刻重开 SSE，`_stream` 看到上一段遗留的 finished=True 会直接发 end，时间线在 resume 段中途断流（真实浏览器验收触发；阶段 3 的 resume 测试在整段结束后才订阅，错过窗口）。/ 已修：resume 拿到 handle 后置 `finished=False`，并加了"决策后立刻订阅"的回归测试（禁用修复验证过会失败）。/ 无需确认。

- [阶段5] `latest_run_id` 要到第一次 human_gate 中断（`save_run_state`）才写入任务和 `runs/*/latest.json`，首次运行的第一个节点进行中刷新页面时，任务详情 `running=true` 但没有任何 run id 可订阅，时间线无法恢复（计划 §9.8 的"运行中刷新页面"验收触发）。/ 已在 `TaskSummary` 增加 `active_run_id` 字段（RunManager 内存态），前端优先用它订阅。/ 这是超出计划 §6.1 字段清单的小扩展，请审查确认接口形态。

- [阶段5] UI 事件协议里没有任何事件会产生 §9.2 的 `assistant` 条目（`model_output` 按 §9.2 归入模型调用行的折叠输出）。/ `AssistantMessage` 组件与 reducer 类型已就位但当前无数据来源，不渲染。/ 等 core 提供 finish 摘要类事件时再接，或确认永远不需要后删除。

- [阶段5] §12.3 的旧任务回放（只有 `conversation.jsonl` 的 Gradio 时代任务显示为纯文本对话 + 顶部提示）没有实现：§6 没有对应 API，属计划未覆盖的接口缺口。/ 阶段 5 先跳过（旧任务在新 UI 里显示为空时间线）。/ 需要决定在阶段 6/7 补一个只读端点还是接受旧任务不可见。

