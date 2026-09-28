# React 前端迁移：执行中发现的问题

格式：`- [阶段N] 问题描述 / 我的临时处理 / 需要确认的点`

- [阶段3] iterate 提示词把全部算子目录（含 `build_periodic_background` 等周期背景算子）开放给模型，但 `_to_dsl_pipeline` 只能生成线性链式 DSL；模型选用这类算子时 `score` 节点执行抛 `ValueError: pipeline operator requires v3 named inputs`，整个 run 直接 failed（真实模型验收第一次触发，见 agent_21a7f99a… 的 stream.jsonl）。/ 未修（计划 §0.3：core 问题只记录）。引导性任务描述（"只用 normalize 和阈值类算子"）可稳定绕开，第二次真实运行完整走到 target_reached + awaiting_review。/ 需要决策：要么在提示词里收敛可用算子集合，要么让 score 把执行失败当作该轮 0 分继续迭代，要么让 DSL 支持 v3 named inputs——属于 core 行为选择，不在本计划范围内改。
