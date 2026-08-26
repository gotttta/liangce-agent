# Liangce Agent Handover

更新时间：2026-08-18

## 当前产品决策

当前流程不使用 Ground Truth，也不自动计算准确率或候选质量分数。算法由视觉 Agent 自主生成、执行和复查，最终由用户查看结果后确认。

Handbook 中甲方已经标注的图片作为 few-shot 视觉参考，只用于学习：

- 哪些对象需要标注
- 标注边界如何放置
- 标注颜色、线宽和风格

参考图与当前目标图没有像素坐标对应关系，不复制坐标，也不作为当前目标图的标准 mask。

## 已完成

- Handbook 参考图从单张扩展为 `reference_examples[]`，最多 3 张。
- CLI 支持重复使用 `--reference` 参数。
- Web UI 支持上传多张 Handbook few-shot 参考图。
- 参考图复制到任务目录的 `references/`，任务 JSON 保存图片路径和描述。
- Qwen 在任务理解和候选结果视觉复查两个阶段接收 Handbook 图片。
- Agent 流程为：

  ```text
  目标图 + Handbook few-shot 图
      -> Qwen 生成候选算法
      -> 本地 DSL 校验和沙箱执行
      -> Qwen 视觉复查并选择候选
      -> 明确要求 revise 时最多自动重试一轮
      -> 等待用户确认
  ```

- 用户点击确认后保存完整 Pipeline 和算法记录，不保存人工标准 mask。
- 算子库默认为空。生成的自定义算子只属于当前候选，不会自动进入可复用算子库。
- 只有用户明确测试批准后，算子才允许通过 `OperatorLibrary.publish(..., user_tested=True, tested_by=...)` 发布。
- 旧的 Ground Truth 保存 API、对应 UI 入口和测试已删除。
- 项目文档已改为“用户确认与算法发布”流程。

## 重要入口

- `core/agent_graph.py`：CLI 和 Web 共用的主 LangGraph。
- `core/agent_loop.py`：候选 Pipeline 执行、事实统计和产物保存。
- `core/operator_library.py`：用户测试批准的可复用算子库。
- `core/algorithm_registry.py`：用户确认后的完整算法库。
- `core/task_store.py`：任务、反馈和确认结果的文件存储。
- `providers/vision.py`：Mock/Qwen Provider、few-shot Prompt 和视觉复查。
- `ui/annotation_app.py`：Gradio Web UI。
- `agent.py`：CLI/API 入口。

## 运行方式

运行测试：

```bash
./.venv/bin/python -m pytest -q
```

当前验证结果：`115 passed`。

启动 Web UI：

```bash
./.venv/bin/python -m ui.app
```

CLI 示例：

```bash
./.venv/bin/python agent.py \
  --target path/to/target.png \
  --description "用荧光绿色提取目标轮廓" \
  --reference path/to/handbook_a.png \
  --reference path/to/handbook_b.png
```

## 输出和状态

单次运行输出通常在 `outputs/<run_id>/iteration_0/`，包括：

- `pipeline.json`
- `operator_trace.json`
- `quality_report.json`：事实统计和执行异常，不是质量评分
- `result_annotated.png`
- `mask.png`
- `measurements.json`
- `agent_trajectory.json`
- `graph_state.json`

用户确认后的算法写入 `workspace/algorithms/<algorithm_id>/algorithm.json`。算子库目录为 `workspace/operators/`，没有用户测试批准时应保持没有可复用自定义算子。

历史任务迁移

如需把旧任务中明确确认过的完整 Pipeline 导入当前算法库，先执行 dry-run：

```bash
PYTHONPATH=. .venv/bin/python scripts/migrate_accepted_algorithms.py --dry-run
```

确认报告后去掉 `--dry-run` 执行。脚本只接受任务状态为 `accepted`、存在 `acceptance/latest.json` 用户确认记录且 Pipeline 通过 DSL 校验的记录；空结果、健康失败、非法或重复 Pipeline 会被跳过。

## 兼容性和残留

- `reference_annotation_path` 参数仍保留在部分旧 API 中，用于兼容旧调用；当前语义只是参考图，不是 Ground Truth。
- `core/measurement/evaluation.py` 和部分旧评估测试仍存在，当前主 Agent 流程不调用它们。若后续要求彻底移除所有旧评估能力，再单独删除该模块和对应测试。
- `data/`、`workspace/`、输出目录中可能有历史任务和样例，不要为了清理术语而删除用户数据。
- 当前工作区是未提交的开发状态，包含本轮 Agent v2 大量新增和修改文件。新窗口开始时先运行 `git status --short`，不要重置或覆盖已有改动。

## 建议下一步

1. 用真实 Handbook 示例跑一次 Web UI，确认 Qwen 能正确理解示例语义和边界风格。
2. 检查用户确认后的 `algorithm.json` 是否包含足够的回放信息。
3. 设计单独的“测试算子并批准”界面或 CLI；在此之前不要向 `workspace/operators/` 发布任何算子。
4. 后续若要彻底移除 Ground Truth 旧代码，先评估 `core/measurement/evaluation.py`、`data/benchmarks/` 和历史文档的兼容需求。
