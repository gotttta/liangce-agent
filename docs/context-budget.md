# 模型上下文与证据读取

模型输入与持久化报告分开：执行器仍完整保存 outputs.json、measurements.json、
quality_report.json、pipeline.json、operator_trace.json 和图像。算法、候选数量、
像素计算、用户约束、验收门禁和原图编码不因上下文压缩改变。

execute_pipeline、compare_candidates 与视觉评审共用 core/experiments/context.py：

- 图像和 Mask 仅展开尺寸、元素数、范围、非有限值数量及前景像素数。
- 轮廓仅展开轮廓数量、点数、图像尺寸及元数据。
- 小型结构化测量完整保留；大型表格明确标记 partial、总数与非代表性样本。
- 质量、错误、健康状态、用户约束、评估指标和测量汇总原样保留。
- 每个候选带 experiment_id；省略字段不是零值，也不代表没有缺陷。
- outputs 描述用户约束应用前的数据；quality 统计、最终 Mask 和叠加图描述约束后的结果。

模型可调用 inspect_experiment：

- report 选择 measurements、outputs、quality_report、pipeline 或 operator_trace。
- selector 使用 JSON Pointer（例如 /results 或 /measurements/data/components），
  offset/limit 分页读取，next_offset 指向下一页。大型嵌套字段使用返回的 selector 深入读取。
- 长字符串和算子源码按字符分页，不作不可见的截断；数组原文不作为像素证据回传。
- region=[left,top,right,bottom] 读取对齐的原图、叠加图和最终 Mask 局部图，
  右下边界不包含，最大 1024×1024，保持原分辨率。全图仍用于判断整体遗漏。
- 只接受当前会话已有实验 ID，不接受任意文件路径。所有检查共享每阶段 3 次 inspection 预算。
- 评审只开放读取工具；不能执行新算法。证据不充分时要求返回 revise 并说明原因。

每次流式模型请求前记录 llm_input_size 到现有统一日志，包括角色文本字符数、
工具参数字符数、工具 Schema 字符数、图片数量及编码长度；不记录提示词或图片内容。
ALIYUN 返回的 usage 仍是实际 token 统计，字符预算不冒充 token 精确值。

LIANGCE_LLM_MAX_TEXT_CHARS 默认 120000，包含文本、历史工具调用参数和工具 Schema，
不把图片 base64 算作文本。超限在发送 API 前明确失败，保留用户要求与证据，
不会静默删除或截断它们。此阈值是防止意外巨大请求的保护，不是模型窗口大小，
也不限制视觉 token；实际图像开销仍查看 usage。修改服务代码或环境变量后需重启生效。

复现验证：

```bash
.venv/bin/python -m pytest -q tests/test_context_evidence.py tests/test_experiment_artifacts.py tests/test_vision_provider.py tests/test_planning_protocol.py tests/test_runtime_logging.py
```

测试覆盖输入不含完整像素数组、原始报告不变、质量和验收信息保留、明细分页可恢复、
小目标/边缘目标/粘连目标局部图逐像素一致、空结果/执行失败不被掩盖，以及请求超限时 API 不被调用。
这些检查证明证据保真和输入缩减；模型实际识别准确率仍需使用固定数据集在线评估。
