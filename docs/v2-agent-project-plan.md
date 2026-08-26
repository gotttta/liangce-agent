# DRAM 缺陷轮廓量测 Agent v2 项目计划

## 0. 产品定义更新

v2 的最终产品不是单一 particle/bridge 缺陷检测器，而是本机单用户的通用缺陷检测算法开发 Agent。周期背景 bridge defect 仅作为第一个端到端验收案例。

用户从一张缺陷图片和自然语言描述开始。Qwen 负责视觉理解、检索已有算法与经用户测试批准的算子、规划候选实验；确定性本地执行器负责运行受约束的 Pipeline DSL。算子库不足时模型可以生成受限的自定义算子源码，但源码必须通过 AST 白名单校验且只能在沙箱中执行；验收算法不会自动发布其中的自定义算子，只有用户单独测试并明确批准后才进入可复用算子库。用户通过文字和漏检/误检标记提供反馈，最终只有用户可以执行“验收并保存”。

已确认的产品约束：

- 图片允许发送至配置化的阿里云 Qwen 视觉模型。
- 每个 Agent 工作流节点最多运行 3 分钟，完整任务可以包含多轮节点。
- 新任务只有一张缺陷图片也可以启动，后续逐步补充同类样本。
- 任务全过程保存对话、样本、节点输入输出、候选算法、生成代码、中间图、反馈和验收记录。
- V1 使用文件目录存储任务和算法，并以 Git 管理版本，不引入数据库和多人权限。
- 验收后的算法进入算法库，支持单张和文件夹批量运行；日常执行不再依赖 Qwen。
- 最终算法交付包括算法实现、适用范围、版本、验收样本和验收结果。

首个纵向关卡为：`创建任务 -> 样本落盘 -> Qwen结构化任务理解 -> 候选实验计划 -> 节点耗时/产物落盘 -> 对话展示并等待用户确认`。

## 1. 项目目标

v2 聚焦一个可验证的任务：从具有周期背景的 DRAM/SEM 图像中排除正常重复结构，提取 in-film particle 缺陷的闭合 mask 和真实轮廓。

系统必须保持 Agent 形态，而不是固定参数表单：Agent 负责理解用户目标、分析图像、规划并调用 CV 算子、检查结果、自动重试，以及在信息不足时请求用户通过文字或画笔反馈。像素级处理由经过测试的确定性算子执行，避免每次生成任意 Python 代码带来的随机性和不可复现性。

## 2. v2 范围

### 2.1 第一阶段范围

- 支持单张 DRAM/SEM 图片和自然语言目标描述。
- 重点处理周期线条背景中的 particle/residue 异常。
- 输出二值 mask、闭合缺陷轮廓、面积、数量和运行记录。
- Agent 能组合算子、调整参数并自动尝试最多 3 个候选方案。
- 支持文字反馈以及 include/exclude 画笔约束。
- 支持用户在结果图上直接确认或提出漏检/误检反馈。
- 保存每轮 pipeline、参数、中间图、结果和用户反馈，保证可重放。

### 2.2 暂不包含

- CD、overlay、roughness、深度或体积量测。
- 未通过 AST 白名单和沙箱边界的 LLM Python 代码执行。
- 模型训练或微调。
- wafer 级批量生产处理。
- 将模型输出自动当作最终答案。

## 3. 总体架构

```text
原图 + 用户描述 + 可选参考标注
              |
              v
        Agent 理解任务和图像
              |
              v
       生成受约束 Pipeline DSL
              |
              v
         确定性 CV 算子执行
              |
              v
       mask / 轮廓 / 事实统计
              |
       +------+----------------+
       |                       |
   结果可信                 结果不确定
       |                       |
   展示与量测       自动改参数或询问用户
                               |
                     文字/画笔反馈后重跑
```

职责边界：

- **LLM/Agent**：理解缺陷、识别背景类型、复用或生成原子算子、组合 pipeline、解释决策、决定重试或询问用户。
- **Pipeline 执行器**：校验 DSL、按顺序调用算子、记录中间结果、捕获失败。
- **CV 算子**：执行确定性的图像处理，不访问网络，不自行改变工作流。
- **结果检查器**：只检查空结果、尺寸错误、过大覆盖和明显边缘/周期结构命中等事实问题，不生成质量分数。
- **用户**：通过自然语言、include/exclude 画笔或候选结果选择消除歧义。

## 4. Agent 工作流

当前 CLI 与 Web UI 共用 `core/agent_graph.py` 中的唯一 LangGraph：

```text
prepare_inputs
  -> understand_task
  -> retrieve_algorithms
  -> plan_candidates
  -> execute_candidates
  -> review_candidates
  -> (revise_candidates -> execute_candidates，最多一轮)
  -> decide_next_action
  -> wait_for_human
       -> END
```

每个节点将耗时、候选来源、事实统计与最终决策写入 `agent_trajectory.json`。
候选内部仍由确定性 Pipeline 执行器完成算子校验、沙箱执行和量测，不生成综合质量分数，也不
替用户判断“可接受/不确定/失败”。候选完成后使用 LangGraph `MemorySaver` 创建
`human_review` interrupt，并通过 `Command(resume=...)` 恢复；
任务存储层保留文件化产物与状态，进程重启后的 checkpoint 持久化将改用外部存储实现。

Agent 默认行为：

1. 从图片与描述中提取目标缺陷、正常背景、排除区域和期望输出。
2. 选择一个主 pipeline，并说明选择理由。
3. 在沙箱中执行最多 3 个候选，并记录覆盖率、连通域数量和边缘占比等事实统计。
4. 由视觉模型比较原图与候选标注叠加图，按文字理由选择候选；不生成候选质量分数。
5. 视觉复查明确要求修订时，自动重新规划并执行最多一轮；仍不能确定时停止自动迭代。
6. 只有结果图生成后才请求用户确认，用户可接受、退出或提供文字/画笔反馈。
7. 用户反馈后保留历史，生成下一轮 pipeline，不覆盖前一轮产物。

## 5. Pipeline DSL 与算子接口

Agent 只输出结构化 pipeline，不直接输出可执行代码：

```json
{
  "task": "particle_contour",
  "background": "periodic_lines",
  "target": "破坏周期规律的局部颗粒",
  "steps": [
    {"op": "exclude_regions", "regions": ["scale_bar"]},
    {"op": "normalize", "method": "percentile"},
    {"op": "periodic_background_residual", "axis": "auto"},
    {"op": "threshold", "method": "otsu", "polarity": "absolute"},
    {"op": "morphology", "method": "close", "radius": 2},
    {"op": "filter_components", "min_area": 20},
    {"op": "apply_user_constraints"},
    {"op": "extract_contours"}
  ]
}
```

统一数据契约：

```text
ImageArtifact: float32 灰度图、原图尺寸、处理历史
MaskArtifact: bool 二维数组、与原图同尺寸、生成步骤元数据
ContourArtifact: 闭合轮廓列表、对应 component_id
OperatorResult: artifact、metadata、warnings、debug_images
```

校验规则：

- 只允许注册表内的算子和参数。
- 参数必须符合类型和范围约束。
- mask 必须为二维数组且尺寸与原图一致。
- Pipeline 必须以图像输入开始，并在量测前产生 mask/contours。
- 未知算子、非法顺序或异常输出必须返回明确错误，不静默回退。

## 6. 第一批算子

### 6.1 预处理

- `grayscale`：统一灰度输入。
- `normalize`：分位数归一化，降低不同图像亮度差异。
- `gaussian_denoise`：抑制高频成像噪声。
- `clahe`：局部对比度增强，仅在低对比图中使用。
- `exclude_regions`：排除比例尺、图像边框和用户指定区域。

### 6.2 背景与异常提取

- `global_threshold`：保留现有亮/暗阈值作为 baseline。
- `adaptive_threshold`：处理不均匀照明。
- `period_estimation`：通过自相关或频域分析估计重复方向与周期。
- `periodic_background_residual`：对齐周期单元，以中位数生成正常模板并计算残差图。
- `template_difference`：有正常参考图或正常区域时执行配准差分。
- `residual_threshold`：对残差图执行 Otsu、分位数或稳健统计阈值。

### 6.3 mask 与轮廓后处理

- `morphology`：开、闭、膨胀和腐蚀。
- `fill_holes`：得到闭合缺陷区域。
- `filter_components`：按面积、位置、长宽比筛除噪声。
- `apply_user_constraints`：确定性应用 include/exclude mask。
- `extract_contours`：从最终 mask 提取闭合轮廓，不画编号和矩形框。

优先完成 `period_estimation + periodic_background_residual`，因为该能力决定系统能否区分正常周期线条和局部 particle 缺陷。

## 7. 用户交互设计

### 7.1 自然语言

支持用户表达目标和反馈，例如：

- “只提取中间颗粒的轮廓。”
- “正常的周期线条不要算。”
- “这里漏了。”
- “下方比例尺不要处理。”

Agent 将反馈解析为目标变化、误检、漏检、边界偏紧或排除区域，不直接把自然语言映射为不透明的阈值变化。

### 7.2 画笔约束

前端提供明确模式：

- `include`：最终 mask 必须包含用户涂画区域。
- `exclude`：最终 mask 必须移除用户涂画区域。
- 后续可扩展 `boundary_hint`，第一阶段不作为必需能力。

画笔保存为独立二值 PNG，并记录模式、原图尺寸和轮次。用户反馈不得仅作为图片发送给 LLM，必须由 `apply_user_constraints` 确定性执行。

### 7.3 候选方案

当 Agent 无法确定紧边界或宽边界时，可展示最多 3 个候选 mask。用户选择后，Agent 将选择结果和对应 pipeline 保存到历史记录并继续量测。

## 8. Handbook Few-shot 参考与验收方式

### 8.1 Handbook 参考图

- Handbook 图是客户已经画好标注的 few-shot 视觉示例，用于理解目标类别、边界和标注风格。
- 参考图与当前目标没有像素坐标对应关系，不复制坐标，不作为当前图片的像素标注。
- 参考图在任务理解和候选视觉复查两个阶段发送给视觉模型。
- 每个任务最多使用 3 张相似示例，并复制到任务目录的 `references/` 下保存。
- 最终只验收当前 Pipeline 和当前结果，不由参考图计算准确率。

本地标注工具运行方式：

```bash
python3 -m ui.app
```

工具只读取画笔图层并保存纯黑白 PNG，默认写入 `data/annotations/`；默认禁止覆盖已有标注，不调用 LLM。

### 8.2 v2 第一阶段验收标准

- 全部测试样本均能完成 pipeline，不能崩溃或产生尺寸不一致的 mask。
- 正常无缺陷样本不产生大面积周期结构误检。
- 同一图片、同一 pipeline 和参数重复运行，mask 必须完全一致。
- 用户 include/exclude 反馈应用后，对应约束必须 100% 生效。
- 每轮输出可由保存的 pipeline 和参数离线重放。

## 9. 输出结构

```text
outputs/<run_id>/
  input_manifest.json
  iteration_0/
    task_understanding.json
    pipeline.json
    operator_trace.json
    debug/
      normalized.png
      background.png
      residual.png
    mask.png
    contours.json
    result_contour.png
    measurements.json
    quality_report.json
    graph_state.json
  iteration_1/
    feedback.json
    include_mask.png
    exclude_mask.png
    ...
```

`pipeline.json` 是可重放的正式执行定义；`operator_trace.json` 记录每个算子的耗时、参数、警告和中间产物；`quality_report.json` 只记录事实统计和执行异常，不代表算法质量评分。

## 10. 实施里程碑

### M1：数据契约与 Pipeline 执行器

- 定义 artifact、算子注册表、参数校验和 Pipeline DSL。
- 将现有阈值和形态学逻辑迁移为第一组注册算子。
- 支持保存、加载和重放 pipeline。
- 保持现有 CLI/API 的输入方式兼容。

完成标准：固定 pipeline 可稳定生成与当前 baseline 等价的 mask，非法 pipeline 有明确错误。

### M2：周期背景粒子检测

- 实现周期方向/周期估计。
- 实现周期模板重建和残差图。
- 增加比例尺/边框排除、残差阈值与组件过滤。
- 用现有 `in_film_particle` 样本建立可视化调试输出。

完成标准：正常周期结构不再被整体提取，主要候选集中于局部异常区域。

### M3：Agent 规划与自检循环

**状态**：进行中。已完成唯一 LangGraph 入口、已验收算法检索、候选规划与去重、最多 3 个候选执行、
沙箱边界、Agent trajectory 和 human interrupt/resume；持久化 checkpoint 尚未完成。

- 将视觉模型输出从单个 threshold strategy 升级为 Pipeline DSL。
- 增加 `inspect_image`、`evaluate_result` 和 `decide_next_action` 节点。
- 支持最多 3 次自动调整，并避免重复失败方案。
- 输出简明的决策解释和具体的用户问题。

完成标准：Agent 能针对正常图、周期异常图选择不同 pipeline，并在 mask 失控时自动恢复或请求用户反馈。

### M4：用户反馈闭环

- 前端增加 include/exclude 模式和清晰的当前模式状态。
- 将画笔图层保存为确定性约束 mask。
- 支持历史轮次、前后结果对比和候选选择。
- 反馈后重新规划并保留全部运行记录。

完成标准：用户可通过一次文字或画笔反馈修正明显误检/漏检，约束不会在下一轮丢失。

### M5：用户确认与算法发布

- 完成结果图、Mask、轮廓和量测的人工复核闭环。
- 保存用户的接受、退出、文字反馈和 include/exclude 画笔记录。
- 用户确认后发布完整、可重放的 Pipeline；算子仍需单独测试批准才进入算子库。

完成标准：用户可复核结果、要求修订并确认发布，所有过程产物可重放。

## 11. 测试计划

- **算子单元测试**：输入输出类型、参数边界、空图、常量图、尺寸保持和确定性。
- **Pipeline 测试**：合法顺序、非法算子、缺失 mask、异常返回、重放一致性。
- **Agent 测试**：使用 Mock provider 验证规划、自动重试、等待反馈和最大迭代限制。
- **反馈测试**：include/exclude 的像素级强约束、坐标一致性和多轮累积。
- **指标测试**：完全一致、完全不相交、空 mask、ignore 区和尺寸不匹配。
- **端到端测试**：样本上传、Agent 执行、轮廓展示、反馈重跑、文件导出。
- **视觉测试**：桌面与移动端检查原图、mask、轮廓和候选结果不变形、不重叠。

## 12. 分阶段测试关卡

v2 按最小纵向闭环推进。每一关都必须先通过 Go/No-Go 检查，才进入下一层；失败时只修当前层，不同时引入新的架构变量。

### Gate 0：Baseline 冻结

**状态**：已通过。冻结记录位于 `data/baselines/v1_threshold/manifest.json`，可复现测试位于 `tests/test_v1_baseline.py`。

**输入**：现有 `in_film_particle` 样本、当前 threshold pipeline 和当前输出。

**验证**：保存 baseline 的 pipeline、mask、轮廓图、运行耗时和已知误检；确认相同输入重复运行产生完全相同的 mask。

**Go 条件**：baseline 可重放，所有输出文件齐全，已知失败样本和失败原因有记录。

**No-Go**：输入无法复现、输出缺失或当前结果没有可比较的记录。

### Gate 1：单算子正确性

**状态**：已通过。统一 Artifact/OperatorResult、算子注册表和 9 个 P0 算子位于 `core/operators/`；单元与真实样本回归位于 `tests/test_operators.py`，Gate 0 等价性由 `tests/test_v1_baseline.py` 持续验证。

**输入**：构造的小图、常量图、噪声图和真实样本。

**验证**：每个算子的输入输出类型、尺寸、参数边界、空结果、异常路径和确定性；保存必要的 debug image。

**Go 条件**：算子单元测试全部通过，mask 始终与原图同尺寸，重复执行结果一致。

**No-Go**：算子隐式改变尺寸、参数越界无错误、或同一输入产生不同结果。

### Gate 2：固定新 Pipeline

**状态**：进行中，尚未放行。已实现周期背景模型、残差、有效区域阈值和 ROI-seeded 固定 Pipeline；结果边界由用户复核和反馈迭代确定。紧裁图因缺少足够正常周期而明确返回 unsupported。执行脚本为 `scripts/run_gate2_pipeline.py`，当前结果写入 `outputs/gate2_fixed_roi/`。

**输入**：不接 LLM 的固定流程：`normalize -> periodic_background_residual -> threshold -> morphology -> contours`。

**验证**：与 baseline 对比正常周期结构误检、缺陷召回、轮廓可视化和运行日志。

**Go 条件**：固定 pipeline 能在样本上稳定提取局部异常，且达到当前阶段设定的指标目标；中间结果可人工解释。

**No-Go**：周期背景仍被整体提取、mask 为空/失控，或无法判断失败发生在哪一步。

### Gate 3：Pipeline DSL 与执行器

**输入**：Gate 2 已验证的固定流程及其 DSL 表达。

**验证**：DSL 执行结果与直接调用算子结果一致；非法算子、非法参数、非法顺序和异常输出被明确拒绝；pipeline 可以保存、加载和重放。

**Go 条件**：DSL 重放得到完全一致的 mask、轮廓和量测结果，operator trace 完整。

**No-Go**：DSL 与直接执行结果不一致、错误被静默吞掉，或 pipeline 无法离线重放。

### Gate 4：Agent 选择与自检

**输入**：少量已验证的候选 pipeline，先使用 Mock provider，再接入真实视觉模型。

**验证**：Agent 能根据图片和描述选择候选、调整允许范围内的参数、识别空/过大/边缘命中等失败，并在最多 3 次尝试后停止或询问用户。

**Go 条件**：Agent 的决策可由 strategy/pipeline/trace 解释，重复输入不会无边界尝试，也不会选择未注册算子。

**No-Go**：Agent 生成非法 pipeline、重复失败方案、无法说明选择原因，或模型输出直接绕过执行器。

### Gate 5：用户反馈闭环

**输入**：真实 UI 中的自然语言、include mask 和 exclude mask。

**验证**：include 像素必须保留，exclude 像素必须移除；反馈轮次、原结果、新结果和决策记录完整保存；Agent 在不确定时能提出具体问题。

**Go 条件**：一次用户反馈能够修正对应误检/漏检，约束 100% 生效，下一轮结果可重放。

**No-Go**：画笔只被传给 LLM 但不改变 mask，反馈丢失，或多轮后无法区分各轮结果。

只有 Gate 0～5 按顺序通过后，才进入新的缺陷类型或批量处理范围。

## 13. 风险与控制

- **样本过少导致过拟合**：训练/调参样本与最终验收样本分开，记录数据集版本。
- **LLM pipeline 不稳定**：严格 schema、算子白名单、参数范围、Mock 回归和 pipeline 重放。
- **周期估计失败**：保留模板差分、用户 ROI 和请求用户标记正常区域的降级路径。
- **模糊边界导致标注争议**：使用 ignore mask，并记录边界容差。
- **Agent 无限尝试**：单轮最多 3 个候选，总迭代次数受控，相同 pipeline 不重复执行。
- **用户反馈只影响语言不影响结果**：include/exclude 由确定性算子强制应用并测试。

## 14. 交付结果

v2 第一阶段完成后，系统应具备以下能力：用户上传周期背景 DRAM/SEM 图并描述目标，Agent 自主选择和执行可重放的 CV pipeline，排除正常结构，输出 particle 缺陷 mask 和闭合轮廓；当结果不确定时，Agent 能自动调整或向用户提出具体问题，用户可以通过文字和 include/exclude 画笔完成修正，并确认发布最终算法。
