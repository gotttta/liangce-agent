# Agent 无法生成可标注算法：问题诊断与修改方案

更新时间：2026-08-20

## 结论

当前“经常无法生成可标注算法”不是单一的 Qwen 能力问题，而是以下几类问题叠加：

1. Web UI 每轮只执行一个候选，模型提示词也要求只返回一个候选，导致没有同轮兜底方案。
2. 通用 Pipeline DSL 不能表达项目中已经验证过的周期背景消除流程，因此周期性 DRAM/SEM 图像仍经常退化为全局阈值。
3. 执行器只把“Mask 非空”当成执行成功，覆盖率达到 85% 的失控结果也会被标记为 `completed`。
4. 阈值、去边界和面积过滤可以连续地把前景清空，但系统没有在中间步骤建立健康检查、备选分支或自动回退。
5. 自动重试没有把所有失败 Pipeline 纳入去重；`no_annotation` 的 Pipeline 可能再次被模型生成并再次执行。
6. Qwen 输出缺少 Pipeline 时没有确定性默认候选，模型一次 JSON 结构错误就会让整轮直接失败。
7. 当前本地算法库和算子库为空，历史检索节点没有可复用成功方案，任务实际上长期处于冷启动。

因此，优先级最高的修改不是继续扩大提示词，而是先建立“可执行、可诊断、可回退”的候选执行闭环。

## 已确认的证据

### 1. Web 实际只有一个候选

Web 调用 [ui/annotation_app.py](../ui/annotation_app.py#L986) 时固定传入 `max_candidates=1`。同时任务理解提示词在 [providers/vision.py](../providers/vision.py#L443) 明确要求只生成一个 `candidate_pipelines` 条目。

这使得“候选 A 为空、候选 B 可用”的情况只能等下一轮自动修订，不能在当前轮比较和选择。`plan_candidate_definitions` 虽然支持最多三个候选，但 Web 没有使用这个能力，见 [core/agent_loop.py](../core/agent_loop.py#L436)。

### 2. 已有的周期背景 Pipeline 没有接入 Agent DSL

仓库已有确定性的周期背景流程 [core/pipelines/periodic_particle.py](../core/pipelines/periodic_particle.py#L21)，并且 Gate 2 固定流程在 `outputs/gate2_fixed/report.json` 中可以处理多张样例。

但通用 DSL 的允许算子列表 [core/pipelines/dsl.py](../core/pipelines/dsl.py#L17) 没有包含：

- `period_estimation`
- `periodic_background_model`
- `periodic_background_residual`
- `residual_threshold`
- `exclude_regions`
- `apply_valid_mask`

这些算子虽然在 [core/operators/image.py](../core/operators/image.py#L111) 和其他算子模块中注册，但没有进入模型看到的 Pipeline Tool Catalog，也无法直接通过当前单输入 DSL 表达完整的周期背景流程。

结果是：模型知道图像有周期结构，也无法调用已验证的周期背景方案，只能在 `statistical_threshold`、`adaptive_threshold` 或 `local_background_residual` 中猜参数。

### 3. 空 Mask 的典型路径是“前景过大后被清空”

最近一次产物 `outputs/20260820_225903_942830_agent_v2/iteration_2/candidate_0/` 记录了：

```text
statistical_threshold: coverage = 0.8338498571
remove_border_components: removed_pixels = 84934
filter_components: kept_components = 0
final coverage = 0
```

这不是模型没有找到任何像素，而是初始暗阈值把大部分图像当成前景，之后去边界组件和面积过滤把整个前景删除。

另一个常见路径是阈值只产生很小的碎片，最终 `filter_components(min_area=100/200)` 将所有组件移除。类似记录见 `outputs/20260820_121309_598168_agent_v2/iteration_0/` 和 `outputs/20260820_141906_058452_agent_v2/iteration_1/`。

### 4. 失控的大 Mask 仍被认为成功

在 `outputs/20260820_225903_942830_agent_v2/iteration_0/candidate_summary.json` 中，候选被标记为 `completed`，但事实统计为：

```text
coverage = 0.852886
component_count = 2
largest bbox touches the image boundary
```

当前成功条件只是 [core/agent_loop.py](../core/agent_loop.py#L110) 中的 `np.any(mask)`。`evaluate_mask_quality` 只记录 coverage、组件数和边界比例，不做健康路由，见 [core/quality.py](../core/quality.py#L4)。

因此，覆盖率 0.85 的明显失控结果会进入视觉复查，而不是立即触发本地回退。

### 5. 自动重试没有排除空结果 Pipeline

`revise_candidates` 生成 `failed_fingerprints` 时只收集 `status == "completed"` 的候选，见 [core/agent_graph.py](../core/agent_graph.py#L560)。

`no_annotation` 候选不会进入去重集合。于是模型即使收到“不能原样重复”的提示，也可能重新生成相同 Pipeline；代码层面没有阻止它再次执行。

### 6. 没有候选时没有确定性回退

模型没有返回可执行 Pipeline 时，`normalize_task_understanding` 会直接抛出错误，见 [providers/vision.py](../providers/vision.py#L625)。即使任务理解成功，只要候选列表为空，`run_planned_agent` 也会直接报错，见 [core/agent_loop.py](../core/agent_loop.py#L55)。

这会把“模型格式错误”和“图像确实难以分割”混成同一种失败，且没有任何本地 baseline 可以继续尝试。

### 7. 当前没有可复用的本地成功知识

截至 2026-08-20，`workspace/algorithms/` 下没有 `algorithm.json`，`workspace/operators/` 下也没有已批准算子文件。虽然工作流包含历史算法检索，但当前新任务找不到任何可执行 baseline，只能完全依赖本轮 Qwen 生成结果。

需要检查旧任务中的用户验收记录为什么没有迁移或发布到当前算法库，并建立一次性迁移脚本；迁移时只能导入确实由用户确认过的完整 Pipeline，不能把历史输出自动视为已验收算法。

### 8. 近期运行快照说明问题不是偶发

对顶层 `outputs/20260820*/iteration_*/candidate_summary.json` 的 22 个候选做事实统计：

- 15 个状态为 `completed`；
- 7 个状态为 `no_annotation`，空结果占 31.8%；
- 15 个 `completed` 中有 2 个 coverage 超过 0.35，其中一个达到 0.852886。

这不是准确率统计，也没有 Ground Truth；它只说明当前执行层同时存在较高的空结果比例和明显失控结果被当成成功的问题。

## 修改目标

修改后，每个任务必须满足以下行为：

1. Qwen 返回 0 个、1 个或多个候选都不会让任务直接崩溃。
2. 每轮至少执行两个语义不同的候选，除非任务明确指定单一已确认算法。
3. 每个候选都记录每一步的输入/输出覆盖率、组件数、边界比例和警告。
4. 空 Mask、过大 Mask、全边界命中和异常组件数量属于“执行健康失败”，自动进入下一个候选。
5. 已验证的周期背景 Pipeline 可以被 Agent 选择或作为确定性兜底。
6. 相同 Pipeline 无论是 `failed`、`no_annotation` 还是健康检查失败，都不能在同一任务中重复执行。
7. 所有候选都失败时，系统仍保存完整诊断，并明确提示用户需要 ROI、include/exclude 约束或新的参考图，而不是只显示“没有标出目标”。

## 推荐修改方案

### P0：先修执行闭环

#### P0.1 增加候选健康检查，不把 `np.any` 当成功条件

**文件**：`core/quality.py`、`core/agent_loop.py`

新增一个事实型健康检查，不称为质量评分，也不替用户判断最终视觉准确性：

```python
def inspect_mask_health(mask, constraints=None):
    # 返回事实和路由建议，不返回综合质量分数
    return {
        "coverage": ...,
        "component_count": ...,
        "border_fraction": ...,
        "largest_component_fraction": ...,
        "issues": ["empty_mask", "coverage_too_large", ...],
        "usable_for_review": False,
    }
```

建议的默认保护条件：

- `coverage == 0`：`empty_mask`
- `coverage > 0.35`：`coverage_too_large`
- `border_fraction > 0.50`：`border_dominated`
- 最大组件覆盖率异常高：`dominant_component`
- 组件数量超过任务约束：`too_many_components`

阈值应作为执行健康边界，可被任务理解结果中的 `target_constraints` 收紧或放宽；不能用它替代最终人工验收。

`run_planned_agent` 应按以下顺序处理：

1. Pipeline 执行。
2. 应用用户 include/exclude 约束。
3. 计算健康事实。
4. 健康失败则保存候选完整产物，状态设为 `health_failed`，继续执行下一个候选。
5. 只有 `usable_for_review=True` 才进入视觉复查候选列表。

如果希望把明显失控结果交给用户查看，也可以保留结果图，但不能把它混入“可复查候选”或标记为普通 `completed`。

#### P0.2 每个 Pipeline step 保存可诊断的 Mask 统计

**文件**：`core/pipelines/dsl.py`

在 `execute_pipeline` 的 trace 中，对每个 `MaskArtifact` 自动记录：

```json
{
  "coverage": 0.123,
  "component_count": 4,
  "border_fraction": 0.02,
  "largest_component_fraction": 0.08
}
```

不要只依赖个别算子自己写 metadata。这样可以明确定位：

- 阈值阶段已经过大；
- 形态学阶段把小目标抹掉；
- `remove_border_components` 删除了目标；
- `filter_components` 的最小面积或最大面积把全部组件筛掉。

同时保留算子现有的 `warnings`，并为 `coverage_exceeded`、`empty_mask`、`kept_components=0` 增加标准化 warning。

#### P0.3 把已验证的周期流程接入 Agent

当前 DSL 是单输入、单输出链，`periodic_background_model` 还需要从周期估计结果和图像同时取参数，不能简单地把几个算子名字加入白名单。

推荐新增一个受约束的内建 Pipeline 类型，例如：

```json
{
  "name": "periodic_particle_builtin",
  "kind": "builtin_pipeline",
  "params": {
    "percentile": 70,
    "min_area": 20,
    "max_components": 3,
    "roi": null
  }
}
```

执行器将其映射到 [core/pipelines/periodic_particle.py](../core/pipelines/periodic_particle.py#L21)，仍然经过参数边界校验、超时限制和 trace 记录。这样可以直接复用 Gate 2 已验证逻辑，不需要模型重新拼接多输入算子。

如果坚持所有流程都必须使用通用 DSL，则需要同时扩展 DSL 的数据契约，支持：

- MetadataArtifact 作为后续算子的输入；
- 一个 step 引用多个输入 artifact；
- `periodic_background_model` 从 `period_estimation` 读取 `period_px` 和 `axis`；
- `periodic_background_residual` 引用 background artifact；
- `exclude_regions` 和 `apply_valid_mask` 的显式输入。

只增加白名单而不扩展这些契约，会得到“模型可以选择，但 Pipeline 不能执行”的新失败。

#### P0.4 Web 每轮执行多个候选

**文件**：`ui/annotation_app.py`、`providers/vision.py`

- Web 默认改为 `max_candidates=3`。
- Prompt 的 schema 改为要求 2 到 3 个语义不同的候选，而不是“只生成一个”。
- 每个候选必须说明不同假设，例如：周期残差、局部残差、亮/暗阈值或 ROI 约束。
- 候选必须经过 Pipeline fingerprint 去重。

如果只执行一个候选，Agent 的“选择算法”实际上没有选择空间，只是在重试同一个猜测。

### P1：修复模型输出和重试策略

#### P1.1 增加确定性 fallback candidate

**文件**：`core/agent_loop.py`、`core/agent_graph.py`

当 Qwen 返回空候选、JSON 校验失败或候选全部不合法时，按任务特征加入本地 fallback：

1. 有周期背景特征：`periodic_particle_builtin`。
2. 有明显局部异常但无周期支持：局部背景残差 + `residual_threshold`。
3. 亮/暗极性明确：归一化 + bright/dark threshold。
4. 极性不明确：同时加入 bright 和 dark 两个候选，而不是默认选择 bright。

fallback 仍然必须经过同一个执行器和人工确认，不能绕过安全校验。

#### P1.2 所有失败状态都参与去重

**文件**：`core/agent_graph.py`

将以下状态都加入 `failed_fingerprints`：

```python
failed_statuses = {"failed", "no_annotation", "health_failed", "duplicate_pipeline"}
```

同时在候选执行前检查 fingerprint，重复时记录 `duplicate_pipeline`，不要再次启动沙箱。

去重还应覆盖历史 `experiment_history`，而不是只看当前候选列表。

#### P1.3 把模型错误和图像失败分开

建议统一为以下错误类型：

| 类型 | 含义 | 是否自动重试 |
|---|---|---:|
| `provider_invalid_json` | Qwen 没有返回可解析结构 | 是，最多一次并追加 schema 错误 |
| `pipeline_invalid` | 算子、参数、输入输出类型不合法 | 是，要求模型修复后换 fingerprint |
| `pipeline_execution_failed` | 沙箱执行异常或超时 | 是，换候选 |
| `empty_mask` | 最终 Mask 为空 | 是，换候选 |
| `health_failed` | Mask 非空但覆盖率/边界/组件结构失控 | 是，换候选 |
| `unsupported_image` | 当前图像不满足周期/ROI 等前提 | 否，向用户请求 ROI 或参考图 |

现在 `_build_failure_state` 将多种情况压缩成 `empty_annotation` 或 `pipeline_execution_failed`，会让后续提示无法针对原因调整。

### P1：改善算子和 Pipeline 的可用性

#### P1.4 处理模型常见参数别名

现有 `normalize_pipeline` 主要只兼容 morphology 的参数别名。历史产物中已经出现：

- `kernel_size` / `iterations` 被传给不支持的算子；
- `operation` 与 `op` 混用；
- 参数值类型与注册函数签名不一致。

建议在模型输出进入执行器前增加显式 schema repair：

1. 只对已知安全别名做转换；
2. 转换后重新验证；
3. 无法安全转换时返回结构化错误并要求模型生成不同 Pipeline；
4. 不要静默删除未知参数。

#### P1.5 暴露可用的 ROI/排除区域算子

`exclude_regions` 已在算子中实现，但当前 DSL 白名单没有将它提供给模型。周期图像、比例尺、边缘大块背景经常需要 ROI 或排除区域；仅依赖 `remove_border_components` 会误删贴边目标或保留大块内部背景。

建议让 Pipeline 能引用任务级 ROI 和用户 exclude mask，且在阈值前应用，而不是只在最终 Mask 上处理。

## 建议的执行状态

当前 `completed` 状态过于宽泛。建议改为：

```text
candidate_status:
  - planned
  - executed
  - health_failed
  - no_annotation
  - pipeline_failed
  - selected_for_review
  - rejected_by_vision
```

`selected_for_review` 只表示“本地健康检查通过并交给视觉复查”，不代表最终视觉正确。最终仍由人工执行 `accept`。

## 测试和验收

### 必须新增的单元测试

1. 全图暗阈值 coverage > 0.35 时，候选状态为 `health_failed`，不会进入普通复查。
2. `remove_border_components` 清空 Mask 时，trace 能指出删除前后的 coverage 和 removed pixels。
3. `filter_components` 清空 Mask 时，trace 能指出 `kept_components=0` 和输入组件数量。
4. `no_annotation` Pipeline 不会在自动重试中再次执行。
5. Qwen 返回空 `candidate_pipelines` 时能执行本地 fallback。
6. Qwen 返回非法参数时，能区分 `pipeline_invalid`，并保留原始错误。
7. Web 默认至少执行两个不同 fingerprint 的候选。
8. 周期背景样例可以通过 Agent 选择 `periodic_particle_builtin`。
9. 所有候选失败时，输出中包含每个候选的 Pipeline、状态、失败 step、统计和建议动作。

### Gate 验收建议

使用以下样例做回归：

- `data/samples/in_film_particle_left_pattern.jpg`
- `data/samples/in_film_particle_middle_defect.jpg`
- `data/samples/in_film_particle_right_zoom.jpg`
- `data/samples/in_film_particle_middle_defect_tight.jpg`
- `data/samples/in_film_particle_right_defect_tight.jpg`

验收不应只看“是否生成了非空 PNG”，至少应检查：

- 空 Mask 率；
- 过大覆盖率；
- 边界主导率；
- 每类图像的候选成功率；
- 重试后的 fingerprint 重复率；
- 失败是否能定位到具体 step；
- 周期背景样例是否优先使用周期残差流程；
- 用户反馈后的 include/exclude 像素是否被确定性遵守。

## 推荐实施顺序

1. P0.1：健康检查和标准化 step trace。
2. P1.2：所有失败状态 fingerprint 去重。
3. P0.4：Web 多候选和 Prompt schema 修正。
4. P1.1：确定性 fallback candidate。
5. P0.3：接入 `periodic_particle_builtin`，先复用已有 Gate 2 逻辑。
6. P1.5：ROI/排除区域进入 Pipeline。
7. 增加测试和五张样例回归，再调整阈值边界。

在完成第 1 至第 5 项前，不建议继续扩大模型提示词或增加更多自定义算子；那会增加候选数量，但不会解决“候选为空、候选重复、失控 Mask 被当成功、已验证周期流程不可调用”这四个结构性问题。
