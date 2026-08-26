# Agent 改进路线图

## 问题诊断

当前 Agent 能生成初版标注并等待人工审核，但**人机协作的反馈闭环不完整**：

1. **"修改"功能缺失用户输入的传递**：用户点"继续修改"后，新增的描述和画笔标注没有传回 Agent
2. **实例图（Handbook）作用有限**：只作为视觉理解的参考，不影响算法参数的生成
3. **拒绝原因未记录**：用户点"退出"时没有保存失败原因，下次遇到类似任务无法避坑

---

## 改进计划

### Phase 1: 完整的修改反馈闭环（P0，必须）

#### 1.1 UI 层：收集用户的增量输入

**文件**：`ui/annotation_app.py`

**当前行为**：
- 用户点"继续修改"后，UI 触发 `resume_agent_graph(thread_id, {"action": "continue"})`
- 只传递了 action，没有传递用户新增的内容

**目标行为**：
- 增加一个"修改说明"文本框，让用户输入："左上角漏了一个颗粒"、"比例尺区域不要标注"
- 保留画笔工具的 include/exclude mask
- 点"继续修改"时，传递：
  ```python
  {
      "action": "continue",
      "incremental_description": "左上角漏了一个颗粒",
      "include_mask_path": "path/to/include.png",  # 用户涂的"必须包含"区域
      "exclude_mask_path": "path/to/exclude.png",  # 用户涂的"必须排除"区域
  }
  ```

**具体修改**：
```python
# ui/annotation_app.py 中的 resume 处理函数

def handle_user_feedback(thread_id, action, feedback_text, include_mask, exclude_mask, iteration_dir):
    """
    action: "accept" | "continue" | "exit"
    feedback_text: 用户输入的增量描述
    include_mask: PIL Image or None
    exclude_mask: PIL Image or None
    """
    response = {"action": action}

    if action == "continue":
        # 保存用户新增的描述
        if feedback_text and feedback_text.strip():
            response["incremental_description"] = feedback_text.strip()

        # 保存用户画笔标注
        if include_mask is not None:
            include_path = iteration_dir / "user_include_mask.png"
            include_mask.save(include_path)
            response["include_mask_path"] = str(include_path)

        if exclude_mask is not None:
            exclude_path = iteration_dir / "user_exclude_mask.png"
            exclude_mask.save(exclude_path)
            response["exclude_mask_path"] = str(exclude_path)

    elif action == "exit":
        # 收集拒绝原因（可选）
        if feedback_text and feedback_text.strip():
            response["rejection_reason"] = feedback_text.strip()

    return resume_agent_graph(thread_id, response)
```

---

#### 1.2 Graph 层：将用户输入传递给下一轮理解

**文件**：`core/agent_graph.py`

**当前行为**：
- `wait_for_human` 收到 `{"action": "continue"}` 后，只更新 `agent_status`
- 下一轮调用 `run_agent_graph` 时，`previous_state` 只包含上一轮的 pipeline 和 quality_report

**目标行为**：
- 将用户的增量描述和画笔区域保存到 `human_response`
- 传递给下一轮的 `understand_task`，让 Qwen 知道"用户圈了这些区域，说明了这些问题"

**具体修改**：

```python
# core/agent_graph.py:472-504

def _wait_for_human(state):
    request = {
        "kind": "human_review",
        "thread_id": state.get("graph_thread_id"),
        "agent_status": state.get("agent_status"),
        "decision": state.get("decision", {}),
        "selected_candidate": state.get("selected_candidate"),
        "quality_report": state.get("quality_report", {}),
        "annotated_image_path": state.get("annotated_image_path"),
        "predicted_mask_path": state.get("predicted_mask_path"),
    }
    response = interrupt(request)
    action = str((response or {}).get("action", "continue"))

    # === 新增：保存用户的增量输入 ===
    human_feedback = {}
    if action == "continue":
        if response.get("incremental_description"):
            human_feedback["incremental_description"] = response["incremental_description"]
        if response.get("include_mask_path"):
            human_feedback["include_mask_path"] = response["include_mask_path"]
        if response.get("exclude_mask_path"):
            human_feedback["exclude_mask_path"] = response["exclude_mask_path"]

    if action == "exit" and response.get("rejection_reason"):
        human_feedback["rejection_reason"] = response["rejection_reason"]

    started = time.monotonic()
    status_by_action = {
        "accept": "accepted",
        "continue": "waiting_for_feedback",
        "exit": "exited",
    }
    result = _with_event(
        {
            **state,
            "human_response": dict(response or {}),
            "human_feedback": human_feedback,  # === 新增字段 ===
            "agent_status": status_by_action[action],
        },
        "resume_after_human",
        started,
        {"action": action, "has_feedback": bool(human_feedback)},
    )
    _write_trajectory(result)
    return result
```

---

#### 1.3 Provider 层：让 Qwen 理解用户反馈

**文件**：`providers/vision.py`

**当前行为**：
- `understand_task` 接收 `previous_context`，包含上一轮的 pipeline 和 quality_report
- 没有读取用户的增量描述和画笔区域

**目标行为**：
- 将用户的增量描述追加到 prompt：
  ```
  用户审核后的反馈：
  - 文字说明：左上角漏了一个颗粒
  - 用户圈出的必须包含区域：[显示 include mask 叠加在原图上]
  - 用户圈出的必须排除区域：[显示 exclude mask 叠加在原图上]

  请基于用户反馈，调整 pipeline 参数或增加预处理步骤。
  ```

**具体修改**：

```python
# providers/vision.py 中的 AliyunVisionProvider.understand_task

def understand_task(self, target_image_path, description, previous_context=None, reference_examples=None):
    messages = [{"role": "system", "content": UNDERSTAND_TASK_SYSTEM_PROMPT}]

    # 原图
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": _encode_image(target_image_path)}},
            {"type": "text", "text": f"任务描述：{description}"},
        ]
    })

    # === 新增：用户反馈 ===
    if previous_context and previous_context.get("human_feedback"):
        feedback = previous_context["human_feedback"]
        feedback_parts = []

        if feedback.get("incremental_description"):
            feedback_parts.append(f"用户补充说明：{feedback['incremental_description']}")

        if feedback.get("include_mask_path"):
            feedback_parts.append({
                "type": "image_url",
                "image_url": {"url": _encode_image(feedback["include_mask_path"])},
            })
            feedback_parts.append({"type": "text", "text": "↑ 用户圈出的必须包含区域（绿色）"})

        if feedback.get("exclude_mask_path"):
            feedback_parts.append({
                "type": "image_url",
                "image_url": {"url": _encode_image(feedback["exclude_mask_path"])},
            })
            feedback_parts.append({"type": "text", "text": "↑ 用户圈出的必须排除区域（红色）"})

        if feedback_parts:
            messages.append({"role": "user", "content": feedback_parts})

    # 上一轮结果（如果有）
    if previous_context and previous_context.get("previous_result_image_path"):
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _encode_image(previous_context["previous_result_image_path"])}},
                {"type": "text", "text": f"上一轮结果存在问题，用户要求修改。"},
            ]
        })

    # ... 其余逻辑保持不变
```

---

#### 1.4 Pipeline 执行层：强制应用 include/exclude 约束

**文件**：`core/agent_loop.py`

**当前行为**：
- `_apply_feedback_quality` 只检查上一轮的误检/漏检区域是否还在
- 没有强制修改 mask

**目标行为**：
- 如果用户提供了 include_mask，最终 mask 必须包含这些像素
- 如果用户提供了 exclude_mask，最终 mask 必须排除这些像素

**具体修改**：

```python
# core/agent_loop.py 中新增函数

def apply_user_constraints(predicted_mask, previous_state):
    """
    强制应用用户的 include/exclude 约束。

    Returns:
        modified_mask: 应用约束后的 mask
        constraint_report: {"included_pixels": int, "excluded_pixels": int}
    """
    if not isinstance(previous_state, dict):
        return predicted_mask, {}

    mask = np.asarray(predicted_mask, dtype=bool).copy()
    report = {}

    # 强制包含用户圈出的区域
    human_feedback = previous_state.get("human_feedback") or {}
    include_path = human_feedback.get("include_mask_path")
    if include_path and Path(include_path).exists():
        include_mask = np.asarray(Image.open(include_path).convert("L")) > 127
        if include_mask.shape == mask.shape:
            added = include_mask & ~mask
            mask |= include_mask
            report["included_pixels"] = int(np.sum(added))

    # 强制排除用户圈出的区域
    exclude_path = human_feedback.get("exclude_mask_path")
    if exclude_path and Path(exclude_path).exists():
        exclude_mask = np.asarray(Image.open(exclude_path).convert("L")) > 127
        if exclude_mask.shape == mask.shape:
            removed = exclude_mask & mask
            mask &= ~exclude_mask
            report["excluded_pixels"] = int(np.sum(removed))

    return mask, report


# 在 run_planned_agent 的执行循环中调用（第 68 行之后）

execution = execute_pipeline_sandbox(image, pipeline)
predicted_mask = execution.mask.data

# === 新增：强制应用用户约束 ===
constrained_mask, constraint_report = apply_user_constraints(predicted_mask, previous_state)
execution.mask.data = constrained_mask  # 替换原始 mask

quality = evaluate_mask_quality(constrained_mask)
quality["user_constraints"] = constraint_report  # 记录约束应用情况
```

---

### Phase 2: 增强实例图的作用（P1，重要）

**问题**：当前实例图只作为"视觉理解的参考"，不影响算法参数。

**目标**：如果用户提供的实例图已经包含标注（比如用红色轮廓圈出了缺陷），让 Agent 提取这个标注作为"目标模板"，在新图上生成类似的标注。

#### 2.1 提取实例图的标注 mask

**文件**：新增 `core/reference_extraction.py`

```python
import cv2
import numpy as np
from pathlib import Path
from PIL import Image


def extract_annotation_from_reference(reference_image_path, color_ranges=None):
    """
    从实例图中提取标注 mask。

    color_ranges: 标注颜色范围，默认检测红色、绿色、荧光色

    Returns:
        mask: bool array，标注区域为 True
        confidence: 0-1，提取置信度
    """
    if color_ranges is None:
        # 默认检测常见标注颜色（HSV 空间）
        color_ranges = [
            {"name": "red", "lower": (0, 100, 100), "upper": (10, 255, 255)},
            {"name": "green", "lower": (40, 100, 100), "upper": (80, 255, 255)},
            {"name": "cyan", "lower": (80, 100, 100), "upper": (100, 255, 255)},
        ]

    img = cv2.imread(str(reference_image_path))
    if img is None:
        return None, 0.0

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros(img.shape[:2], dtype=bool)

    for color_range in color_ranges:
        lower = np.array(color_range["lower"])
        upper = np.array(color_range["upper"])
        color_mask = cv2.inRange(hsv, lower, upper)
        mask |= (color_mask > 0)

    # 置信度：标注区域占比 0.1-5% 认为合理
    coverage = np.mean(mask)
    confidence = 1.0 if 0.001 < coverage < 0.05 else 0.5

    return mask, confidence
```

#### 2.2 将实例图的 mask 传递给 understand_task

**文件**：`core/agent_graph.py` 中的 `_prepare_inputs`

```python
def _prepare_inputs(state):
    started = time.monotonic()
    target = Path(state["target_image_path"])
    if not target.exists():
        raise FileNotFoundError(f"Image not found: {target}")

    # === 新增：提取实例图的标注 ===
    reference_masks = []
    for ref_example in state.get("reference_examples", []):
        ref_path = ref_example.get("image_path")
        if ref_path and Path(ref_path).exists():
            mask, confidence = extract_annotation_from_reference(ref_path)
            if mask is not None and confidence > 0.3:
                reference_masks.append({
                    "image_path": ref_path,
                    "mask": mask,
                    "confidence": confidence,
                })

    return _with_event(state, "prepare_inputs", started, {
        "target_image_path": str(target),
        "reference_masks": reference_masks,  # === 新增字段 ===
    })
```

#### 2.3 让 Qwen 知道实例图的标注特征

**文件**：`providers/vision.py`

在 `understand_task` 的 prompt 中补充：

```python
if reference_masks:
    for ref in reference_masks:
        stats = {
            "region_count": len(cv2.findContours((ref["mask"] * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]),
            "total_area_px": int(np.sum(ref["mask"])),
            "coverage": f"{np.mean(ref['mask']) * 100:.2f}%",
        }
        messages.append({
            "role": "user",
            "content": f"实例图标注特征：{stats['region_count']} 个区域，总面积 {stats['total_area_px']} 像素，覆盖率 {stats['coverage']}。请生成类似规模的标注。"
        })
```

---

### Phase 3: 记录拒绝原因（P2，优化）

**文件**：`core/task_store.py`

**目标**：用户点"退出"时，保存拒绝原因和失败的 pipeline，作为下次检索时的负样本过滤。

```python
def save_rejection_record(task_id, pipeline, rejection_reason, quality_report):
    """
    保存用户拒绝的算法记录。

    用途：
    1. 下次遇到类似任务时，不推荐这个 pipeline
    2. 分析失败模式，改进算子库
    """
    rejection_dir = TASK_ROOT / task_id / "rejections"
    rejection_dir.mkdir(exist_ok=True)

    record = {
        "rejected_at": datetime.now().isoformat(),
        "pipeline": pipeline,
        "quality_report": quality_report,
        "rejection_reason": rejection_reason or "用户未说明原因",
    }

    record_path = rejection_dir / f"{datetime.now():%Y%m%d_%H%M%S}_rejection.json"
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
```

**调用位置**：`core/agent_graph.py` 中的 `_wait_for_human`

```python
if action == "exit":
    if state.get("human_feedback", {}).get("rejection_reason"):
        save_rejection_record(
            task_id=state.get("graph_thread_id"),
            pipeline=state.get("pipeline"),
            rejection_reason=state["human_feedback"]["rejection_reason"],
            quality_report=state.get("quality_report"),
        )
```

---

## 实施优先级

| Phase | 功能 | 影响 | 工作量 | 优先级 |
|-------|------|------|--------|--------|
| 1.1 | UI 收集用户增量输入 | 直接影响用户体验 | 2-3 小时 | **P0** |
| 1.2 | Graph 传递用户输入 | 核心闭环 | 1 小时 | **P0** |
| 1.3 | Qwen 理解用户反馈 | 核心闭环 | 2-3 小时 | **P0** |
| 1.4 | 强制应用画笔约束 | 保证用户意图生效 | 1-2 小时 | **P0** |
| 2.1-2.3 | 实例图标注提取 | 提升 few-shot 效果 | 4-6 小时 | P1 |
| 3 | 拒绝原因记录 | 优化长期体验 | 1 小时 | P2 |

**总工作量估算**：Phase 1（P0）约 6-9 小时，Phase 2（P1）约 4-6 小时，Phase 3（P2）约 1 小时。

---

## 测试验证

### 测试用例 1：用户圈出漏检区域

1. 运行 agent，生成初版标注
2. 用户发现左上角漏了一个颗粒
3. 用户用画笔在左上角涂一个圈（include mode）
4. 用户输入文字："左上角漏了一个颗粒"
5. 点击"继续修改"

**预期结果**：
- 下一轮标注**必须包含**用户圈出的区域
- Agent 日志显示：`user_constraints: {"included_pixels": 1234}`

---

### 测试用例 2：用户排除误检区域

1. 运行 agent，生成初版标注
2. 用户发现比例尺被误标注了
3. 用户用画笔涂掉比例尺（exclude mode）
4. 用户输入文字："比例尺不要标注"
5. 点击"继续修改"

**预期结果**：
- 下一轮标注**不包含**用户涂掉的区域
- Agent 日志显示：`user_constraints: {"excluded_pixels": 5678}`

---

### 测试用例 3：提供实例图

1. 上传一张已标注的实例图（红色轮廓圈出缺陷）
2. 上传一张新的待标注图
3. 输入描述："提取类似的颗粒缺陷"

**预期结果**：
- Agent 提取实例图的标注区域（面积、数量、分布）
- 生成的 pipeline 在新图上产生类似规模的标注
- 日志显示：`reference_template: {"region_count": 3, "total_area_px": 1200}`

---

## 代码修改清单

### 必须修改（Phase 1）

- [ ] `ui/annotation_app.py`：增加"修改说明"文本框，保存 include/exclude mask
- [ ] `core/agent_graph.py`：`_wait_for_human` 保存 `human_feedback` 字段
- [ ] `core/agent_graph.py`：`_make_understand_task_node` 传递 `human_feedback`
- [ ] `providers/vision.py`：`understand_task` 读取并展示用户反馈
- [ ] `core/agent_loop.py`：新增 `apply_user_constraints` 函数
- [ ] `core/agent_loop.py`：在 pipeline 执行后调用约束应用
- [ ] `tests/test_agent_graph.py`：新增"用户反馈传递"测试

### 可选修改（Phase 2）

- [ ] `core/reference_extraction.py`：新增实例图标注提取模块
- [ ] `core/agent_graph.py`：`_prepare_inputs` 提取实例图 mask
- [ ] `providers/vision.py`：将实例图特征传递给 Qwen
- [ ] `tests/test_reference_extraction.py`：测试标注提取准确性

### 优化修改（Phase 3）

- [ ] `core/task_store.py`：新增 `save_rejection_record` 函数
- [ ] `core/agent_graph.py`：`_wait_for_human` 调用拒绝记录保存
- [ ] `core/algorithm_registry.py`：检索时过滤用户拒绝的 pipeline

---

## 验收标准

完成 Phase 1 后，系统应满足：

1. ✅ 用户点"继续修改"时，能输入文字说明和画笔标注
2. ✅ 下一轮 agent 的 prompt 中包含用户反馈
3. ✅ 用户圈出的 include 区域**必须出现**在最终 mask 中
4. ✅ 用户圈出的 exclude 区域**必须不出现**在最终 mask 中
5. ✅ `quality_report.json` 中记录 `user_constraints` 应用情况
6. ✅ 端到端测试：从初版标注 → 用户修改 → 生成符合约束的新标注

完成 Phase 2 后，系统应满足：

7. ✅ 上传包含标注的实例图时，能提取标注区域
8. ✅ Agent 生成的标注在规模（面积、数量）上接近实例图
9. ✅ `agent_trajectory.json` 中记录 `reference_template` 信息

完成 Phase 3 后，系统应满足：

10. ✅ 用户点"退出"时，能输入拒绝原因
11. ✅ 拒绝记录保存到 `workspace/tasks/<task_id>/rejections/`
12. ✅ 算法检索时，过滤掉用户在类似任务中拒绝过的 pipeline
