# 代码任务说明（第二批）：agent_workflow 替换

> 第一批任务（`core/reference_store.py`、`core/scoring.py`、`core/iteration_tracker.py`、`tools/gen_reference.py`）已完成并通过审查。
> 本批任务在同一分支 `refactor/reference-scoring` 继续，完成后由 Claude Code 审查。

---

## 前置修复（先做，再开始主任务）

### 修复 A：`IterationTracker.record` 的 TOCTOU 问题

**文件：** `core/iteration_tracker.py`

**问题：** `record` 方法先调 `len(self.history)`（读文件一次），再 `open("a")`（写文件一次）。两次操作之间如果有另一个进程也在写，会产生重复的 iteration 编号。

**修法：** 把计数和写入放在同一个 `with` 块里，用文件内实际行数确定 iteration 编号：

```python
def record(self, run_score: RunScore, algorithm_spec: dict) -> IterationRecord:
    with self.history_path.open("a", encoding="utf-8") as handle:
        # 在同一个 open 调用内数行数，避免 TOCTOU
        handle.seek(0)
        iteration = sum(1 for line in handle if line.strip())
        record = IterationRecord(
            iteration=iteration,
            run_score=run_score,
            algorithm_spec=algorithm_spec,
            timestamp=_utc_now_iso(),
        )
        handle.seek(0, 2)  # 回到末尾再追加
        handle.write(json.dumps(_record_to_json(record), ensure_ascii=False) + "\n")
    return record
```

注意：`open("a")` 模式下 `seek(0)` 可以读，但写入始终在末尾。验证修复后，原来的 46 个测试仍然全部通过。

### 修复 B：`should_stop` 中 `no_improvement` 语义澄清

**文件：** `core/iteration_tracker.py`

**问题：** 当前 `delta < 0.01` 会把分数下降（delta 为负）也判为"没有提升"，语义不明确。

**修法：** 改为 `abs(delta) < NO_IMPROVEMENT_DELTA`，并在 docstring 里说清楚：

```python
NO_IMPROVEMENT_DELTA = 0.01  # 相邻轮次 composite_mean 变化量（含下降）小于此值视为无改善
```

docstring 里加一行：
```
"no_improvement"  最近 patience 轮，每相邻两条 composite_mean 变化的绝对值 < 0.01（含下降）
```

**补测试**（加到 `tests/test_iteration_tracker.py`）：

```python
def test_should_stop_no_improvement_also_triggered_on_consistent_decline(tmp_path):
    tracker = IterationTracker(tmp_path / "run")
    for composite in (0.5, 0.495, 0.490, 0.488):   # 持续小幅下降
        _add(tracker, composite)
    assert tracker.should_stop(patience=4) == (True, "no_improvement")
```

---

## 主任务：替换 `core/agent_workflow.py`

### 目标

用新图替换 v2 图。新图以 IoU 打分为核心，去掉 `quality_review`、`review_evidence`、`validate_submission` 节点，增加 `gen_reference`、`iterate`、`score`、`promote` 节点。

**设计图（见 `docs/2026-09-28-new-agent-design.md` "新的图结构"一节）：**

```
START → prepare → [has_reference?]
    ├── 否 → gen_reference → human_gate（等用户确认参考掩膜）→ iterate
    └── 是 ──────────────────────────────────────────────────→ iterate
iterate → score → [improved?]
    ├── 是 → promote → iterate
    └── 否 → iterate
[should_stop?] → human_gate（展示最优结果）→ finish → END
```

### 状态类型定义

新建 `core/workflow_state.py`，内容：

```python
from dataclasses import dataclass, field
from typing import Annotated
from operator import or_ as dict_merge

import numpy as np

from core.scoring import RunScore


@dataclass
class AlgorithmSpec:
    pipeline: list[dict]   # 算子序列，每个 dict 有 "op" 和参数键值对
    notes: str = ""


@dataclass
class WorkflowState:
    # 输入
    image_paths: list[str] = field(default_factory=list)
    task: str = ""
    target_type: str = "defect"       # "defect" | "array"

    # 参考掩膜
    reference_confirmed: bool = False  # 用户已确认参考掩膜
    pending_reference_image_id: str = ""  # 等待用户确认的那张图

    # 迭代状态
    current_spec: AlgorithmSpec | None = None
    best_spec: AlgorithmSpec | None = None
    best_score: float = 0.0
    iteration: int = 0

    # 路由
    next_node: str = ""               # 由各节点设置，route() 读取
    stop_reason: str = ""             # "target_reached" | "no_improvement" | "max_iterations" | ""

    # 用户交互
    human_message: str = ""           # 展示给用户的消息
    user_feedback: str = ""           # 用户上一轮的反馈文字
```

状态用 `Annotated[WorkflowState, _replace]` 包装，`_replace` 是 `lambda prev, cur: cur`（全量替换，不合并）。

### 节点说明

#### `prepare`

- 从输入读取 `image_paths`、`task`、`target_type`
- 调 `ReferenceStore(workspace).list_scoreable()`，判断是否已有参考掩膜
- 有参考掩膜（>= 1 张）→ `next_node = "iterate"`
- 没有 → `next_node = "gen_reference"`

#### `gen_reference`

- 取第一张还没有参考掩膜的图
- 调视觉模型（用 `providers/` 下现有的 provider，不要自己重新初始化 OpenAI client）生成粗框
- 调 SAM（`SamSession`，从 `tools/gen_reference.py` 里提取成 `core/sam_session.py`，见下方说明）生成掩膜
- 把叠加图和提示文字写入 `human_message`，设 `next_node = "human_gate"`
- 用户确认后（`user_feedback` 不为空），调 `ReferenceStore.save()` 写入
- 如果用户说"不对"/"这里漏了"，解析反馈并用 SAM 负点/正点重新生成，再次发给用户确认

#### `iterate`

- 如果没有 `current_spec`：让模型根据 `task` 和 `target_type` 提出一个初始算子序列
- 如果有 `current_spec` 和上一轮的分数反馈：告诉模型哪里多检/漏检/边界偏差，让它提出修改方案
- 模型输出算子序列，写入 `current_spec`
- `next_node = "score"`

算子序列的格式：

```json
{"pipeline": [
  {"op": "local_contrast", "clip_limit": 0.02, "tile_grid": 8},
  {"op": "bright_threshold", "sensitivity": 1.6, "min_area_px": 30}
]}
```

模型可以使用的算子名称来自 `core/operators/defect.py` 和 `core/operators/image.py`。把可用算子名称和参数范围列入系统提示，不要让模型自由发明算子名。

#### `score`

- 从 `ReferenceStore` 加载所有 `list_scoreable()` 返回的参考掩膜
- 对每张图，用 `current_spec.pipeline` 里的算子序列处理图片，得到预测掩膜列表
- 调 `score_image()`，汇总成 `RunScore`
- 调 `IterationTracker.record()` 记录
- 如果 `RunScore.composite_mean > best_score` → `next_node = "promote"`
- 否则 → 检查 `should_stop()`，`True` → `next_node = "human_gate"`；`False` → `next_node = "iterate"`

#### `promote`

- 更新 `best_spec = current_spec`，`best_score = RunScore.composite_mean`
- 再检查 `should_stop()`，`True` → `next_node = "human_gate"`；`False` → `next_node = "iterate"`

#### `human_gate`

- `stop_reason` 不为空时：展示 `best_spec` 和最终分数，等用户决定是继续还是接受
- `pending_reference_image_id` 不为空时：展示参考掩膜叠加图，等用户确认
- 用户回复写入 `user_feedback`，继续路由

#### `finish`

- 把 `best_spec` 序列化为 JSON，写入 `workspace/outputs/<run_id>/algorithm.json`
- 写入 `workspace/outputs/<run_id>/score.json`（`RunScore` 序列化）
- 返回最终状态

### `core/sam_session.py`（从 `tools/gen_reference.py` 提取）

把 `tools/gen_reference.py` 里的 `SamSession` 类、`_grid_candidates`、`_mask_from_point`、`_select_array_masks`、`_select_defect_masks` 移到 `core/sam_session.py`，`tools/gen_reference.py` 改为从 `core.sam_session` import。

`core/sam_session.py` 顶部加注释：

```python
# SAM 依赖（transformers, torch）只在 tmp/sam-venv 里有，不在主 venv 里。
# 所有 import 都在方法内部，主 venv 下 import 本文件不会报错。
```

### 对现有代码的改动范围

| 文件 | 改动 |
|------|------|
| `core/agent_workflow.py` | 用新图全量替换，保留文件名 |
| `core/workflow_state.py` | 新建 |
| `core/sam_session.py` | 新建（从 `tools/gen_reference.py` 提取） |
| `tools/gen_reference.py` | 改为从 `core.sam_session` import，其余不动 |
| `core/orchestration_runtime.py` | 删除 `quality_review`、`review_evidence`、`validate_submission` 相关方法和常量；其余不动 |

不动：`core/operators/`、`core/graph_nodes.py`、`skills/`、`agent_types.py`、`providers/`。

### 测试文件

新建 `tests/test_agent_workflow.py`，覆盖以下场景：

- `prepare` 节点：有参考掩膜时 `next_node = "iterate"`，无参考掩膜时 `next_node = "gen_reference"`
- `score` 节点：`composite_mean` 提升时 `next_node = "promote"`，不提升且未触发停止时 `next_node = "iterate"`，不提升且触发停止时 `next_node = "human_gate"`
- `promote` 节点：更新 `best_spec` 和 `best_score`
- `should_stop` 路由的三种触发条件通过图跑通（target_reached、no_improvement、max_iterations）
- 用 `unittest.mock` mock 掉 SAM 调用和视觉模型调用，不做真实推理

---

## 关键注意事项

**1. SAM import 仍然只放在方法内部**

`core/sam_session.py` 里的 `torch`、`transformers` import 必须放在方法内部，不能放模块顶层。原因同第一批任务说明里的关键点 1：主 venv 里没有这两个包。

**2. 不要改 `providers/` 下的视觉模型调用方式**

`gen_reference` 节点调视觉模型时，用 `providers/` 下现有的 provider，不要自己重新初始化 OpenAI client。现有调用方式参考 `core/agent_workflow.py` 里的 `ToolAgentRuntime`。

**3. `core/orchestration_runtime.py` 只删不改**

只删除 `quality_review`、`review_evidence`、`validate_submission` 相关方法。其余代码（预算管理、ActionStore、RunLimits 等）一行不动——后续可能还用得到。

**4. `WorkflowState` 用全量替换，不用 LangGraph 的 `Annotated` 合并**

LangGraph 的 `Annotated` 默认会合并 state 字段（比如列表追加）。`WorkflowState` 的每个字段都是整体替换的，用 `lambda prev, cur: cur` 作为 reducer，避免列表意外追加。

---

## 验收标准

1. `pytest tests/` 全部通过（含第一批的 46 个 + 修复 A/B 的新测试 + `test_agent_workflow.py`）
2. `tools/gen_reference.py --help` 在主 venv 下正常运行
3. `core/agent_workflow.py` 不再包含 `quality_review`、`review_evidence`、`validate_submission` 字样
4. `core/orchestration_runtime.py` 不再包含上述三个字样，其余方法签名未变

---

## 文件清单

新增：
```
core/workflow_state.py
core/sam_session.py
tests/test_agent_workflow.py
```

修改：
```
core/agent_workflow.py       （全量替换）
core/orchestration_runtime.py（删除三个已废弃节点的方法）
tools/gen_reference.py       （改 import，其余不动）
core/iteration_tracker.py    （修复 A 和 B）
tests/test_iteration_tracker.py（补 B 的测试）
```
