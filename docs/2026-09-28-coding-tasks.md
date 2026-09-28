# 代码任务说明：liangce-agent 新架构

> 写给 GLM 的详细施工说明。完成后由 Claude Code 审查。
> 基于分支：`refactor/tool-agent-v2`，新建分支 `refactor/reference-scoring` 开工。

---

## 整体原则

- 所有新文件放在 `core/` 下，除非另有说明
- 代码风格参照 `core/operators/types.py`：有 dataclass、有类型注解、无多余注释
- 注释只写代码本身无法表达的约束，不写"这行做了什么"
- 每个模块都要有对应的测试文件，放在 `tests/` 下
- 现有代码不删，只新增；对现有代码的修改仅限于本说明明确列出的部分
- 所有路径都用 `pathlib.Path`，不用字符串拼接

---

## 任务 1：`core/reference_store.py`

### 功能

存取参考掩膜。每张样本图对应一条记录，存在：

```
workspace/references/
  <image_id>/
    mask.npy        # bool 数组，shape (H, W)
    meta.json
```

### `meta.json` 字段

```json
{
  "image_id": "in_film_particle_middle_defect",
  "image_path": "data/samples/in_film_particle_middle_defect.jpg",
  "task": "标出图中膜内的颗粒缺陷",
  "target_type": "defect",
  "sam_iou_score": 0.91,
  "skip_scoring": false,
  "skip_reason": "",
  "confirmed_at": "2026-09-28T12:00:00Z",
  "confirmed_by": "user"
}
```

`target_type` 枚举值：`"defect"` 或 `"array"`。

### 接口

```python
from pathlib import Path
import numpy as np

class ReferenceStore:
    def __init__(self, root: str | Path) -> None:
        """root 不存在时自动创建。"""

    def save(
        self,
        image_id: str,
        mask: np.ndarray,    # dtype 必须可转为 bool，shape (H, W)
        meta: dict,
    ) -> Path:
        """
        写入 mask.npy 和 meta.json，返回 meta.json 的路径。
        image_id 已存在时覆盖。
        meta 必须包含：image_id, image_path, task, target_type,
                       sam_iou_score, confirmed_at, confirmed_by。
        缺少任何必填字段时 raise ValueError。
        mask 不是 2D 时 raise ValueError。
        mask 存为 bool 型。
        """

    def load(self, image_id: str) -> tuple[np.ndarray, dict]:
        """
        返回 (mask, meta)。
        mask 保证是 bool 型 ndarray。
        image_id 不存在时 raise FileNotFoundError。
        """

    def list_scoreable(self) -> list[str]:
        """返回所有 skip_scoring=false 的 image_id，按字母序排序。"""

    def mark_skip(self, image_id: str, reason: str) -> None:
        """
        把 skip_scoring 设为 true，skip_reason 写入 reason。
        image_id 不存在时 raise FileNotFoundError。
        """
```

### 测试文件：`tests/test_reference_store.py`

覆盖以下场景：
- save 后 load，mask 和 meta 完整还原，mask dtype 是 bool
- 覆盖写入同一 image_id，读回的是新数据
- save 缺少必填字段时抛 ValueError
- save 非 2D mask 时抛 ValueError
- load 不存在的 image_id 时抛 FileNotFoundError
- mark_skip 后 load 回来 skip_scoring 是 true
- list_scoreable 只返回 skip_scoring=false 的条目
- 用 `tmp_path` fixture 隔离，不写真实 workspace

---

## 任务 2：`core/scoring.py`

### 功能

给一次算法预测结果打分，对比参考掩膜。

### 接口

```python
from dataclasses import dataclass
import numpy as np

def mask_iou(pred: np.ndarray, ref: np.ndarray) -> float:
    """
    两个 bool 掩膜的 IoU。
    两个都是空（全 False）时返回 1.0（视为"正确地标了空"）。
    一个空一个非空时返回 0.0。
    输入不是 bool 型时自动转换（非零视为 True）。
    """

def match_instances(
    pred_masks: list[np.ndarray],
    ref_masks: list[np.ndarray],
    iou_threshold: float = 0.3,
) -> dict:
    """
    用匈牙利算法对 pred 和 ref 做最优匹配，只保留 IoU >= iou_threshold 的对。
    返回：
    {
        "matched": [(pred_idx, ref_idx, iou_value), ...],
        "false_positives": [pred_idx, ...],  # pred 有，ref 没有
        "false_negatives": [ref_idx, ...],   # ref 有，pred 没有
    }
    pred_masks 或 ref_masks 为空时返回空 matched，其余填满对应列表。
    """

@dataclass
class ImageScore:
    image_id: str
    iou_mean: float           # matched 对的 IoU 均值；无匹配时 0.0
    false_positive_count: int
    false_negative_count: int
    ref_count: int            # 参考掩膜里的目标数量
    composite: float          # 见下方公式

@dataclass
class RunScore:
    image_scores: list[ImageScore]
    composite_mean: float     # 所有 ImageScore 的 composite 均值

def score_image(
    image_id: str,
    pred_masks: list[np.ndarray],
    ref_whole_mask: np.ndarray,    # 整张图的参考掩膜（所有目标合并）
) -> ImageScore:
    """
    先用 skimage.measure.label 把 ref_whole_mask 拆成实例列表，
    再调 match_instances，最后算 composite。
    """

def score_run(scores: list[ImageScore]) -> RunScore:
    """composite_mean 是所有 scores 的 composite 的均值。"""
```

### `composite` 公式

```python
fp_penalty = false_positive_count / max(ref_count, 1)
fn_penalty = false_negative_count / max(ref_count, 1)
composite  = iou_mean * max(0.0, 1.0 - fp_penalty) * max(0.0, 1.0 - fn_penalty)
```

### 匈牙利算法实现

用 `scipy.optimize.linear_sum_assignment`，cost matrix 是 `1 - iou`，iou 低于 `iou_threshold` 的对不算匹配（视为 false positive + false negative）。

### 测试文件：`tests/test_scoring.py`

覆盖以下场景：
- `mask_iou`：完全重叠返回 1.0，完全分离返回 0.0，部分重叠计算正确，两个空掩膜返回 1.0，一空一非空返回 0.0
- `match_instances`：完美匹配（1 pred 1 ref）、多目标匹配、全误检（pred 有 ref 没有）、全漏检（ref 有 pred 没有）、空列表输入
- `score_image`：单目标完美预测，composite 应接近 1.0；有误检时 composite 下降；ref_whole_mask 全空时（无目标）且 pred_masks 也空时 composite 为 1.0
- `score_run`：composite_mean 是多张图的均值

---

## 任务 3：`core/iteration_tracker.py`

### 功能

跟踪多轮迭代历史，判断何时停止。

### 接口

```python
from dataclasses import dataclass
from pathlib import Path
from core.scoring import RunScore

@dataclass
class IterationRecord:
    iteration: int           # 从 0 开始自增
    run_score: RunScore
    algorithm_spec: dict     # 算子序列，原样存，不做验证
    timestamp: str           # ISO 8601，UTC

class IterationTracker:
    def __init__(self, workspace_dir: str | Path) -> None:
        """
        历史存在 workspace_dir/iteration_history.jsonl，每行一条 JSON。
        workspace_dir 不存在时自动创建。
        文件不存在时视为空历史，不报错。
        """

    def record(self, run_score: RunScore, algorithm_spec: dict) -> IterationRecord:
        """
        追加写入一条记录，返回该记录。
        iteration 编号 = 当前历史条数（第一条是 0）。
        """

    @property
    def best(self) -> IterationRecord | None:
        """composite_mean 最高的记录；没有历史时返回 None。"""

    @property
    def history(self) -> list[IterationRecord]:
        """返回全部历史，按 iteration 升序。"""

    def should_stop(
        self,
        patience: int = 5,
        target: float = 0.85,
        max_iter: int = 100,
    ) -> tuple[bool, str]:
        """
        返回 (是否停止, 原因字符串)。
        原因字符串取值（按优先级，从高到低）：
          "target_reached"  最新记录的 composite_mean >= target
          "max_iterations"  历史条数 >= max_iter
          "no_improvement"  最近 patience 条记录，每相邻两条 composite_mean 差 < 0.01
          ""                不停止
        不足 patience 条时 "no_improvement" 不触发。
        """
```

### 序列化说明

`IterationRecord` 写入 jsonl 时，`run_score` 展开为嵌套 dict：

```json
{
  "iteration": 0,
  "composite_mean": 0.72,
  "image_scores": [
    {"image_id": "img_a", "iou_mean": 0.85, "false_positive_count": 0,
     "false_negative_count": 1, "ref_count": 3, "composite": 0.71}
  ],
  "algorithm_spec": {"pipeline": [...]},
  "timestamp": "2026-09-28T12:00:00Z"
}
```

load 时重建 `RunScore` 和 `ImageScore` dataclass。

### 测试文件：`tests/test_iteration_tracker.py`

覆盖以下场景：
- record 后 history 长度加 1，iteration 编号正确
- best 返回 composite_mean 最高的记录
- best 在历史为空时返回 None
- should_stop：target_reached（最新分数 >= target）
- should_stop：max_iterations（已达上限）
- should_stop：no_improvement（连续 patience 轮提升 < 0.01）
- should_stop：不足 patience 条时不触发 no_improvement
- should_stop：不触发时返回 ("", False) — 注意返回顺序是 (bool, str)
- 写入再重新实例化，历史能从文件正确恢复
- 用 `tmp_path` 隔离

---

## 任务 4：`tools/gen_reference.py`（开发工具，不进机台）

### 功能

独立脚本，给一张样本图生成 SAM 参考掩膜，存入 `ReferenceStore`。

仅在开发机上运行，依赖 `tmp/sam-venv`（`transformers`、`torch`、`openai` 已安装）。

### 命令行接口

```
python tools/gen_reference.py \
  --image data/samples/in_film_particle_middle_defect.jpg \
  --task "标出图中膜内的颗粒缺陷" \
  --target-type defect \
  [--workspace workspace/references] \
  [--sam-model facebook/sam-vit-base] \
  [--show]   # 生成后用 cv2.imshow 展示叠加图，等待键盘确认
```

### 流程（内部实现细节供参考，不是死规定）

1. 读图，初始化 SAM（`facebook/sam-vit-base`，优先用 MPS，否则 CPU）
2. 根据 `--target-type`：
   - `array`：走"全图候选 + 模型选编号"（参考 `tmp/select_probe2.py`）
   - `defect`：走"模型给框 + SAM 候选 + 模型挑"（参考 `tmp/box_first_probe.py`）
3. 把选出的掩膜合并成一张 bool 掩膜（多目标做 OR）
4. 计算 SAM IoU score 均值，写入 meta
5. 如果 SAM IoU score 均值 < 0.7，打印警告，`meta["skip_scoring"] = True`
6. 调 `ReferenceStore.save()` 写入
7. 如果指定了 `--show`，用 cv2 展示叠加图

### 依赖

脚本顶部加：

```python
# 需要在 tmp/sam-venv 下运行：
#   tmp/sam-venv/bin/python tools/gen_reference.py ...
# 依赖：transformers, torch, openai（已在 tmp/sam-venv 里安装）
```

不需要写测试（这是手动工具）。

---

## 任务 5：修改 `core/agent_workflow.py`——暂缓

**本次不动 `core/agent_workflow.py`。**

原因：任务 1–4 完成后先跑通测试，确认 `ReferenceStore`、`scoring`、`IterationTracker` 三个模块的接口稳定，再动工作流。避免接口还没稳定就改图，白做工。

---

## 关键注意事项

**1. `tools/gen_reference.py` 里 SAM 的 import 放在函数内部，不能在模块顶层**

该脚本依赖 `transformers`、`torch`，这些包只在 `tmp/sam-venv` 里有，主 venv 没有。
如果把 import 写在模块顶层，在主 venv 里运行 `python tools/gen_reference.py --help` 就会立刻报 `ModuleNotFoundError`。

正确做法：

```python
# 顶层只 import 标准库和已在主 venv 里的包（pathlib, argparse, numpy, cv2）
import argparse
from pathlib import Path
import numpy as np

def _load_sam(model_name: str):
    # SAM 和 torch 的 import 放在这里
    import torch
    from transformers import SamModel, SamProcessor
    ...
```

**2. 任务 5（改 `core/agent_workflow.py`）本次不动**

等任务 1–4 的测试全部通过、接口稳定后，再另开一个任务改工作流。
本次 PR 里不应该有对 `core/agent_workflow.py` 的任何修改。

**3. `score_image` 里参考掩膜需要先拆成实例列表再匹配**

`ReferenceStore` 存的是整张图所有目标合并后的一个 bool 掩膜（多目标做 OR）。
`score_image` 收到的是这个合并掩膜，必须先用 `skimage.measure.label` + `skimage.measure.find_contours` 或直接用连通域分析拆成单个实例的掩膜列表，再传给 `match_instances`。
不能把整张合并掩膜当成"一个目标"直接和 pred_masks 里的每一个比 IoU——否则多目标场景的 false positive / false negative 计算会完全错误。

---

## 验收标准

1. `pytest tests/test_reference_store.py tests/test_scoring.py tests/test_iteration_tracker.py` 全部通过
2. 没有引入新的第三方依赖（`scipy` 和 `skimage` 已在 `requirements.txt`，可以用；`numpy` 同）
3. `tools/gen_reference.py --help` 能正常打印帮助，不报 import 错误（在主 venv 里运行）
4. `core/agent_workflow.py` 没有任何改动

---

## 文件清单

新增文件：

```
core/reference_store.py
core/scoring.py
core/iteration_tracker.py
tools/__init__.py          # 空文件，让 tools/ 成为 package（如果还不存在）
tools/gen_reference.py
tests/test_reference_store.py
tests/test_scoring.py
tests/test_iteration_tracker.py
```

不改动：`core/agent_workflow.py`、`core/operators/`、`skills/`、`agent_types.py`。
