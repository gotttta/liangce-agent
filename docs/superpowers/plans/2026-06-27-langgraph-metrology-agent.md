# LangGraph Metrology Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a LangGraph-based, Codex-like DRAM/SEM defect metrology agent that uses an Alibaba Cloud multimodal model to generate segmentation strategies, executes reproducible local image algorithms, and supports brush-plus-text feedback loops.

**Architecture:** Keep the existing lightweight Python project and custom HTTP UI. Add focused modules for strategy/state types, evaluation metrics, vision providers, and LangGraph orchestration; preserve the existing CLI and `/api/run` entry points while routing them through the new graph.

**Tech Stack:** Python, NumPy, Pillow, OpenCV/skimage baseline segmentation, LangGraph, OpenAI-compatible Alibaba Cloud API client, pytest, custom `http.server` UI.

---

## File Structure

- `requirements.txt`: add runtime and test dependencies.
- `agent_types.py`: define `AgentState`, strategy defaults, and validation helpers.
- `providers/__init__.py`: provider package marker.
- `providers/vision.py`: mock provider and Alibaba multimodal provider.
- `graph_workflow.py`: LangGraph nodes, graph builder, and graph runner.
- `core/segmentation.py`: keep existing `segment_anomalies`; add strategy-aware segmentation.
- `core/measurement/area.py`: add `area_ratio` to the summary.
- `core/measurement/evaluation.py`: compare predicted mask with optional reference annotation.
- `agent.py`: preserve CLI API and call the LangGraph runner.
- `ui/app.py`: evolve the existing chat UI API for reference annotation and feedback.
- `tests/`: pytest coverage for metrics, strategy defaults, providers, segmentation, and graph flow.

The existing `main.py`, `README.md`, and `docs/design.md` remain stable unless a task explicitly updates them.

---

### Task 1: Dependencies And Measurement Baseline

**Files:**
- Modify: `requirements.txt`
- Modify: `core/measurement/area.py`
- Create: `tests/test_measurement_area.py`

- [ ] **Step 1: Add dependencies**

Replace `requirements.txt` with:

```text
numpy
Pillow
opencv-python-headless
scikit-image
langgraph
openai
pytest
```

- [ ] **Step 2: Write the failing measurement test**

Create `tests/test_measurement_area.py`:

```python
import numpy as np

from core.measurement.area import measure_components


def test_measure_components_reports_count_area_bbox_and_area_ratio():
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:3, 2:5] = True
    mask[6:8, 7:9] = True

    result = measure_components(mask, min_area=1, unit="pixel")

    assert result["summary"] == {
        "count": 2,
        "total_area": 10,
        "unit": "pixel",
        "area_ratio": 0.1,
    }
    assert result["results"][0]["bbox"] == [2, 1, 3, 2]
    assert result["results"][0]["area"] == 6
    assert result["results"][1]["bbox"] == [7, 6, 2, 2]
    assert result["results"][1]["area"] == 4


def test_measure_components_filters_small_noise():
    mask = np.zeros((8, 8), dtype=bool)
    mask[1, 1] = True
    mask[3:6, 3:6] = True

    result = measure_components(mask, min_area=4, unit="pixel")

    assert result["summary"]["count"] == 1
    assert result["summary"]["total_area"] == 9
    assert result["summary"]["area_ratio"] == round(9 / 64, 6)
```

- [ ] **Step 3: Run the test and verify it fails**

Run:

```bash
pytest tests/test_measurement_area.py -v
```

Expected: `test_measure_components_reports_count_area_bbox_and_area_ratio` fails because `area_ratio` is missing from `summary`.

- [ ] **Step 4: Implement the minimal measurement change**

In `core/measurement/area.py`, change the returned summary block in `measure_components` to:

```python
    total_area = int(sum(item["area"] for item in results))
    image_area = int(mask.size)
    area_ratio = round(total_area / image_area, 6) if image_area else 0.0
    return {
        "results": results,
        "summary": {
            "count": len(results),
            "total_area": total_area,
            "unit": unit,
            "area_ratio": area_ratio,
        },
    }
```

- [ ] **Step 5: Run the test and verify it passes**

Run:

```bash
pytest tests/test_measurement_area.py -v
```

Expected: both tests pass.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt core/measurement/area.py tests/test_measurement_area.py
git commit -m "test: cover area measurement summary"
```

---

### Task 2: Reference Annotation Evaluation Metrics

**Files:**
- Create: `core/measurement/evaluation.py`
- Create: `tests/test_evaluation_metrics.py`

- [ ] **Step 1: Write the failing evaluation tests**

Create `tests/test_evaluation_metrics.py`:

```python
import numpy as np

from core.measurement.evaluation import evaluate_prediction


def test_evaluate_prediction_computes_iou_count_error_and_area_error():
    predicted = np.zeros((6, 6), dtype=bool)
    predicted[1:3, 1:3] = True
    predicted[4:6, 4:6] = True

    reference = np.zeros((6, 6), dtype=bool)
    reference[1:3, 1:3] = True
    reference[0:2, 4:6] = True

    metrics = evaluate_prediction(predicted, reference)

    assert metrics["status"] == "ok"
    assert metrics["iou"] == round(4 / 12, 6)
    assert metrics["predicted_count"] == 2
    assert metrics["reference_count"] == 2
    assert metrics["count_error"] == 0
    assert metrics["predicted_area"] == 8
    assert metrics["reference_area"] == 8
    assert metrics["area_error"] == 0


def test_evaluate_prediction_rejects_shape_mismatch():
    predicted = np.zeros((4, 4), dtype=bool)
    reference = np.zeros((5, 4), dtype=bool)

    metrics = evaluate_prediction(predicted, reference)

    assert metrics["status"] == "invalid"
    assert metrics["reason"] == "shape_mismatch"
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
pytest tests/test_evaluation_metrics.py -v
```

Expected: import fails because `core.measurement.evaluation` does not exist.

- [ ] **Step 3: Implement evaluation metrics**

Create `core/measurement/evaluation.py`:

```python
import numpy as np

from core.measurement.area import measure_components


def evaluate_prediction(predicted_mask, reference_mask, min_area=1):
    predicted = np.asarray(predicted_mask, dtype=bool)
    reference = np.asarray(reference_mask, dtype=bool)

    if predicted.shape != reference.shape:
        return {
            "status": "invalid",
            "reason": "shape_mismatch",
            "predicted_shape": list(predicted.shape),
            "reference_shape": list(reference.shape),
        }

    intersection = int(np.count_nonzero(predicted & reference))
    union = int(np.count_nonzero(predicted | reference))
    predicted_area = int(np.count_nonzero(predicted))
    reference_area = int(np.count_nonzero(reference))
    predicted_count = measure_components(predicted, min_area=min_area)["summary"]["count"]
    reference_count = measure_components(reference, min_area=min_area)["summary"]["count"]

    return {
        "status": "ok",
        "iou": round(intersection / union, 6) if union else 1.0,
        "predicted_count": predicted_count,
        "reference_count": reference_count,
        "count_error": predicted_count - reference_count,
        "predicted_area": predicted_area,
        "reference_area": reference_area,
        "area_error": predicted_area - reference_area,
    }
```

- [ ] **Step 4: Run the tests and verify they pass**

Run:

```bash
pytest tests/test_evaluation_metrics.py -v
```

Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add core/measurement/evaluation.py tests/test_evaluation_metrics.py
git commit -m "feat: add reference annotation metrics"
```

---

### Task 3: Strategy And Agent State Types

**Files:**
- Create: `agent_types.py`
- Create: `tests/test_agent_types.py`

- [ ] **Step 1: Write the failing strategy tests**

Create `tests/test_agent_types.py`:

```python
from agent_types import default_strategy, normalize_strategy


def test_default_strategy_is_area_count_bright_threshold():
    strategy = default_strategy()

    assert strategy["measurement_type"] == "area_count"
    assert strategy["visual_observation"]["polarity"] == "bright_on_dark"
    assert strategy["segmentation"]["method"] == "bright_threshold"
    assert strategy["segmentation"]["min_area_px"] == 20
    assert strategy["measurement"]["metrics"] == ["count", "area", "bbox", "area_ratio"]


def test_normalize_strategy_fills_missing_fields_without_losing_llm_values():
    raw = {
        "defect_type": "residue",
        "segmentation": {
            "sensitivity": 2.4,
            "min_area_px": 12,
        },
        "confidence": 0.81,
    }

    strategy = normalize_strategy(raw)

    assert strategy["defect_type"] == "residue"
    assert strategy["measurement_type"] == "area_count"
    assert strategy["segmentation"]["method"] == "bright_threshold"
    assert strategy["segmentation"]["sensitivity"] == 2.4
    assert strategy["segmentation"]["min_area_px"] == 12
    assert strategy["confidence"] == 0.81
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
pytest tests/test_agent_types.py -v
```

Expected: import fails because `agent_types.py` does not exist.

- [ ] **Step 3: Implement strategy helpers and state type**

Create `agent_types.py`:

```python
from copy import deepcopy
from typing import Optional, TypedDict


class AgentState(TypedDict, total=False):
    target_image_path: str
    description: str
    reference_annotation_path: Optional[str]
    run_dir: str
    iteration: int
    strategy: dict
    predicted_mask_path: str
    annotated_image_path: str
    measurements: dict
    metrics: dict
    feedback_brush_path: Optional[str]
    feedback_text: Optional[str]
    conversation: list[dict]
    run_history: list[dict]
    status: str
    errors: list[str]


def default_strategy():
    return {
        "defect_type": "particle_residue",
        "measurement_type": "area_count",
        "visual_observation": {
            "defect_appearance": "small anomaly regions",
            "background_pattern": "unknown",
            "polarity": "bright_on_dark",
        },
        "segmentation": {
            "method": "bright_threshold",
            "sensitivity": 1.8,
            "min_area_px": 20,
            "max_area_px": None,
            "morphology": "open_then_close",
        },
        "measurement": {
            "metrics": ["count", "area", "bbox", "area_ratio"],
            "unit": "pixel",
        },
        "confidence": 0.5,
        "notes": ["Default strategy used for area/count baseline."],
    }


def normalize_strategy(raw_strategy):
    strategy = default_strategy()
    raw = raw_strategy or {}

    for key in ("defect_type", "measurement_type", "confidence", "notes"):
        if key in raw:
            strategy[key] = raw[key]

    for section in ("visual_observation", "segmentation", "measurement"):
        if isinstance(raw.get(section), dict):
            merged = deepcopy(strategy[section])
            merged.update(raw[section])
            strategy[section] = merged

    return strategy
```

- [ ] **Step 4: Run the tests and verify they pass**

Run:

```bash
pytest tests/test_agent_types.py -v
```

Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add agent_types.py tests/test_agent_types.py
git commit -m "feat: define agent strategy state"
```

---

### Task 4: Strategy-Aware Segmentation Backend

**Files:**
- Modify: `core/segmentation.py`
- Create: `tests/test_strategy_segmentation.py`

- [ ] **Step 1: Write the failing segmentation tests**

Create `tests/test_strategy_segmentation.py`:

```python
import numpy as np

from core.segmentation import segment_with_strategy


def test_segment_with_strategy_finds_bright_defect():
    image = np.full((20, 20), 50, dtype=np.float32)
    image[5:9, 6:10] = 120
    strategy = {
        "segmentation": {
            "method": "bright_threshold",
            "sensitivity": 1.0,
            "min_area_px": 5,
            "morphology": "none",
        }
    }

    mask, meta = segment_with_strategy(image, strategy)

    assert meta["selected"] == "bright"
    assert mask[6, 7]
    assert not mask[0, 0]


def test_segment_with_strategy_finds_dark_defect():
    image = np.full((20, 20), 100, dtype=np.float32)
    image[11:15, 12:16] = 10
    strategy = {
        "segmentation": {
            "method": "dark_threshold",
            "sensitivity": 1.0,
            "min_area_px": 5,
            "morphology": "none",
        }
    }

    mask, meta = segment_with_strategy(image, strategy)

    assert meta["selected"] == "dark"
    assert mask[12, 13]
    assert not mask[0, 0]
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
pytest tests/test_strategy_segmentation.py -v
```

Expected: import fails because `segment_with_strategy` does not exist.

- [ ] **Step 3: Implement `segment_with_strategy`**

Append these functions to `core/segmentation.py` while keeping `segment_anomalies` unchanged:

```python
def segment_with_strategy(image, strategy):
    segmentation = (strategy or {}).get("segmentation", {})
    method = segmentation.get("method", "auto_bright_dark_threshold")
    sensitivity = float(segmentation.get("sensitivity", 1.8))
    morphology = segmentation.get("morphology", "open_then_close")

    mean = float(np.mean(image))
    std = float(np.std(image))
    if std < 1e-6:
        return np.zeros(image.shape, dtype=bool), {
            "selected": "none",
            "mean": mean,
            "std": std,
            "threshold": None,
            "coverage": 0.0,
        }

    if method == "bright_threshold":
        threshold = mean + sensitivity * std
        mask = image >= threshold
        selected = "bright"
    elif method == "dark_threshold":
        threshold = mean - sensitivity * std
        mask = image <= threshold
        selected = "dark"
    else:
        mask, meta = segment_anomalies(image, sensitivity=sensitivity)
        mask = _apply_morphology(mask, morphology)
        meta["method"] = method
        return mask, meta

    mask = _apply_morphology(mask, morphology)
    coverage = float(np.count_nonzero(mask) / mask.size)
    if coverage > 0.35:
        mask = np.zeros(image.shape, dtype=bool)
        selected = "none"
        threshold_value = None
        coverage = 0.0
    else:
        threshold_value = float(threshold)

    return mask, {
        "selected": selected,
        "method": method,
        "mean": mean,
        "std": std,
        "threshold": threshold_value,
        "coverage": coverage,
    }


def _apply_morphology(mask, morphology):
    if morphology == "none":
        return mask.astype(bool)

    try:
        from skimage.morphology import binary_closing, binary_opening, disk
    except ImportError:
        return mask.astype(bool)

    footprint = disk(1)
    if morphology == "open":
        return binary_opening(mask, footprint).astype(bool)
    if morphology == "close":
        return binary_closing(mask, footprint).astype(bool)
    if morphology == "open_then_close":
        return binary_closing(binary_opening(mask, footprint), footprint).astype(bool)
    if morphology == "close_then_open":
        return binary_opening(binary_closing(mask, footprint), footprint).astype(bool)
    return mask.astype(bool)
```

- [ ] **Step 4: Run segmentation and existing tests**

Run:

```bash
pytest tests/test_strategy_segmentation.py tests/test_measurement_area.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add core/segmentation.py tests/test_strategy_segmentation.py
git commit -m "feat: add strategy-aware segmentation"
```

---

### Task 5: Vision Providers

**Files:**
- Create: `providers/__init__.py`
- Create: `providers/vision.py`
- Create: `tests/test_vision_provider.py`

- [ ] **Step 1: Write provider tests with no network calls**

Create `tests/test_vision_provider.py`:

```python
from providers.vision import MockVisionProvider, extract_json_object


def test_mock_provider_returns_normalized_strategy():
    provider = MockVisionProvider()

    strategy = provider.create_strategy(
        target_image_path="target.png",
        description="找亮色残留，量面积和数量",
        reference_annotation_path=None,
        previous_state=None,
    )

    assert strategy["measurement_type"] == "area_count"
    assert strategy["segmentation"]["method"] == "bright_threshold"
    assert strategy["segmentation"]["min_area_px"] == 20


def test_extract_json_object_from_markdown_response():
    text = '观察如下：```json\\n{"defect_type": "residue", "confidence": 0.8}\\n```'

    parsed = extract_json_object(text)

    assert parsed == {"defect_type": "residue", "confidence": 0.8}
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
pytest tests/test_vision_provider.py -v
```

Expected: import fails because `providers.vision` does not exist.

- [ ] **Step 3: Implement providers**

Create `providers/__init__.py`:

```python
"""Vision strategy providers for the metrology agent."""
```

Create `providers/vision.py`:

```python
import base64
import json
import os
from pathlib import Path

from agent_types import normalize_strategy


class MockVisionProvider:
    def create_strategy(
        self,
        target_image_path,
        description,
        reference_annotation_path=None,
        previous_state=None,
    ):
        text = (description or "").lower()
        method = "dark_threshold" if any(word in text for word in ["暗", "dark", "black"]) else "bright_threshold"
        raw = {
            "defect_type": "particle_residue",
            "measurement_type": "area_count",
            "visual_observation": {
                "defect_appearance": "small anomaly regions inferred from user description",
                "background_pattern": "unknown",
                "polarity": "dark_on_bright" if method == "dark_threshold" else "bright_on_dark",
            },
            "segmentation": {
                "method": method,
                "sensitivity": 1.8,
                "min_area_px": 20,
                "morphology": "open_then_close",
            },
            "confidence": 0.5,
            "notes": ["Mock provider used; no remote multimodal model was called."],
        }
        return normalize_strategy(raw)


class AliyunVisionProvider:
    def __init__(self, api_key=None, base_url=None, model=None):
        self.api_key = api_key or os.getenv("ALIYUN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = base_url or os.getenv("ALIYUN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model = model or os.getenv("ALIYUN_VISION_MODEL", "qwen-vl-max-latest")
        if not self.api_key:
            raise ValueError("Missing ALIYUN_API_KEY or DASHSCOPE_API_KEY")

    def create_strategy(
        self,
        target_image_path,
        description,
        reference_annotation_path=None,
        previous_state=None,
    ):
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        messages = build_strategy_messages(
            target_image_path=target_image_path,
            description=description,
            reference_annotation_path=reference_annotation_path,
            previous_state=previous_state,
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
        )
        content = response.choices[0].message.content
        return normalize_strategy(extract_json_object(content))


def build_strategy_messages(target_image_path, description, reference_annotation_path=None, previous_state=None):
    content = [
        {
            "type": "text",
            "text": (
                "你是DRAM/SEM缺陷量测agent。只输出JSON，不要输出Markdown。"
                "任务是为本地OpenCV/skimage分割生成结构化strategy。"
                "LLM不直接生成mask。measurement_type固定为area_count。"
                f"用户描述：{description or ''}"
            ),
        },
        image_content(target_image_path, "target image"),
    ]
    if reference_annotation_path:
        content.append(image_content(reference_annotation_path, "same-image reference annotation"))
    if previous_state:
        content.append({
            "type": "text",
            "text": "上一轮状态：" + json.dumps(previous_state, ensure_ascii=False)[:6000],
        })

    return [{"role": "user", "content": content}]


def image_content(path, label):
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{encode_image(path)}",
            "detail": "high",
        },
        "text": label,
    }


def encode_image(path):
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def extract_json_object(text):
    cleaned = (text or "").strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                return json.loads(candidate)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Provider response did not contain a JSON object")
    return json.loads(cleaned[start:end + 1])
```

- [ ] **Step 4: Run provider tests**

Run:

```bash
pytest tests/test_vision_provider.py -v
```

Expected: both tests pass without network access.

- [ ] **Step 5: Commit**

```bash
git add providers tests/test_vision_provider.py
git commit -m "feat: add multimodal strategy providers"
```

---

### Task 6: LangGraph Workflow

**Files:**
- Create: `graph_workflow.py`
- Create: `tests/test_graph_workflow.py`

- [ ] **Step 1: Write the failing graph test**

Create `tests/test_graph_workflow.py`:

```python
from pathlib import Path

import numpy as np
from PIL import Image

from graph_workflow import run_graph
from providers.vision import MockVisionProvider


def test_run_graph_writes_iteration_outputs(tmp_path):
    target = tmp_path / "target.png"
    image = np.full((30, 30), 40, dtype=np.uint8)
    image[10:16, 12:18] = 180
    Image.fromarray(image, mode="L").save(target)

    state = run_graph(
        target_image_path=target,
        description="找亮色残留，量面积和数量",
        output_root=tmp_path / "outputs",
        provider=MockVisionProvider(),
    )

    assert state["status"] == "ok"
    assert state["iteration"] == 0
    assert Path(state["predicted_mask_path"]).exists()
    assert Path(state["annotated_image_path"]).exists()
    assert state["measurements"]["summary"]["count"] == 1
    assert state["conversation"]
```

- [ ] **Step 2: Run the graph test and verify it fails**

Run:

```bash
pytest tests/test_graph_workflow.py -v
```

Expected: import fails because `graph_workflow.py` does not exist.

- [ ] **Step 3: Implement the graph workflow**

Create `graph_workflow.py`:

```python
import json
from datetime import datetime
from pathlib import Path

from langgraph.graph import END, StateGraph

from agent_types import AgentState
from core.measurement.area import measure_components
from core.measurement.evaluation import evaluate_prediction
from core.preprocessing import load_grayscale
from core.segmentation import segment_with_strategy
from core.visualization import save_annotated_image, save_mask_image
from providers.vision import MockVisionProvider


def run_graph(
    target_image_path,
    description,
    output_root="outputs",
    reference_annotation_path=None,
    feedback_brush_path=None,
    feedback_text=None,
    provider=None,
    unit="pixel",
    max_iterations=2,
    existing_run_dir=None,
):
    graph = build_graph(provider=provider or MockVisionProvider(), max_iterations=max_iterations)
    initial_state = {
        "target_image_path": str(target_image_path),
        "description": description,
        "reference_annotation_path": str(reference_annotation_path) if reference_annotation_path else None,
        "feedback_brush_path": str(feedback_brush_path) if feedback_brush_path else None,
        "feedback_text": feedback_text,
        "run_dir": str(existing_run_dir or make_run_dir(output_root)),
        "iteration": 0,
        "conversation": [],
        "run_history": [],
        "status": "pending",
        "errors": [],
        "unit": unit,
    }
    return graph.invoke(initial_state)


def build_graph(provider, max_iterations=2):
    workflow = StateGraph(AgentState)
    workflow.add_node("prepare_inputs", prepare_inputs)
    workflow.add_node("vision_strategy", make_vision_strategy_node(provider))
    workflow.add_node("segment_defects", segment_defects)
    workflow.add_node("measure_defects", measure_defects)
    workflow.add_node("render_outputs", render_outputs)
    workflow.add_node("prepare_feedback_iteration", prepare_feedback_iteration)

    workflow.set_entry_point("prepare_inputs")
    workflow.add_edge("prepare_inputs", "vision_strategy")
    workflow.add_edge("vision_strategy", "segment_defects")
    workflow.add_edge("segment_defects", "measure_defects")
    workflow.add_edge("measure_defects", "render_outputs")
    workflow.add_conditional_edges(
        "render_outputs",
        lambda state: route_after_render(state, max_iterations=max_iterations),
        {"rerun": "prepare_feedback_iteration", "end": END},
    )
    workflow.add_edge("prepare_feedback_iteration", "vision_strategy")
    return workflow.compile()


def make_run_dir(output_root):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(output_root) / f"{timestamp}_agent"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def prepare_inputs(state):
    target = Path(state["target_image_path"])
    if not target.exists():
        raise FileNotFoundError(f"Image not found: {target}")
    if not state.get("description"):
        raise ValueError("缺少缺陷描述")
    Path(state["run_dir"]).mkdir(parents=True, exist_ok=True)
    state["conversation"].append({"role": "assistant", "content": "我先看图并生成一版量测策略。"})
    return state


def make_vision_strategy_node(provider):
    def vision_strategy(state):
        previous_state = {
            "strategy": state.get("strategy"),
            "measurements": state.get("measurements"),
            "feedback_text": state.get("feedback_text"),
            "feedback_brush_path": state.get("feedback_brush_path"),
        }
        strategy = provider.create_strategy(
            target_image_path=state["target_image_path"],
            description=state["description"],
            reference_annotation_path=state.get("reference_annotation_path"),
            previous_state=previous_state if state.get("iteration", 0) > 0 else None,
        )
        state["strategy"] = strategy
        state["conversation"].append({
            "role": "assistant",
            "content": f"我会使用{strategy['segmentation']['method']}，最小面积阈值为{strategy['segmentation']['min_area_px']} px。",
        })
        return state
    return vision_strategy


def segment_defects(state):
    image = load_grayscale(state["target_image_path"])
    mask, meta = segment_with_strategy(image, state["strategy"])
    iteration_dir = current_iteration_dir(state)
    mask_path = iteration_dir / "mask.png"
    save_mask_image(mask, mask_path)
    state["predicted_mask_path"] = str(mask_path)
    state["segmentation"] = meta
    state["_mask_array"] = mask
    return state


def measure_defects(state):
    min_area = int(state["strategy"]["segmentation"].get("min_area_px", 20))
    unit = state.get("unit") or state["strategy"]["measurement"].get("unit", "pixel")
    measurements = measure_components(state["_mask_array"], min_area=min_area, unit=unit)
    state["measurements"] = measurements

    reference_path = state.get("reference_annotation_path")
    if reference_path:
        reference = load_grayscale(reference_path) > 0
        state["metrics"] = evaluate_prediction(state["_mask_array"], reference, min_area=min_area)
    else:
        state["metrics"] = {"status": "skipped", "reason": "no_reference_annotation"}
    return state


def render_outputs(state):
    iteration_dir = current_iteration_dir(state)
    annotated_path = iteration_dir / "result_annotated.png"
    save_annotated_image(state["target_image_path"], state["measurements"]["results"], annotated_path)
    state["annotated_image_path"] = str(annotated_path)
    state["status"] = "ok"

    write_json(iteration_dir / "strategy.json", state["strategy"])
    write_json(iteration_dir / "measurements.json", state["measurements"])
    write_json(iteration_dir / "metrics.json", state["metrics"])

    serializable = {key: value for key, value in state.items() if key != "_mask_array"}
    write_json(iteration_dir / "graph_state.json", serializable)

    summary = state["measurements"]["summary"]
    state["conversation"].append({
        "role": "assistant",
        "content": f"这一轮标出 {summary['count']} 个区域，总面积 {summary['total_area']} {summary['unit']}。",
    })
    state["run_history"].append({
        "iteration": state["iteration"],
        "directory": str(iteration_dir),
        "summary": summary,
        "metrics": state["metrics"],
    })
    state.pop("_mask_array", None)
    return state


def route_after_render(state, max_iterations):
    has_feedback = bool(state.get("feedback_text") or state.get("feedback_brush_path"))
    if has_feedback and state.get("iteration", 0) + 1 < max_iterations:
        return "rerun"
    return "end"


def prepare_feedback_iteration(state):
    state["iteration"] = state.get("iteration", 0) + 1
    state["conversation"].append({
        "role": "assistant",
        "content": "收到你的画笔和文字反馈，我会修订策略后重新跑一轮。",
    })
    return state


def current_iteration_dir(state):
    iteration_dir = Path(state["run_dir"]) / f"iteration_{state.get('iteration', 0)}"
    iteration_dir.mkdir(parents=True, exist_ok=True)
    return iteration_dir


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
```

- [ ] **Step 4: Run the graph test**

Run:

```bash
pytest tests/test_graph_workflow.py -v
```

Expected: test passes.

- [ ] **Step 5: Commit**

```bash
git add graph_workflow.py tests/test_graph_workflow.py
git commit -m "feat: add LangGraph metrology workflow"
```

---

### Task 7: Preserve CLI And Existing Agent API

**Files:**
- Modify: `agent.py`
- Create: `tests/test_agent_api.py`

- [ ] **Step 1: Write the API compatibility test**

Create `tests/test_agent_api.py`:

```python
from pathlib import Path

import numpy as np
from PIL import Image

from agent import run_from_paths


def test_run_from_paths_returns_latest_iteration_result(tmp_path):
    target = tmp_path / "target.png"
    image = np.full((24, 24), 30, dtype=np.uint8)
    image[8:13, 9:14] = 160
    Image.fromarray(image, mode="L").save(target)

    run_dir, output = run_from_paths(
        target=target,
        description="找亮色残留，统计面积和数量",
        output_root=tmp_path / "outputs",
    )

    assert Path(run_dir).exists()
    assert output["status"] == "ok"
    assert output["summary"]["count"] == 1
    assert (Path(run_dir) / "iteration_0" / "result_annotated.png").exists()
```

- [ ] **Step 2: Run the test**

Run:

```bash
pytest tests/test_agent_api.py -v
```

Expected: it fails because `agent.py` still writes the old flat output format.

- [ ] **Step 3: Modify `agent.py` to call LangGraph**

Replace `run(args)` in `agent.py` with:

```python
def run(args):
    from graph_workflow import run_graph
    from providers.vision import MockVisionProvider

    state = run_graph(
        target_image_path=args.target,
        description=args.description,
        reference_annotation_path=args.reference,
        output_root=args.output_root,
        provider=MockVisionProvider(),
        unit=args.unit,
    )
    return Path(state["run_dir"]), {
        "defect_type": state["strategy"]["defect_type"],
        "measurement_type": state["strategy"]["measurement_type"],
        "unit": state["measurements"]["summary"]["unit"],
        "status": state["status"],
        "results": state["measurements"]["results"],
        "summary": state["measurements"]["summary"],
        "segmentation": state.get("segmentation", {}),
        "metrics": state.get("metrics", {}),
        "notes": state["strategy"].get("notes", []),
        "conversation": state.get("conversation", []),
        "latest_iteration": state.get("iteration", 0),
        "latest_annotated_image": state.get("annotated_image_path"),
        "latest_mask": state.get("predicted_mask_path"),
    }
```

Keep `parse_args`, `run_from_paths`, and `main` signatures unchanged so existing commands still work.

- [ ] **Step 4: Run CLI/API tests**

Run:

```bash
pytest tests/test_agent_api.py tests/test_graph_workflow.py -v
```

Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add agent.py tests/test_agent_api.py
git commit -m "feat: route CLI through LangGraph workflow"
```

---

### Task 8: UI Backend For Reference Annotation And Feedback

**Files:**
- Modify: `ui/app.py`
- Create: `tests/test_ui_handlers.py`

- [ ] **Step 1: Extract response file mapping test**

Create `tests/test_ui_handlers.py`:

```python
from pathlib import Path

from ui.app import build_file_links


def test_build_file_links_points_to_latest_iteration():
    root = Path("/repo")
    run_dir = root / "outputs" / "run_001"
    result = {
        "latest_iteration": 1,
        "latest_annotated_image": str(run_dir / "iteration_1" / "result_annotated.png"),
        "latest_mask": str(run_dir / "iteration_1" / "mask.png"),
    }

    files = build_file_links(root, run_dir, result)

    assert files["annotated"] == "/outputs/run_001/iteration_1/result_annotated.png"
    assert files["mask"] == "/outputs/run_001/iteration_1/mask.png"
    assert files["measurements"] == "/outputs/run_001/iteration_1/measurements.json"
    assert files["strategy"] == "/outputs/run_001/iteration_1/strategy.json"
    assert files["graph_state"] == "/outputs/run_001/iteration_1/graph_state.json"
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
pytest tests/test_ui_handlers.py -v
```

Expected: import fails because `build_file_links` does not exist.

- [ ] **Step 3: Add file-link helper**

In `ui/app.py`, add:

```python
def build_file_links(root, run_dir, result):
    latest_iteration = int(result.get("latest_iteration", 0))
    iteration_dir = Path(run_dir) / f"iteration_{latest_iteration}"
    iteration_rel = iteration_dir.relative_to(root)
    return {
        "annotated": "/" + str(Path(result["latest_annotated_image"]).relative_to(root)),
        "mask": "/" + str(Path(result["latest_mask"]).relative_to(root)),
        "measurements": "/" + str(iteration_rel / "measurements.json"),
        "strategy": "/" + str(iteration_rel / "strategy.json"),
        "metrics": "/" + str(iteration_rel / "metrics.json"),
        "graph_state": "/" + str(iteration_rel / "graph_state.json"),
    }
```

Then replace the old hard-coded `files = { ... }` block in `handle_run` with:

```python
    files = build_file_links(ROOT, run_dir, result)
```

- [ ] **Step 4: Add reference annotation naming**

In `handle_run`, keep accepting the existing form field name `reference` but treat it as same-image reference annotation:

```python
    reference_path = save_upload(form, "reference", required=False)
```

No backend rename is required in this task; the frontend label changes in Task 9.

- [ ] **Step 5: Run UI helper test**

Run:

```bash
pytest tests/test_ui_handlers.py -v
```

Expected: test passes.

- [ ] **Step 6: Commit**

```bash
git add ui/app.py tests/test_ui_handlers.py
git commit -m "feat: expose latest graph outputs in UI API"
```

---

### Task 9: Codex-Like Brush Feedback UI

**Files:**
- Modify: `ui/app.py`

- [ ] **Step 1: Update upload copy**

In `render_app()`, change the reference upload tile copy from:

```html
<span>参考图</span>
<small>可选，用来说明缺陷长什么样</small>
```

to:

```html
<span>参考标注图</span>
<small>可选，同一张原图上的人工标注</small>
```

- [ ] **Step 2: Add feedback controls to result messages**

Inside `renderResultMessage(payload)`, after the downloads section, add this HTML:

```javascript
              <section class="feedback-workspace">
                <p class="label">画笔反馈</p>
                <canvas class="feedback-canvas" width="640" height="420"></canvas>
                <textarea class="feedback-text" rows="3" placeholder="例如：左下角这里漏了；右上角这些小点不要算"></textarea>
                <button type="button" class="feedback-submit">应用反馈并重跑</button>
              </section>
```

- [ ] **Step 3: Add brush initialization JavaScript**

After `timeline.appendChild(node);` in `renderResultMessage`, add:

```javascript
          initializeFeedbackCanvas(node, files.annotated, payload.run_dir);
```

Then add this function before the `form.addEventListener` block:

```javascript
        function initializeFeedbackCanvas(node, imageUrl, runDir) {
          const canvas = node.querySelector('.feedback-canvas');
          const context = canvas.getContext('2d');
          const image = new Image();
          let drawing = false;

          image.onload = () => {
            context.clearRect(0, 0, canvas.width, canvas.height);
            context.drawImage(image, 0, 0, canvas.width, canvas.height);
          };
          image.src = imageUrl;

          canvas.addEventListener('pointerdown', (event) => {
            drawing = true;
            context.beginPath();
            context.lineWidth = 8;
            context.strokeStyle = '#ff3040';
            context.lineCap = 'round';
            context.moveTo(event.offsetX, event.offsetY);
          });

          canvas.addEventListener('pointermove', (event) => {
            if (!drawing) return;
            context.lineTo(event.offsetX, event.offsetY);
            context.stroke();
          });

          canvas.addEventListener('pointerup', () => {
            drawing = false;
          });

          node.querySelector('.feedback-submit').addEventListener('click', async () => {
            const feedbackText = node.querySelector('.feedback-text').value.trim();
            const feedbackBrush = canvas.toDataURL('image/png');
            await submitFeedback(runDir, feedbackText, feedbackBrush);
          });
        }
```

- [ ] **Step 4: Add feedback submit JavaScript**

Add this function before the `form.addEventListener` block:

```javascript
        async function submitFeedback(runDir, feedbackText, feedbackBrush) {
          if (!feedbackText) {
            alert('请写一句反馈，例如“这里漏了”或“这些不要算”。');
            return;
          }

          const thinking = pushThinking();
          statusPill.textContent = '根据反馈重跑';
          statusCard.textContent = '反馈处理中';

          try {
            const response = await fetch('/api/feedback', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({run_dir: runDir, feedback_text: feedbackText, feedback_brush: feedbackBrush}),
            });
            const payload = await response.json();
            thinking.remove();
            if (!payload.ok) throw new Error(payload.error || '反馈运行失败');
            renderResultMessage(payload);
            statusPill.textContent = '反馈已应用';
            statusCard.textContent = `${payload.result.summary.count} 个区域`;
          } catch (error) {
            thinking.remove();
            alert(error.message || String(error));
            statusPill.textContent = '失败';
            statusCard.textContent = '失败';
          }
        }
```

- [ ] **Step 5: Add basic feedback CSS**

In the `<style>` block, add:

```css
    .feedback-workspace {
      margin-top: 14px;
      display: grid;
      gap: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--soft);
      padding: 12px;
    }
    .feedback-canvas {
      width: 100%;
      max-width: 640px;
      aspect-ratio: 64 / 42;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      touch-action: none;
    }
    .feedback-text {
      width: 100%;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      font: inherit;
    }
```

- [ ] **Step 6: Commit**

```bash
git add ui/app.py
git commit -m "feat: add brush feedback controls"
```

---

### Task 10: Feedback API Backend

**Files:**
- Modify: `ui/app.py`
- Modify: `graph_workflow.py`
- Create: `tests/test_feedback_payload.py`

- [ ] **Step 1: Write feedback image decoding test**

Create `tests/test_feedback_payload.py`:

```python
from pathlib import Path

from PIL import Image

from ui.app import save_feedback_brush


def test_save_feedback_brush_writes_png(tmp_path):
    data_url = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
        "AAAADUlEQVR42mP8z8BQDwAFgwJ/lU9nNwAAAABJRU5ErkJggg=="
    )

    path = save_feedback_brush(tmp_path, data_url)

    assert Path(path).exists()
    assert Image.open(path).size == (1, 1)
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
pytest tests/test_feedback_payload.py -v
```

Expected: import fails because `save_feedback_brush` does not exist.

- [ ] **Step 3: Add feedback brush saving helper**

In `ui/app.py`, add imports:

```python
import base64
```

Add:

```python
def save_feedback_brush(run_dir, data_url):
    if not data_url.startswith("data:image/png;base64,"):
        raise ValueError("feedback_brush must be a PNG data URL")
    payload = data_url.split(",", 1)[1]
    feedback_path = Path(run_dir) / "feedback_brush.png"
    feedback_path.write_bytes(base64.b64decode(payload))
    return feedback_path
```

- [ ] **Step 4: Add `/api/feedback` route**

In `AppHandler.do_POST`, replace the route check with:

```python
        if self.path == "/api/run":
            try:
                payload = handle_run(self)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=500)
                return
            self._send_json({"ok": True, **payload})
            return

        if self.path == "/api/feedback":
            try:
                payload = handle_feedback(self)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=500)
                return
            self._send_json({"ok": True, **payload})
            return

        self.send_error(404)
```

Add `handle_feedback`:

```python
def handle_feedback(handler):
    length = int(handler.headers.get("Content-Length", "0"))
    payload = json.loads(handler.rfile.read(length).decode("utf-8"))
    run_dir = (ROOT / payload["run_dir"]).resolve()
    output_root = OUTPUT_ROOT.resolve()
    if output_root not in run_dir.parents and run_dir != output_root:
        raise ValueError("Invalid run_dir")

    feedback_text = payload.get("feedback_text", "").strip()
    feedback_brush_path = save_feedback_brush(run_dir, payload.get("feedback_brush", ""))

    previous_state_path = run_dir / "iteration_0" / "graph_state.json"
    previous_state = json.loads(previous_state_path.read_text(encoding="utf-8"))

    from graph_workflow import run_graph
    from providers.vision import MockVisionProvider

    state = run_graph(
        target_image_path=previous_state["target_image_path"],
        description=previous_state["description"],
        reference_annotation_path=previous_state.get("reference_annotation_path"),
        output_root=run_dir.parent,
        existing_run_dir=run_dir,
        feedback_brush_path=feedback_brush_path,
        feedback_text=feedback_text,
        provider=MockVisionProvider(),
        unit=previous_state.get("unit", "pixel"),
        max_iterations=2,
    )
    run_dir = Path(state["run_dir"])
    result = state_to_result(state)
    return {
        "run_dir": str(run_dir.relative_to(ROOT)),
        "files": build_file_links(ROOT, run_dir, result),
        "result": result,
    }
```

- [ ] **Step 5: Extract `state_to_result`**

In `ui/app.py`, add:

```python
def state_to_result(state):
    return {
        "defect_type": state["strategy"]["defect_type"],
        "measurement_type": state["strategy"]["measurement_type"],
        "unit": state["measurements"]["summary"]["unit"],
        "status": state["status"],
        "results": state["measurements"]["results"],
        "summary": state["measurements"]["summary"],
        "segmentation": state.get("segmentation", {}),
        "metrics": state.get("metrics", {}),
        "notes": state["strategy"].get("notes", []),
        "conversation": state.get("conversation", []),
        "latest_iteration": state.get("iteration", 0),
        "latest_annotated_image": state.get("annotated_image_path"),
        "latest_mask": state.get("predicted_mask_path"),
    }
```

Then in `handle_run`, keep using the `result` returned by `run_from_paths` for now. A later cleanup can share this converter with `agent.py`.

- [ ] **Step 6: Run feedback helper test**

Run:

```bash
pytest tests/test_feedback_payload.py tests/test_ui_handlers.py -v
```

Expected: tests pass.

- [ ] **Step 7: Commit**

```bash
git add ui/app.py tests/test_feedback_payload.py
git commit -m "feat: add feedback rerun API"
```

---

### Task 11: Documentation And Manual Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README run instructions**

Add this section to `README.md` under the existing Web UI instructions:

```markdown
## LangGraph Agent Mode

The agent mode uses a LangGraph state flow:

```text
prepare_inputs -> vision_strategy -> segment_defects -> measure_defects -> render_outputs
```

By default, local development uses `MockVisionProvider`, so it does not call a remote model.
To use Alibaba Cloud's OpenAI-compatible endpoint, set:

```bash
export DASHSCOPE_API_KEY="your-api-key"
export ALIYUN_VISION_MODEL="qwen-vl-max-latest"
```

The model name is configurable because available model names can vary by account and region.

Run tests:

```bash
pytest -v
```

Run the UI:

```bash
python3 -m ui.app
```
```
```

- [ ] **Step 2: Run all tests**

Run:

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 3: Start the local UI**

Run:

```bash
python3 -m ui.app
```

Expected: server prints a local URL such as `http://127.0.0.1:7860` or the existing app port.

- [ ] **Step 4: Manual smoke test**

Use a small image with one bright defect region:

1. Upload the target image.
2. Enter `找亮色残留，量面积和数量`.
3. Run the agent.
4. Confirm an annotated image and mask appear in the chat.
5. Draw a feedback mark and enter `这里漏了`.
6. Confirm a new result message appears.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: document LangGraph agent workflow"
```

---

## Self-Review Notes

- Spec coverage: tasks cover LangGraph orchestration, Alibaba/mock provider boundary, OpenCV/skimage segmentation, area/count measurement, optional reference annotation metrics, Codex-like UI, brush-plus-text feedback, output records, and tests.
- Scope control: CD/space/bridge/contact/overlay/roughness and SAM/SAM2 execution are intentionally outside V1. The plan only reserves the segmentation backend boundary.
- Type consistency: `AgentState`, `strategy`, `reference_annotation_path`, `feedback_brush_path`, `feedback_text`, `measurements`, and `metrics` names are used consistently across tasks.
- Feedback continuity: Task 10 uses `existing_run_dir` so brush feedback appends `iteration_1` under the same run directory.
