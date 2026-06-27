# LangGraph DRAM Defect Metrology Agent Design

## 1. Goal

Build a Codex-like conversational metrology workspace for DRAM/SEM defect images.
The first version focuses on area and count measurement for particle, residue,
contamination, and bright/dark anomaly defects.

The core success criterion is annotation quality: the agent should identify and
mark defect regions accurately enough for a user to trust the area and count
measurements. The agent architecture exists to improve that marking loop, not to
add framework complexity for its own sake.

## 2. Scope

### In Scope

- Required target image upload.
- Required natural-language defect description.
- Optional same-image reference annotation.
- Alibaba Cloud multimodal LLM strategy generation.
- LangGraph orchestration for the agent state flow.
- OpenCV/skimage segmentation baseline.
- Area, count, bounding box, and area-ratio measurement.
- Optional IoU, count error, and area error when a reference annotation is
  provided.
- User feedback through brush annotation plus text description.
- Multi-round rerun after feedback.
- Codex-like conversational UI: concise assistant messages in the foreground and
  detailed experiment records in the background.
- SAM/SAM2 segmentation backend interface reserved for future work.

### Out of Scope For V1

- CD, space, bridge/open, contact, overlay, roughness, height, depth, or volume
  measurement.
- Direct LLM-generated pixel masks.
- Production-grade long-term memory.
- Full LangGraph checkpoint persistence beyond local run records.
- Synthetic SEM data generation.

## 3. Inputs And Outputs

### Inputs

The target image is required. This is the raw image that needs automatic defect
annotation and measurement.

The defect description is required. It tells the agent what kind of defect to
find and which measurement to perform.

The reference annotation is optional. In this project terminology, this means a
same-image human annotation that shows which regions should count as defects. It
can help the agent understand the target defect and can also be used as the
evaluation reference.

### Outputs

Each run writes one output directory with per-iteration records:

```text
outputs/<run_id>/
  iteration_0/
    strategy.json
    mask.png
    result_annotated.png
    measurements.json
    graph_state.json
  iteration_1/
    feedback_brush.png
    feedback_text.txt
    strategy.json
    mask.png
    result_annotated.png
    measurements.json
    graph_state.json
```

The UI shows the latest annotated image, predicted mask, area/count table,
strategy summary, optional metrics, and conversation history.

## 4. Recommended Architecture

V1 uses the effect-first Agent MVP:

1. The user uploads a target image, enters a defect description, and optionally
   uploads a same-image reference annotation.
2. A LangGraph state graph coordinates the task.
3. Alibaba Cloud multimodal LLM reads the image context and outputs a structured
   strategy.
4. OpenCV/skimage executes the strategy to produce a predicted mask.
5. The measurement layer computes count, area, bounding boxes, and area ratio.
6. The UI presents a Codex-like explanation, annotated image, mask, and results.
7. The user can draw on the result image and add feedback text.
8. LangGraph routes the feedback into a new strategy generation step and reruns
   the algorithm.

LangGraph is part of the core architecture because the project needs an
explainable human-in-the-loop agent flow. It gives the internship project a
clear agent story: multimodal strategy generation, deterministic tool execution,
measurement, rendering, and iterative feedback correction.

## 5. LangGraph State

`AgentState` is the shared graph state.

```text
target_image_path: str
description: str
reference_annotation_path: Optional[str]
run_dir: str
iteration: int
strategy: Optional[dict]
predicted_mask_path: Optional[str]
annotated_image_path: Optional[str]
measurements: Optional[dict]
metrics: Optional[dict]
feedback_brush_path: Optional[str]
feedback_text: Optional[str]
conversation: list[dict]
run_history: list[dict]
status: str
errors: list[str]
```

The foreground UI uses `conversation` for concise assistant messages. The
background records use `strategy`, `measurements`, `metrics`, `run_history`, and
`graph_state.json`.

## 6. LangGraph Nodes

### `prepare_inputs`

Validates that the target image and description exist, checks whether the
optional reference annotation is readable, creates the run directory, and
initializes iteration state.

### `vision_strategy`

Calls the Alibaba Cloud multimodal provider. On the first iteration, it uses the
target image, defect description, and optional reference annotation. On feedback
iterations, it also uses the previous annotated image, previous predicted mask,
feedback brush layer, feedback text, and previous strategy.

The node outputs a structured `DefectStrategy`. The LLM does not output the final
mask.

### `segment_defects`

Executes the strategy with the default OpenCV/skimage backend. It handles
grayscale conversion, normalization, bright/dark anomaly thresholding,
adaptive-threshold fallback, morphology, and connected-component filtering.

V1 defines a segmentation backend boundary:

```text
SegmentationBackend = opencv | sam2
```

Only `opencv` is implemented in V1. `sam2` is reserved for later boundary
refinement.

### `measure_defects`

Computes count, area, bounding boxes, and area ratio from the predicted mask. If
the optional reference annotation is present and compatible, it computes IoU,
count error, and area error.

### `render_outputs`

Writes the annotated image, predicted mask, strategy, measurements, metrics, and
graph state to the current iteration directory. It also appends concise user
facing messages to the conversation.

### `route_after_render`

Routes to `vision_strategy` when feedback brush or feedback text is present and
the maximum iteration count has not been reached. Otherwise it ends the graph.

## 7. Graph Flow

```text
START
  -> prepare_inputs
  -> vision_strategy
  -> segment_defects
  -> measure_defects
  -> render_outputs
  -> route_after_render
      -> END
      -> vision_strategy
```

Feedback is modeled as a state update rather than a separate standalone tool.
This keeps the user interaction and experiment history connected.

## 8. Defect Strategy Schema

The LLM produces JSON similar to:

```json
{
  "defect_type": "particle_residue",
  "measurement_type": "area_count",
  "visual_observation": {
    "defect_appearance": "small bright particles on darker background",
    "background_pattern": "regular array texture",
    "polarity": "bright_on_dark"
  },
  "segmentation": {
    "method": "bright_threshold",
    "sensitivity": 1.7,
    "min_area_px": 25,
    "max_area_px": null,
    "morphology": "open_then_close"
  },
  "measurement": {
    "metrics": ["count", "area", "bbox", "area_ratio"],
    "unit": "pixel"
  },
  "confidence": 0.72,
  "notes": ["Ignore tiny isolated noise below min_area_px."]
}
```

The schema should be validated and normalized before execution. If the LLM output
is missing fields, the system fills conservative defaults and records a warning.

## 9. LLM And Algorithm Boundary

The multimodal LLM is responsible for image understanding and strategy
selection:

- Identify whether the defect is bright-on-dark or dark-on-bright.
- Summarize visible defect appearance.
- Choose the segmentation method.
- Recommend sensitivity, area thresholds, and morphology.
- Revise the strategy after brush and text feedback.
- Explain the result in concise user-facing language.

The local algorithm is responsible for reproducible pixel-level execution:

- Load and normalize the target image.
- Apply thresholding or anomaly segmentation.
- Apply morphology and connected-component filtering.
- Produce the final predicted mask.
- Compute measurements and optional metrics.

This separation makes the output explainable: the LLM decides what to try, while
the deterministic image pipeline performs the measurement.

## 10. UI Design

The interface should feel like a Codex-like agent workspace instead of a static
form.

### Layout

Left side: conversation.

- User natural-language requests and feedback.
- Agent observations.
- Strategy explanation.
- Result summary.
- Suggested next action when confidence is low.

Center: image workspace.

- Original image, annotated image, and mask views.
- Brush annotation on the latest annotated image.
- Per-iteration result switching.

Right side: structured results.

- Current strategy summary.
- Count and area table.
- Optional IoU, count error, and area error.
- Run history and export links.

### Interaction

First run:

```text
Upload target image -> enter "find bright residue and measure area/count"
-> agent observes image -> strategy -> segmentation -> measurement -> result
```

Feedback run:

```text
Draw on result image -> enter "this lower-left region was missed"
-> agent explains strategy adjustment -> rerun -> updated result
```

The foreground message style should be concise and engineering-oriented. The
full experiment details are saved in files for reporting.

## 11. Testing

### Algorithm Tests

- Connected-component count is correct.
- Area and bounding boxes are correct.
- Area ratio is correct.
- Empty images and no-defect masks are handled.
- Tiny noise is filtered according to `min_area_px`.
- Bright and dark defect strategies behave differently.

### Agent Flow Tests

- The LangGraph flow completes with target image and description only.
- The flow completes with optional reference annotation.
- Invalid reference annotation dimensions produce a clear warning.
- Feedback brush and feedback text trigger a new strategy iteration.
- Output files are written for each iteration.

### Evaluation Tests

- When a reference annotation is provided, IoU is computed.
- Count error is computed from predicted versus reference connected components.
- Area error is computed from predicted versus reference foreground area.
- Missing reference annotation skips metrics without failing the run.

## 12. Success Criteria

V1 is successful when:

- A user can upload a target image and enter a defect description.
- The agent can generate an explainable strategy.
- The system outputs an annotated image, predicted mask, and area/count results.
- The user can draw feedback on the result image and add text feedback.
- LangGraph routes feedback into a new strategy and reruns the measurement.
- If a reference annotation is provided, the system outputs IoU, count error, and
  area error.
- The UI feels like a collaborative agent: concise in the foreground, detailed
  in the saved experiment records.

## 13. Failure Handling

The agent should report uncertainty instead of overstating success. Common
failure reasons include:

- Defect and background contrast is too low.
- Bright/dark polarity is uncertain.
- Regular array texture is being mistaken for defects.
- The reference annotation is not aligned with the target image.
- The OpenCV baseline cannot handle complex boundaries and SAM/SAM2 refinement
  should be considered.

These reasons should appear in the conversation summary and in the saved
measurements or graph state.

## 14. Resume-Friendly Project Summary

Implemented a LangGraph-based multimodal defect metrology agent for DRAM/SEM
images. The system uses Alibaba Cloud vision-language models to generate
structured defect segmentation strategies, executes reproducible OpenCV/skimage
measurement pipelines, and supports human-in-the-loop brush and text feedback to
iteratively improve defect annotation accuracy.
