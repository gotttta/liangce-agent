# DRAM Defect Metrology Agent

MVP runner for the DRAM defect metrology Agent described in `docs/design.md`.

The current version is a LangGraph-based agent MVP:

- reads an unannotated target image
- generates a structured area/count strategy
- segments bright/dark anomaly regions with OpenCV/skimage
- measures connected components for metrology
- supports optional same-image reference annotation metrics
- provides a Gradio conversational workspace with brush feedback

Local development uses a mock vision provider by default, so it does not call a
remote model unless the provider is switched in code or configuration.

## Run

```bash
python3 main.py \
  --target path/to/target.png \
  --description "bright particle defect, measure area and count"
```

Optional same-image reference annotation:

```bash
python3 main.py \
  --reference path/to/reference.png \
  --target path/to/target.png \
  --description "bridge defect, measure area and gap"
```

## Web UI

```bash
python3 -m ui.app
```

Open `http://127.0.0.1:7860`.

## LangGraph Agent Mode

The agent mode uses a LangGraph state flow:

```text
prepare_inputs -> vision_strategy -> segment_defects -> measure_defects -> render_outputs
```

By default, local development uses `MockVisionProvider`, so it does not call a
remote model. To use Alibaba Cloud's OpenAI-compatible endpoint, set:

```bash
export DASHSCOPE_API_KEY="your-api-key"
export ALIYUN_VISION_MODEL="qwen-vl-max-latest"
```

The model name is configurable because available model names can vary by account
and region.

Run tests:

```bash
python3 -m pytest -v
```

## Outputs

Each run creates:

```text
outputs/<timestamp>_agent/
└── iteration_0/
    ├── strategy.json
    ├── result_annotated.png
    ├── mask.png
    ├── measurements.json
    ├── metrics.json
    └── graph_state.json
```
