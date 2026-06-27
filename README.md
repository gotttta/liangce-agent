# DRAM Defect Metrology Agent

MVP runner for the DRAM defect metrology Agent described in `docs/design.md`.

The current version is a LangGraph-based agent MVP:

- reads an unannotated target image
- generates a structured area/count strategy
- segments bright/dark anomaly regions with OpenCV/skimage
- measures connected components for metrology
- supports optional same-image reference annotation metrics
- provides a Gradio conversational workspace with brush feedback

Runtime uses the Alibaba Cloud multimodal model configured in `.env`. The test
suite still uses a mock provider so tests do not spend tokens or require network
access.

## Run

Create `.env` in the project root:

```bash
DASHSCOPE_API_KEY="your-api-key"
ALIYUN_BASE_URL="your-openai-compatible-url"
ALIYUN_VISION_MODEL="qwen3.7-plus"
```

`ALIYUN_API_KEY` can be used instead of `DASHSCOPE_API_KEY`.

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

By default, the runtime calls Alibaba Cloud's OpenAI-compatible endpoint. Set:

```bash
export DASHSCOPE_API_KEY="your-api-key"
export ALIYUN_BASE_URL="your-openai-compatible-url"
export ALIYUN_VISION_MODEL="qwen3.7-plus"
```

The default model is `qwen3.7-plus`. If the API key or base URL is missing, the
agent fails fast instead of falling back to a local mock.

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
