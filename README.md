# DRAM Defect Metrology Agent

Local single-user industrial vision Agent for developing reproducible defect-detection pipelines.

The current v2 flow:

- runs one canonical LangGraph workflow for CLI and web requests
- uses Alibaba Cloud Qwen for visual task understanding, Pipeline planning, and candidate review
- retrieves matching user-accepted algorithms and combines them with model-generated Pipeline DSL candidates
- validates every operator, artifact type, parameter, and pipeline step locally
- executes candidates deterministically with OpenCV/skimage operators in a bounded sandbox
- applies deterministic user include/exclude constraints before measuring and rendering results
- waits for the user to confirm whether the rendered annotation is accurate
- publishes user-accepted pipelines to a reusable local algorithm registry
- stores the generated algorithm, operator trace, mask statistics, mask, optional contours and measurements

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

The `.env` file is ignored by git and should not be committed. Keep API keys
only in your local `.env`.

```bash
python3 main.py \
  --target path/to/target.png \
  --description "bright particle defect, measure area and count"
```

Optional Handbook few-shot references from other annotated images:

```bash
python3 main.py \
  --reference path/to/reference.png \
  --target path/to/target.png \
  --description "bright residue defect, measure area and count"
```

## Web UI

```bash
python3 -m ui.app
```

Open `http://127.0.0.1:7860`.

`ui.app` is the supported web entry point. Both the CLI and web UI execute the canonical graph in
`core/agent_graph.py`; `graph_workflow.py` remains only as a compatibility import for older callers.

The graph nodes are:

```text
prepare_inputs
  -> understand_task
  -> retrieve_algorithms
  -> plan_candidates
  -> execute_candidates
  -> review_candidates
  -> (revise_candidates -> execute_candidates, bounded by max_auto_revisions)
  -> decide_next_action
  -> wait_for_human
```

The responsibilities are intentionally split:

- **Qwen**: understands the image and description, proposes a structured Pipeline, and compares
  rendered candidate overlays. It does not directly write the final mask or run arbitrary code.
- **Local executor**: validates the Pipeline DSL, runs allowlisted operators, applies brush constraints,
  measures connected components, and writes reproducible artifacts.
- **User**: makes the final visual acceptance decision. The runtime does not replace this decision with
  an automatic quality score or an `acceptable/uncertain/failed` label.

After `decide_next_action`, LangGraph creates a `human_review` interrupt. The web UI resumes that
same graph thread when the user accepts, continues editing, or exits; resuming the interrupt itself
does not rerun a candidate. When the user submits new text or brush feedback, the next iteration uses
the previous state and feedback to plan a new Pipeline. Task artifacts and graph state are also written
to disk for cross-restart result recovery; the LangGraph checkpoint itself currently uses in-memory storage.

The graph API defaults to `max_candidates=3` and `max_auto_revisions=2`. The Web UI currently passes
`max_candidates=1`, so it executes one candidate per iteration and can request up to two bounded automatic
revision rounds. Callers can override `max_auto_revisions` when using `run_agent_graph` directly.

The runtime records factual mask statistics only. After execution, the vision provider compares the
rendered candidate overlays and selects one with a written visual rationale. It may request a bounded
automatic revision, but it never creates a synthetic quality score. Final visual accuracy is still
decided at the human review boundary. Pixel processing remains bounded and auditable. The reusable
operator catalog starts empty and contains only operators that a user has explicitly marked as tested.
If the catalog is insufficient, the model may emit a restricted custom operator
(`apply(data, params)`) for the current candidate only.

That source is AST-validated and runs only in a disposable sandbox with operator, step, timeout, and
resource limits. Accepting an algorithm stores its complete Pipeline for replay, but does not publish
its custom stages into `workspace/operators/`. Publishing a reusable stage is a separate, explicit
user-tested approval action.

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

## Handbook few-shot references

Upload one to three customer-annotated Handbook images as visual examples. The
Agent uses them during task understanding and candidate review to learn what
objects should be marked, where the boundary sits, and which annotation style is
expected. Reference images are not pixel-aligned annotations, are never used to
copy coordinates into the target image, and do not produce an accuracy score.
They are copied into the task's `references/` directory and recorded in
`reference_examples`.

## Outputs

Each run creates:

```text
outputs/<timestamp>_agent_v2/
└── iteration_0/
    ├── candidate_0/
    ├── candidate_summary.json
    ├── pipeline.json
    ├── operator_trace.json
    ├── agent_trajectory.json
    ├── quality_report.json  # factual mask statistics; no quality classification
    ├── result_annotated.png
    ├── mask.png
    ├── measurements.json
    └── graph_state.json
```

## Accepted algorithm registry

When the user accepts a result in the web UI, the replayable pipeline is published to:

```text
workspace/algorithms/<algorithm_id>/algorithm.json
```

Each record includes the pipeline, defect and background characteristics, mask statistics,
measurement summary, source task and acceptance metadata. For a new task, the agent searches
this registry using structured task characteristics such as defect type, background pattern,
measurement type and polarity. Matching pipelines are replayed as candidate baselines and must
execute successfully on the current image; retrieval never bypasses local validation or human
acceptance.
