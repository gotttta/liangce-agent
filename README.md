# DRAM Defect Metrology Agent

Local single-user industrial vision Agent for developing reproducible defect-detection pipelines.

The current v2 flow:

- runs one canonical LangGraph workflow for CLI and web requests
- uses Alibaba Cloud Qwen for visual task understanding, one Pipeline per iteration, and independent visual review
- retrieves matching user-accepted algorithms as a possible starting point for a single experiment
- validates every operator, artifact type, parameter, and pipeline step locally
- executes experiments deterministically with OpenCV/skimage operators in a bounded sandbox
- applies deterministic user include/exclude constraints before measuring and rendering results
- waits for the user to confirm whether the rendered annotation is accurate
- publishes user-accepted pipelines to a reusable local algorithm registry
- stores the generated algorithm, operator trace, mask statistics, mask, optional contours and measurements

## V3 Pipeline Architecture

V3 keeps pixel processing deterministic while making the generated algorithm
less rigid. Terminology is consistent across code, prompts and saved records:

| Term | Meaning | Implementation |
|---|---|---|
| Agent Tool（工具） | Model-facing actions such as `load_skill` and `execute_pipeline` | `core/tools/` |
| Operator（算子） | Atomic CV operations composed into a Pipeline | `core/operators/`, `core/operator_catalog.py` |
| Skill（业务技能） | On-demand workflow instructions and reference resources | `skills/`, `core/skills/` |

The reusable image-processing layers are:

```text
Operator (versioned CV operation)
  -> Pipeline (task-specific operator composition)
Skill instructions guide the planning and verification of that Pipeline.
  -> Algorithm (a task-specific, executed Pipeline instance)
```

- **Operators** are registry-backed, versioned operations with typed input ports,
  parameter contracts and model-visibility metadata. The model receives the
  same catalog the executor validates.
- **Skills** are file-based workflows, not mandatory Pipeline templates. The six
  built-ins are `line_width_spacing`, `hole_diameter`, `position_offset`,
  `area_measurement`, `contour_deviation` and `defect_count`.
  Each lives in `skills/<name>/SKILL.md`, with measurement definitions, reference
  prerequisites and acceptance checks available in the first load. Specialized
  metrics require actual executed computation; references and calibration must
  not be invented.
- **Algorithms** are the concrete Pipeline instances saved when a user accepts
  a result. They retain Operator and Skill references for replay.

New v3 nodes use `operator`, legacy JSON Skill dependencies use
`required_operators`, and replay records use `operator_versions`.
Agent action JSON keeps `tool` alongside
`type: "call_tool"`. Old Pipeline fields `tool` and `tool_versions`, and workspace
Skill field `required_tools`, are converted on read; conflicting Pipeline aliases
are rejected. The legacy sequential `steps` format retains its `op` field.

V3 Pipelines use a typed DAG when an algorithm needs multiple artifacts. For
example, periodic background subtraction consumes both the denoised image and
the separately calculated background model:

```json
{
  "schema_version": 3,
  "name": "periodic_segmentation_baseline",
  "nodes": [
    {
      "id": "period",
      "operator": "period_estimation",
      "inputs": {"image": "$image"},
      "params": {"axis": "auto"}
    },
    {
      "id": "background",
      "operator": "build_periodic_background",
      "inputs": {"image": "$image", "period": "period"},
      "params": {}
    },
    {
      "id": "residual",
      "operator": "subtract_periodic_background",
      "inputs": {"image": "$image", "background": "background"},
      "params": {"mode": "absolute"}
    }
  ]
}
```

Before execution the runtime rejects unknown Operators, unknown parameters,
unmatched input ports, invalid artifact types, cyclic graphs, missing final
output references. Node budgets are configurable (256 by default). It executes a topological ordering in the
same sandbox limits as the legacy DSL. Existing `steps` Pipelines and the
`periodic_particle_builtin` replay path remain supported for old records.

The model loads Skill guidance when relevant and constructs a Pipeline from
appropriate operators. Skills do not require a fixed Pipeline. Generated operators may use ordinary Python, imports, loops
and helpers with NumPy, OpenCV, SciPy, scikit-image and Pillow. Every Pipeline runs
in a disposable, network-disabled Docker container; there is no host fallback.
Build the runtime image before using the application: see [Docker sandbox setup](docs/docker-sandbox.md).
Publishing a reusable operator still requires explicit user-tested approval.

Runtime uses the Alibaba Cloud multimodal model configured in `.env`. The test
suite still uses a mock provider so tests do not spend tokens or require network
access.

## Run

Use Python 3.12 (the tested baseline is 3.12.14 on macOS arm64). Create an
environment and install the locked runtime dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements.lock
```

For development and tests, install `requirements-dev.lock` instead. The `.txt`
files declare direct dependencies; the `.lock` files pin the resolved versions
and distribution hashes. Other platforms require their own installation and
regression verification before being treated as supported environments.

Create `.env` in the project root:

```bash
DASHSCOPE_API_KEY="your-api-key"
ALIYUN_BASE_URL="your-openai-compatible-url"
ALIYUN_VISION_MODEL="deepseek-flash"
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

Web requests have no overall time limit by default. To enable one, set
`LIANGCE_REQUEST_TIMEOUT_SECONDS` to a positive number of seconds before starting
the service; unset it or use `0` to disable it. Restart an existing service after
changing this setting. Model network timeouts, Docker sandbox limits, planning
round limits, and cooperative cancellation still apply.

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
  -> END
```

A failed review or an exhausted revision budget routes to `report_failure`, which also waits at the same
`wait_for_human` interrupt.

The responsibilities are intentionally split:

- **Qwen**: understands the image and description, proposes a structured Pipeline, and compares
  rendered candidate overlays. Generated Python runs only in the Docker execution environment.
- **Local executor**: validates the Pipeline DSL, dispatches registered and generated operators to Docker, applies brush constraints,
  measures connected components, and writes reproducible artifacts.
- **User**: makes the final visual acceptance decision. The runtime does not replace this decision with
  an automatic quality score or an `acceptable/uncertain/failed` label.

After `decide_next_action`, LangGraph creates a `human_review` interrupt. The web UI resumes that
same graph thread when the user accepts, continues editing, or exits; resuming the interrupt itself
does not rerun a candidate. When the user submits new text or brush feedback, the next iteration uses
the previous state and feedback to plan a new Pipeline. Task artifacts and graph state are also written
to disk for cross-restart result recovery; LangGraph checkpoints now use SQLite (`workspace/checkpoints.sqlite3`, override with `LIANGCE_CHECKPOINT_PATH`). See [memory design](docs/MEMORY.md).

The graph and Web UI default to one experiment per iteration (`max_candidates=1`), with at most
two automatic revisions (`max_auto_revisions=2`). The model writes one final Pipeline and uses
execution evidence, intermediate images, and visual review to explain the next targeted change.
The planner never fills a candidate quota or reruns the previous Pipeline as a mandatory baseline.
Repeated rejected Pipelines are not executed again within the automatic revision loop.

`compare_candidates` remains an on-demand tool for existing experiment IDs: compare two supported
methods, investigate stalled revisions, or check a new version against a verified baseline.
Comparisons require matching input and task/feedback constraints. Legacy API callers can explicitly
raise `max_candidates` for batch experiments or `max_calibration_candidates` for Ground Truth
calibration; calibration now defaults to **0**. Ground Truth evaluation and visual review remain active.

Each experiment's `experiment.json` preserves the algorithm version, parent experiment, hypothesis,
change reason, expected change, outputs, and subsequent review/acceptance status. Graph state keeps
`experiment_records` across revisions and user follow-ups. A `verified_baseline` is retained only
after all automatic acceptance checks pass or the user explicitly accepts a rendered result.
Failed revisions may roll back to that baseline only under identical input and acceptance conditions;
the failed experiment and its review stay in history. Metrics alone never establish a verified baseline.

The runtime records factual mask statistics only. After execution, the vision provider checks the
rendered experiment against the task criteria and writes a visual rationale. A single result still
requires review; unavailable review leaves acceptance pending. It may request a bounded
automatic revision, but it never creates a synthetic quality score. Final visual accuracy is still
decided at the human review boundary. Pixel processing remains bounded and auditable. The reusable
operator catalog starts empty and contains only operators that a user has explicitly marked as tested.
If the catalog is insufficient, the model may emit a custom operator
(`apply(data, params)`) for the current candidate only.

That source is checked for syntax and the entrypoint contract, and runs only in a disposable Docker
container with network, filesystem, output, step, timeout and resource limits. Accepting an algorithm stores its complete Pipeline for replay, but does not publish
its custom stages into `workspace/operators/`. Publishing a reusable stage is a separate, explicit
user-tested approval action.

By default, the runtime calls Alibaba Cloud's OpenAI-compatible endpoint. Set:

```bash
export DASHSCOPE_API_KEY="your-api-key"
export ALIYUN_BASE_URL="your-openai-compatible-url"
export ALIYUN_VISION_MODEL="deepseek-flash"
```

The default model is `deepseek-flash`. If the API key or base URL is missing, the
agent fails fast instead of falling back to a local mock.

Run tests:

```bash
python -m pip install --require-hashes -r requirements-dev.lock
python -m pytest -v
```

To deliberately refresh dependencies with `uv`, regenerate both locks and rerun
the test suite, including the frozen pixel baseline:

```bash
uv pip compile requirements.txt --python-version 3.12 --generate-hashes --upgrade --output-file requirements.lock
uv pip compile requirements-dev.txt --constraint requirements.lock --python-version 3.12 --generate-hashes --upgrade --output-file requirements-dev.lock
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
    ├── evaluation_report.json  # Ground Truth metrics; only written when Ground Truth is supplied
    ├── graph_state.json
    ├── candidate_budget.json
    └── runtime_environment.json
```

## Accepted algorithm registry

Replay records preserve Operator versions and Skill provenance through normalization.
Recorded Operator versions must match the installed implementation; mismatches or
incomplete version maps fail validation instead of silently upgrading the
algorithm. Unversioned legacy pipelines are allowed and receive
`version_provenance: recorded_at_execution` when first executed, which does not
claim knowledge of their original environment. Generated Operator versions come
from their embedded definitions. Each iteration also writes
`runtime_environment.json` with Python and dependency versions and source hashes;
this records the environment but does not automatically recreate it.

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

## Agent experiment tools

The model receives a compact operator index and can request details with
`query_operators`, load workflow guidance with `load_skill`,
and inspect intermediate images with `inspect_artifact`.

Planning uses a single editable algorithm draft and submits a saved experiment:

- `save_task`: persist understanding and acceptance criteria before editing code.
- `create_draft`, `read_draft`, `edit_draft`: save versioned drafts, inspect them,
  and apply exact local edits against the current revision. Every edit runs static
  validation; syntax failures include the operator, line, column and source excerpt.
- `execute_pipeline(draft_id, revision)`: run a validated draft against the current image in the existing
  sandbox, apply user constraints, and return an experiment ID, factual
  statistics, intermediate artifacts and the rendered overlay.
- `compare_candidates`: compare two or three scoped experiment IDs using an
  aligned overlay sheet and added/removed pixel counts. These counts are not
  accuracy scores; comparison never accepts or publishes an algorithm.
- `submit_experiment(experiment_id, reason)`: load the actual executed algorithm
  from backend storage. Final model output no longer repeats Python source in a
  large JSON document. Submission requires matching input, task/feedback scope,
  code hash and runtime; it does not imply visual acceptance.

Each planning request defaults to three definition queries, eight draft edits,
six validation/execution requests, two actual executions and two comparisons.
Static failures and unavailable Docker do not spend execution budget; runtime
algorithm failures do. Identical pipelines, versions, scopes and runtimes can reuse
results within the request. Docker and the sandbox image are checked before planning;
infrastructure failures stop the session, and cleanup failures preserve the original cause.
Artifacts live under `<output_root>/agent_experiments/<session_id>/`: `task.json`,
`drafts/<draft_id>/revision_<n>.json`, `execution_<n>/`, `session.json`, and
`submission.json` (or `failure.json`). Drafts and execution snapshots stay separate.
Submitted pipelines still pass through formal execution, independent visual review,
and human acceptance. Exploration budgets are
additional to the existing candidate budgets, not a unified global limit.


## Planning protocol and implementation boundaries

`ALIYUN_TOOL_MODE=native` (default) uses function schemas, streamed tool-call
arguments and `role=tool` replies linked by call ID. For endpoints that do not
support this protocol, explicitly set `ALIYUN_TOOL_MODE=text`; this compatibility
mode uses the same schemas, dispatcher and budgets. There is no silent fallback.
Live provider compatibility is not established by the mock protocol tests.

- `core/planning.py`: bounded planning state machine (24 model turns for algorithm
  development, reserving the last two for ID submission; 10 by default for other
  sessions). `LIANGCE_PLANNING_TIMEOUT_SECONDS` defaults to 900 and also respects
  a shorter enclosing request deadline. `LIANGCE_MAX_DRAFT_EDITS` defaults to 8.
  Exhausted tool categories are disabled; full
  exhaustion or repeated requests for unavailable tools triggers finalization.
- `core/tools/contracts.py`: `ToolSpec` input schemas and `ToolResult` envelope
  (`call_id`, `status`, `data`, `error`). Errors contain `code`, `message`, and
  `retryable`, with structured `details` for diagnostics; dispatch also returns
  remaining `budget` and `available_tools`.
- `core/tools/budget.py`: request-scoped counters enforced by dispatch and the
  execution/comparison entry points. Invalid known calls consume interaction
  budget, failed experiments consume execution budget, rejected calls do not
  execute. JSON/final-output validation is separate from budget errors.
- `core/experiments/runner.py`: shared single-candidate execution, constraints,
  quality facts and rendering. Neither the tools nor Runner import agent_loop
  or providers. The workflow still owns selection, calibration and acceptance.

Error codes include `invalid_arguments`, `unknown_tool`, `budget_exhausted`,
`pipeline_invalid`, `timeout`, `resource_limit`, `worker_terminated`,
`execution_failed`, `artifact_missing`, and `io_error`. A dead worker without an
error payload is not assumed to have exceeded memory. Finalization rejects tools
with `finalization_required`; malformed final JSON uses `invalid_json` and invalid
candidate content uses `invalid_final_output`.

Skills use progressive disclosure:

1. Initial planning/revision context includes only each Skill's name and description.
2. `load_skill({"name": "area_measurement"})` returns its complete `SKILL.md`
   with the core measurement rules and acceptance checks, plus a list of optional
   resource paths, without inlining references or operator contracts. The six
   built-ins are self-contained and currently have no separate resources.
3. For a Skill that lists an optional resource, a subsequent `load_skill` call
   with the same `name` and a relative `resource` path reads that file only.
   Script resources return source text; they are not executed by this read tool.
   Execution stays in the existing sandbox.

The model selects Skills from their catalog descriptions; the registry does not
parse description prose for keyword search or instantiate Pipelines.

`SKILL.md` uses scalar frontmatter fields `name`, `description`, and optional
`version` (default `1.0.0`). This parser supports these simple scalar fields,
not general YAML. Workspace skills may use `<skill_root>/<directory>/SKILL.md`.
Resources must resolve inside the selected Skill directory, including symlinks,
and are limited to 64 KiB of Markdown, text, JSON, or Python source.
`load_skill` accepts optional `version`; otherwise the highest semantic version
is used. Duplicate name/version pairs cannot replace built-ins. Accepted legacy
`skill_*/skill.json` templates remain a compatibility format: `load_skill` reads
them through `get()`, validates their schema and dependencies, and returns their
template and operator contracts. New Skills use Markdown workflow guidance.
Each body/resource load consumes one of the three discovery calls per planning
request. For Markdown Skills, operator contracts are queried separately with
`query_operators`, which accepts up to 30 names per call.
The periodic segmentation fallback is an internal CV baseline, not a business Skill.

Historical algorithm retrieval remains a deterministic workflow step in
`core/agent_graph.py`; `search_algorithms` is not an exposed Agent Tool. Whole-task
budgets across planning, revision and calibration, provider prompt/normalization
extraction, and executable Skill-specific verification policies remain follow-up
work; Skill checklists (and legacy `verification` metadata) do not imply these
policies are all automatically enforced.


## Runtime logs / 调试日志

CLI (`python main.py ...`) and web (`python -m ui.app`) write application logs
both to stderr and to `workspace/logs/agent.log` under the project root.
Restart an already running server to load this logging configuration.

```bash
tail -f workspace/logs/agent.log
LIANGCE_LOG_LEVEL=DEBUG python -m ui.app
```

Each line includes timestamp, level, task ID, request/run ID and stage. A web
request receives a new run ID; nested graph and model calls retain that ID.
CLI calls have a run ID and use `task=-` because no UI task is created.
Background UI workers copy the caller context. Node and tool events record
progress; each actual model completion call records model, duration and failure
traceback. Graph completion records its output directory and graph thread ID.
Recoverable provider/candidate failures also retain their traceback.

Configuration uses environment variables (set before starting the process):

| Variable | Default | Meaning |
|---|---|---|
| `LIANGCE_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `LIANGCE_LOG_DIR` | project `workspace/logs` | Log directory |
| `LIANGCE_LOG_MAX_BYTES` | `10485760` | Rotate at approximately 10 MiB |
| `LIANGCE_LOG_BACKUP_COUNT` | `5` | Number of rotated files retained |

DEBUG adds tool argument summaries. Model prompts, full replies and streaming
chunks are not logged. Known environment secrets, credential fields and image
base64 are filtered in both output destinations, including exception tracebacks.
Existing task events, node JSON records and algorithm artifacts remain available
for detailed inspection; node-save logs include their paths. Logs can still
contain user descriptions, filenames and error details; review before sharing.
Rotation supports one application process; use separate `LIANGCE_LOG_DIR` values
for multiple server processes. Programmatic callers can explicitly call
`core.runtime_logging.configure_logging()` to enable the same output.
