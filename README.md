# DRAM Defect Metrology Agent

Local single-user industrial vision Agent for developing reproducible defect-detection pipelines.

The production flow:

- runs one canonical LangGraph workflow for CLI and web requests
- uses the configured Alibaba Cloud model for one proposal action at a time and independent visual review
- rebuilds model context from the current draft, experiment records, fixed requirements and accumulated read evidence
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
Production Agent actions use `kind: "tool"` with `tool` and `arguments`, or
`kind: "needs_input"`. Independent review returns `review` or read-only `read` actions.
The legacy planning protocol retains `type: "call_tool"`. Old Pipeline fields `tool` and `tool_versions`, and workspace
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
ALIYUN_VISION_MODEL="deepseek-v4.1-flash"
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
cd web && npm install && npm run build && cd ..
python3 -m api
```

Open `http://127.0.0.1:8765`. `python3 -m api` serves both the REST API and the
built frontend; run only one instance at a time. For frontend development, keep
`python3 -m api` running and start the Vite dev server in `web/` (`npm run dev`,
http://127.0.0.1:5173, proxies `/api` to 8765).

Production controller runs have a 600-second deadline by default. An optional
`LIANGCE_REQUEST_TIMEOUT_SECONDS` adds an enclosing request deadline; unset or
`0` disables that extra limit, not the controller deadline. Restart an existing
service after changing configuration. Runs execute in background threads and
outlive the browser connection: closing or refreshing the page does not cancel
a run; only the stop control (cancel endpoint) does.

`python3 -m api` is the supported web entry point. Both the CLI and web UI execute the canonical graph in
`core/agent_graph.py`.

The production Aliyun provider uses explicit LangGraph business nodes:

```text
initialize_run -> prepare_task -> agent_decision <-> tool_execution
                                      |
                              validate_submission
                                      |
                                quality_review <-> review_evidence
                                  |       |
                       agent_decision   wait_for_human -> finish

budget exhausted / cancelled / failed -> finish (retains artifacts and reason)
```

There is one Agent loop. The model decides when to create/edit a draft, execute,
inspect intermediate evidence, compare experiments, and submit an executed result.
Saving a draft does not execute it. Executing a draft returns to the Agent without
automatic review. Submission verifies persisted scope, algorithm, runtime and
artifact hashes before independent quality review; it never reruns the algorithm.
The shortest successful path takes four model calls: create, execute, submit,
review. Reads, edits and comparisons use the same global call allowance.
No production node invokes an inner `PlanningSession`.

Providers exposing `agent_action` use this graph (version 2). Providers exposing
only `propose_action`, and version-one persisted checkpoints, retain the earlier
controller/effect graph. Legacy/mock providers and explicit `understanding` retain
the legacy graph. Recovery selects the persisted graph version instead of silently
migrating an unfinished run. `max_candidates` and calibration remain legacy options.

The responsibilities are intentionally split:

- **Vision provider**: understands the image and description, proposes a structured Pipeline, and compares
  rendered candidate overlays. Generated Python runs only in the Docker execution environment.
- **Local executor**: validates the Pipeline DSL, dispatches registered and generated operators to Docker, applies brush constraints,
  measures connected components, and writes reproducible artifacts.
- **User**: makes the final visual acceptance decision. The runtime does not replace this decision with
  an automatic quality score or an `acceptable/uncertain/failed` label.

When the controller enters `awaiting_feedback`, LangGraph creates a `human_review` interrupt. The web UI resumes that
same graph thread when the user accepts, continues editing, or exits; resuming the interrupt itself
does not rerun a candidate. When the user submits new text or brush feedback, the next iteration uses
the previous state and feedback to plan a new Pipeline. Task artifacts and graph state are also written
to disk for explicit cross-restart recovery. SQLite checkpoints use
`durability='sync'` (`workspace/checkpoints.sqlite3`, overridden with
`LIANGCE_CHECKPOINT_PATH`); a task file lock prevents simultaneous writers.
Action receipts distinguish prepared, completed and unknown work. Completed
receipts are replayed without another call; an unknown model request stops
conservatively. Execution recovery verifies saved artifacts, scope and runtime,
and checks/cleans the action's deterministically named Docker container first.
There is no automatic startup scan or automatic resubmission of unknown work.
Checkpoints currently include full dictionaries, not only artifact references.
See [current Agent workflow](docs/2026-09-22-tool-agent-workflow.md).

### Controller limits

| Variable | Default | Meaning |
|---|---|---|
| `LIANGCE_RUN_TIMEOUT_SECONDS` | `600` | Overall automatic-run deadline, retained across recovery |
| `LIANGCE_RUN_MAX_MODEL_CALLS` | `10` | Shared proposal, review, read follow-up and retry allowance |
| `LIANGCE_RUN_MAX_EXECUTIONS` | `3` | Maximum reserved executions; also capped by `max_auto_revisions + 1` |
| `LIANGCE_MODEL_CALL_TIMEOUT_SECONDS` | `120` | Controller ceiling for one model action |
| `LIANGCE_ACTION_MAX_OUTPUT_TOKENS` | `8192` | Model response limit, valid from `256` to `16384` |
| `LIANGCE_REQUEST_TIMEOUT_SECONDS` | unset | Optional enclosing deadline; `0` disables only this extra limit |

The action provider shares the 120-second configured default; an explicit provider
`timeout_seconds` remains respected. Legacy provider methods retain their 90-second
default. The effective action deadline is the minimum of the provider limit,
controller ceiling and remaining run/request time. Proposal calls additionally leave a completion reserve, 50 seconds
under defaults. Single JSON requests use a cancellable total deadline and no SDK
retries; the controller allows at most one eligible network retry per run and
charges it to the shared allowance. Execution reservations are counted before
the effect begins; static draft validation does not consume one. Limits are
persisted with the run, and stopping never requires an extra model call.

Each experiment's `experiment.json` preserves the algorithm version, parent experiment, hypothesis,
change reason, expected change, outputs, and subsequent review/acceptance status. Graph state keeps
`experiment_records` across revisions and user follow-ups. A `verified_baseline` is retained only
after all automatic acceptance checks pass or the user explicitly accepts a rendered result.
The controller retains the most recent usable execution when a later attempt
fails; retention does not imply that the result passed review. Scope includes
input content, requirements, feedback masks and reference images/descriptions.
Old acceptance cannot transfer to a changed scope. Metrics alone never establish
a verified baseline.

The runtime records factual mask statistics only. After execution, the vision provider checks the
rendered experiment against the task criteria and writes a visual rationale. A single result still
requires review; unavailable review leaves acceptance pending. It may request a bounded
automatic revision, but it never creates a synthetic quality score. Without Ground Truth,
these checks do not establish an accuracy improvement. Final visual accuracy is still
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
export ALIYUN_VISION_MODEL="deepseek-v4.1-flash"
```

The default model is `deepseek-v4.1-flash`. If the API key or base URL is missing, the
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
outputs/<graph_thread_id>/
├── actions/action_<n>/  # started record, saved result and completion receipt
├── drafts/<draft_id>/revision_<n>.json
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
measurement summary, source task and acceptance metadata. During preparation, the
production controller retrieves up to two accepted algorithms using the current
description and supplies their full Pipelines as procedural memory without an
extra model call. A follow-up also retains the previous task Pipeline. The legacy
graph uses its structured retrieval node. Reusing a saved method never bypasses
validation or acceptance on the current image.

## Agent experiment tools

The Agent returns one validated JSON tool call per request:

```json
{"kind":"tool","tool":"execute_pipeline","arguments":{"draft_id":"saved_draft","revision":1}}
```

- `create_draft`, `edit_draft`: persist and validate source; no automatic execution.
  The first `create_draft` also contains `understanding` in its arguments.
- `execute_pipeline`: run an explicitly named draft/revision and return an experiment ID.
- `query_operators`, `load_skill`, `read_draft`, `inspect_experiment`,
  `inspect_artifact`: retrieve exact definitions, source, reports and images.
- `compare_candidates`: compare two or three compatible executed experiments.
- `submit_experiment`: submit an actual experiment ID to independent review.
  Earlier successful experiments from the same run may be submitted.
- `needs_input` is a separate action for essential missing user information.

The controller saves and validates drafts before execution. Invalid source remains
available with exact diagnostics for repair. Read results are accumulated and
deduplicated. Operator definitions and Skill text remain available across proposal,
execution and review. Exploration evidence remains available during edits;
independent review starts from submitted artifacts and static definitions.
A failed patch retains its evidence for repair.
Repeating the same request without a state change stops the loop. Every model
follow-up consumes the shared call allowance.
Requests are rebuilt from canonical state, including the exact current draft,
experiment identities, rejected results, review issues and remaining budget.
Oversized essential context fails before transmission instead of silently
discarding requirements or source. Images and report pages remain bounded.

Only the first draft of a new user run may contain source-backed
`contract_updates` and `memory_updates`. The program validates the quoted user
instruction and then freezes the contract for automatic revisions. Later model
actions cannot lower acceptance requirements. Docker availability is checked
before the first model request, and infrastructure failures retain their cause.


## Planning protocol and implementation boundaries

- `core/agent_workflow.py`: explicit Agent/tool/submission/review graph nodes.
- `core/agent_protocol.py`: tool action validation and Agent instructions.
- `core/orchestration.py`: shared durable action execution, contract handling and
  version-one controller/effect compatibility.
- `core/orchestration_runtime.py`: persisted global limits, deadlines, action
  receipts and task locks.
- `providers/vision.py`: one cancellable JSON request per production action,
  schema validation and fresh evidence context. No native mutation tools are sent.
- `core/tools/contracts.py`: shared schemas for bounded read arguments and draft edits.
- `core/experiments/runner.py`: shared execution, user constraints, quality facts
  and rendering; `drafts.py` persists versioned source and validation diagnostics.

The production graph reuses existing draft, execution, inspection and comparison
implementations without constructing the legacy planning session.
`core/planning.py` and its separate budgets remain for legacy callers.
`ALIYUN_TOOL_MODE=native|text`,
`LIANGCE_PLANNING_TIMEOUT_SECONDS` and `LIANGCE_MAX_DRAFT_EDITS` configure that
legacy path, not production action calls. Mock tests do not establish live
endpoint compatibility or image accuracy.

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
Each body/resource load is a bounded read action in the production controller;
legacy planning retains its discovery counter. Operator contracts are queried separately with
`query_operators`, which accepts up to 30 names per call.
The periodic segmentation fallback is an internal CV baseline, not a business Skill.

Historical algorithm retrieval is deterministic: production performs it in
`prepare`, while the legacy graph retains its retrieval node.
`search_algorithms` is not an exposed Agent Tool.
Skill checklists and legacy `verification` metadata do not imply that every
domain-specific verification rule is automatically enforced.


## Runtime logs / 调试日志

CLI (`python main.py ...`) and web (`python3 -m api`) write application logs
both to stderr and to `workspace/logs/agent.log` under the project root.
Restart an already running server to load this logging configuration.

```bash
tail -f workspace/logs/agent.log
LIANGCE_LOG_LEVEL=DEBUG python3 -m api
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
