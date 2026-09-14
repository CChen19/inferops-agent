# Architecture

One pass through the system, in the order things actually happen. File
references are to master at `95229d0` (through merged PR #59).

## 1. Task confirmation (`inferops/task.py`)

Natural language only *drafts* an `OptimizationTask`; code validates it. A task
carries the model, engine, workload spec, primary metric, constraints, budget,
and service control mode (`managed` or `external`), and it has a status:
`ready`, `confirmed`, `needs_clarification`, or `rejected`.

`require_confirmed` gates the run: experiment budget cannot be spent until the
task is `confirmed`. Unsupported models or workloads are rejected or sent back
for clarification, never silently substituted. The same `task_conditions`
mapping feeds the confirmation page, the executor, and the final report, so
what the user approved and what the report claims cannot drift apart.

## 2. Durable task state and compatible history (`inferops/resume.py`, `inferops/memory/`)

Production runs use a disk-backed LangGraph `SqliteSaver` and a stable thread id
derived from the session prefix. `production_checkpointer` is a context manager
that closes that SQLite connection when the run ends (PR #45). The
`run_eval` session test uses `cwd=tmp_path` so a default-db open cannot leak
`inferops_memory.db` into the worktree. The same SQLite database has a `tasks`
table that preserves the confirmed task payload, task id, session prefix,
thread id, and lifecycle status. `inferops agent --resume-task <task_id>`
reloads that identity and continues from the stored checkpoint instead of
rerunning the baseline. Eval recovery goldens deliberately keep `MemorySaver`
and stay disk-free, so this production persistence claim is not being
projected onto the fixture harness.

PR #57 adds Chainlit-independent parsing and formatting helpers in
`inferops/resume.py`; `tests/test_resume.py` does not import `app.py`. In chat,
`resume <12-hex>`, `resume-task <12-hex>`, and a bare 12-hex task id bypass task
drafting and confirmation and call `run_agent(..., resume_task_id=...)`. A
resume command with no id is rejected before GPU budget is spent.

When `memory_db_path` is present, PR #58 queries compatible history from another
session with the same `model_name` and `workload_name`. GPU SKU is not stored or
matched. These rows are `prior_session_hint` only: they cannot be cited as a
this-run `citations.metric.run_id`, enter `experiment_summaries`, promotion, or
`best_summary`, skip confirmation, or turn a prior success into a confirmed
result. Failed, invalid, and OOM parameter pairs do count as duplicates. The
CLI passes its database path into `run_agent`; the Chainlit streaming
`prepare_initial_state` path at this SHA does not, so the UI does not yet load
compatible history.

## 3. Planner (`inferops/agent/planner.py`, `inferops/citations.py`, `inferops/rag/`)

The compiled graph is small and deliberately so:

```
START → planner → executor → reflector → {planner | executor | END}
```

The planner is the only LLM-owned node. In the case run it was
`deepseek/deepseek-chat` over OpenRouter (`llm_boundary: live_openrouter`). The
LLM may propose a hypothesis and an explanation (`LLM_MAY_PROPOSE =
{hypothesis_text, explanation}`) and is explicitly barred from owning control
flow or verdicts (`LLM_MUST_NOT_OWN`); the reflector asserts that the next
action is one of `continue`, `remeasure`, `rollback`, `stop` regardless of what
the model said.

Hypotheses are drawn from a fixed search space
(`AGENT_SEARCH_SPACE` in `inferops/agent/state.py`):

```python
AGENT_SEARCH_SPACE = {
    "max_num_batched_tokens": [2048, 3072, 4096],
    "max_num_seqs":           [64, 128, 256],
    "enable_chunked_prefill": [False, True],
    "enable_prefix_caching":  [False, True],
}
```

A small RAG corpus over vLLM concepts (PagedAttention, chunked prefill, prefix
caching, scheduling) is available to the planner for phrasing hypotheses. It
does not decide anything. Every proposed hypothesis must carry a structured
metric citation whose `run_id`, metric name, and numeric value exactly exist in
the summaries shown to the planner. Retrieval now returns each Chroma
`chunk_id`, source, and corpus version; index metadata uses
`CORPUS_VERSION = "inferops-corpus-1"` (PR #59). The production source header
remains `[source: {source}] §{section}`, followed by
`chunk_id=... version=...` on the next line.

When retrieval returns sources, a document citation and matching `[source: ...]`
rationale tag are required, and `chunk_id`, source, and version must match one
retrieved tuple exactly. Forged or missing fields are rejected with one retry.
When retrieval returns none, document fields must still be omitted. This is an
existence gate only; it does not claim that the cited chunk semantically
supports the hypothesis.

## 4. Managed vLLM child (`inferops/tools/vllm_process.py`, `inferops/tools/managed_lifecycle.py`)

In `managed` mode the agent first takes a non-blocking file lock for the single
GPU, then launches and owns the vLLM server as a subprocess with an explicit
argv (`_build_cmd`) and benchmarks against it. A second managed task fails
closed with holder information; it does not start or stop anything.

Readiness is not trust. After `/health` returns 200, the agent calls
`assert_listener_bound_to_child`: the PID listening on the port must equal the
managed child PID, or the run is never valid. A healthy listener this task did
not spawn is treated as an unknown occupant and is not stopped. The sole
recovery exception is a child strictly matched to a stale InferOps lease record,
and even that stop-and-relaunch path is opt-in via
`INFEROPS_ADOPT_STALE_MANAGED`; the default is off.

A managed child is registered as owned immediately when it is spawned, before
readiness polling begins. Stop during model load therefore fails closed: the
in-flight experiment is never valid, the owned child is stopped, and its lease
is released (PR #32). Readiness polling also aborts when cancellation is
requested or `stop()` has cleared the managed process, rather than continuing
to poll a dead port until `STARTUP_TIMEOUT_S` (PR #35). `_wait_should_abort`
binds `proc = self._proc` once before calling `poll()`, so a racing `stop()`
cannot AttributeError on a cleared handle (PR #37). `is_crashed()`,
`exit_code()`, `pid`, and `stop()` do the same one-bind (PRs #39 and #43).
`test_stop_binds_proc_once` now also asserts that this bound fake child was
terminated (PR #52), strengthening the CPU coverage of the existing production
`stop()` behavior rather than adding a GPU timing. These are merged code paths
covered by deterministic CPU tests, not new GPU measurements.

The readiness `/health` GET is bounded by `CANCEL_CHECK_S` (0.25 s), not a
hardcoded 3 s (PR #48). Cancel/Stop does not interrupt that GET mid-call:
`_wait_should_abort()` is checked between polls, and the request timeout only
caps how long recognition of an abort can be delayed. `health_ok()` remains a
separate one-shot with `timeout_s=2.0`.

## 5. Evidence and the promotion gate (`inferops/schemas.py`, `state.py`)

The `actual_config` is not the requested config. It is parsed back out of the
launch command line (`parse_vllm_cli_knobs`) and filtered to
`MANAGED_CLI_EVIDENCED_KEYS`: `model_name`, `max_num_seqs`,
`max_num_batched_tokens`, `max_model_len`, `gpu_memory_utilization`,
`enforce_eager`, `enable_chunked_prefill`, `enable_prefix_caching`. Anything
outside that set — notably `scheduler_policy` and `tensor_parallel_size` — stays
on the schema for future search but can never be recorded as applied, and shows
up in reports as `requested X → measured <missing>`.

Evidence kinds that are never sufficient on their own include
`config_file_only`, `http_ok_only`, `health_check_only`,
`performance_delta_only`, and `external_unverified`. Promotion reads a single
`promotable` flag (`is_promotable_summary`) that callers must populate from the
full gate; a row claiming `valid` without `promotable=True` is rejected.
Missing metrics stay `None` and are never rewritten to `0.0`.

## 6. Confirmation (`inferops/metrics/confirm.py`, `agent/reflect_constraints.py`)

A candidate that looks good enough is not adopted on one measurement. The
reflector routes to `remeasure`, which builds a repeat campaign of interleaved
baseline/candidate pairs. A verdict can only be produced by
`verdict_from_ledgers` / `evaluate_campaign`, and requires at least
`DEFAULT_MIN_PAIRS = 3` usable pairs with a median relative delta of at least
`DEFAULT_MIN_REL_DELTA = 0.05`. `ConfirmationDecision` is frozen, so a
`too_noisy` or `no_diff` verdict cannot be mutated into a confirmation.

`maybe_promote_best` advances `best_summary` only through a decision *bound* to
that exact candidate — a confirmation computed for candidate A cannot promote
candidate B. Below the reflect threshold (`IMPROVEMENT_THRESHOLD_PCT = 5.0`),
confirmation is never even attempted, and three consecutive non-improving
trials (`MAX_STREAK = 3`) stop the run. PR #57's reflector formatter makes the
UI distinguish `continue`, `remeasure`, `rollback`, and `stop`; it changes
presentation, not who owns the action.

## 7. DecisionReport (`inferops/decision.py`)

Every run ends in exactly one of four outcomes:

1. `confirmed_and_meets_goals` — confirmed improvement that satisfies the task
   goals; adopt the measured configuration.
2. `improved_but_unmet_goals` — reliable improvement over baseline, but the
   serving goals are still unmet.
3. `no_reliable_improvement` — keep the baseline. (This is what the case run
   returned.)
4. `inconclusive` — evidence insufficient or execution failed; nothing can be
   recommended.

The report carries goal checks (missing observed values fail closed), the
adopted measured config, the diff versus baseline, a `why_not_others` list
naming every rejected candidate and the reason, and evidence links with
`run_id`, `mlflow_run_id`, and ledger path. The same renderer
(`render_decision_markdown`) drives both the Markdown report and the Chainlit
UI, so the UI cannot show a friendlier story than the file.

`ExperimentSummary` now persists `baseline_primary`, the denominator used for
`vs_baseline_pct` (PR #42). The Experiment Log prints that stored percentage
at stored precision and includes a `ttft_p99` column (PR #44). The Executive
Summary headline uses the same `_fmt_vs_baseline` helper (PR #50), so 3.77
prints as `+3.77%` and a missing value prints as `n/a`, never `+0.0%`. PR #54
routes the Chainlit live-result and all-experiments displays through tested
helpers in `inferops.tools.final_report`, and updates the graph run summary,
executor completion line, and `scripts/run_agent.py` result to preserve the
stored precision; their missing-value output is `n/a` or `unavailable`, not an
invented `+0.0%`. The published live case artifact predates these changes and
is not rewritten by them. A zero baseline denominator in `compare_experiments`
raises `ValueError` instead of inventing `0.0%`; the executor records compare
as `tool_unavailable` and leaves `vs_baseline_pct` as `None` (PR #49).

## Known limits of this architecture

- Only the eight `MANAGED_CLI_EVIDENCED_KEYS` knobs can be applied and proven.
  `scheduler_policy` and `tensor_parallel_size` are schema-only.
- The load generator is concurrency-limited `asyncio.gather`. There is no
  offered arrival rate and no queueing model, so `target_qps` is a measured
  throughput goal only.
- Confirmation is the only path to adoption, and it is expensive in budget
  slots; below the 5% reflect threshold it never triggers, so small real gains
  are systematically left unadopted.
- Production checkpointing and confirmed-task metadata are local SQLite state;
  eval recovery goldens still use in-memory `MemorySaver` fixtures.
- The file lock serializes managed work on one GPU, but it does not make the
  hardware measurements more generalizable or eliminate run-to-run drift.
- Trials run sequentially on one GPU. Thermal and clock drift across a session
  are not measured or compensated.
- Hardware scope is one RTX 3060 Laptop with 6 GB VRAM, so larger models and
  multi-GPU paths are untested here.
- `external` service mode cannot prove config application the way `managed` can;
  externally-launched servers fall back to weaker evidence kinds.
- `_fmt_vs_baseline` has separate copies in `planner.py` and
  `final_report.py`. They agree for reachable float/`None` inputs today, but
  could silently diverge later.
- CI tests the Chainlit-facing helpers in `final_report.py`, not the `app.py`
  call sites because Chainlit is absent from the dev environment; a future
  local re-round at an app call site could escape CI.
- `eval/runner.py` and `scripts/run_comparison.py` retain one-decimal formatting
  for `gap_pct` / `vs_default`; those are different metrics, not
  `vs_baseline_pct` regressions.
- A remote `VLLM_HOST` whose RTT exceeds `CANCEL_CHECK_S` (0.25 s) would never
  succeed `wait_ready`.
- `_run_resumed_task` catches every `ValueError` and can say no GPU budget was
  spent even if a mid-run `ValueError` occurred after GPU work. `resume nope`
  also takes the missing-id error path rather than becoming a task draft.
- Resume calls `run_agent` without streaming, so reflector updates are not shown
  while a resumed run is in progress.
- Chainlit's streaming `prepare_initial_state` path does not pass
  `memory_db_path`, so compatible history is currently CLI-only. GPU SKU is not
  stored or matched for compatible-history lookup.
- `_recover_param_value` can mis-attribute a failed historical row with two
  non-default knobs when its id has no `_<knob>_` token.
- A Chroma index built before PR #59 lacks version metadata and fail-closes
  document-citing hypotheses until rebuilt. No index is checked into the repo.
- A corpus body containing line-start `[source: x] §y` immediately followed by
  a `chunk_id=... version=...` line can still be parsed as an available
  document ref; this extends the pre-existing `sources_from_context` exposure
  to the full tuple.
