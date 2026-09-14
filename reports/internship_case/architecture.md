# Architecture

One pass through the system, in the order things actually happen. File
references are to master at `ca8e3fb` (through merged PRs #51–#53).

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

## 2. Durable task state (`inferops/agent/graph.py`, `inferops/memory/db.py`)

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

## 3. Planner (`inferops/agent/graph.py`, `inferops/agent/planner.py`, `inferops/citations.py`)

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
the summaries shown to the planner. A document citation and matching
`[source: ...]` rationale tag are required if and only if retrieval returned
sources; when retrieval returned none, inventing a source is rejected. Forged
run ids, values, or document sources are rejected, with one retry. This is an
existence gate, not a claim that the cited evidence semantically proves the
hypothesis.

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
trials (`MAX_STREAK = 3`) stop the run.

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
prints as `+3.77%` and a missing value prints as `n/a`, never `+0.0%`. The
published live case artifact predates these changes and is not rewritten by
them. A zero baseline denominator in `compare_experiments` raises `ValueError`
instead of inventing `0.0%`; the executor records compare as
`tool_unavailable` and leaves `vs_baseline_pct` as `None` (PR #49).

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
- CLI/UI vs-baseline sites in `app.py`, `graph.py`, `executor.py`, and
  `scripts/run_agent.py` still format as `+:.1f`. Those are not covered by
  PR #50.
- A remote `VLLM_HOST` whose RTT exceeds `CANCEL_CHECK_S` (0.25 s) would never
  succeed `wait_ready`.
