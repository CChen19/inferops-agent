# Architecture

One pass through the system, in the order things actually happen. File
references are to this worktree at `11d19b7`.

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

## 2. Planner (`inferops/agent/graph.py`, `inferops/agent/planner.py`)

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
does not decide anything.

## 3. Managed vLLM child (`inferops/tools/vllm_process.py`)

In `managed` mode the agent launches and owns the vLLM server as a subprocess
with an explicit argv (`_build_cmd`), then benchmarks against it.

Readiness is not trust. After `/health` returns 200, the agent calls
`assert_listener_bound_to_child`: the PID listening on the port must equal the
managed child PID, or the run is never valid. If a previous occupant of the port
cannot be stopped (`still_listening = True`), the agent refuses to spawn a
replacement rather than race a stale server.

## 4. Evidence and the promotion gate (`inferops/schemas.py`, `state.py`)

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

## 5. Confirmation (`inferops/metrics/confirm.py`, `agent/reflect_constraints.py`)

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

## 6. DecisionReport (`inferops/decision.py`)

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

## Known limits of this architecture

- Only the eight `MANAGED_CLI_EVIDENCED_KEYS` knobs can be applied and proven.
  `scheduler_policy` and `tensor_parallel_size` are schema-only.
- The load generator is concurrency-limited `asyncio.gather`. There is no
  offered arrival rate and no queueing model, so `target_qps` is a measured
  throughput goal only.
- Confirmation is the only path to adoption, and it is expensive in budget
  slots; below the 5% reflect threshold it never triggers, so small real gains
  are systematically left unadopted.
- The LangGraph checkpointer is an in-process `MemorySaver`. Resume works within
  a process via a session thread id, not across restarts.
- Trials run sequentially on one GPU. Thermal and clock drift across a session
  are not measured or compensated.
- Hardware scope is one RTX 3060 Laptop with 6 GB VRAM, so larger models and
  multi-GPU paths are untested here.
- `external` service mode cannot prove config application the way `managed` can;
  externally-launched servers fall back to weaker evidence kinds.
