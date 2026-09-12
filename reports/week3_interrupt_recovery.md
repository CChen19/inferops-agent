# Week-3 P0-⑧: Interrupt / tool-error recovery

Tune owns this slice. Extends existing Reflect / executor failure paths.
Does **not** redefine ④/⑤ schema, ⑦ goldens, or Week-1
`is_promotable` / `is_confirmed_promotable` (both stay fail-closed).

Hard-kill orphan GPU cleanup, ⑨ controlled对照, and Runtime vanity are
out of scope.

## Recovery / failure event (not a metrics schema)

Agent-local fact on `last_recovery` and on the executor / Reflect
trajectory step. Required fields:

| Field | Meaning |
|---|---|
| `attempt_id` | Stable id for this try (`experiment_id`, or confirm `{prefix}confirm_{param}_{value}_rN`) |
| `experiment_id` | Session-scoped experiment name (empty if none yet) |
| `hypothesis` | `{id, param, value, text}` |
| `stage` / `tool` | `propose_config` · `run_benchmark` · `analyze_bottleneck` · `compare_experiments` · `confirmation_slot` · `confirmation_campaign` |
| `reason` / `code` | Human string + machine code (`propose_rejected`, `tool_exception`, `benchmark_error`, `confirmation_slot_failed`, …) |
| `result_persisted` | True only when a real ① contract row exists |
| `budget_consumed` | Whether `experiments_remaining` was decremented |
| `retryable` | Confirmation-slot fail under remasure/budget cap |
| `next_action` | Suggested `rollback` / `remeasure` (Reflect re-checks) |

No parallel metrics schema. Generic exceptions do **not** invent
`run_id`, perf, GPU, ledger, or a success summary.

## State table (this-attempt failure)

| Incoming | Executor write | Reflect `latest` | Promote? |
|---|---|---|---|
| Propose `ValueError` | `last_recovery` + trajectory; **no** budget; `last_result` / ⑤ bind **cleared**; no forged summary | Prior `experiment_summaries[-1]` is **not** current | No |
| Generic tool exception (no result) | Budget −1; recovery fact; **no** forged row; stale bind cleared | Prior success ignored | No |
| `BenchmarkError` + `exc.result` | Keep that persisted failed contract row as `last_result` + summary | That failed row **is** current | No (`is_promotable` fail-closed) |
| Confirmation mid-slot fail | Stage + completed `cited_run_ids`; `confirmation_decision=None`; budget −1 | Prior search success ignored | No confirm / no promote |
| `analyze_bottleneck` / `compare_experiments` degrade | Trajectory `tools.*.status=unavailable`; vs from result metrics, **never silent 0** | Unchanged success/fail of the bench itself | No promotion change |
| `KeyboardInterrupt` / `SystemExit` / `GraphInterrupt` | **Re-raised** — not a success patch | n/a | n/a |

`current_attempt_latest()` only treats `summaries[-1]` as current when
`result_persisted` and `experiment_id` match this attempt.

## Checkpoint boundaries

Production `run_agent` compiles with `MemorySaver` and a stable
`thread_id = session_prefix.rstrip("_")`.

```
START → planner → │ executor │ → reflector ⇄ planner | executor | END
                  ▲
                  interrupt_before=["executor"]  (node boundary, before tool)
```

| Boundary | What is committed | Resume |
|---|---|---|
| `interrupt_before=["executor"]` | Planner patch + checkpoint | Executor runs **once**; tool not called during the interrupt |
| Tool persist, node not returned | SQLite/store row exists; LangGraph state still pre-executor | `get_result_by_id` reuses the row; **no second benchmark** |
| Executor / Reflect return | Budget, trajectory, `last_result` / recovery | Same attempt is not double-counted |

Eval / `build_graph(llm)` without a checkpointer stays config-free
(`graph.invoke(state)`), so ③ real-graph eval is unchanged.

## Retry / budget

| Path | Budget | Remeasure | Cap |
|---|---|---|---|
| Propose reject | Not consumed | No (`retryable=False`) | n/a |
| Generic exception / `BenchmarkError` | −1 | No | Budget / streak |
| Confirmation slot fail | −1 | Yes only if `remeasure_count < MAX_REMEASURES` **and** budget remains | `MAX_REMEASURES` (= ⑤ `DEFAULT_MIN_PAIRS`) and budget — **no infinite loop** |

Reflect re-validates `retryable`. Leaving remasure still clears ⑤ bind
(`clear_confirmation_fields`). Cross-candidate confirmation cannot
survive a failure or a new hyp.

## Tests (fixture / CPU only)

- `test_propose_rejection_emits_recovery_and_does_not_promote`
- `test_generic_benchmark_exception_does_not_forge_or_read_stale_success`
- `test_benchmark_error_keeps_persisted_row_and_does_not_promote`
- `test_benchmark_error_without_result_does_not_forge_summary`
- `test_confirmation_mid_slot_fail_records_ids_and_does_not_confirm`
- `test_confirmation_failures_are_capped_no_infinite_remeasure`
- `test_analyze_and_compare_unavailable_are_recorded_not_silent_zero`
- `test_persist_then_interrupt_before_commit_reuses_result`
- `test_interrupt_before_tool_runs_benchmark_once_on_resume`
- `test_persist_then_interrupt_graph_reuses_result_no_second_benchmark`
- `test_uninterrupted_vs_interrupted_resumed_terminal_equivalence`
- `test_no_confirmed_promotion_on_fail_or_resume_paths`
- `test_keyboardinterrupt_and_systemexit_are_not_swallowed_as_success`

## Evidence

```bash
pytest -q
```

```text
342 passed in 13.53s
```

Fixture / CPU only. GPU was not run in this environment — **GPU-not-run ≠ pass**.
No invented vLLM / GPU numbers.

## Out of scope

- Hard-kill orphan GPU cleanup
- Redefining ④ ledger / ⑤ confirmation / ⑦ goldens
- Loosening Week-1 `is_promotable` / `is_confirmed_promotable`
- ⑨ controlled对照
- Runtime vanity
