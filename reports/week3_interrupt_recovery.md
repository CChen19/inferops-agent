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
| `validity_status` | **Additive residual.** Week-1 status on the contract row, or `""` when none. Unconfirmable incomplete uses `insufficient_evidence` (not a new enum). |
| `retry_count` | **Additive residual.** `remeasure_count` for this attempt (0 on first search). |

No parallel metrics schema. Generic exceptions do **not** invent
`run_id`, perf, GPU, ledger, or a success summary.

Residual P0 (this Tune PR) does **not** rewrite Eval goldens (#10).
`RECOVERY_FIELDS` grew by two keys only. Existing goldens already iterate
the tuple; `recovery_event()` always emits the new keys. Eval can follow
this freeze surface without a schema rewrite:

```text
RECOVERY_FIELDS += validity_status, retry_count
TRAJECTORY_AUDIT_FIELDS = retry_count, budget_consumed, next_action, stop_reason
codes += ack_lost
Week-1 status for unconfirmable incomplete = insufficient_evidence
(optional additive event.incomplete=true — not a ① enum)
```

## State table (this-attempt failure)

| Incoming | Executor write | Reflect `latest` | Promote? |
|---|---|---|---|
| Propose `ValueError` / generic propose exception | `last_recovery` + trajectory; **no** budget; `last_result` / ⑤ bind **cleared**; no forged summary | Prior `experiment_summaries[-1]` is **not** current | No |
| Generic tool exception (no result) | Budget −1; recovery fact; **no** forged row; stale bind cleared | Prior success ignored | No |
| Startup-ok + receipt/ack lost, **row already persisted** | Fact-check `get_result_by_id(attempt_id)`; reuse; **no second bench** | Reused row is current | No (⑥ still owns promote) |
| Startup-ok + ack lost, **lookup miss** | Persist ① row via `derive_status` → `insufficient_evidence` (`code=ack_lost`, `incomplete=true`); not only `result_persisted=false` | That insuff. row **is** current | No (`is_promotable` fail-closed) |
| `BenchmarkError` + `exc.result` | Keep that persisted failed contract row as `last_result` + summary | That failed row **is** current | No (`is_promotable` fail-closed) |
| Confirmation mid-slot fail | Stage + completed `cited_run_ids`; `confirmation_decision=None`; budget −1 | Prior search success ignored | No confirm / no promote |
| `analyze_bottleneck` / `compare_experiments` degrade | Trajectory `tools.*.status=unavailable`; vs from result metrics, **never silent 0** | Unchanged success/fail of the bench itself | No promotion change |
| `KeyboardInterrupt` / `SystemExit` / `GraphInterrupt` | **Re-raised** — not a success patch | n/a | n/a |

`current_attempt_latest()` only treats `summaries[-1]` as current when
`result_persisted` and `experiment_id` match this attempt.

Last-slot budget (`experiments_remaining==0`) is judged **after**
validity / SLO. `would_promote` / `is_confirmed_promotable` alone cannot
bypass a high or missing `error_rate`.

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
| vLLM/startup ok, receipt/ack lost | Result may already be at the stable `experiment_id` / confirm slot id | Fact-check that id; reuse if present; else persist `insufficient_evidence`; **never blindly re-start** |
| Confirm slot persist, campaign not committed | Slot rows at `{prefix}confirm_{param}_{value}_r{N}_{b\|c}{i}` | Resume reuses those slot ids; remaining slots run; budget −1 once on commit |
| Executor / Reflect return | Budget, trajectory, `last_result` / recovery | Same attempt is not double-counted |

Eval / `build_graph(llm)` without a checkpointer stays config-free
(`graph.invoke(state)`), so ③ real-graph eval is unchanged.

## Retry / budget

| Path | Budget | Remeasure | Cap |
|---|---|---|---|
| Propose reject / generic propose error | Not consumed | No (`retryable=False`) | n/a |
| Generic exception / `BenchmarkError` | −1 | No | Budget / streak |
| Confirmation campaign (success **or** fail) | −1 once per attempt | Fail: remasure only if `remeasure_count < MAX_REMEASURES` **and** budget remains | `MAX_REMEASURES` (= ⑤ `DEFAULT_MIN_PAIRS`) and budget — **no infinite loop**; last-slot confirm may promote **only after** validity + SLO (high / missing `error_rate` never promotes) |

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
- `test_confirmation_slot_ids_include_remeasure_identity`
- `test_confirm_persist_then_crash_reuses_slots_budget_once`
- `test_successful_confirm_promotes_and_consumes_budget_once`
- `test_successful_confirm_on_last_budget_slot_still_promotes`
- `test_last_budget_slot_high_error_rate_does_not_promote`
- `test_last_budget_slot_missing_error_rate_does_not_promote`
- `test_generic_propose_tool_error_emits_recovery_no_forge`
- `test_graphinterrupt_is_reraised_not_swallowed`
- `test_is_ack_lost_matches_receipt_and_ack_wording`
- `test_unconfirmable_row_uses_week1_insufficient_evidence`
- `test_ack_lost_reuses_persisted_row_no_second_benchmark`
- `test_ack_lost_unconfirmable_persists_insufficient_evidence`
- `test_confirm_slot_ack_lost_reuses_persisted_no_rebench`
- `test_trajectory_records_retry_budget_stop_and_next_action`

## Evidence

```bash
pytest -q
```

```text
(pending this PR pytest -q line)
```

Fixture / CPU only. GPU was not run in this environment — **GPU-not-run ≠ pass**.
No invented vLLM / GPU numbers.

Executor + Reflect trajectory steps now carry `retry_count`,
`budget_consumed`, `next_action`, and `stop_reason` (Reflect fills
`stop_reason`; executor failure records the suggested `next_action`).

## Out of scope

- Hard-kill orphan GPU cleanup
- Redefining ④ ledger / ⑤ confirmation / ⑦ goldens
- Loosening Week-1 `is_promotable` / `is_confirmed_promotable`
- ⑨ controlled对照
- Runtime vanity
