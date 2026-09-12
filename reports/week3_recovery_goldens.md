# Week-3 P0-⑧: recovery goldens + deterministic CI gate

Eval owns this thin slice. Consumes Tune Week-3 ⑧ recovery only
(failure-event fields, checkpoint/resume entrypoints, retry/budget
semantics). Does **not** invent a second recovery schema, loosen Week-1
`is_promotable` / `derive_status`, or rewrite Tune's state machine.

## Freeze status

**Unblocked.** Tune ⑧ residual is on **master**. Consumes #13 freeze
`fcb0f48` → merge `a0c7061` (not a second schema; not `b109ee6`):

| Surface | Value |
|---|---|
| Master merge | `a0c7061ef82fc32b68ae78f3ad504ac33eda191d` (`#13`) |
| Tune tip | `fcb0f48a5252c8e8c6b265a19579cc2c37b048e6` |
| PR | https://github.com/CChen19/inferops-agent/pull/13 |
| Report | `reports/week3_interrupt_recovery.md` |

Codex P1s on that tip (consumed, not reimplemented): confirm persist-resume
slot reuse, successful-confirm budget −1 once, last-slot SLO/validity
before any budget-exhausted promote.

Consumed import surface (Tune-owned names, not renamed):

```python
from inferops.agent.recovery import (
    RECOVERY_FIELDS,
    recovery_event,
    current_attempt_latest,
)
from inferops.agent.graph import (
    build_graph,              # checkpointer + interrupt_before=["executor"]
    graph_invoke_config,
    session_thread_id,
    production_checkpointer,
)
from inferops.agent.confirm_campaign import confirmation_slot_experiment_id
from inferops.schemas import is_promotable, derive_status
from inferops.metrics import is_confirmed_promotable
from inferops.agent.reflect_constraints import conclude_experiment
from inferops.agent.executor import tool_boundary_overrides
```

## Consume (do not loosen / do not rewrite)

| Stack | Frozen rule Eval keeps |
|---|---|
| ① `is_promotable` / `derive_status` | `valid` + critical evidence + full actual cover + `successful_requests > 0` |
| ④ ledger v2 | Missing ≠ 0. Incomplete stays in the error denominator |
| ⑤ `is_confirmed_promotable` | Search winner / partial campaign ≠ confirm |
| ⑥ Reflect | `too_noisy` remasures; `no_diff` → `no_reliable_improvement`. Best only via bind + confirm |
| Tune ⑧ | `last_recovery`, MemorySaver + `thread_id`, propose/no-budget, generic/`BenchmarkError` −1, confirm campaign −1 once, slot ids `{prefix}confirm_{param}_{value}_rN_{b\|c}{i}` |

Stubs are allowed at `tool_boundary_overrides` (propose /
`run_benchmark` / confirmation `run_arm`). Some recovery drivers
also patch persist/lookup and analysis helpers
(`get_result_by_id`, `analyze_bottleneck`, `compare_experiments`,
`propose_config_patch`, `confirmation_run_arm_override`) so
persist-reuse and mid-slot failure can be scripted without a second
schema. Planner, executor, and reflector stay production.

## Tune ⑧ contract (consumed)

### Failure-event fields

`attempt_id`, `experiment_id`, `hypothesis`, `stage`/`tool`,
`reason`/`code`, `result_persisted`, `budget_consumed`, `retryable`,
`next_action`, **`validity_status`**, **`retry_count`**.
Codes used by goldens: `propose_tool_error`, `benchmark_error`,
`tool_exception`, `confirmation_slot_failed`, **`ack_lost`**.
Unconfirmable persist uses Week-1 **`insufficient_evidence`** (not a
new enum). Executor + Reflect carry `TRAJECTORY_AUDIT_FIELDS`
(`retry_count`, `budget_consumed`, `next_action`, `stop_reason`).
Do not invent `receipt_lost` / `incomplete_receipt` as recovery fields.

### Checkpoint / resume

- `interrupt_before=["executor"]` — planner committed; tool not called
  until resume
- Persist-then-crash — `get_result_by_id` reuses the row; no second
  benchmark
- Confirm slot persist — `confirmation_slot_experiment_id` + reuse;
  remaining slots run; budget −1 once on commit
- `graph_invoke_config(session_prefix)` → `thread_id = session_prefix.rstrip("_")`

### Retry / budget

| Path | Budget | Remeasure |
|---|---|---|
| Propose reject / generic propose error | not consumed | no |
| Generic / `BenchmarkError` | −1 | no |
| Confirmation campaign (success or fail) | −1 once | fail: remasure only under `MAX_REMEASURES` + remaining budget |
| Last-slot confirm | −1 to 0 | may promote **only after** validity + SLO |

## Comparable terminal fields

Interrupted+resumed **R** and uninterrupted **U** (same scripted tool
outcomes) must agree on:

```text
stop_reason, should_stop, next_action,
best.experiment_id / run_id / promotable / confirmed_promotable,
confirmed_gate (= is_confirmed_promotable),
experiments_remaining, tried_experiment_ids,
last_result run_id + validity_status,
confirmation verdict/phase/search_winner/bound_run_ids,
reflect.cited_run_ids,
trajectory_identity (node, action_kind, experiment_id, run_id,
                     validity_status, promoted_to_best)
```

Step numbers may differ if Tune records an interrupt marker; remaining
trajectory identity keys (including `run_id`, `validity_status`,
`promoted_to_best`) must match — not only
`(node, action_kind, experiment_id)`.

## Acceptance matrix

Shared gate: no false promote (including `best_summary` swapped to a
search winner / partial campaign even when `confirmed_promotable`
stays false), no invented GPU/metrics
(`gpu_utilization_pct` / `gpu_memory_used_gb` / `cost_usd`), empty /
skipped set **FAIL**. `REQUIRED_GOLDEN_IDS` is a floor:
`catalog.required_ids` must not shrink below it; a non-empty catalog
subset + deleted fixtures still **FAIL** missing ids. `skip`/`skipped`
on a fixture also fails. `INFEROPS_GPU_GOLDENS` unset → CPU/fixture only.
This thin set does **not** claim `confirmed_promotable=true`.

| id | Fail-closed asserts |
|---|---|
| `propose_tool_error` | Tune `propose_tool_error`; no budget; no forged summary; stale latest ignored; no promote |
| `benchmark_no_result` | `BenchmarkError` without `exc.result`; no forged row/metrics; budget −1 once; no promote |
| `benchmark_error_failed_result` | real failed contract row kept; `is_promotable` false; `current_attempt_latest` is that row |
| `confirmation_mid_fail` | `confirmation_slot_failed`; completed slot `run_id` recorded; no ⑤ decision; remasure (under cap); `best_summary` identity unchanged |
| `prior_success_current_fail` | `tool_exception`; Reflect does not treat prior success as current; bind cleared |
| `pre_tool_interrupt` | tool not called at interrupt; one benchmark on resume; no confirmed promote; extra `confirm_` calls fail |
| `post_persist_pre_commit_interrupt` | search persist reused; confirm slots reuse `_rN_` ids; campaign budget −1 once |
| `idempotent_re_resume` | second resume matches budget / tried ids / run_ids; no duplicate attempt |
| `resume_equivalence` | U vs R match the full comparable end-state (`next_action`, best run/promotable, summary/last-result ids, confirmation/bindings/Reflect refs, full trajectory identity) |
| `ack_lost_unconfirmable` | `code=ack_lost`; persist ① `insufficient_evidence`; no promote; resume does not re-bench |
| `ack_lost_save_failure` | `code=ack_lost`; `result_persisted=false`; `validity_status=insufficient_evidence`; no forged row |
| `trajectory_audit_executor_reflect` | Executor + Reflect have `retry_count`, `budget_consumed`, `next_action`, `stop_reason`; planner must not |

## Fixtures + runner

```text
tests/fixtures/recovery_goldens/
inferops/eval/recovery_goldens.py
scripts/run_recovery_goldens.py
tests/test_recovery_goldens.py
```

## CI

`.github/workflows/eval-mock.yml`:

```bash
pytest -q
python scripts/run_measurement_goldens.py
python scripts/run_recovery_goldens.py
```

Labels: **CPU/fixture**. GPU-not-run ≠ pass. No self-started GPU.
No invented vLLM numbers.

## Out of scope

Rewriting Tune's recovery state machine. New ledger / confirmation /
Reflect schemas. Runtime / vanity / UI. Real GPU acceptance. Auto-merge
— Chris merges manually.

## Evidence

```bash
pytest -q
python scripts/run_measurement_goldens.py
python scripts/run_recovery_goldens.py
```

```text
375 passed in 14.20s
measurement-trust golden gate passed (CPU/fixture; GPU-not-run ≠ pass)
recovery golden gate passed (CPU/fixture; GPU-not-run ≠ pass)
```

Eval tip after Codex P1: `35cc642` on master `45d2d4e` (Tune tip `d1e5e82`).
Proving tests:
`test_catalog_subset_and_deleted_fixtures_fail_floor`,
`test_resume_equivalence_compares_full_end_state`,
`test_search_winner_best_swap_fails_without_confirmed_flag`,
`test_confirmation_mid_fail_keeps_best_identity`.
