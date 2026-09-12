# Week-3 P0-⑧: recovery goldens + deterministic CI gate

Eval owns this thin slice. Consumes Tune Week-3 ⑧ recovery only
(failure-event fields, checkpoint/resume entrypoints, retry/budget
semantics). Does **not** invent a second recovery schema, loosen Week-1
`is_promotable` / `derive_status`, or rewrite Tune's state machine.

## Freeze status

**Unblocked.** Tune ⑧ is on **master**. Consumes the merged #11 contract
(not a second schema; not the pre-review `4f02584`):

| Surface | Value |
|---|---|
| Master merge | `45d2d4ed5253fa29ae98cd25826b894597c16288` (`#11`) |
| Tune tip | `d1e5e8259601ec3eca69e5852cdb3774dfd9881d` |
| PR | https://github.com/CChen19/inferops-agent/pull/11 |
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

Stubs are allowed **only** at `tool_boundary_overrides` (propose /
`run_benchmark` / confirmation `run_arm`). Planner, executor, and
reflector stay production.

## Tune ⑧ contract (consumed)

### Failure-event fields

`attempt_id`, `experiment_id`, `hypothesis`, `stage`/`tool`,
`reason`/`code`, `result_persisted`, `budget_consumed`, `retryable`,
`next_action`. Codes used by goldens: `propose_tool_error`,
`benchmark_error`, `tool_exception`, `confirmation_slot_failed`.

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

Step numbers may differ if Tune records an interrupt marker; executor
identity keys must match.

## Acceptance matrix

Shared gate: no false promote, no invented GPU/metrics
(`gpu_utilization_pct` / `gpu_memory_used_gb` / `cost_usd`), empty /
skipped set **FAIL**, `INFEROPS_GPU_GOLDENS` unset → CPU/fixture only.
This thin set does **not** claim `confirmed_promotable=true`.

| id | Fail-closed asserts |
|---|---|
| `propose_tool_error` | Tune `propose_tool_error`; no budget; no forged summary; stale latest ignored; no promote |
| `benchmark_no_result` | `BenchmarkError` without `exc.result`; no forged row/metrics; budget −1 once; no promote |
| `benchmark_error_failed_result` | real failed contract row kept; `is_promotable` false; `current_attempt_latest` is that row |
| `confirmation_mid_fail` | `confirmation_slot_failed`; completed slot `run_id` recorded; no ⑤ decision; remasure (under cap) |
| `prior_success_current_fail` | `tool_exception`; Reflect does not treat prior success as current; bind cleared |
| `pre_tool_interrupt` | tool not called at interrupt; one benchmark on resume; no confirmed promote |
| `post_persist_pre_commit_interrupt` | search persist reused; confirm slots reuse `_rN_` ids; campaign budget −1 once |
| `idempotent_re_resume` | second resume matches budget / tried ids / run_ids; no duplicate attempt |
| `resume_equivalence` | U vs R match on `stop_reason` / best / budget / executor trajectory identity |

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
369 passed in 13.74s
measurement-trust golden gate passed (CPU/fixture; GPU-not-run ≠ pass)
recovery golden gate passed (CPU/fixture; GPU-not-run ≠ pass)
```
