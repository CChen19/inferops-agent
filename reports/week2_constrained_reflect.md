# Week-2 P1-⑥: Minimal constrained Reflect

Tune owns this slice. Consumes ④ ledger + ⑤ confirmation APIs. Does **not**
redefine `confirmed_improvement` minting, Week-1 `is_promotable` /
`derive_status`, or eval gates.

## Deterministic vs LLM

| Owner | Fields / decisions |
|---|---|
| **Code** (`reflect_constraints` / `confirm_campaign` / `reflector_node`) | `validity_status` (fail-closed), SLO (`error_rate` vs `MAX_ERROR_RATE`, env `INFEROPS_REFLECT_MAX_ERROR_RATE`), budget, duplicate identity, ⑤ verdict / bind, `is_confirmed_promotable`, `next_action` |
| **LLM (planner only)** | Hypothesis `param` / `value` / rationale text |
| **LLM in Reflect** | **Not wired.** `optional_llm_explanation` is a deterministic string. An LLM may only paraphrase hypothesis text / `reason` as a side-channel — never validity, SLO, budget, duplicates, or confirmation. |

`LLM_MUST_NOT_OWN` in `reflect_constraints.py` is the encoded split.
Production `reflector_node` has no LLM call and no empty-plan heuristic.

## Next actions

| Action | When | Route |
|---|---|---|
| `continue` | Executor `duplicate_candidate` skip; confirmed+promoted; search streak still alive | pending → executor, else planner |
| `remeasure` | ⑤ `too_noisy` (under cap) or search winner | executor runs a **⑤ interleaved confirmation campaign** (`run_interleaved_repeats` + `evaluate_campaign`). Production: per-slot `run_benchmark` at the tool boundary. CI: fixture `run_arm` or a stubbed tool edge |
| `rollback` | OOM / `failed` / `invalid` / `insufficient_evidence` / missing or high `error_rate` | restore current `best_summary`; pending → executor, else planner |
| `stop` | Budget; `too_noisy` at remasure cap; **`no_reliable_improvement`** | END |

`no_reliable_improvement` is first-class: ⑤ `no_diff` / `regression`, or the
legacy three-in-a-row search streak. It is not only a streak label.

## Best promotion

Executor **records** search results. It does **not** write `best_summary` for
candidates.

Reflect calls `maybe_promote_best` → **bind** (decision candidate `run_id`s +
`confirmation_target` fingerprint) **and** `is_confirmed_promotable(result, decision)`.

A confirmed decision for candidate A cannot promote candidate B. Confirmation
state is cleared when Reflect leaves remasure and when Executor starts a
different hyp.

Refused (stay at current / baseline):

- search winner (`phase=search`)
- remasure campaign that stays `no_diff` / `too_noisy` / unbound
- unevidenced / Week-1 fail + beautiful ⑤ numbers
- forged `ConfirmationDecision` (⑤ schema rejects minting)
- missing `last_result` or missing / stale decision

`MAX_ERROR_RATE = 0.05` is the agent-local SLO hook (same 5% bar as ⑤
`DEFAULT_MIN_REL_DELTA`). Override with `INFEROPS_REFLECT_MAX_ERROR_RATE`.
Missing `error_rate` is fail-closed (rollback), not SLO-ok.

## Remeasure / confirmation (offline vs production)

**Production default is not “unavailable.”** Remeasure always has a per-slot
`run_arm`:

1. **Search exec** — after a successful search benchmark, if baseline and
   candidate both carry a ④ `request_ledger`, Tune calls
   `verdict_from_ledgers(..., phase=search, min_pairs=1)`. A genuine
   `search_winner` is recorded (`repeat_ledgers` / `confirmation_decision` /
   bind fingerprint). Reflect then chooses `remeasure`. Metric-only rows
   (no ledger) do **not** invent `search_winner`; they stay on the streak path.
2. **Confirm campaign** — executor runs `run_interleaved_repeats` +
   `evaluate_campaign` (phase=confirmation, `n_pairs=DEFAULT_MIN_PAIRS`).
   Production `run_arm` calls `run_benchmark` once per interleave slot
   (baseline = empty patch, candidate = `{param: value}`) at the existing
   tool boundary. The last candidate `ExperimentResult` is written to
   `last_result` and appended as `experiment_summaries[-1]` so Reflect can
   call `is_confirmed_promotable` without test-side rewriting.
3. **CI / offline** — inject fixture ledgers via
   `confirmation_run_arm_override` / `tool_boundary_overrides`, or stub
   `run_benchmark` at that same tool edge. **No invented GPU numbers.**
   Fixture `run_arm` may attach `.last_candidate_result` so the graph still
   emits a bound result + summary.

A missing ledger on a production slot fails closed (`confirmation_campaign_failed`).
Nothing is minted to look like a GPU run.

Baseline seeding in `prepare_initial_state` still uses Week-1
`is_promotable_summary` — that is the starting current config, not a candidate
promotion.

`summary.promotable` remains the Week-1 flag. `confirmed_promotable=True` is
stamped only on a Reflect-promoted best.

## Trajectory contract

Every Reflect step records:

- `hypothesis` (id / param / value / text)
- `config_diff` (param, from, to, summary)
- `cited_run_ids`
- `constraint_checks` (validity, SLO, budget, duplicate, ⑤ verdict, gate)
- `next_action`
- `stop_reason` when stopping

Examples (also under `tests/fixtures/reflect_trajectories/`):

| Case | `next_action` | `stop_reason` |
|---|---|---|
| duplicate candidate | `continue` | — |
| OOM / exec fail | `rollback` | — |
| SLO breach (`error_rate=0.40`) | `rollback` | — |
| budget exhaust | `stop` | `budget_exhausted` |
| `too_noisy` (1 pair, `min_pairs=3`) | `remeasure` | — |
| ⑤ `no_diff` | `stop` | `no_reliable_improvement` |

```json
{
  "node": "reflector",
  "hypothesis": {"id": "h1", "param": "max_num_batched_tokens", "value": 4096},
  "config_diff": {"summary": "max_num_batched_tokens: baseline → 4096"},
  "cited_run_ids": ["nrb_00", "nrc_00"],
  "constraint_checks": {"verdict": "no_diff", "gate": "is_confirmed_promotable"},
  "next_action": "stop",
  "stop_reason": "no_reliable_improvement"
}
```

## What ⑥ does not do

- Invent ④/⑤ schemas or loosen Week-1 gates
- Own eval thresholds, GPU fleets, goldens (⑦), interrupt (⑧)
- Start a GPU. CI uses fixture ledgers / a stubbed tool edge only.

## Evidence

```bash
pytest -q
```

```text
311 passed in 14.29s
```

P1 proofs (this revision — residual search→confirm production path):

- `test_search_exec_records_search_winner_without_injected_ledgers`
- `test_search_exec_without_ledgers_does_not_invent_search_winner`
- `test_search_winner_remeasure_campaign_promotes` (no post-hoc `last_result` rewrite)
- `test_search_winner_remeasure_unconfirmed_does_not_promote`
- `test_fixture_run_arm_hook_still_drives_campaign`

Prior P1s kept green:

- `test_confirmed_decision_for_a_does_not_promote_b`
- `test_executor_clears_confirmation_when_candidate_changes`
- `test_invalid_and_insufficient_evidence_rollback`
- `test_missing_error_rate_rollback_fail_closed`

No real vLLM / GPU numbers are claimed in this environment. Trajectory JSON
fixtures share the live Reflect key set (`hypothesis`, `config_diff`,
`cited_run_ids`, `constraint_checks`, `next_action`, `stop_reason`,
`reasoning`, `result`); live steps may carry extra check fields.
