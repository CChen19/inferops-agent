# Week-2 P1-⑥: Minimal constrained Reflect

Tune owns this slice. Consumes ④ ledger + ⑤ confirmation APIs. Does **not**
redefine `confirmed_improvement` minting, Week-1 `is_promotable` /
`derive_status`, or eval gates.

## Deterministic vs LLM

| Owner | Fields / decisions |
|---|---|
| **Code** (`inferops/agent/reflect_constraints.py` + `reflector_node`) | `validity_status`, SLO (`error_rate` vs `MAX_ERROR_RATE=0.05`), budget, duplicate identity, ⑤ verdict / `search_winner`, `is_confirmed_promotable`, `next_action` |
| **LLM (planner only)** | Hypothesis `param` / `value` / rationale text |
| **LLM in Reflect** | **Not wired.** `optional_llm_explanation` is a deterministic string. An LLM may only paraphrase hypothesis text / `reason` as a side-channel — never validity, SLO, budget, duplicates, or confirmation. |

`LLM_MUST_NOT_OWN` in `reflect_constraints.py` is the encoded split.
Production `reflector_node` has no LLM call and no empty-plan heuristic.

## Next actions

| Action | When | Route |
|---|---|---|
| `continue` | Executor `duplicate_candidate` skip; confirmed+promoted; search streak still alive | pending → executor, else planner |
| `remeasure` | ⑤ `too_noisy` (under cap) or search winner (queue confirmation) | executor (same hyp re-pended; new `run_id` / `_rN` experiment id) |
| `rollback` | OOM / exec `failed` or SLO breach | restore current `best_summary`; pending → executor, else planner |
| `stop` | Budget; `too_noisy` at remasure cap; **`no_reliable_improvement`** | END |

`no_reliable_improvement` is first-class: ⑤ `no_diff` / `regression`, or the
legacy three-in-a-row search streak. It is not only a streak label.

## Best promotion

Executor **records** search results. It does **not** write `best_summary` for
candidates.

Reflect calls `maybe_promote_best` → **`is_confirmed_promotable(result, decision)`**
(Week-1 `is_promotable` **and** confirmation-phase `confirmed_improvement`).

Refused (stay at current / baseline):

- search winner (`phase=search`)
- unevidenced / Week-1 fail + beautiful ⑤ numbers
- forged `ConfirmationDecision` (⑤ schema rejects minting)
- missing `last_result` or missing decision

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
- Start a GPU. Fixture ledgers only.

## Evidence

```bash
pytest -q
```

```text
(pending — filled after suite run)
```

No real vLLM / GPU numbers are claimed in this environment.
