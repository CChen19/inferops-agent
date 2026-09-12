# Week-2 P0-⑤: Repeat measurement + candidate confirmation

Confirmation sits on the P0-④ ledger. It does **not** invent a second
metrics schema. Aggregates come only from `recalculate_from_ledger`.
Week-1 `is_promotable` / `derive_status` are unchanged.

## Consume ④

```python
from inferops.metrics import (
    RequestLedger,
    RunConditions,
    recalculate_from_ledger,
    report_from_ledger,
    report_from_result,
    interleave_schedule,
    run_interleaved_repeats,
    verdict_from_ledgers,
    evaluate_campaign,
    is_confirmed_promotable,
    ConfirmationVerdict,
    RepeatPhase,
)

# LEDGER_SCHEMA_VERSION == "2"
# is_promotable / derive_status — do not loosen
```

Missing metrics stay `None`. `incomplete` / `timeout` / `cancel` are never
success. `token_count_source=missing` cannot drive TPOT or tok/s.

## Protocol

Independent repeats, interleaved under the same `RunConditions`:

```text
B0, C0, B1, C1, B2, C2, …
```

`interleave_schedule(n_pairs, phase=…)` / `run_interleaved_repeats(run_arm, …)`
produce that order. A conditions mismatch raises — it is not a silent verdict.
The same `RequestLedger` / `run_id` cannot count as multiple pairs
(`require_unique_repeat_identities`). `evaluate_campaign` also requires
`campaign.schedule` to match `interleave_schedule` (B0 C0 B1 C1 …).
`min_pairs` must be `> 0`. `min_rel_delta` must be finite and `> 0`.
`ConfirmationDecision` is frozen; mutating a computed `too_noisy` /
`no_diff` cannot pass `is_confirmed_promotable`.

Search and confirmation are different phases:

| Phase | What it may record | May set `confirmed_improvement`? | May pass `is_confirmed_promotable`? |
|---|---|---|---|
| `search` | `search_winner=True` when numbers look better | **no** | **no** |
| `confirmation` | official verdict | yes, if the numeric test holds | only if Week-1 gate also holds |

A lucky search pair must be re-run as confirmation. Tune should treat
`search_winner` as “queue confirmation”, never as deploy.

## Verdicts (deterministic, from ledger aggregates)

Positive relative delta means the candidate is better on the chosen metric.

```text
rel_delta = (cand − base) / |base|     # higher-is-better
rel_delta = −(cand − base) / |base|    # lower-is-better
```

`None` baseline/candidate, or baseline `0`, → pair class `missing` (not `0`).

Defaults: `min_pairs=3`, `min_rel_delta=0.05`.

| Verdict | When |
|---|---|
| `too_noisy` | **any** missing primary (even if other pairs would suffice), usable pairs `< min_pairs`, or mixed better+worse pairs |
| `regression` | every usable pair worse and median ≤ −threshold |
| `no_diff` | ties / below threshold; **also** the confirmation verdict for a search-phase numeric win (`reason=search_phase_unconfirmed`) |
| `confirmed_improvement` | confirmation phase **only**; every usable pair better and median ≥ threshold |

Encoded in `ConfirmationDecision` and `verdict_from_ledgers`:

- Schema rejects `confirmed_improvement` on `phase=search`.
- Hand-built `ConfirmationDecision(verdict=confirmed_improvement, …)` is
  **forged** and rejected. Only `verdict_from_ledgers` / `evaluate_campaign`
  may mint that verdict, and only with `numeric_signal=improvement` plus
  enough unique usable pairs.

## Promotion (Tune must use this)

```python
is_confirmed_promotable(result, decision)
```

is `True` only when **all** of:

1. `is_promotable(result)` — critical evidence + full actual-config cover +
   `successful_requests > 0` (Week-1, untouched)
2. `decision.phase == confirmation`
3. `decision.verdict == confirmed_improvement`

Search winners, unevidenced “best”, missing tok/s, and one-shot lucky repeats
stay `False`.

## Metric names Tune may pass

`throughput_rps`, `tokens_per_second`, `error_rate`,
`ttft_p50_ms`, `ttft_p99_ms`, `tpot_p50_ms`, `tpot_p99_ms`,
`e2e_p50_ms`, `e2e_p99_ms`.

GPU / cost are **not** confirmable. GPU-not-run stays `None` and is not a pass.

## Evidence (this PR)

CPU / ledger fixtures in `tests/test_repeat_confirmation.py`:

- Interleaved independent repeats under one `RunConditions`
- Search-phase +50% on one pair → `search_winner`, not confirmed, not promotable
- One-shot confirmation pair → `too_noisy` (min_pairs=3)
- Disagreeing pairs → `too_noisy`
- Missing token provenance cannot confirm a tok/s gain
- `incomplete` / `timeout` / `cancel` do not inflate success / RPS
- Unevidenced row + beautiful confirmation numbers → still not promotable
- GPU-not-run ≠ pass
- Hand-built / forged `confirmed_improvement` rejected (including `usable_pairs=0`)
- Duplicate `run_id` / reused ledger cannot count as independent repeats
- `min_pairs <= 0` and `min_rel_delta <= 0` rejected
- Any missing primary → `too_noisy` even when other pairs would suffice
- Mutating a computed `no_diff` / `too_noisy` decision cannot pass the gate
- `evaluate_campaign` without / with a non-interleaved schedule is rejected
- Non-finite `min_rel_delta` (`inf` / `-inf` / `nan`) is rejected

```bash
pytest -q
```

```text
275 passed in 18.28s
```

No real vLLM / GPU numbers are claimed in this environment.

## Out of scope

⑥ Reflect, ⑦ goldens, ⑧ interrupt, Runtime, vanity features.
