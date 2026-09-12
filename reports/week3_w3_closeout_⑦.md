# Week-3 ⑦ closeout: unified goldens (original 验收)

Eval owns this residual. Thin W2 ⑦ (6 measurement-trust cases) did **not**
meet original ⑦ 验收. This closeout expands it without loosening
thresholds, rewriting Tune Reflect, or starting ⑨.

Baseline: `origin/master` @ `a0c7061` (Tune ⑧ residual freeze `fcb0f48`).

## Consume existing stacks (do not redefine)

| Stack | Frozen rule Eval keeps |
|---|---|
| ① `is_promotable` / `derive_status` | `valid` + critical evidence + full actual cover + `successful_requests > 0` |
| ④ ledger v2 | Missing ≠ 0. Incomplete stays in the error denominator |
| ⑤ `is_confirmed_promotable` | Search winner / partial campaign ≠ confirm |
| ⑥ Reflect | `too_noisy` remasures; `no_diff` → `no_reliable_improvement` |
| Tune ⑧ recovery | `last_recovery` fields on master; checkpoint/resume; budget semantics |
| Week-1 memory | `save_result` / `query_results(promotable_only=True)` / `get_promotable_result` |

## Unified manifest

Structured data consumed by the CI gate:

```text
tests/fixtures/unified_goldens/manifest.json
inferops/eval/unified_goldens.py
scripts/run_unified_goldens.py
```

27 cases (inside 20–30). Each records `source`, `expected_behavior`,
`judge_rule`, `reviewer`, `holdout`.

| Source | Count | Holdout |
|---|---:|---:|
| measurement_goldens (W2 ⑦ floor) | 6 | `tpot_na`, `no_reliable_improvement` |
| recovery_goldens (W3 ⑧ + #13 freeze) | 12 | `idempotent_re_resume`, `resume_equivalence`, `trajectory_audit_executor_reflect` |
| error_memory_goldens (this closeout) | 9 | `unevidenced_high_score_not_promotable`, `promotable_only_excludes_errors` |

Holdout cases are unused for prompt tuning. The gate fails if those ids
appear in `inferops.eval.judge` few-shot / rubric or `data/corpus/tuning_notes.md`.

## Error-memory goldens

New deterministic set. Consumes ① + experiment memory only:

| id | Judge rule |
|---|---|
| `failed_row_remembered_not_promotable` | Failed unrun row is stored; not promotable |
| `unevidenced_high_score_not_promotable` | High RPS without evidence ≠ best |
| `invalid_actual_mismatch_not_promotable` | Mismatched actual → `invalid` |
| `insufficient_evidence_cli_only_not_promotable` | CLI-only actual stays insufficient |
| `zero_success_failed_not_promotable` | `successful_requests=0` → `failed` |
| `legacy_incomplete_row_not_promotable` | Legacy default `insufficient_evidence` |
| `duplicate_failed_config_stays_remembered` | Upsert keeps one failed row |
| `oom_failed_row_not_best` | OOM-at-start failed ≠ best |
| `promotable_only_excludes_errors` | Mixed set → only Week-1-valid row |

## Layer separation

| Layer | How it is labeled | Pass rule |
|---|---|---|
| Offline / fixture | `offline_fixture` — measurement + recovery + error-memory | Fail-closed. Empty / skipped **FAIL** |
| Real LLM | `scripts/run_real_llm_goldens.py` only. `llm_boundary=live` | N≥3 when a key exists. `pass_rate` = accepted/n_requested via `judge_live_run` (not “call didn’t throw”). Fake/offline / missing / unknown boundary **must not** be labeled live. External campaign JSON that claims pass without live + N≥3 + consistent counts **FAIL** |
| GPU | `GPU: 未执行` / `Blocked` unless a 3060 worker archives real logs | GPU-not-run ≠ pass. No invented numbers |

See `reports/week3_real_llm.md` for the live-LLM campaign (blocked in this
environment — no `OPENROUTER_API_KEY`). A live-boundary report with empty
rows, zero quality, or `eval_empty_plan` is **not** accepted and cannot
inflate `pass_rate` to 3/3.

## Tune ⑧ freeze (`fcb0f48` → `a0c7061`)

Matching recovery goldens consume only this surface:

- `RECOVERY_FIELDS += validity_status, retry_count`
- `codes += ack_lost`
- unconfirmable → Week-1 `insufficient_evidence`
- `TRAJECTORY_AUDIT_FIELDS = retry_count, budget_consumed, next_action, stop_reason` (executor + Reflect)

Do **not** invent `receipt_lost` / `incomplete_receipt` as recovery fields.
The unified gate still fails if a case claims those names.

## CI

`.github/workflows/eval-mock.yml`:

```bash
pytest -q
python scripts/run_measurement_goldens.py
python scripts/run_recovery_goldens.py
python scripts/run_error_memory_goldens.py
python scripts/run_unified_goldens.py
python scripts/run_real_llm_goldens.py   # blocker report if no API key
```

## Out of scope

⑨ controlled对照. Tune Reflect rewrite. Threshold / ① loosening.
Invented GPU / vLLM numbers. Auto-merge — Chris merges manually.

## Evidence

```bash
pytest -q
python scripts/run_measurement_goldens.py
python scripts/run_recovery_goldens.py
python scripts/run_error_memory_goldens.py
python scripts/run_unified_goldens.py
python scripts/run_real_llm_goldens.py
```

```text
419 passed in 16.86s
measurement-trust golden gate passed (CPU/fixture; GPU-not-run ≠ pass)
recovery golden gate passed (CPU/fixture; GPU-not-run ≠ pass)
error-memory golden gate passed (CPU/fixture; GPU-not-run ≠ pass)
unified golden gate passed (offline fixtures; GPU-not-run ≠ pass; real-LLM is separate)
real-LLM: BLOCKED (not a pass; not fake-LLM)
```

```text
GPU: 未执行 / Blocked — no 3060 / no GPU worker in this environment.
real-LLM: BLOCKED — OPENROUTER_API_KEY missing. Not a pass. Not fake-LLM.
```

CPU / fixture only for the deterministic gates. No invented vLLM / GPU
numbers.
