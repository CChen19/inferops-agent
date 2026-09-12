# Week-2 P0-⑦: measurement-trust goldens + deterministic CI gates

Eval owns this thin slice. Consumes ④ ledger, ⑤ confirmation, and ⑥
Reflect conclusions already on master. Does **not** redefine ledger
schema, confirmation minting, Reflect actions, or Week-1 `is_promotable`.

## Consume existing stacks

```python
from inferops.metrics import (
    LEDGER_SCHEMA_VERSION,          # "2"
    recalculate_from_ledger,
    compute_tpot_ms,
    verdict_from_ledgers,
    is_confirmed_promotable,
)
from inferops.agent.reflect_constraints import conclude_experiment
from inferops.eval.measurement_goldens import measurement_trust_gate
```

- Missing metrics stay `None` (missing ≠ 0)
- `incomplete` / timeout / cancel are never success
- `token_count_source=missing` is ignored for TPOT / tok-s
- Search winner ≠ `confirmed_improvement` ≠ `is_confirmed_promotable`
- ⑥ `too_noisy` remasures; `no_diff` / `regression` stop as
  `no_reliable_improvement`

## Golden set (thin — 6 cases)

Fixtures under `tests/fixtures/measurement_goldens/`:

| id | Trust rule |
|---|---|
| `missing_requests` | Empty ledger stays `None`; lost request is incomplete; `token_count_source=missing` cannot invent tok/s |
| `failures_not_dropped` | fail / timeout / cancel / incomplete stay in the error numerator and measured denominator |
| `tpot_na` | zero-token / single-token / missing provenance → TPOT N/A, never `0` |
| `search_win_unconfirmed` | search-phase +50% is `search_winner` only; Reflect remasures; not promotable |
| `too_noisy` | disagreeing confirmation pairs → `too_noisy`; Reflect remasures; not promotable |
| `no_reliable_improvement` | confirmation `no_diff` → ⑥ `stop_reason=no_reliable_improvement` |

All fixtures are synthetic (`gpu_sampled=false`). Inventing a GPU/util/cost
number in a fixture fails the gate.

## Deterministic CI gates

`measurement_trust_gate` + `scripts/run_measurement_goldens.py`:

- Required ids must be present — an empty / skipped set is **not** a pass
- GPU-not-run ≠ pass (`INFEROPS_GPU_GOLDENS` unset → CPU goldens only)
- GPU-sampled goldens without that env fail closed
- Expected `confirmed_promotable=true` is refused on this thin set
- Goldens fail if someone loosens missing→0, drops failures, writes TPOT=0,
  treats search as confirm, or turns `no_diff` into a promote

CI (`.github/workflows/eval-mock.yml`) runs the script after `pytest -q`.

```bash
pytest -q
python scripts/run_measurement_goldens.py
```

## Out of scope

⑧ interrupt, Runtime, vanity features. Thresholds and ① promotable gates
are unchanged.

## Evidence

```bash
pytest -q
python scripts/run_measurement_goldens.py
```

```text
327 passed in 15.89s
measurement-trust golden gate passed (CPU/fixture; GPU-not-run ≠ pass)
```

CPU / fixture only. No real vLLM / GPU numbers are claimed in this
environment.

## Week-3 ⑦ closeout

The thin 6-case set above is still the measurement-trust floor. Week-3
closeout expands original ⑦ 验收 via a unified manifest (~24 cases)
spanning these goldens + recovery goldens + error-memory goldens.

See `reports/week3_w3_closeout_⑦.md` and `reports/week3_real_llm.md`.
Real-LLM and GPU layers are **separate** and are not labeled from
these CPU fixtures.
