# Week-2 P0-④: Benchmark metric definitions + request ledger

Measurement trust only. Reuses Week-1 experiment contract (`run_id`, validity,
promotable gate) and config evidence. Does **not** loosen ① gates. Item ⑤
(repeat / confirmation) consumes this ledger and does **not** invent a second
schema. ⑥ / ⑧ remain out of scope here.

Rules live in `inferops/metrics/` (schema + functions), not prose alone.

## Recalculation entrypoint

```python
from inferops.metrics import recalculate_from_ledger, report_from_ledger, report_from_result

agg = recalculate_from_ledger(ledger)          # source of truth
agg, md = report_from_ledger(ledger)           # same numbers + Markdown
agg, md = report_from_result(experiment_result)
```

MLflow uses the **same** `run_id`. Ledger JSON is written to
`logs/ledger_<run_id>.json` and attached as an MLflow artifact under `ledger/`.

Missing metrics are `None` / `n/a`. They are **never** defaulted to `0`.

The session Markdown from `write_final_report` is an agent-level experiment log
(validity / deploy gate). It does **not** recompute bench aggregates; those go
through `report_from_ledger` / `report_from_result` only.

## Metric definitions (encoded)

| Metric | Formula | Denominator / sample scope | Never |
|---|---|---|---|
| **Client TTFT** | `t_first_token − t_start` (streaming only) | `measured_requests_with_client_ttft` (SUCCESS+TRUNCATE with observed first token) | Invent TTFT from E2E on non-stream / no-token |
| **Per-request TPOT** | `(e2e − ttft) / (output_tokens − 1)` | `success_or_truncate_with_usage_tokens_ge_2` | Write `0` when `output_tokens < 2`; use tokens when `token_count_source=missing` |
| **Throughput RPS** | `successful / window_s` | `successful` = SUCCESS+TRUNCATE; window = load wall clock (`window_start_s`→`window_end_s`) | Count fail/timeout/cancel/incomplete as success |
| **Token throughput** | `sum(usage output_tokens of SUCCESS+TRUNCATE) / window_s` | Only `token_count_source=usage`; missing provenance → `None` | Phantom / SSE-chunk / unprovenanced tokens |
| **Error rate** | `failed / total_measured` | `total_measured` = non-warmup rows; failed = FAIL+TIMEOUT+CANCEL+INCOMPLETE | Drop failures from the denominator |
| **E2E percentiles** | nearest-rank on eligible e2e_ms | `success_or_truncate_with_e2e` | Package incomplete as success |
| **GPU util / mem** | monitor average / max | Only if `samples > 0` | Invent 0% / 0 GB |
| **Cost** | n/a | Only if `pricing_assumptions` / `cost_usd` provided | Invent dollars |

Latency percentiles always carry `sample_n` + `sample_scope`.

## Outcomes (cannot fake success)

`success | fail | timeout | cancel | truncate | incomplete`

- **truncate** finishes with length limit — still measured for latency / throughput.
- **incomplete** (lost / partial / HTTP 200 with no tokens and no first token) enters the **error** numerator. Never packaged as success.
- Warmup rows are stored but excluded from all denominators.

## Workload / run conditions (on the ledger)

`RunConditions`: workload name, `num_requests`, concurrency, input/output targets,
arrival (`rps` or closed-loop), warmup count, stream flag, sampling temperature,
cache flag (prefix cache if known). No invented GPU/cost/perf.

## Validity (unchanged from ①)

`is_promotable` still requires `status==valid` + critical evidence + complete
actual-config coverage + `successful_requests > 0`. A rich ledger on an
unevidenced row **cannot** promote best.

## Evidence (this PR)

- Deterministic timestamp fixtures in `tests/test_request_ledger.py`
- Cases: success, fail, timeout, cancel, truncate, zero output, single output token
- Independent recalculation from persisted ledger matches report
- Incomplete ≠ success
- `pytest -q` (see PR body; correction round 2)

Synthetic example (not a real vLLM measurement):

- [`week2_request_ledger_examples/minimal_ledger.json`](./week2_request_ledger_examples/minimal_ledger.json)
- [`week2_request_ledger_examples/minimal_report.md`](./week2_request_ledger_examples/minimal_report.md)

No thin real-vLLM check was run in this environment (no GPU claimed).

## Field mapping (audit → ledger)

See [`week2_field_mapping.md`](./week2_field_mapping.md).

## Stable interfaces for item ⑤

See [`week2_ledger_interfaces.md`](./week2_ledger_interfaces.md).
