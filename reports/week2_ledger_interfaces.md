# Stable interfaces for item ⑤ (consumers)

Item ⑤ (not this PR) should consume these names. Do not invent parallel schemas.

## Identity

- `run_id: str` — opaque hex; same value on `ExperimentResult`, `RequestLedger`, MLflow tags/params.
- `request_id: str` — unique within a run (`req-NNNN` / `warmup-NNNN`).
- Key: `(run_id, request_id)`.

## Schema versions

- `EXPERIMENT_SCHEMA_VERSION = "1"` (`inferops.schemas`) — Week-1 contract, unchanged.
- `LEDGER_SCHEMA_VERSION = "2"` (`inferops.metrics.ledger`).
  v2: `token_count_source=missing` makes `output_tokens` unusable for TPOT / tok-s.

## Types

| Name | Module | Role |
|---|---|---|
| `RequestOutcome` | `inferops.metrics.ledger` | `success\|fail\|timeout\|cancel\|truncate\|incomplete` |
| `TerminationReason` | `inferops.metrics.ledger` | `stop\|length\|timeout\|cancel\|error\|incomplete\|zero_output` |
| `RequestRecord` | `inferops.metrics.ledger` | One row (`token_count_source`: `usage` \| `missing`) |
| `TokenCountSource` | `inferops.metrics.ledger` | Never invent tokens from SSE chunks |
| `RequestLedger` | `inferops.metrics.ledger` | Rows + `RunConditions` + window |
| `RunConditions` | `inferops.metrics.ledger` | Workload / arrival / warmup / cache / sampling |
| `AggregateMetrics` | `inferops.metrics.aggregate` | Recalculated summary |
| `LatencyStat` | `inferops.metrics.aggregate` | Percentiles + `sample_n` + `sample_scope` |
| `LatencyPercentiles` | `inferops.schemas` | Result field; `p*` may be `None` |
| `ExperimentResult.request_ledger` | `inferops.schemas` | Embedded list of record dicts |
| `ExperimentResult.ledger_path` | `inferops.schemas` | On-disk JSON |

## Functions

| Name | Contract |
|---|---|
| `compute_tpot_ms(e2e_ms, ttft_ms, output_tokens, token_count_source)` | `None` if source ≠ `usage` or `output_tokens < 2` — never `0` |
| `classify_http_outcome(...)` | Maps HTTP / timeout / cancel / finish_reason → outcome |
| `recalculate_from_ledger(ledger, gpu_*=None, cost_usd=None)` | **Only** aggregate entrypoint |
| `report_from_ledger` / `report_from_result` | Report entrypoint (same numbers) |
| `persist_ledger` / `load_ledger` | JSON round-trip |
| `ledger_from_result` | Rebuild ledger from a result |
| `is_promotable` / `derive_status` | Week-1 gates — **do not loosen** |

## Persistence

- File: `logs/ledger_<run_id>.json`
- MLflow: tags/params `run_id`; artifact path `ledger/`
- SQLite: `result_json` may embed `request_ledger`; contract columns unchanged

## What ⑤ must not do

- Treat `incomplete` / timeout / cancel as success.
- Fill missing TTFT/TPOT/GPU/cost with `0`.
- Promote `best` without `is_promotable`.
- Invent a second `run_id`.
