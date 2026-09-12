# P0-④ field mapping: existing fields → request ledger

Audit of benchmark / report / MLflow / test paths on current `master`, and how
each maps to the Week-2 ledger. Legacy rows without a ledger stay readable;
aggregates on those rows are **not** independently recalculable.

## Identity (unchanged)

| Existing | Ledger / result | Notes |
|---|---|---|
| `ExperimentResult.run_id` | `RequestLedger.run_id` + `RequestRecord.run_id` | Same UUID hex; MLflow tag `run_id` |
| `experiment_id` | (result only) | Human name; not a ledger key |
| `mlflow_run_id` | (result + MLflow) | Aligned via the same `run_id` tags/params |
| `session_id` | (result) | Session prefix |

Primary ledger key: **`(run_id, request_id)`**.

## Per-request (was missing or aggregated away)

| Old path | New ledger field | Mapping rule |
|---|---|---|
| `RequestMetrics.success` | `outcome` | `True` → `success` only if tokens + clean finish; else classify |
| `RequestMetrics.ttft_ms` (always float; E2E used if no first token) | `ttft_ms` + `t_first_token_s` | **Client TTFT only** when first token observed. Else `None` (never E2E-as-TTFT) |
| `RequestMetrics.e2e_ms` | `e2e_ms` + `t_start_s` / `t_end_s` | Wall duration; kept on fail/timeout |
| *(none — TPOT was `e2e_p[k] - ttft_p[k]` on percentiles)* | `tpot_ms` | Per-request `(e2e−ttft)/(n−1)`; `None` if `n<2` |
| `RequestMetrics.output_tokens` (`max(..., 1)` phantom) | `output_tokens` | Actual count; `0` allowed |
| *(none)* | `input_tokens` | From `usage.prompt_tokens` when present |
| `RequestMetrics.error` | `error` + `http_status` + `finish_reason` | Transport / API signals |
| *(none)* | `termination_reason` | `stop\|length\|timeout\|cancel\|error\|incomplete\|zero_output` |
| *(none)* | `is_warmup` | Warmup stored, excluded from denominators |
| *(none)* | `request_id` | `req-NNNN` / `warmup-NNNN` |

## Aggregates (LoadResult / ExperimentResult / MLflow)

| Old field | New source | Change |
|---|---|---|
| `total_requests` | `len(measured)` | Unchanged meaning; now excludes warmup explicitly |
| `successful_requests` / `LoadResult.successful` | SUCCESS+TRUNCATE | Failures no longer able to inflate by being dropped only from numerator |
| `total_time_s` | `window_end_s − window_start_s` | Documented as **load wall-clock window**, not sum of e2e |
| `throughput_rps` | `successful / window_s` | `None` if window missing/0 |
| `tokens_per_second` | success output tokens / window | No phantom +1 tokens |
| *(none)* | `error_rate` | `failed / total_measured`; failures **in** denominator |
| `ttft` / `tpot` / `e2e_latency` (`LatencyPercentiles` floats, empty→0) | same + `sample_n` + `sample_scope` | Empty → `None`, not `0.0` |
| `raw_ttft_ms` / `raw_e2e_ms` | still filled from eligible rows | Compat for bootstrap CI; ledger is source of truth |
| `gpu_utilization_pct` / `gpu_memory_used_gb` | same | Only if GPU monitor `samples > 0`; else `None` (was `0`) |
| *(none)* | `cost_usd` | Only if pricing assumptions exist |
| *(none)* | `request_ledger` / `ledger_path` | Persisted rows + `logs/ledger_<run_id>.json` |

### Old TPOT bug (removed)

`bench_runner` computed `tpot_p[k] = max(0, e2e_p[k] - ttft_p[k])` — percentile
difference, not per-request TPOT, and forced `0` on missing. Replaced by
`compute_tpot_ms` over ledger rows.

### Old TTFT bug (removed)

Non-stream path set `ttft = e2e`. Streaming without a first token also set
`ttft = e2e`. Both invented client TTFT. Now `ttft_ms=None`.

### Old token bug (removed)

`max(output_tokens, 1)` on every “success” inflated tok/s and hid zero-output
failures. Actual `0` is stored; zero-output classifies as fail/`zero_output`.

## Report / MLflow / tests

| Path | Before | After |
|---|---|---|
| `observability.log_experiment_result` | Always logged rps/ttft/e2e (0 on fail) | Logs only non-`None` metrics; tags `run_id`; artifact `ledger/` |
| `write_report_section` / `write_final_report` | Session summaries | Unchanged UI; metric Markdown via `report_from_ledger` / `format_aggregate_report` |
| `analyze_bottleneck` | `gpu or 0.0`; crash on None latency | Missing latency → `unknown`; unsamped GPU not treated as 0% |
| `compare_experiments` | Fake samples from p50=0 | Missing percentiles → empty samples / error, not 0-ms |
| `agent.state.summary_from_result` | `round(float)` | Missing → `0.0` **in the summary table only**; `validity_status` / `promotable` carry trust (gate unchanged) |
| `tests/test_traffic.py` | Fake `RequestMetrics` | Fake `RequestRecord` with outcomes |
| `eval/metrics.py` (agent eval gaps) | Unchanged | Out of scope (agent-eval, not bench aggregates) |

## Workload conditions

| `WorkloadSpec` | `RunConditions` |
|---|---|
| `name` | `workload_name` |
| `num_requests` | `num_requests` |
| `concurrency` | `concurrency` |
| `input_len` / `output_len` | `input_len_target` / `output_len_target` |
| `distribution` / `rps` | `distribution` / `arrival_rps` |
| *(runner)* `warmup_requests` | `warmup_requests` |
| *(runner)* stream flag | `stream_response` |
| `enable_prefix_caching` (config) | `cache_enabled` when known |
