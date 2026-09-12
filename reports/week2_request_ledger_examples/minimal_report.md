# Minimal recalculated report (synthetic fixture — not real perf)

### Metrics for `run_id=fixture_run_aaaaaaaaaaaaaaaaaaaa`

- **total_requests** (error-rate denominator): 7
- **successful_requests** (SUCCESS+TRUNCATE): 3
- **failed_requests** (error numerator): 4
- **error_rate**: 0.5714
- **throughput window (s)**: 10.0000
- **throughput_rps** (successful / window): 0.300
- **tokens_per_second** (success output tokens / window): 2.100
- **TTFT (ms)**: p50=100.0 p90=200.0 p95=200.0 p99=200.0 (n=3, scope=`measured_requests_with_client_ttft`)
- **TPOT (ms)**: p50=200.0 p90=333.3 p95=333.3 p99=333.3 (n=3, scope=`success_or_truncate_with_output_tokens_ge_2`)
- **E2E (ms)**: p50=1100.0 p90=1200.0 p95=1200.0 p99=1200.0 (n=3, scope=`success_or_truncate_with_e2e`)
- **outcome_counts**: `{'success': 2, 'truncate': 1, 'fail': 1, 'timeout': 1, 'cancel': 1, 'incomplete': 1}`
- **gpu_utilization_pct**: n/a (not sampled)
- **gpu_memory_used_gb**: n/a (not sampled)
- **cost_usd**: n/a (no pricing assumptions)

Ledger: `reports/week2_request_ledger_examples/minimal_ledger.json`
