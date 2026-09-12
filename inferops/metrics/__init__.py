"""Benchmark metric definitions + recalculable per-request ledger (Week-2 P0-④).

Public surface consumed by bench_runner, reports, MLflow, and (later) item ⑤:
  - RequestRecord / RequestLedger / RunConditions
  - TerminationReason / RequestOutcome
  - AggregateMetrics / LatencyStat
  - compute_tpot_ms, recalculate_from_ledger, ledger_to_experiment_fields
  - persist_ledger / load_ledger
"""

from inferops.metrics.aggregate import (
    AggregateMetrics,
    LatencyStat,
    apply_aggregate_to_result_fields,
    format_aggregate_report,
    ledger_to_experiment_fields,
    recalculate_from_ledger,
)
from inferops.metrics.definitions import (
    ERROR_OUTCOMES,
    LATENCY_SUCCESS_OUTCOMES,
    TPOT_MIN_OUTPUT_TOKENS,
    TTFT_SAMPLE_SCOPE,
    TPOT_SAMPLE_SCOPE,
    E2E_SAMPLE_SCOPE,
    compute_tpot_ms,
    is_error_for_rate,
    is_latency_eligible,
    is_tpot_eligible,
    is_ttft_eligible,
    percentile,
)
from inferops.metrics.ledger import (
    LEDGER_SCHEMA_VERSION,
    RequestLedger,
    RequestOutcome,
    RequestRecord,
    RunConditions,
    TokenCountSource,
    TerminationReason,
    load_ledger,
    persist_ledger,
)
from inferops.metrics.report import (
    ledger_from_result,
    report_from_ledger,
    report_from_result,
)

__all__ = [
    "ERROR_OUTCOMES",
    "LATENCY_SUCCESS_OUTCOMES",
    "TPOT_MIN_OUTPUT_TOKENS",
    "TTFT_SAMPLE_SCOPE",
    "TPOT_SAMPLE_SCOPE",
    "E2E_SAMPLE_SCOPE",
    "AggregateMetrics",
    "LatencyStat",
    "LEDGER_SCHEMA_VERSION",
    "RequestLedger",
    "RequestOutcome",
    "RequestRecord",
    "RunConditions",
    "TokenCountSource",
    "TerminationReason",
    "apply_aggregate_to_result_fields",
    "compute_tpot_ms",
    "format_aggregate_report",
    "is_error_for_rate",
    "is_latency_eligible",
    "is_tpot_eligible",
    "is_ttft_eligible",
    "ledger_from_result",
    "ledger_to_experiment_fields",
    "load_ledger",
    "percentile",
    "persist_ledger",
    "recalculate_from_ledger",
    "report_from_ledger",
    "report_from_result",
]
