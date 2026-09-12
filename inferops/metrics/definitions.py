"""Encoded metric rules (not prose alone).

Every aggregate must be recalculable from per-request ledger rows.
Failures / timeouts / cancels / truncations / short outputs must not invent
fake gains (no phantom tokens, no TPOT=0 for single-token, no missing→0).
"""

from __future__ import annotations

from typing import Iterable, Sequence

from inferops.metrics.ledger import RequestOutcome, RequestRecord, TerminationReason

# ---------------------------------------------------------------------------
# Constants (stable for item ⑤ consumers)
# ---------------------------------------------------------------------------

# TPOT requires at least this many *decode* intervals after the first token.
# With N output tokens there are (N - 1) inter-token gaps. N <= 1 → TPOT N/A.
TPOT_MIN_OUTPUT_TOKENS: int = 2

# Outcomes that enter the error-rate numerator.
ERROR_OUTCOMES: frozenset[RequestOutcome] = frozenset(
    {
        RequestOutcome.FAIL,
        RequestOutcome.TIMEOUT,
        RequestOutcome.CANCEL,
        RequestOutcome.INCOMPLETE,
    }
)

# Outcomes eligible for latency percentiles (explicitly excludes incomplete).
LATENCY_SUCCESS_OUTCOMES: frozenset[RequestOutcome] = frozenset(
    {
        RequestOutcome.SUCCESS,
        RequestOutcome.TRUNCATE,  # finished with length limit; still measured
    }
)

TTFT_SAMPLE_SCOPE = "measured_requests_with_client_ttft"
TPOT_SAMPLE_SCOPE = "success_or_truncate_with_output_tokens_ge_2"
E2E_SAMPLE_SCOPE = "success_or_truncate_with_e2e"


# ---------------------------------------------------------------------------
# Per-request formulas
# ---------------------------------------------------------------------------

def compute_tpot_ms(
    *,
    e2e_ms: float | None,
    ttft_ms: float | None,
    output_tokens: int,
) -> float | None:
    """Per-request TPOT = (e2e − ttft) / (output_tokens − 1).

    Returns None (N/A / missing) when:
      - output_tokens < 2 (single or zero output → NEVER write 0)
      - e2e or ttft missing
      - e2e < ttft (clock anomaly)
    """
    if output_tokens < TPOT_MIN_OUTPUT_TOKENS:
        return None
    if e2e_ms is None or ttft_ms is None:
        return None
    if e2e_ms < ttft_ms:
        return None
    return (e2e_ms - ttft_ms) / (output_tokens - 1)


def is_error_for_rate(record: RequestRecord) -> bool:
    """True if this measured request counts in the error-rate numerator."""
    if record.is_warmup:
        return False
    return record.outcome in ERROR_OUTCOMES


def is_latency_eligible(record: RequestRecord) -> bool:
    if record.is_warmup:
        return False
    return record.outcome in LATENCY_SUCCESS_OUTCOMES


def is_ttft_eligible(record: RequestRecord) -> bool:
    """Client TTFT sample: latency-eligible AND ttft actually measured."""
    return is_latency_eligible(record) and record.ttft_ms is not None


def is_tpot_eligible(record: RequestRecord) -> bool:
    """TPOT sample: latency-eligible AND tpot computed (never invent 0)."""
    return is_latency_eligible(record) and record.tpot_ms is not None


# ---------------------------------------------------------------------------
# Percentiles
# ---------------------------------------------------------------------------

def percentile(data: Sequence[float], p: float) -> float | None:
    """Nearest-rank percentile. Empty → None (never 0.0 as a fake value)."""
    if not data:
        return None
    if p <= 0:
        return float(data[0])
    if p >= 100:
        return float(data[-1])
    sorted_data = sorted(data)
    # nearest-rank: index = ceil(p/100 * n) - 1  ≈ int(n * p / 100) clamped
    idx = int(len(sorted_data) * p / 100)
    idx = min(max(idx, 0), len(sorted_data) - 1)
    return float(sorted_data[idx])


def percentiles(data: Sequence[float]) -> dict[str, float | None]:
    s = sorted(data)
    return {
        "p50": percentile(s, 50),
        "p90": percentile(s, 90),
        "p95": percentile(s, 95),
        "p99": percentile(s, 99),
    }


def classify_http_outcome(
    *,
    status_code: int | None,
    timed_out: bool,
    cancelled: bool,
    finish_reason: str | None,
    output_tokens: int,
    first_token_seen: bool,
    error: str = "",
) -> tuple[RequestOutcome, TerminationReason]:
    """Map raw transport/API signals → outcome + termination (no fake success)."""
    if cancelled:
        return RequestOutcome.CANCEL, TerminationReason.CANCEL
    if timed_out:
        return RequestOutcome.TIMEOUT, TerminationReason.TIMEOUT
    if status_code is not None and status_code != 200:
        return RequestOutcome.FAIL, TerminationReason.ERROR
    if error:
        return RequestOutcome.FAIL, TerminationReason.ERROR
    # Streaming finished without a clean stop/length and no usable body.
    if not first_token_seen and output_tokens <= 0:
        return RequestOutcome.INCOMPLETE, TerminationReason.INCOMPLETE

    fr = (finish_reason or "").lower()
    if fr in {"length", "max_tokens", "max_token"}:
        return RequestOutcome.TRUNCATE, TerminationReason.LENGTH
    if output_tokens <= 0:
        # Completed HTTP but zero completion tokens — not a latency success.
        return RequestOutcome.FAIL, TerminationReason.ZERO_OUTPUT
    if fr in {"stop", "end_turn", "eos", ""}:
        return RequestOutcome.SUCCESS, TerminationReason.STOP
    # Unknown finish_reason with tokens: treat as success but keep reason.
    return RequestOutcome.SUCCESS, TerminationReason.STOP


def measured_records(records: Iterable[RequestRecord]) -> list[RequestRecord]:
    return [r for r in records if not r.is_warmup]
