"""Per-request request ledger keyed by run_id + request_id."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field, model_validator

LEDGER_SCHEMA_VERSION = "1"


class RequestOutcome(str, Enum):
    """High-level request fate used for denominators / latency eligibility."""

    SUCCESS = "success"
    FAIL = "fail"
    TIMEOUT = "timeout"
    CANCEL = "cancel"
    TRUNCATE = "truncate"
    INCOMPLETE = "incomplete"  # lost / partial — never packaged as success


class TerminationReason(str, Enum):
    """Why the request ended (finer grain than outcome)."""

    STOP = "stop"
    LENGTH = "length"
    TIMEOUT = "timeout"
    CANCEL = "cancel"
    ERROR = "error"
    INCOMPLETE = "incomplete"
    ZERO_OUTPUT = "zero_output"


class RunConditions(BaseModel):
    """Workload / concurrency / arrival / sampling / warmup / cache context.

    Attached to the ledger so aggregates are interpretable without inventing
    numbers. Missing fields stay None — never invent GPU/cost/perf.
    """

    workload_name: str = ""
    num_requests: int | None = None
    concurrency: int | None = None
    input_len_target: int | None = None
    output_len_target: int | None = None
    distribution: str | None = None
    arrival_rps: float | None = None  # None = closed-loop / as-fast-as-possible
    warmup_requests: int = 0
    stream_response: bool = True  # client TTFT requires True
    sampling_temperature: float | None = 0.0
    cache_enabled: bool | None = None  # prefix cache / engine cache flag if known
    pricing_assumptions: dict[str, Any] | None = None  # cost only if present


class RequestRecord(BaseModel):
    """One measured (or warmup) request. Primary key: (run_id, request_id)."""

    run_id: str
    request_id: str
    is_warmup: bool = False

    # Timestamps (epoch seconds, float). Prefer these for recalculation.
    t_start_s: float
    t_first_token_s: float | None = None
    t_end_s: float | None = None

    # Derived durations (ms). Stored for convenience; recalculation prefers
    # timestamps when both ends are present.
    ttft_ms: float | None = None
    e2e_ms: float | None = None
    tpot_ms: float | None = None  # None when output_tokens < 2 — NEVER 0

    # Actual token counts (never invent; 0 is allowed and meaningful)
    input_tokens: int | None = None
    output_tokens: int = 0

    outcome: RequestOutcome
    termination_reason: TerminationReason
    http_status: int | None = None
    finish_reason: str | None = None
    error: str = ""

    @model_validator(mode="after")
    def _derive_durations_and_guard_tpot(self) -> RequestRecord:
        from inferops.metrics.definitions import compute_tpot_ms

        # Derive durations from timestamps when possible.
        if self.ttft_ms is None and self.t_first_token_s is not None:
            self.ttft_ms = (self.t_first_token_s - self.t_start_s) * 1000.0
        if self.e2e_ms is None and self.t_end_s is not None:
            self.e2e_ms = (self.t_end_s - self.t_start_s) * 1000.0

        # Recompute TPOT from canonical formula; never leave a fake 0 for N=1.
        computed = compute_tpot_ms(
            e2e_ms=self.e2e_ms,
            ttft_ms=self.ttft_ms,
            output_tokens=self.output_tokens,
        )
        if self.output_tokens < 2:
            # Hard rule: single / zero output → TPOT missing, never 0.
            self.tpot_ms = None
        elif computed is not None:
            self.tpot_ms = computed
        elif self.tpot_ms == 0.0 and self.output_tokens < 2:
            self.tpot_ms = None

        # Incomplete / fail paths must not look like success with TTFT=0.
        if self.outcome == RequestOutcome.INCOMPLETE:
            # Keep measured partials, but never invent TTFT from e2e.
            pass
        return self

    @property
    def key(self) -> tuple[str, str]:
        return (self.run_id, self.request_id)


class RequestLedger(BaseModel):
    """All per-request rows for one experiment run, keyed by run_id."""

    run_id: str
    schema_version: str = LEDGER_SCHEMA_VERSION
    conditions: RunConditions = Field(default_factory=RunConditions)
    records: list[RequestRecord] = Field(default_factory=list)
    # Wall-clock window used for throughput (set by load runner).
    window_start_s: float | None = None
    window_end_s: float | None = None

    def add(self, record: RequestRecord) -> None:
        if record.run_id != self.run_id:
            raise ValueError(
                f"record.run_id={record.run_id!r} does not match ledger "
                f"run_id={self.run_id!r}"
            )
        self.records.append(record)

    def extend(self, records: Iterable[RequestRecord]) -> None:
        for r in records:
            self.add(r)

    def measured(self) -> list[RequestRecord]:
        return [r for r in self.records if not r.is_warmup]

    def by_request_id(self, request_id: str) -> RequestRecord | None:
        for r in self.records:
            if r.request_id == request_id:
                return r
        return None


def persist_ledger(ledger: RequestLedger, path: Path | str) -> Path:
    """Write ledger JSON (stable for independent recalculation)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(ledger.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return p


def load_ledger(path: Path | str) -> RequestLedger:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return RequestLedger.model_validate(data)
