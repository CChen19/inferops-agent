"""Observability bootstrap: MLflow experiment tracking + OpenTelemetry spans."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Generator

import mlflow
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

_tracer: trace.Tracer | None = None


# ---------------------------------------------------------------------------
# OpenTelemetry
# ---------------------------------------------------------------------------

def init_otel(service_name: str = "inferops-agent") -> trace.Tracer:
    """Configure a console-exporting OTel tracer (swap exporter later for LangSmith/OTLP)."""
    global _tracer
    if _tracer is not None:
        return _tracer

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    # Console exporter — zero external deps, easy to grep
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name)
    return _tracer


def get_tracer() -> trace.Tracer:
    if _tracer is None:
        return init_otel()
    return _tracer


@contextmanager
def span(name: str, attributes: dict[str, Any] | None = None) -> Generator[trace.Span, None, None]:
    """Thin context manager: `with span("my-step", {"k": v}): ...`"""
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as s:
        if attributes:
            for k, v in attributes.items():
                s.set_attribute(k, str(v))
        yield s


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------

_MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")  # local SQLite


def init_mlflow(experiment_name: str = "inferops") -> str:
    """Point MLflow at the local tracking store and create/return the experiment."""
    mlflow.set_tracking_uri(_MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    exp = mlflow.get_experiment_by_name(experiment_name)
    assert exp is not None
    return exp.experiment_id


@contextmanager
def mlflow_run(
    run_name: str,
    tags: dict[str, str] | None = None,
) -> Generator[mlflow.ActiveRun, None, None]:
    """Start an MLflow run and yield it; logs are auto-flushed on exit."""
    with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
        yield run


def log_experiment_result(result: Any) -> None:
    """Log an ExperimentResult to the active MLflow run."""
    from inferops.schemas import ExperimentResult  # avoid circular at module level

    assert isinstance(result, ExperimentResult)
    cfg = result.config

    mlflow.log_params(
        {
            "model": cfg.model_name,
            "model_size": cfg.model_size.value,
            "max_num_seqs": cfg.max_num_seqs,
            "max_num_batched_tokens": cfg.max_num_batched_tokens,
            "gpu_memory_utilization": cfg.gpu_memory_utilization,
            "enforce_eager": cfg.enforce_eager,
            "enable_chunked_prefill": cfg.enable_chunked_prefill,
            "enable_prefix_caching": cfg.enable_prefix_caching,
            "scheduler_policy": cfg.scheduler_policy.value,
            "workload": cfg.workload.name,
            "concurrency": cfg.workload.concurrency,
        }
    )
    mlflow.log_metrics(
        {
            "throughput_rps": result.throughput_rps,
            "tokens_per_second": result.tokens_per_second,
            "ttft_p50_ms": result.ttft.p50,
            "ttft_p99_ms": result.ttft.p99,
            "e2e_p50_ms": result.e2e_latency.p50,
            "e2e_p99_ms": result.e2e_latency.p99,
        }
    )
    if result.gpu_memory_used_gb is not None:
        mlflow.log_metric("gpu_memory_used_gb", result.gpu_memory_used_gb)
