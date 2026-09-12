# Week-1 Experiment Contract (P0-①)

This document defines the minimal experiment contract for InferOps. Rules are
enforced in schema helpers and selection gates — not prose alone.

**Scope note:** Completing ① (this contract) does **not** mean ② (GPU config
application / restart acceptance) is complete. Item ② is specified in
[`week1_config_application_acceptance.md`](./week1_config_application_acceptance.md)
and owned by Chris for GPU work.

## Identity mapping

Every persisted experiment row carries:

| Field | Role |
|---|---|
| `run_id` | Unique opaque id for this execution (UUID hex). Aligns MLflow tags / params. |
| `experiment_id` | Human-readable / session-scoped name (e.g. `agent_chat_abc_baseline`). |
| `session_id` | Session prefix shared by all runs in one agent session. |
| `mlflow_run_id` | MLflow active-run id when logging succeeded. |

MLflow tags/params always include `run_id`, `experiment_id`, `session_id`,
`schema_version`, `status`, `workload_hash`, and `code_sha` when available.

## Provenance

| Field | Meaning |
|---|---|
| `schema_version` | Contract version (`"1"` for Week-1). |
| `code_sha` | Git short SHA of the code that produced the result. |
| `hardware` | Model name, engine, optional vLLM / GPU / CUDA fingerprint. |
| `workload_hash` | Stable hash of workload shape (name, counts, lengths, rps). |

## Requested vs actual config + evidence

- `config` / `requested_config` — what the agent *asked* to run.
- `actual_config` — knobs verified on the live instance (or `null` if unknown).
- `config_evidence` — structured proof (`kind`, `verified`, `instance_id`, …).

### Status enum

```text
valid | invalid | failed | insufficient_evidence
```

Derivation (`inferops.schemas.derive_status`):

1. Explicit failure → `failed`
2. Missing / non-critical evidence → `insufficient_evidence`
3. Critical evidence but `actual_config` missing → `insufficient_evidence`
4. Critical evidence + actual knobs disagree with requested → `invalid`
5. Critical evidence + actual matches requested → `valid`

**Legacy / incomplete rows default to `insufficient_evidence`.** Old data must
never auto-promote to `valid`.

### Evidence that is NEVER sufficient alone

Encoded in `INSUFFICIENT_EVIDENCE_KINDS` / `ConfigEvidence.is_critical_evidence()`:

- config file alone
- HTTP 200 / health check alone
- performance change alone
- `external_unverified` (healthy external server without instance identity)

## Best-candidate gate

A candidate may become `best_summary` / deploy recommendation **only if**
`is_promotable(result)` is true:

```text
status == valid
AND actual_config is not None
AND config_evidence.is_critical_evidence()
```

High primary-metric scores alone never promote. Executor, baseline seeding,
eval `_best_agent_result`, and final-report deploy text all honor this gate.

### Failing path closed by this change

Before: `executor_node` compared only primary metric and promoted DB-resumed
rows with no actual-config evidence (`inferops/agent/executor.py` best update).
External vLLM path stored requested config as if applied (`bench_runner`).

After: external → `insufficient_evidence`; executor refuses promotion; reports
withhold deploy; eval best uses `promotable_only=True`.

## Sample status reports

See `reports/week1_status_samples/` for Markdown samples covering all four
statuses (synthetic fixtures — **not** real performance claims).
