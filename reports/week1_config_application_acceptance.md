# Week-1 Item ② — Config Application Acceptance Spec

**Owner:** Chris (GPU / vLLM restart path)  
**Status:** Acceptance specification only — **do not treat P0-① as completing ②**.  
**Related:** [`week1_experiment_contract.md`](./week1_experiment_contract.md)

This document is the acceptance gate for proving that a *requested* experiment
config was actually applied to a live vLLM instance. Implementation of GPU
restart / instance-identity verification is out of scope for P0-①.

## Non-negotiable evidence rules

The following are **NEVER** sufficient evidence of config application, alone or
in combination with each other:

1. **Config file alone** — writing or observing a config file does not prove the
   running process loaded those knobs.
2. **HTTP 200 alone** — `/health` (or any OK response) proves liveness, not
   parameter identity.
3. **Performance change alone** — metric deltas can come from noise, warmup,
   contention, or a different workload; they do not identify the live config.

Contract encoding: `INSUFFICIENT_EVIDENCE_KINDS` and
`ConfigEvidence.is_critical_evidence()` in `inferops/schemas.py`. Any path that
only has the above MUST yield `status=insufficient_evidence` and MUST NOT
promote the candidate to best.

## Managed vLLM (InferOps starts / owns the process)

### Requirement

When InferOps manages the lifecycle and the requested knobs differ from the
currently running instance:

1. The managed instance **MUST restart** (or be freshly started) with the
   requested CLI / engine knobs.
2. After restart, acceptance **MUST verify instance identity** (e.g. new PID /
   generation / start token) so a stale process cannot be mistaken for the new
   config.
3. Only then may `actual_config` be recorded and `status` become `valid`
   (assuming knobs match).

### Acceptance checks (Chris)

- [ ] Restart is triggered whenever requested knobs ≠ observed live knobs.
- [ ] Identity after restart differs from the pre-restart identity.
- [ ] Health OK on the *new* identity is recorded, but health alone is not the
      sole evidence.
- [ ] `config_evidence.kind` is a critical kind (e.g. `managed_process_start` or
      `instance_identity`) with `verified=True`.
- [ ] Mismatch between requested and observed knobs → `invalid`, never `valid`.
- [ ] Failure to start / OOM / crash → `failed` (not silently reused).

### Current P0-① interim behavior

`bench_runner` marks managed starts with `managed_process_start` evidence when
**this process** launched vLLM with CLI args matching requested knobs. Full
“healthy-but-different → must restart + verify identity” acceptance remains
Chris’s ② work when an already-running managed instance needs a knob change.

## External vLLM (pre-existing / shared server)

### Requirement

`healthy ≠ config applied`.

If InferOps cannot independently verify that the live external instance is
running the requested knobs (and confirm instance identity), the result MUST be:

```text
status = insufficient_evidence
actual_config = null
```

and MUST NOT be selected as best / deployable.

### Acceptance checks (Chris)

- [ ] External path never sets `status=valid` based on `/health` alone.
- [ ] If knob introspection / identity probe is unavailable → `insufficient_evidence`.
- [ ] If probe exists and knobs disagree → `invalid`.
- [ ] If probe exists, identity confirmed, knobs match → may be `valid` with
      critical evidence (not health-only).

### Current P0-① behavior

External healthy detection sets `external_unverified` evidence (`verified=False`)
and `insufficient_evidence`. This satisfies the “cannot verify → insufficient”
rule until ② adds a verified probe.

## Promotion / reporting implications

| Outcome | Best candidate? | Deploy recommendation? |
|---|---|---|
| `valid` + critical evidence | Eligible | Allowed |
| `invalid` | No | No |
| `failed` | No | No |
| `insufficient_evidence` | No | No |

Eval best-of-session (`eval/runner._best_agent_result`) only considers
`promotable_only=True` rows.

## Out of scope for this doc / for P0-①

- Items ④⑤⑦
- Strategy / search-space changes
- Loosening any regression or safety thresholds
- Implementing the GPU restart machinery itself (Chris)
