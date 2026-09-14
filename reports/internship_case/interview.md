# Interview freeze

**Interview SHA:** `2731bed` (master through merged PR #61).

**One-sentence story:** This is a constrained experiment-decision system that
keeps the baseline when evidence is insufficient — not a LangGraph/RAG
auto-tune win.

## Read these, in this order

| Doc | Why |
|---|---|
| [`case.md`](case.md) | One live planner run: kept baseline at 18.135 rps (`26dd06f`) |
| [`fair_compare.md`](fair_compare.md) | Exploratory planner vs search compare; different SHAs and claim levels |
| [`demo.md`](demo.md) | 3–5 min script; no live GPU; pinned to this freeze SHA |
| [`architecture.md`](architecture.md) | How confirmation, promotion, and evidence gates work |
| [`limits.md`](limits.md) | Verified claims vs known gaps |

## Fair-compare numbers (do not rank strategies)

| Arm | Best observed | SHA | Claim |
|---|---|---|---|
| Planner (adopted) | **18.135 rps** baseline | `26dd06f` | production decision: keep baseline |
| Search (observed) | **19.145 rps** | `277167e` | observe-after-pick protocol score; **unconfirmed** |

`confirmed_gain` is `null`. The compare is exploratory (different SHAs, different
`claim_level`s), **not** a strategy ranking. Do not say the planner beat search,
or that search beat the planner at adoption.

## Capability boundary

| Capability | Status |
|---|---|
| Live planner run returns `no_reliable_improvement` and keeps baseline | **Proven live** (`live_3060_case_v2`, SHA `26dd06f`) |
| Search arm observes 19.145 rps with `confirmed_gain: null` | **Proven live** (protocol score only; SHA `277167e`) |
| SLO-invalid trial rejected while engine stays healthy | **Proven live** (`livefair_search_01`) |
| Confirmation campaign on the live case | **Not proven** — best candidate was 3.77% < 5% threshold; campaign never ran |
| Memory reducing wasted trials | **Not proven live** — PR #64 CPU pre-experiment only (after freeze); no live waste reduction; see below |
| Agent vs search at adoption | **Not proven** — different claim levels and SHAs; no adoption winner |
| Document citations | **Existence-only** — structured `(chunk_id, source, version)` match when retrieved; not semantic support |
| Resume, citation gates, managed lifecycle, report precision | **Proven CPU** — merged tests/code; not new GPU measurements |

Detail and citations for each row live in [`limits.md`](limits.md).

## After the freeze (current master notes)

Interview claims stay pinned to **`2731bed`**. Later merges on master are
recorded here so the pack does not go stale, but they are **not** part of the
interview SHA and do not retarget the demo.

| Item | Status on master |
|---|---|
| PR #63 interview freeze | Merged at `61374af` (docs that pinned `2731bed`) |
| PR #62 closeout small-fixes | Merged at `5cb9e20` — resume copy / history attribution / index version / hardware fingerprint. **After** the freeze; not claimed in the interview SHA |
| PR #65 interview note for #62 | Merged (docs hygiene after freeze) |
| PR #64 memory pre-experiment | Merged at `cec13ef` — **After** the freeze; CPU fixture result only; **not** part of the interview claim surface |

### PR #64 CPU pre-experiment (honest result only)

Scripted proposal order; outcomes from `HiddenResultFixture` over ground truth.
Not a paid LLM using memory. Not a live GPU waste-reduction claim. Reusable OOM
on `t2048_c1_p0` is a **scenario override** of GT (GT row is valid 16.0 rps),
not a published GPU failure.

Goal: `valid` + SLO and `throughput_rps >= 17.2` — not “a valid baseline exists.”

| Arm | Reusable OOM waste | Notes |
|---|---|---|
| A — no cross-session memory | 1 extra execute | Baseline waste |
| B — exact lasting config-failure filter | 0 | Timeout/spawn are **not** lasting blacklists; OOM **is** lasting |
| C — `query_compatible_history` + `is_history_failure` | 0 | On the transient-timeout scenario, C **wrongly filters a feasible goal config** (`is_history_failure` treats failed timeout as history failure). Over-filter is a disclosed result |

**Continue-threshold decision:** B is enough. C adds no extra help and over-filters
transients → keep the simple lasting exact-config filter; do **not** upgrade RAG;
do **not** start a GPU memory rerun.

## Still not claimed

- Memory reducing **live** wasted trials (PR #64 is CPU-only; no live waste-reduction result).
- Confirmation campaign on the live case (3.77% < 5%; never ran).
- Agent vs search at adoption (exploratory compare only; `confirmed_gain` null).
- That a paid LLM planner uses memory to cut waste (pre-experiment was scripted).

Anything else merged after `2731bed` is outside this freeze unless a later
freeze doc says otherwise.
