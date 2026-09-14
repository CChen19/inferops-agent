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
| Memory reducing wasted trials | **Not proven** — compatible-history / memory work is in flight or CPU-scoped; no live waste reduction shown |
| Agent vs search at adoption | **Not proven** — different claim levels and SHAs; no adoption winner |
| Document citations | **Existence-only** — structured `(chunk_id, source, version)` match when retrieved; not semantic support |
| Resume, citation gates, managed lifecycle, report precision | **Proven CPU** — merged tests/code; not new GPU measurements |

Detail and citations for each row live in [`limits.md`](limits.md).

## In flight — not claimed in this freeze

- **PR #62 closeout** (resume copy / history attribution / index version /
  hardware fingerprint): Opus eval **FAIL** on planner fingerprint wiring;
  being fixed. Not part of the interview claim surface.
- **Memory pre-experiment:** in flight, not done. Do not claim memory already
  reduces wasted trials on live hardware.

Anything merged after `2731bed` is outside this freeze unless a later freeze
doc says otherwise.
