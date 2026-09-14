# InferOps

**Local vLLM Serving Optimization & Experiment-Decision Assistant**

InferOps drafts, tests, and evaluates serving configurations for local vLLM workloads. It benchmarks candidate parameter assignments against your baseline under concurrency limits, and recommends keeping the baseline whenever measured evidence does not reliably demonstrate an improvement.

---

### Key Operational Guarantees

- **Confirm-Before-GPU:** Every proposed optimization task requires explicit user confirmation before any GPU budget is spent or benchmark process is started.
- **Evidence-Based Evaluation:** Candidates must reliably improve throughput while strictly respecting TTFT and TPOT SLOs. If no candidate reliably beats the baseline, InferOps advises keeping the baseline configuration.
- **Fail-Closed Lifecycle:** Managed child processes spawned by InferOps are tracked and cleanly stopped on cancellation. External or unmanaged servers are never modified.
- **Checkpointed & Resumable:** Session states are saved to SQLite. Interrupted runs can be resumed anytime using `resume <task_id>`.

---

### Example Queries

Type a serving scenario into the composer below to begin:

- *"I have Qwen2.5-1.5B on RTX 3060, chat scenario, target QPS=10, TTFT p99 under 200ms"*
- *"Long document QA, concurrency=4, keep TTFT p99 <= 400ms"*
- *"High concurrency short outputs, 32 users, maximize throughput"*

*Note: Target QPS specifies a measured throughput goal under concurrency-limited load. To resume a previous run, use `resume <task_id>`.*
