# Day 1 Agent Interview Review / 第一天 Agent 面试复习总结

## 1. Day 1 Main Theme / 第一天主线

### Knowledge Points / 知识点

Day 1 的重点是把 InferOps 项目本身讲清楚，尤其是：

- 这个项目解决什么问题
- vLLM serving optimization 的基本背景
- baseline、workload、metrics 分别是什么
- Plan -> Execute -> Reflect workflow 怎么跑
- Planner、Executor、Reflector 分别做什么
- 为什么这个项目不是简单 prompt demo

Day 2 会重点讲 JD 对齐、Agent Reliability、Evaluation、Observability、MCP、Electron 和 Production Guardrails，所以 Day 1 不在这些点上展开太多。

### Related Questions / 相关题目

- Can you walk me through your project?
- What problem does InferOps solve?
- Why is this an agent project?

### Answer / 答案

中文：

InferOps 是一个面向本地 vLLM 推理服务的自动优化 Agent。它把自然语言 serving goal 转成具体 workload，先在当前 hardware、model 和 workload 下跑 baseline，然后进入 LangGraph 的 Plan -> Execute -> Reflect 循环。Planner 根据实验历史、当前瓶颈和 RAG 检索到的 vLLM 知识提出配置假设；Executor 校验配置、运行 benchmark、记录指标；Reflector 根据结果和预算决定继续、重规划或停止。这个项目的核心是把手动调参流程变成一个可控的闭环实验系统。

English:

InferOps is an autonomous optimization agent for local vLLM inference. It turns a natural-language serving goal into a concrete workload, first runs a baseline under the current hardware, model, and workload, and then enters a LangGraph Plan -> Execute -> Reflect loop. The planner proposes configuration hypotheses based on experiment history, current bottleneck analysis, and RAG-retrieved vLLM knowledge. The executor validates the configuration, runs the benchmark, and records metrics. The reflector decides whether to continue, replan, or stop based on results and budget. The core idea is to turn manual tuning into a controlled closed-loop experiment system.

## 2. Baseline / 基线实验

### Knowledge Points / 知识点

Baseline 是当前环境下的起点性能。它不是全局常量，而是和具体 context 绑定：

- hardware
- model
- workload
- default config
- benchmark setting

Agent 一开始没有实验数据，所以先跑 baseline，获得可比较的起点。后续每个 config change 都要和 baseline 或当前 best result 对比。

常见 baseline metrics：

- throughput
- TTFT
- E2E latency
- p95 / p99 latency
- GPU utilization
- VRAM usage
- success rate

### Related Questions / 相关题目

- What is a baseline?
- Why do you run a baseline first?
- Does the baseline change if hardware or workload changes?

### Answer / 答案

中文：

Baseline 是当前机器、模型、workload 和默认配置下的起点性能。Agent 不能一开始就猜最优参数，因为没有实验依据，所以先跑 baseline，收集 throughput、TTFT、latency、GPU utilization 和 VRAM usage 等指标。如果换了机器、模型或 workload，就应该重新跑 baseline，因为 baseline 是 experiment context 的一部分，不是固定常量。

English:

The baseline is the starting performance under the current hardware, model, workload, and default configuration. The agent should not guess the optimal parameters at the beginning because it has no experimental evidence yet. It first runs a baseline to collect metrics such as throughput, TTFT, latency, GPU utilization, and VRAM usage. If the hardware, model, or workload changes, the baseline should be rerun because it is part of the experiment context, not a fixed global constant.

## 3. Workload / 工作负载

### Knowledge Points / 知识点

Workload 是用来模拟真实 serving 场景的请求模式。它描述的不是模型本身，也不是 GPU 配置，而是请求如何到来、请求有多长、输出有多长、并发有多高，以及目标更偏吞吐还是延迟。

项目里的 workloads：

| Workload | Meaning / 含义 | Main Stress / 主要压力 |
|---|---|---|
| `chat_short` | 短聊天请求 | scheduler throughput |
| `long_context_qa` | 长上下文问答 | prefill, KV cache |
| `high_concurrency_short_out` | 高并发短输出 | scheduling, p99 latency |
| `long_generation` | 长输出生成 | decode throughput, KV cache |
| `mixed_traffic` | 长短请求混合 | fairness, tail latency |

### Related Questions / 相关题目

- What is a workload?
- Why do you need different workloads?
- How does workload affect optimization?

### Answer / 答案

中文：

Workload 是一组模拟真实请求模式的 benchmark 场景。不同 workload 会暴露不同瓶颈：短聊天更考验 scheduler throughput，长上下文问答更考验 prefill 和 KV cache，高并发请求更容易影响 p99 latency，长生成更偏 decode throughput。优化参数必须和 workload 绑定，否则一个场景下有效的配置不一定适合另一个场景。

English:

A workload is a benchmark scenario that simulates a real request pattern. Different workloads expose different bottlenecks: short chat stresses scheduler throughput, long-context QA stresses prefill and KV cache, high concurrency affects p99 latency, and long generation stresses decode throughput. Optimization must be tied to the workload because a configuration that works well for one scenario may not work well for another.

## 4. Prefill, Decode, and Scheduling / Prefill、Decode 和调度

### Knowledge Points / 知识点

LLM inference 可以分成两个阶段：

- Prefill: 读取 prompt，计算已有 token 的 KV cache
- Decode: 一个 token 一个 token 生成输出

基本规律：

- prompt 越长，prefill 压力越大
- output 越长，decode 压力越大
- concurrency 越高，scheduler 压力越大
- context 越长，KV cache 和显存压力越大

TTFT 通常和 prefill、排队、调度有关。E2E latency 同时受到 prefill、decode、排队和输出长度影响。

### Related Questions / 相关题目

- What is the difference between prefill and decode?
- Why do long prompts increase TTFT?
- Why does high concurrency stress the scheduler?

### Answer / 答案

中文：

Prefill 是模型读取 prompt 并为已有 token 计算 KV cache 的阶段，长 prompt 会让 prefill 更重，可能增加 TTFT。Decode 是模型逐 token 生成答案的阶段，长输出会让 decode 持续更久。高并发会让 scheduler 需要在更多 active sequences 之间分配 GPU 计算和 KV cache，因此容易影响 throughput 和 tail latency。

English:

Prefill is the phase where the model reads the prompt and computes KV cache for existing tokens. Long prompts make prefill heavier and can increase TTFT. Decode is the phase where the model generates the answer token by token. Long outputs make decode last longer. High concurrency forces the scheduler to allocate GPU compute and KV cache across more active sequences, which can affect throughput and tail latency.

## 5. KV Cache / KV 缓存

### Knowledge Points / 知识点

KV Cache 存储 Transformer attention 中历史 token 的 Key 和 Value。这样 decode 新 token 时，不需要重复计算整段上下文。

KV Cache 变大的原因：

- active sequences 增加
- prompt 更长
- output 更长
- 模型更大
- context window 更长
- `max_num_seqs` 更高

它的 trade-off：

- 好处：加速 decode
- 代价：占用 GPU memory

### Related Questions / 相关题目

- What is KV cache?
- Why does KV cache use GPU memory?
- How does concurrency affect KV cache?

### Answer / 答案

中文：

KV Cache 缓存历史 token 在 attention 里的 Key 和 Value，这样模型在 decode 新 token 时可以复用历史上下文，不需要重复计算。它会随着 active sequences、prompt length、output length、model size 和 context window 增长。KV Cache 能提高 decode 效率，但会消耗 GPU 显存，所以它是 vLLM serving optimization 中非常重要的资源。

English:

KV cache stores the Key and Value tensors of previous tokens in attention, so the model can reuse the context when decoding new tokens instead of recomputing everything. It grows with active sequences, prompt length, output length, model size, and context window. KV cache improves decode efficiency, but it consumes GPU memory, so it is a critical resource in vLLM serving optimization.

## 6. Concurrency vs Active Sequences / 并发与活跃序列

### Knowledge Points / 知识点

Concurrency 是外部请求压力，表示同时有多少请求在等待或进行中。

Active sequences 是 scheduler 当前真正接纳进推理循环、正在占用 KV cache 和 GPU 调度资源的序列。

例子：

如果外部来了 100 个请求，但 `max_num_seqs = 16`，那么可能只有 16 个 active sequences，剩下的请求在队列里等待。

### Related Questions / 相关题目

- What is the difference between concurrency and active sequences?
- Why can high concurrency increase latency?
- Does every external request immediately become an active sequence?

### Answer / 答案

中文：

Concurrency 是外部同时到来的请求数量，而 active sequences 是 vLLM scheduler 当前实际放进推理循环的序列数量。并不是所有外部请求都会立刻变成 active sequence。如果 concurrency 很高，但 `max_num_seqs` 有限制，一部分请求会排队等待，因此 tail latency 可能升高。

English:

Concurrency is the number of external requests happening at the same time, while active sequences are the sequences that the vLLM scheduler has actually admitted into the inference loop. Not every external request immediately becomes an active sequence. If concurrency is high but `max_num_seqs` is limited, some requests will wait in the queue, which can increase tail latency.

## 7. Request, Sequence, and Complex Tasks / 请求、序列与复杂任务

### Knowledge Points / 知识点

普通 completion 请求通常对应一个 sequence。长 prompt 可能通过 chunked prefill 分块处理，但逻辑上仍是同一个 sequence。

一个外部任务可能变成多个 sequence 或多个 LLM call 的情况：

- `n > 1`
- best-of sampling
- beam search
- agent 应用层拆任务
- 多轮 tool calling

### Related Questions / 相关题目

- Does one request always map to one sequence?
- Can one complex task become multiple LLM calls?
- Is chunked prefill the same as multiple sequences?

### Answer / 答案

中文：

普通 completion 请求通常对应一个 sequence。长 prompt 可能通过 chunked prefill 分成多个调度步骤，但它仍然是同一个逻辑 sequence。一个外部任务在应用层可能被拆成多个 LLM call，比如 agent 规划、工具调用、反思、多候选生成等，但这属于应用层 workflow，不等于 vLLM 内部每次都自动变成多个 sequence。

English:

A normal completion request usually maps to one sequence. A long prompt may be processed through chunked prefill across multiple scheduling steps, but it is still one logical sequence. An external task may be split into multiple LLM calls at the application layer, such as planning, tool calling, reflection, or generating multiple candidates, but that is an application-level workflow and does not mean vLLM automatically turns every request into multiple sequences.

## 8. Plan -> Execute -> Reflect / 规划、执行、反思

### Knowledge Points / 知识点

InferOps 的主循环是 Plan -> Execute -> Reflect：

1. Plan: 根据目标、baseline、历史实验、瓶颈和 RAG 知识提出下一步 hypothesis
2. Execute: 校验配置，运行 benchmark，记录 metrics
3. Reflect: 根据预算、提升、瓶颈变化和 pending hypotheses 决定下一步

这个设计的价值：

- 把 LLM reasoning 和实际执行分开
- 让 benchmark 结果回到下一轮决策
- 让 workflow 有明确状态和控制流
- 方便追踪每次实验为什么发生

### Related Questions / 相关题目

- Why did you use Plan -> Execute -> Reflect?
- What does each stage do?
- Why not use a single prompt?

### Answer / 答案

中文：

我使用 Plan -> Execute -> Reflect 是因为调参是一个需要反馈的闭环过程。Planner 负责根据目标和历史数据提出假设；Executor 负责把假设变成真实 benchmark；Reflector 负责根据结果决定继续、重规划或停止。相比单个 prompt，这种结构更清晰，也更容易控制实验预算、记录状态和解释每一步决策。

English:

I used Plan -> Execute -> Reflect because tuning is a feedback-driven closed-loop process. The planner proposes hypotheses based on the goal and history, the executor turns hypotheses into real benchmarks, and the reflector decides whether to continue, replan, or stop based on results. Compared with a single prompt, this structure is clearer and makes it easier to control experiment budget, track state, and explain each decision.

## 9. Planner / 规划器

### Knowledge Points / 知识点

Planner 主要负责提出下一步实验假设。它会看：

- user goal
- baseline result
- experiment history
- current bottleneck
- RAG-retrieved vLLM knowledge
- current search space

Planner 输出不是直接执行命令，而是 structured hypothesis，例如：

- parameter
- proposed value
- rationale
- metric evidence
- source citation

### Related Questions / 相关题目

- What does the planner use to decide the next parameter?
- Why use RAG in the planner?
- What does the planner output?

### Answer / 答案

中文：

Planner 根据 user goal、baseline、实验历史、当前瓶颈和 RAG 检索到的 vLLM 知识提出下一步配置假设。它不是直接执行命令，而是输出结构化 hypothesis，包括要调整的参数、候选值、理由、相关指标和 source citation。RAG 的作用是让 planner 使用项目知识库里的 serving optimization 知识，而不是只依赖模型记忆。

English:

The planner proposes the next configuration hypothesis based on the user goal, baseline, experiment history, current bottleneck, and RAG-retrieved vLLM knowledge. It does not directly execute commands. It outputs a structured hypothesis, including the parameter to tune, the proposed value, rationale, relevant metrics, and source citation. RAG allows the planner to use serving optimization knowledge from the project corpus instead of relying only on the model's memory.

## 10. Executor / 执行器

### Knowledge Points / 知识点

Executor 负责把 hypothesis 变成真实实验。

主要步骤：

1. 取一个 pending hypothesis
2. 检查配置是否合法
3. 检查是否重复
4. 生成 experiment id
5. 运行 benchmark
6. 解析 metrics
7. 保存到 experiment memory
8. 分析 bottleneck
9. 更新 best result 和 history

### Related Questions / 相关题目

- What does the executor do?
- Does the executor use the LLM?
- What happens after a benchmark finishes?

### Answer / 答案

中文：

Executor 主要是确定性执行层，不依赖 LLM 自由发挥。它拿到 planner 的 hypothesis 后，会检查配置是否合法、是否重复，然后运行 benchmark、解析 metrics、保存实验结果、分析 bottleneck，并更新 best result 和 experiment history。它的职责是把想法变成可比较的实验数据。

English:

The executor is mainly a deterministic execution layer, not a place where the LLM freely acts. After receiving a planner hypothesis, it checks whether the configuration is valid and non-duplicate, runs the benchmark, parses metrics, stores the experiment result, analyzes the bottleneck, and updates the best result and experiment history. Its job is to turn ideas into comparable experimental data.

## 11. Reflector / 反思与路由节点

### Knowledge Points / 知识点

Reflector 不负责生成新参数。它负责控制 workflow 的下一步：

- continue: 继续执行已有 pending hypotheses
- replan: 回到 planner 生成新 hypothesis
- stop: 停止实验并生成报告

它主要看：

- budget 是否用完
- latest experiment 是否有提升
- 是否连续无提升
- bottleneck 是否变化
- 是否还有 pending hypotheses

### Related Questions / 相关题目

- What does the reflector do?
- Does the reflector generate new parameters?
- When does the agent stop or replan?

### Answer / 答案

中文：

Reflector 是 workflow 的路由和控制节点，不直接生成新参数。它根据实验预算、最新结果、连续无提升次数、瓶颈变化和 pending hypotheses 判断下一步：继续执行、回到 planner 重规划，或者停止并生成报告。简单说，Planner 决定尝试什么，Executor 负责执行，Reflector 决定当前计划是否还值得继续。

English:

The reflector is the routing and control node of the workflow. It does not directly generate new parameters. It decides the next step based on experiment budget, latest results, no-improvement streak, bottleneck changes, and pending hypotheses: continue execution, go back to the planner, or stop and generate a report. In short, the planner decides what to try, the executor runs it, and the reflector decides whether the current plan is still worth following.

## 12. RAG Knowledge Base / RAG 知识库

### Knowledge Points / 知识点

RAG 在项目中的作用是给 planner 提供 vLLM serving optimization 的外部知识。

知识库内容包括：

- vLLM scheduler notes
- PagedAttention
- prefix caching
- chunked prefill
- speculative decoding
- tuning notes

RAG 不是直接执行新技术。要真正调某个技术或参数，还需要它出现在 search space 或 tool layer 里。

### Related Questions / 相关题目

- Why use RAG?
- Can the agent use new papers?
- Does adding a document automatically make a new action executable?

### Answer / 答案

中文：

我使用 RAG 是因为 serving optimization 的知识更新较快，而且 planner 需要结合项目知识库来提出更合理的调参方向。新论文或 tuning note 可以加入 RAG corpus，让 planner 检索到新的思路。但如果要真正执行某个新技术，还需要把它暴露为工具层或 search space 中的安全 action。

English:

I use RAG because serving optimization knowledge changes quickly, and the planner needs project-specific knowledge to propose better tuning directions. New papers or tuning notes can be added to the RAG corpus so the planner can retrieve new ideas. But to actually execute a new technique, it must be exposed as a safe action in the tool layer or search space.

## 13. Metrics and Report / 指标与报告

### Knowledge Points / 知识点

项目中常用指标：

- throughput: 单位时间处理多少请求或 token
- TTFT: time to first token，首 token 延迟
- E2E latency: 整个请求从开始到完成的延迟
- p95 / p99 latency: tail latency
- GPU utilization: GPU 使用率
- VRAM usage: 显存占用
- success rate: 成功请求比例

最终 report 应该说明：

- baseline 是什么
- 尝试了哪些 config
- best result 是什么
- 哪些指标改善或变差
- 当前瓶颈是什么
- 为什么停止

### Related Questions / 相关题目

- What metrics do you track?
- What does the final report include?
- How do you decide whether a result is better?

### Answer / 答案

中文：

我主要跟踪 throughput、TTFT、E2E latency、p95/p99 latency、GPU utilization、VRAM usage 和 success rate。最终报告会包含 baseline、每次尝试的配置、benchmark 结果、best result、当前瓶颈和停止原因。判断一个结果是否更好不能只看 throughput，还要看它是否满足 latency 和成功率约束。

English:

I mainly track throughput, TTFT, end-to-end latency, p95/p99 latency, GPU utilization, VRAM usage, and success rate. The final report includes the baseline, each tried configuration, benchmark results, the best result, current bottleneck, and stopping reason. A result should not be judged only by throughput; it also needs to satisfy latency and success-rate constraints.

## 14. Common Defense Questions / 常见防御问题

### Q1. Why not pure grid search? / 为什么不用纯 grid search？

中文：

Grid search 可以找到好配置，但成本高，而且不适合作为在线优化策略。我更适合用 grid sweep 作为 ground truth 或离线评估参考；在线优化时，agent 在有限实验预算下根据反馈做 targeted search。

English:

Grid search can find good configurations, but it is expensive and not ideal as an online optimization strategy. I would rather use grid sweep as ground truth or an offline evaluation reference. During online optimization, the agent performs targeted search under a limited experiment budget based on feedback.

### Q2. Why not a free-form ReAct agent? / 为什么不用完全自由的 ReAct agent？

中文：

自由 ReAct agent 灵活，但调参任务会真实消耗资源，也可能造成 OOM 或无效实验。我选择显式 LangGraph workflow，是为了把 planning、execution 和 reflection 分开，让执行、预算和停止逻辑更可控。

English:

A free-form ReAct agent is flexible, but tuning consumes real resources and may cause OOM or invalid experiments. I chose an explicit LangGraph workflow to separate planning, execution, and reflection, making execution, budget, and stopping logic more controllable.

### Q3. What if the LLM proposes a wrong parameter? / 如果 LLM 生成错误参数怎么办？

中文：

Planner 输出不会直接执行。系统会检查参数是否在 search space 中，值是否在合法范围内，配置是否重复，理由是否包含指标证据。错误或不支持的参数会被拒绝。

English:

Planner output is not executed directly. The system checks whether the parameter is in the search space, whether the value is within the valid range, whether the configuration is a duplicate, and whether the rationale contains metric evidence. Wrong or unsupported parameters are rejected.

### Q4. What if the hardware or model changes? / 如果换机器或模型怎么办？

中文：

Baseline 和优化结果都和环境绑定。换 hardware、model 或 workload 后，需要重新跑 baseline，并基于新的实际测量结果优化，而不是假设旧参数可以直接迁移。

English:

The baseline and optimization results are environment-specific. If the hardware, model, or workload changes, the baseline should be rerun and optimization should be based on the new measured results instead of assuming old parameters transfer directly.

## 15. Final 90-Second Answer / 90 秒总结

### 中文

我做的 InferOps 是一个面向本地 vLLM 推理服务的自动优化 Agent。它从自然语言 serving goal 开始，抽取或选择对应 workload，在当前 hardware、model 和 workload 下先跑 baseline，然后进入 LangGraph 的 Plan -> Execute -> Reflect 闭环。Planner 利用实验历史、当前瓶颈和 RAG 检索到的 vLLM 知识提出下一步配置假设；Executor 校验配置、运行 benchmark、记录 throughput、TTFT、latency、GPU utilization 和 VRAM usage 等指标；Reflector 根据预算、提升情况、瓶颈变化和 pending hypotheses 决定继续、重规划或停止。最终系统生成报告，说明 baseline、尝试过的配置、best result、瓶颈和停止原因。这个项目的核心是把手动 vLLM 调参流程变成一个有状态、有反馈、可控制的 agent workflow。

### English

I built InferOps, an autonomous optimization agent for local vLLM inference. It starts from a natural-language serving goal, extracts or selects the corresponding workload, runs a baseline under the current hardware, model, and workload, and then enters a LangGraph Plan -> Execute -> Reflect loop. The planner uses experiment history, current bottleneck analysis, and RAG-retrieved vLLM knowledge to propose the next configuration hypothesis. The executor validates the configuration, runs the benchmark, and records metrics such as throughput, TTFT, latency, GPU utilization, and VRAM usage. The reflector decides whether to continue, replan, or stop based on budget, improvement, bottleneck changes, and pending hypotheses. Finally, the system generates a report showing the baseline, tried configurations, best result, bottleneck, and stopping reason. The core idea is to turn manual vLLM tuning into a stateful, feedback-driven, controllable agent workflow.

