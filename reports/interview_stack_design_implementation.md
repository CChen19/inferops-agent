# InferOps 面试深挖：用了什么、怎么设计、怎么实现

这份材料按代码现状整理，用来回答 Agent 开发岗位最常见的三层追问：

1. 你在项目里到底用到了什么？
2. 你是怎么设计的？为什么这样设计？
3. 你是怎么实现的？相关知识点是什么？

仓库里已有 Day 1（项目主线）和 Day 2（JD / reliability / eval）笔记。本文补的是**对照源码的技术深挖**，面试时优先背这里的数字、接口和决策。

核心定位一句话：

> InferOps 不是 chatbot。它是一个把 vLLM 调参变成可控闭环实验的 Agent workflow：LLM 提假设，代码做校验，工具确定性执行，规则控制循环，eval 衡量质量。

---

## 0. 90 秒开口稿

InferOps 是一个本地 vLLM serving 优化 Agent。用户用自然语言描述场景，系统先抽 workload、跑 baseline，再进入 LangGraph 的 Plan → Execute → Reflect 循环。

- Planner 用 RAG 检索 vLLM 知识，结合实验历史和 bottleneck，输出结构化 hypothesis。
- Executor 不让 LLM 自由调工具；它校验 search space、跑 benchmark、写 SQLite、做瓶颈分类和 bootstrap 对比。
- Reflector 是纯规则：预算用完、连续 3 次提升不足 5%、或 bottleneck 切换后重规划。

硬件约束是 RTX 3060 Laptop 6 GB，所以 search space 被刻意收窄。项目还有 Chainlit UI、Chroma RAG、MLflow / OpenTelemetry、以及 CI-safe eval harness。

---

## 1. 我在项目中到底用到了什么？

面试官问 “你用了什么框架”，不要报一串名字。按**职责层**讲：每一层解决什么问题、为什么选它。

### 1.1 技术栈总表

| 层 | 实际用到的 | 职责 | 面试怎么说 |
|---|---|---|---|
| Agent orchestration | LangGraph 1.x `StateGraph` | 显式节点图 + 条件边 | 把 planning / execution / reflection 拆开，而不是一个 ReAct 循环 |
| LLM | OpenRouter / DeepSeek / Claude（LangChain ChatModel） | 只用于 intent + planner | LLM 不是 source of truth |
| Schema | Pydantic v2 | 工具 I/O、实验 config、结果 | 缩小 LLM 输出面，拒绝非法参数 |
| State | `TypedDict` `AgentState` | 图上共享状态 | LangGraph reducer + 可序列化 |
| Serving | vLLM 0.18，Qwen2.5 | 被优化的对象 | 真实 GPU 实验，不是 mock demo |
| Benchmark | `httpx` 异步压测 + `nvidia-ml-py` | TTFT / E2E / RPS / GPU | 一次实验 = 一次可比较观测 |
| Memory | SQLite | 实验可查询、可复用 | resume、去重、eval |
| Tracking | MLflow + OpenTelemetry | params/metrics + span | observability ≠ evaluation |
| RAG | Chroma + `BAAI/bge-base-zh-v1.5`（CPU） | planner grounding | 知识可更新，不必改 orchestration |
| UI | Chainlit | NL → 逐步展示 → 报告 | 验证 workflow，不是生产桌面端 |
| Eval | pytest + mock harness + regression gate | outcome / efficiency / trajectory | CI 不依赖 GPU |
| CLI | Typer | `inferops` 入口 | 工程化，不只 notebook |

`pyproject.toml` 里还有 `tenacity`、`langchain-community`。主路径真正驱动循环的是：LangGraph 节点 + 类型化工具函数 + SQLite。

### 1.2 面试容易说错的两点

**1. 图并没有 `bind_tools`。**

`inferops/tools/registry.py` 把 9 个工具包成 LangChain `@tool`，测试会检查 `ALL_TOOLS`。真正跑起来的图不走 ReAct function calling。Planner 调 LLM 产 JSON；Executor / Planner 直接调用 typed Python 函数。

这是设计，不是漏实现。调参会烧 GPU、可能 OOM。不能让模型自己决定调哪个工具、传什么参数。

可以说：

> 我有 tool registry，但生产路径是显式节点调用。LLM 只在 planner 里提案，执行层是确定性的。

**2. 真正的 AgentState 在 `inferops/agent/state.py`。**

`inferops/schemas.py` 里还有一个 Pydantic `AgentState`，当前图没有用。面试只讲 TypedDict 那份。

### 1.3 9 个工具分别干什么

| Tool | 谁调用 | 干什么 | 不干什么 |
|---|---|---|---|
| `propose_config_patch` | Executor | 单参数 patch + 安全范围校验 | 不跑实验 |
| `run_benchmark` | Executor | 启 vLLM / 压测 / 落库 / MLflow | 不决策 |
| `analyze_bottleneck` | Executor / baseline | 规则分类瓶颈 | 不调 LLM |
| `compare_experiments` | Executor | bootstrap CI，判断提升是否噪声 | 不改配置 |
| `query_experiment_memory` | 工具层 / eval | 按 workload 查历史 | 不执行 |
| `knowledge_retriever` | Planner | 向量检索 corpus | 不直接改 serving |
| `read_gpu_metrics` | 工具层 | 窗口采样 GPU util / VRAM | 不替代 bottleneck 规则 |
| `profile_with_pyspy` | 工具层 | CPU hotspot | GPU 低、CPU 高时才用 |
| `write_report_section` / `write_final_report` | UI / 收尾 | Markdown 报告 | 不参与路由 |

主循环高频路径只有四步：`propose_config_patch` → `run_benchmark` → `analyze_bottleneck` → `compare_experiments`。其余是检索、观测和报告。

### 1.4 四个可调参数（Agent search space）

Planner 只能在这个离散空间里提案：

```text
max_num_batched_tokens : [2048, 3072, 4096]
max_num_seqs           : [64, 128, 256]
enable_chunked_prefill : [False, True]
enable_prefix_caching  : [False, True]
```

`propose_config_patch` 的允许范围更宽（含 `max_model_len`、`gpu_memory_utilization`），那是工具层保护。Planner prompt 只暴露上面四个，减少胡编参数。

硬件约束写在 `configs/search_space.py`：

- RTX 3060 Laptop 6 GB，WSL2 下 Windows 还占一部分显存
- `gpu_memory_utilization=0.80`
- `max_model_len=2048`，避免按 Qwen 32k 上下文预留 KV
- `max_num_seqs=128` 作为默认，贴近本地 concurrency

### 1.5 五个 golden workloads

| Workload | 请求 / 并发 / in-out | 主要压什么 | 主指标 |
|---|---|---|---|
| `chat_short` | 60 / 16 / 128→128 | scheduler throughput | `throughput_rps` |
| `long_context_qa` | 20 / 4 / 1024→256 | prefill + KV | `throughput_rps` |
| `high_concurrency_short_out` | 120 / 32 / 64→32 | 调度、p99 | `throughput_rps` |
| `long_generation` | 10 / 2 / 256→512 | decode + KV | `tokens_per_second` |
| `mixed_traffic` | 40 / 8 / 短长混合 | fairness、tail latency | `throughput_rps` |

Prompt 生成在 `workloads/definitions.py`，每个 workload 额外 10 条 warmup。优化必须和 workload 绑定：短聊天上好的配置，长上下文不一定好。

### 1.6 RAG corpus

`data/corpus/` 六篇 Markdown：

- PagedAttention
- chunked prefill
- prefix caching
- speculative decoding
- vLLM scheduler
- tuning notes

RAG 只给 planner 提供 citation。新文档进 corpus 不会自动变成可执行 action。要真正调一个新技术，必须进入 search space 或 tool 层。

---

## 2. 我是怎么设计的？为什么要这样设计？

### 2.0 这是 workflow 还是 Agent？（高频追问）

面试官如果抓住 “tool 调用顺序是写死的”，不要硬辩成完全自主 Agent。先承认，再划清 **控制流** 和 **决策内容**。

Anthropic《Building Effective Agents》的定义：

- **Workflow**：LLM 和 tools 走预先写好的代码路径，下一步由代码决定。
- **Agent**：LLM 根据环境反馈，自己决定下一步做什么、调哪个 tool。

按这个标准，InferOps 的 **执行层是 workflow**。Executor 里固定是：

```text
propose_config_patch → run_benchmark → analyze_bottleneck → compare_experiments
```

Reflector 的 continue / replan / stop 也是规则，不是模型选边。这和 “LLM 自己决定调哪个 tool” 的 ReAct Agent 确实不是一类东西。

但它也不是普通 DAG workflow。普通 workflow 在编译期就知道要跑哪几步、试哪个配置。InferOps 在运行期才决定：

- 下一步改哪个参数、改成什么值（Planner + RAG + 实验历史）
- 还要不要继续、要不要丢掉当前 plan（测量结果 + 瓶颈是否切换）
- 循环转几圈（预算上限内，由反馈停，不是写死 3 次实验）

更准确的名字：

> **Agentic workflow / constrained agent**：agency 在 “下一步试什么”，不在 “下一步调哪个函数”。

对应 Anthropic 的 **evaluator-optimizer** 变体：生成假设 → 环境给出可验证反馈 → 再生成。区别是 evaluator 我用规则而不是第二个 LLM，因为吞吐、延迟、是否重复、是否超预算都可以确定性判断。

可以把 agency 画成谱，不要画成非黑即白：

```text
纯脚本 / grid search
  → 固定 DAG + LLM 填空
  → InferOps：固定 tool 协议 + LLM 选实验 + 反馈闭环     ← 这里
  → allowlist 内 LLM 选 tool
  → 自由 ReAct
```

为什么 agency 只放在 Planner：

| 部分 | 不确定性 | 失败代价 | 谁来做 |
|---|---|---|---|
| 下一步试什么配置 | 高：取决于瓶颈、历史、workload | 一次浪费的实验 | LLM |
| 工具怎么调、什么顺序 | 低：实验室协议是固定的 | GPU OOM、无效实验、不可复现 | 代码 |

类比：调参工程师也不会每次现场决定 “要不要先跑 nvidia-smi”。实验协议是 workflow；**根据上次结果决定下一个 hypothesis** 才是他们的判断力。InferOps 自动化的是后一件事。

如果对方坚持 “Agent 必须自己选 tool”，就明确说：

> 若标准是 open-ended tool calling，这不是那种 Agent。若标准是闭环里由模型根据环境反馈决定下一步行动内容，这是 constrained agent。我故意不把 tool 选择交给模型，因为这边的 tool 有真实副作用。

不要说的话：

- “这就是完全自主 Agent”
- “LangGraph 写了所以一定是 Agent”
- 把 `@tool` registry 说成正在用的 ReAct

### 2.1 系统分层

```text
User NL
  → Chainlit / CLI
  → intent extraction          # LLM，但输出受限 JSON
  → baseline benchmark         # 确定性
  → LangGraph
        planner    # 唯一强依赖 LLM 的节点
        executor   # 确定性工具链
        reflector  # 纯规则路由
  → SQLite + MLflow + Markdown report
```

设计原则：

> The LLM proposes; code validates; tools execute deterministically; rules control the loop; evaluation measures reliability.

### 2.2 为什么是 Plan → Execute → Reflect，而不是单 prompt 或自由 ReAct

调参是**有反馈的闭环**，不是一次生成答案。

| 方案 | 为什么没选 |
|---|---|
| 单 prompt 让模型直接给最优配置 | 没有测量，无法验证 |
| 纯 grid search | 离线 ground truth 可以，在线太贵 |
| 自由 ReAct | 灵活，但会乱调工具、重复实验、OOM |
| 显式三节点图 | planning、执行、停止逻辑可分开测试 |

图拓扑在 `inferops/agent/graph.py`：

```text
START → planner → executor → reflector
              ↑         ↑
              └─────────┴──── 条件边：planner / executor / END
```

Reflector 之后的路由：

1. `should_stop` → END
2. 还有 pending hypothesis → executor（把当前计划跑完）
3. 否则 → planner（重新提案）

这比 ReAct 的 “模型自己决定下一步” 更可控：预算、停止、重规划都是代码。

### 2.3 为什么 baseline 必须先跑

Agent 一开始没有证据。Baseline 不是全局常数，而是绑定：

- hardware
- model
- workload
- default config
- benchmark 设置

换机器 / 模型 / workload 就要重跑。后续每个 config 都相对 baseline 或当前 best 比较。`prepare_initial_state()` 会占掉一个实验名额。

### 2.4 为什么一次只改一个参数

为了因果归因。一次改多个可能更快，但无法解释是哪个 knob 导致变化。Planner prompt 写死 “Change exactly ONE parameter”；Executor 的 patch 也是单 key。

### 2.5 为什么 Reflector 不用 LLM

预算、停止、去重、重规划必须 **deterministic and testable**。LLM 适合提出方向，不适合当控制平面。

规则：

| 条件 | 动作 |
|---|---|
| `experiments_remaining <= 0` | stop，`budget_exhausted` |
| 相对 baseline 提升 ≥ **5%** | streak 清零 |
| 否则 | streak + 1 |
| streak ≥ **3** | stop，`no_improvement_3_consecutive` |
| bottleneck 从旧类型切到新类型 | 把剩余 pending 标 `skipped`，逼 planner 按新瓶颈重规划 |

5% 是抗噪声阈值。README 里 chat_short 只提升 1%，会被当成无提升，这是刻意的：小幅波动不应当成成功。

### 2.6 为什么 RAG 接在 Planner，而不是 Executor

Executor 需要确定性。RAG 影响的是“下一步试什么”，不是“怎么跑实验”。

检索 query 由 bottleneck + workload 拼出：

```text
"{bottleneck} optimization {workload} vLLM"
```

Hypothesis rationale 必须同时有：

1. 历史里的数字指标（正则 `\d+(\.\d+)?`）
2. `[source: ...]` citation

没有证据或没有 citation 的假设会被丢掉。知识过期时更新 corpus 即可，不必改图。

### 2.7 为什么 search space 这么小

这是本地 6 GB 卡上的可靠实验，不是云上 autotuner。空间小的好处：

- 不容易 OOM
- 去重和 resume 简单
- 能先用 grid sweep 做出 ground truth
- 面试能讲清 trade-off：可靠性、可解释性优先于搜索速度

### 2.8 状态设计为什么用 TypedDict

LangGraph 节点返回 **state patch**，不是替换整个对象。`messages` 用 `Annotated[..., add_messages]` reducer，避免覆盖历史。

`AgentState` 真正管循环的字段：

- `hypotheses`：pending / running / success / failed / skipped
- `experiment_summaries` / `baseline_summary` / `best_summary`
- `current_bottleneck`
- `experiments_remaining` / `no_improvement_streak` / `should_stop`
- `trajectory`：给 eval / LLM-as-judge 用

State 是控制面。LLM 只往里面写 hypothesis 和一段 analysis。

### 2.9 可靠性五层

1. **Structured state**：goal、baseline、history、budget、best 都在 state 里
2. **Validated actions**：allowlist、离散值、范围、去重、证据正则
3. **Deterministic execution**：Executor 只 apply、benchmark、parse、store
4. **Rule-based reflection**：停止和重规划可单测
5. **Eval + observability**：长期看质量，单次失败可追踪

### 2.10 评估为什么分三层

| 层 | 测什么 | 谁算 |
|---|---|---|
| Outcome | 离 ground truth 多远、主指标好坏 | 代码 |
| Efficiency | 实验次数、墙钟、相对 random/greedy | 代码 |
| Trajectory | 是否基于证据、是否重复、是否随瓶颈重规划 | LLM judge 或 heuristic |

Composite：

```text
0.50 * quality + 0.30 * efficiency + 0.20 * trajectory
```

- quality = `1 - gap_pct/100`
- efficiency = `1 - n_experiments/budget`
- trajectory 缺省时回退到 quality

LLM-as-judge **不**打 throughput / latency。那些必须由代码算。Judge 只打推理质量，rubric：

| 维度 | 权重 |
|---|---|
| evidence_based | 0.30 |
| no_repeat | 0.25 |
| replan | 0.25 |
| efficient | 0.20 |

CI 用 mock ground truth + random/greedy baseline，不碰真 GPU。Regression gate：主策略 gap 回退超过 5pp，或 composite 掉超过 0.05，就失败。

---

## 3. 我是怎么实现的？相关知识点

下面按一次真实请求的路径讲，穿插会被追问的知识点。

### 3.1 入口：自然语言 → Intent

`app.py` / `inferops/agent/intent.py`：

1. LLM 抽 JSON：`workload_name`、`model_hint`、`target_qps`、`budget`
2. workload 不在五选一里就回退 `chat_short`
3. 检查 vLLM `/health`，没起来就只展示计划、不跑实验
4. `prepare_initial_state` 跑或加载 baseline
5. `graph.astream(..., stream_mode=["updates", "values"])` 把每个 node 推到 Chainlit

相关知识点：

- Intent 是 **constrained structured extraction**，不是聊天
- UI 和 CLI 共用 `prepare_initial_state` / `build_graph`，避免两套状态机

### 3.2 Baseline 实现

`_run_baseline()`：

1. `experiment_id = {session_prefix}baseline`
2. SQLite 已有就复用，否则 `make_configs(workload)[0]` 跑默认配置
3. `analyze_bottleneck` 填初始 bottleneck
4. `best_summary = baseline_summary`，`vs_baseline_pct = 0`

相关知识点：resume 靠稳定 experiment_id，不是靠 LLM 记忆。

### 3.3 Planner 实现

`planner_node(state, llm)`：

1. 按剩余预算决定假设数量：`>4 → 3`，`>2 → 2`，否则 `1`
2. RAG top-4 chunks
3. Prompt 塞入：workload 描述、主指标、bottleneck、baseline/best、最近 8 次历史、已试 (param, value)、离散 search space、bottleneck guidance
4. `llm.invoke`；JSON 解析失败再 retry 一次
5. `_validate_hypotheses` 过滤
6. 合法假设标 `pending`，写入 `trajectory`

Bottleneck guidance（写在 prompt 里，也是面试记忆点）：

| 瓶颈 | 优先试 |
|---|---|
| compute-bound | 增大 `max_num_batched_tokens` |
| scheduling-bound | `enable_chunked_prefill=true` |
| memory-bound | 减小 `max_num_seqs` |
| kv-bound | prefix caching 或减小 `max_num_seqs` |

相关知识点：

- **Structured output + schema validation**：先 JSON，再代码过滤，不信任模型
- **Grounding**：citation 是硬约束，不是文案
- temperature=0.3，`max_tokens=1024`：要探索，但输出要短、要可解析

### 3.4 RAG 实现

Pipeline：

1. Markdown 按 H2/H3 切段，再按 **400 word / 50 overlap** 切 chunk
2. `bge-base-zh-v1.5` **强制 CPU**，避免和 vLLM 抢 VRAM
3. Chroma cosine / HNSW
4. 返回 `1 - distance` 作为 similarity
5. index 为空时 planner 降级，不把整个图打挂

相关知识点：

- RAG 更新的是 planner 的知识源，不是 action 空间
- 中文 embedding 模型用于中英混合 corpus；normalize embeddings 后用 cosine

### 3.5 Executor 实现

`executor_node` 每次只取 **第一个 pending**：

1. 重复 (param, value) → `skipped`，**不扣预算**
2. 同 id 已在 DB → 直接加载（resume）
3. `propose_config_patch`：参数必须在 allowlist，数值在 RTX 3060 安全范围
4. `run_benchmark`：默认 config + 单参数 patch
5. `analyze_bottleneck`
6. `compare_experiments`：对主指标做 **1000 次 bootstrap**
7. 主指标更好则更新 `best_summary`
8. 扣 1 次预算，写 trajectory

`run_benchmark` 额外约束：`max_num_batched_tokens >= max_model_len`，否则 vLLM 不接受。

相关知识点：

- Executor 失败会标 `failed` 并扣预算，避免死循环
- hypothesis `status=failed` 在代码里也用于 “相对 baseline 为负” 的成功跑完的实验，面试时说清：这是结果标签，不一定是工具崩溃

### 3.6 Benchmark 实现

`bench_runner.run_experiment`：

1. `/health` 已通 → 复用外部 vLLM（Chainlit 场景），避免每次启停
2. 否则 `VLLMProcess` 拉起，检测 OOM / crash / startup timeout
3. `GPUMonitor` 0.5s 采样
4. `run_load`：warmup + measure；Chainlit 下 **非流式**，避开 httpx/anyio streaming deadlock
5. 算 TTFT / TPOT / E2E percentiles、RPS、tok/s
6. 原始 per-request latency 写入 `raw_ttft_ms` / `raw_e2e_ms`，给 bootstrap 用
7. MLflow log params + metrics

压测请求打 `POST /v1/chat/completions`，`temperature=0.0`。

相关知识点：

- **TTFT**：首 token 时间，主要反映 prefill + 排队
- **E2E**：整请求延迟，含 decode 和输出长度
- **TPOT**：近似 `(E2E - TTFT) / (output_tokens - 1)`
- **Concurrency vs active sequences**：外部并发 32 不等于 scheduler 里同时有 32 条；`max_num_seqs` 会把多余请求堵在队列里，抬高 tail latency
- **Prefill vs decode**：长 prompt 压 prefill/TTFT；长输出压 decode/KV；高并发压 scheduler

### 3.7 Bottleneck 规则实现

`analyze_bottleneck` 按优先级，不是 LLM 分类：

1. TTFT p99/p50 **> 3** → `scheduling-bound`（>5 则 high confidence）  
   长 prefill 堵短请求，建议 chunked prefill
2. GPU util **< 80%** 且 E2E p50 **> 1500ms** 且 E2E p99/p50 **> 1.5** → `kv-bound`  
   建议减 `max_num_seqs` 或开 prefix caching
3. GPU util **≥ 88%** → `compute-bound`  
   建议加大 `max_num_batched_tokens`；接近硬件天花板
4. GPU mem **> 5.5 GB**（6 GB 卡）→ `memory-bound`
5. 否则 `unknown`

面试可以说：这是可测试的启发式，不是学习出来的分类器。阈值按 3060 经验设定，换硬件要重标定。

### 3.8 统计比较实现

`compare_experiments`：

- 延迟：用 raw samples；没有 raw 时用 p50/p99 拟合 log-normal 再采样
- 吞吐：围绕标量做小高斯抖动再 bootstrap（实现上的近似，面试承认即可）
- 95% CI 不跨 0 → `significant`
- `|delta| < 2%` 视为 tie

相关知识点：同一 config 跑两次也会抖。GPU warmup、后台进程、dynamic batching 都会引入噪声。所以：固定 context、尽量区分 warmup/measure、一次一参、小提升要过阈值。

### 3.9 Memory 实现

SQLite 表 `experiments`：

- unique `experiment_id`
- `config_hash`：对可调 knobs 做 sha256（不含 experiment_id/tags）
- 完整 `config_json` / `result_json`
- 常用指标列方便 ORDER BY

Eval 用 `session_prefix` 把一次 agent run 的行捞出来，和 grid-sweep ground truth 比。

### 3.10 Observability 实现

- 每个 tool `with span("tool.xxx", attrs)`
- 默认 ConsoleSpanExporter；`OTEL_EXPORTER=otlp` 可打到 Jaeger
- MLflow 默认 `sqlite:///mlruns.db`

面试区分：

- Observability：发生了什么（trace、metrics、state）
- Evaluation：这样做好不好（gap、composite、judge）

Debug 三层：workflow trace → experiment metrics → decision trajectory。

### 3.11 Chainlit 实现要点

- `asyncio.to_thread(prepare_initial_state)` 避免阻塞 event loop
- `astream` 的 `updates` 用来推 planner/executor/reflector 卡片
- 最终 `write_final_report`，并从 hypothesis rationale 收集 `[source:]`

Chainlit 是 Python-native 的 agent UI，用来验证 workflow。生产级本地 dashboard 更适合 Electron（main / preload / renderer，`contextIsolation`，窄 IPC）。项目没做 Electron，但设计意图要能讲。

### 3.12 相关 vLLM 知识点（会被顺着项目问）

**PagedAttention**  
KV 按 block 按需分配，用 block table 映射，减少碎片。`gpu_memory_utilization` 控制 KV pool 占比；`max_model_len` 决定每条请求最多占多少 block。

**Chunked prefill**  
把长 prefill 切块，和 decode 交错，减轻 head-of-line blocking，降低 TTFT 方差。对应 scheduling-bound。

**Prefix caching**  
共享 system prompt / 前缀时复用 KV，摊销 prefill。对应 kv-bound 且 prompt 有公共前缀。

**max_num_batched_tokens**  
每个 scheduler step 最多处理多少 token。增大通常提高 GPU 饱和与吞吐，但可能让长 prefill 更“胖”，影响短请求延迟。

**max_num_seqs**  
同时进入推理循环的最大序列数。增大提高并发容量，但 KV 压力上升。

**KV cache**  
存历史 token 的 K/V，decode 时不用重算整段。随 batch、prompt、output、模型变大。加速 decode，但吃显存。

---

## 4. 一条请求的时序（背这个）

```text
"I have Qwen2.5-1.5B on RTX 3060, chat scenario, target QPS=10"
        │
        ▼
extract_intent → workload=chat_short, budget=6
        │
        ▼
vLLM health check
        │
        ▼
baseline (default config) → metrics + bottleneck
        │
        ▼
planner: RAG(bottleneck) + LLM JSON → validated hypotheses
        │
        ▼
executor: propose → benchmark → bottleneck → bootstrap vs baseline
        │
        ▼
reflector:
  budget=0? stop
  improvement>=5%? reset streak
  streak>=3? stop
  bottleneck switched? skip pending, replan
  else pending? executor : planner
        │
        ▼
write_final_report + Chainlit 表格
```

---

## 5. 高频追问（短答）

**这不就是 workflow 吗？和 Agent 定义冲突怎么办？**  
执行层确实是 workflow：tool 顺序写死，Reflector 用规则路由。Agency 在 Planner：下一步试哪个配置、要不要因瓶颈切换而重规划，这些在运行前不知道。更准确的叫法是 agentic workflow / constrained agent。Anthropic 也把 evaluator-optimizer 归在 workflow，同时承认它是 agentic system。我把模型放在高不确定性的规划上，把高副作用的执行留给代码。

**为什么不用纯 grid search？**  
Grid 当离线 ground truth。在线用有限预算做 targeted search。

**为什么不用自由 ReAct？**  
真实验会 OOM、重复、烧时间。显式图把执行和停止逻辑从 LLM 手里拿回来。

**LLM 胡编参数怎么办？**  
不直接执行。Allowlist、离散值、范围、去重、证据/citation 正则，非法直接丢。

**提升 1% 算成功吗？**  
Reflector 要 ≥5% 才重置 streak。1% 可能是噪声。Best 仍按主指标记录，但停止逻辑不把它当显著改进。

**换 GPU 怎么办？**  
Baseline、search space、bottleneck 阈值都和环境绑定，必须重测，不能迁移旧最优。

**RAG 加一篇新论文，agent 会自动用这项技术吗？**  
只会进入 planner 的检索上下文。要执行，还得进 search space / tool。

**Tool registry 为啥存在却没 bind？**  
保留 LangChain 兼容封装和测试契约；运行时走显式节点，降低 blast radius。

**项目局限？**  
UI 是 Chainlit 不是 Electron；search space 小；生产还缺 approval、canary、rollback、权限、audit；OTel 默认 console。下一步可以做 MCP tool wrapper、更完整 trace export、multi-objective、confidence-aware planning。

---

## 6. 和源码对齐的记忆清单

背数字：

- 提升阈值 **5%**，连续无提升 **3** 次
- TTFT 方差比 **>3** → scheduling-bound
- GPU util **≥88%** → compute-bound
- VRAM **>5.5 GB** → memory-bound
- Composite **50 / 30 / 20**
- Judge 权重 **0.30 / 0.25 / 0.25 / 0.20**
- RAG chunk **400 / 50**，top_k **4**
- Bootstrap 执行路径 **1000** 次（工具默认 2000）
- 测试约 **153**，工具 **9**，workload **5**，corpus **6**

背文件：

- 图：`inferops/agent/graph.py`
- 状态：`inferops/agent/state.py`
- 三节点：`planner.py` / `executor.py` / `reflector.py`
- 工具：`inferops/tools/*.py`
- RAG：`inferops/rag/`
- Eval：`inferops/eval/`
- UI：`app.py`

背一句：

> LLM 负责提出方向，代码负责验证边界，工具负责确定性执行，规则负责停或重规划，eval 负责长期质量。
