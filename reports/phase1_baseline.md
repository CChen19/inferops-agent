# InferOps Phase 1 基线报告

生成时间：2026-04-28  
模型：Qwen/Qwen2.5-0.5B-Instruct  
硬件：NVIDIA RTX 3060 Laptop 6GB，WSL2 Ubuntu，CUDA 12.8  
vLLM 版本：0.18.0

---

## 实验配置

**固定参数**

| 参数 | 值 |
|---|---|
| gpu_memory_utilization | 0.80 |
| max_model_len | 2048 |
| max_num_seqs | 128 |
| dtype | auto (bfloat16) |
| enforce_eager | False |

**变体（搜索空间）**

| 变体 | max_num_batched_tokens | enable_chunked_prefill | enable_prefix_caching |
|---|---|---|---|
| default | 2048 | False | False |
| chunked | 2048 | True | False |
| prefix_cache | 2048 | False | True |
| big_batch | 4096 | False | False |

**Workload**

| Workload | 请求数 | 并发 | 输入 token | 输出 token | 场景 |
|---|---|---|---|---|---|
| chat_short | 60 | 16 | ~15 | 128 | 高并发短对话 |
| long_context_qa | 20 | 4 | ~1024 | 256 | 长文档问答 |

---

## 实验结果

### chat_short

| 变体 | RPS | Tok/s | TTFT p50 | TTFT p99 | E2E p50 | E2E p99 | GPU 利用率 |
|---|---|---|---|---|---|---|---|
| default | 14.96 | 1929 | 48ms | 69ms | 1015ms | 1032ms | 86% |
| chunked | 16.68 | 2151 | 67ms ↑ | 79ms | 914ms | 951ms | 84% |
| prefix_cache | 16.84 | 2167 | 49ms | 175ms ↑↑ | 878ms | 1001ms | 81% |
| **big_batch** | **17.23** | **2222** | **52ms** | **65ms** | **883ms** | **892ms** | **86%** |

成功率：4 × 60/60 ✓

### long_context_qa

| 变体 | RPS | Tok/s | TTFT p50 | TTFT p99 | E2E p50 | E2E p99 | GPU 利用率 |
|---|---|---|---|---|---|---|---|
| default | 2.43 | 625 | 39ms | 46ms | 1643ms | 1658ms | 92% |
| **chunked** | **2.90** | **745** | **38ms** | **43ms** | **1379ms** | **1392ms** | **89%** |
| **prefix_cache** | **2.90** | **745** | 39ms | 51ms | 1374ms | 1397ms | 92% |
| big_batch | 2.35 | 604 | 41ms | 52ms | 1703ms ↑ | 1711ms ↑ | 94% |

成功率：4 × 20/20 ✓

---

## 分析

### 1. big_batch：短序列利器，长序列毒药

chat_short 上 big_batch 以 +15% RPS、最优尾延迟拿下第一。背后原因：较大的 token 批次让 GPU 矩阵运算更饱和，decode kernel 效率更高。

但在 long_context_qa 上，big_batch 反而垫底，RPS 比 default 还低 3%（2.35 vs 2.43），E2E 延迟最差（1703ms）。原因：每步处理 4096 token 时，长序列的 KV 访问开销急剧上升，GPU 反而被卡在内存带宽瓶颈上（利用率 94%，但 effective throughput 下降）。

**结论：max_num_batched_tokens 需要根据序列长度匹配，没有万能最优值。**

### 2. chunked_prefill：短序列负优化，长序列强优化

- chat_short：TTFT p50 从 48ms 涨到 67ms（+40%）——短序列 prefill 本身很快，拆块反而引入调度开销
- long_context_qa：RPS +19%，E2E 从 1643ms 降到 1379ms（-16%）——1024 token 的 prefill 被拆成小块和 decode 交替执行，消除了头行阻塞（Head-of-Line Blocking）

**结论：chunked_prefill 的收益和序列长度正相关，短序列场景应关闭。**

### 3. prefix_cache：有共享前缀才有收益

- chat_short：TTFT p99 从 69ms 飙到 175ms（+153%）——prompt 各不相同，缓存命中率低，哈希+查询的 overhead 反而拖累了尾延迟
- long_context_qa：与 chunked 几乎并列第一（2.90 rps，E2E 1374ms）——20 个请求共享同一段 ~5000 字的背景段落，前缀缓存命中，prefill 成本被分摊

**结论：prefix_caching 适合有固定系统 prompt 或长共享上下文的场景；随机 prompt 场景下应关闭。**

### 4. 两类 workload 的本质差异

| 指标 | chat_short | long_context_qa |
|---|---|---|
| TTFT | ~48ms（prefill 轻） | ~39ms（prefill 虽长但 GPU 高效） |
| E2E | ~900ms | ~1400ms |
| E2E 构成 | 95% decode | decode 为主，prefill 占比更大 |
| GPU 利用率 | 81–86% | 89–94% |
| 瓶颈 | decode throughput | decode + prefill 内存带宽 |

long_context_qa 的 TTFT 反而比 chat_short 更低（39ms vs 48ms），原因是低并发（4 vs 16）下每批次竞争更少，prefill 能整块快速执行。

---

## 结论：最优配置推荐

| 场景 | 推荐变体 | 关键参数 |
|---|---|---|
| 高并发短对话 | **big_batch** | max_num_batched_tokens=4096 |
| 长文档/长上下文 | **chunked** 或 **prefix_cache** | enable_chunked_prefill=True 或 enable_prefix_caching=True |
| 混合场景 | chunked | 长短序列都不差 |

---

## WSL2 环境注意事项

- `nvidia-smi` 显示 ~5.9 GB 已用，但 vLLM 实际只占约 3.7 GB；另外 ~2.2 GB 是 Windows 桌面合成器占用，PyTorch/CUDA 侧不可见
- `gpu_memory_utilization=0.80` 是 6GB WSL2 环境的安全上限
- `max_model_len=2048` 必须显式设置；Qwen2.5 默认 32768 token 上下文，不设会 OOM
- vLLM 日志提示 `pin_memory=False`（WSL 检测），属于正常现象，不影响功能

---

## 下一步（Phase 2）

- 将本次 benchmark 链路接入 LangGraph agent（plan → run → analyze 三节点）
- Agent 自动根据上一轮结果选择下一组参数，实现闭环搜索
- 引入多目标权衡（吞吐 vs 尾延迟），探索 Pareto 最优配置
