# Global KV Buffer: Cross-Layer Shared Temporary Buffer for KV Cache Compression

## 1. 问题：朴素压缩方案的显存浪费

在标准 LLM serving 中，KV cache pool 按 `num_layers × max_total_num_tokens` 静态预分配。
如果在 prefill 后对 KV cache 做压缩（如 SnapKV），朴素方案存在一个矛盾：

> **压缩需要先看到完整 KV，但完整 KV 需要在每一层都分配 pool 空间。**

```
朴素方案内存布局:
┌──────────────────────────────────────────────────────────────┐
│ Real KV Pool:  T_full × L × H × D × B                       │
│ (每层都为完整序列分配 slot, 压缩后大量 slot 闲置等待释放)      │
└──────────────────────────────────────────────────────────────┘
```

压缩后只保留 `T_full × (1-r)` 个 token，但在压缩完成前，所有 `T_full × L` 个 slot 都必须被占用。
这意味着在 prefill 高峰期，pool 必须能容纳全量 token，压缩带来的显存节省**无法体现在 pool 的最大容量上**。

## 2. 核心思想：分离 "临时计算" 与 "持久存储"

**关键观察**：Transformer 是逐层计算的。在任一时刻，只有**当前层**需要完整 KV，其他层要么还没算到，要么已经压缩完毕。

因此，我们将 KV 存储分为两个池：

| 池 | 生命周期 | 大小 | 用途 |
|---|---|---|---|
| **GlobalKVPool** (临时) | 单层内 → 用完即释放 | `T_global × 1层` | 存放当前层的完整 KV，供 FlashAttention 和压缩算法使用 |
| **Real KV Pool** (持久) | 跨层 → decode 期间持续使用 | `T_compressed × L层` | 存放压缩后的 KV，供 decode 阶段使用 |

```
Global/Real Split 内存布局:
┌────────────────────────────────┐
│ GlobalKVPool: T_global × 1层   │  ← 跨层共享，每层复用
│ (临时: 写入→FA→压缩→释放)      │
├────────────────────────────────┤
│ Real KV Pool: T_real × L层     │  ← 只存压缩后的 KV
│ (持久: decode 阶段使用)         │
└────────────────────────────────┘
```

## 3. 三阶段 Forward 流水线

每一层 transformer 的 `forward_extend` 分为三个阶段：

```
Phase 1: Assemble                Phase 2: Attention            Phase 3: Compress
┌─────────────────────┐         ┌──────────────────┐         ┌──────────────────────┐
│ CPU prefix KV ──┐   │         │                  │         │ importance(Q,K,V)    │
│                 ▼   │         │ flash_attn_      │         │       ↓              │
│ GlobalKVPool ← merge│    →    │ varlen_func()    │    →    │ top-k keep_indices   │
│                 ▲   │         │ (full KV)        │         │       ↓              │
│ Model K,V ──────┘   │         │                  │         │ K,V[keep] → RealPool │
└─────────────────────┘         └──────────────────┘         │ free(global_slots)   │
                                       ↓                     └──────────────────────┘
                                  output (2D)
                                  [total_q, H*D]
```

**Phase 1 — 组装完整 KV**:
- 将 CPU prefix cache (若命中) 的 KV 加载到 GlobalKVPool
- 将当前层 model 新产生的 K,V 写入 GlobalKVPool
- 读出连续的完整 K,V tensor

**Phase 2 — FlashAttention**:
- 使用 `flash_attn_varlen_func` 直接在连续 tensor 上计算（绕过 paged pool）
- 输出是模型的 attention output，形状与标准 FA 一致

**Phase 3 — 压缩 + 写入 + 释放**:
- SnapKV 重要性估计 → 选出 top-k token 索引
- 将选中的 K,V 写入 Real KV Pool 的预分配 slot
- **立即释放 GlobalKVPool slots** → 下一层复用同一块显存

## 4. 预分配策略：Scheduler 层的压缩长度计算

在 `prepare_for_extend` 阶段（**进入 model forward 之前**），scheduler 预计算压缩后长度并只分配该数量的 real pool slot：

```python
compressed_total_len = max(min_tokens, int(full_seq_len × (1 - ratio)))
# full_seq_len = prefix_len + extend_len
```

这使得 real pool 的占用在 **分配时** 就已经是压缩后的大小，而非全量。

## 5. 形式化显存分析

### 5.1 符号定义

| 符号 | 含义 |
|------|------|
| $L$  | Transformer 层数 |
| $H$  | KV heads 数 (GQA 下为 `num_kv_heads`) |
| $D$  | `head_dim + v_head_dim` (K 和 V 的维度之和) |
| $B$  | 每个元素的字节数 (bf16=2, fp8=1) |
| $T$  | 给定显存预算下可分配的最大 token 数 |
| $r$  | 压缩率 (0.5 表示丢弃 50% 的 token) |
| $S$  | 当前 batch 的最大序列长度 |
| $M$  | 可用 KV cache 显存总量 (由 `mem_fraction_static` 决定) |

### 5.2 每 token 的 cell size

```
cell_size = H × D × B    (单层单 token 的 KV 存储)
```

### 5.3 朴素方案 (Baseline, 无压缩)

全部显存分配给 Real KV Pool:

$$M_{baseline} = T_{base} \times L \times cell$$

$$T_{base} = \frac{M}{L \times cell}$$

每个请求占用 `seq_len` 个 pool slot, 可同时服务的总 token 数:

$$C_{base} = T_{base}$$

### 5.4 Global/Real Split 方案

显存分为两部分:

$$M = M_{global} + M_{real}$$

$$M_{global} = T_{global} \times 1 \times cell \quad \text{(单层)}$$

$$M_{real} = T_{real} \times L \times cell \quad \text{(L层)}$$

其中 $T_{global} = S$ (当前 batch 最大序列长度), 则:

$$T_{real} = \frac{M - S \times cell}{L \times cell} = T_{base} - \frac{S}{L}$$

**关键**: 每个请求只占用 `seq_len × (1-r)` 个 real pool slot, 等效可服务的总 token 数:

$$C_{split} = \frac{T_{real}}{1 - r}$$

### 5.5 显存效率增益

$$\text{Gain} = \frac{C_{split}}{C_{base}} = \frac{T_{real}}{T_{base} \times (1-r)} = \frac{T_{base} - S/L}{T_{base} \times (1-r)}$$

当 $S \ll T_{base} \times L$ (通常成立), 近似:

$$\boxed{\text{Gain} \approx \frac{1}{1-r}}$$

- $r = 0.5 \Rightarrow \text{Gain} \approx 2.0\times$
- $r = 0.7 \Rightarrow \text{Gain} \approx 3.33\times$

### 5.6 GlobalKVPool 的显存开销占比

$$\text{Overhead} = \frac{M_{global}}{M} = \frac{S \times cell}{T_{base} \times L \times cell} = \frac{S}{T_{base} \times L}$$

对于典型配置 ($S = 8192$, $T_{base} = 80000$, $L = 32$):

$$\text{Overhead} = \frac{8192}{80000 \times 32} \approx 0.32\%$$

**GlobalKVPool 的额外显存开销可以忽略不计。**

## 6. 数值 Demo

### 配置: Llama-3-8B, 24GB GPU, bf16

```
L  = 32                    # num_layers
H  = 8                     # num_kv_heads (GQA: 32 query heads, 8 kv heads)
D  = 128 + 128 = 256       # head_dim + v_head_dim
B  = 2                     # bf16
r  = 0.5                   # 压缩率

cell_size = 8 × 256 × 2 = 4096 bytes/token/layer = 4 KB/token/layer

mem_fraction_static = 0.8
可用 GPU 显存 ≈ 24GB × 0.8 - model_weights ≈ 10 GB (估算)
M = 10 × 1024³ bytes = 10,737,418,240 bytes

context_len = 8192 (最大序列长度)
```

#### Baseline (无压缩)

```
T_base = M / (L × cell) = 10,737,418,240 / (32 × 4096)
       = 81,920 tokens

每请求占用 seq_len 个 slot
并发 8192-token 请求数 = 81920 / 8192 = 10 个
```

#### Global/Real Split (r=0.5)

```
GlobalKVPool:
  M_global = 8192 × 4096 = 33,554,432 bytes = 32 MB (单层!)
  占总 KV 显存: 32MB / 10GB = 0.32%

Real KV Pool:
  M_real = M - M_global = 10,737,418,240 - 33,554,432 = 10,703,863,808 bytes
  T_real = M_real / (L × cell) = 10,703,863,808 / 131,072 = 81,664 tokens

每请求占用 seq_len × (1-r) = 8192 × 0.5 = 4096 个 slot
并发 8192-token 请求数 = 81664 / 4096 = 19.9 ≈ 19 个

提升: 19 / 10 = 1.9×
```

#### 汇总表

| 指标 | Baseline | Global/Real Split (r=0.5) | 变化 |
|------|----------|--------------------------|------|
| Real pool tokens | 81,920 | 81,664 | -0.3% |
| GlobalKVPool | 0 | 32 MB | +0.3% |
| 每请求 slot 占用 | 8,192 | 4,096 | **-50%** |
| 并发 8K 请求数 | 10 | 19 | **+90%** |
| Decode per-token 计算量 | ∝ 8192 | ∝ 4096 | **-50%** |

### 配置: Llama-3-70B, 80GB GPU (A100), bf16

```
L  = 80
H  = 8  (GQA)
D  = 256
B  = 2
cell = 8 × 256 × 2 = 4096 bytes

M ≈ 30 GB (80GB GPU, 扣除 ~50GB 模型权重)
T_base = 30GB / (80 × 4KB) = 30 × 1024³ / 327,680 = 98,304 tokens

context_len = 8192
GlobalKVPool = 8192 × 4096 = 32 MB (占 0.1%)
```

| 指标 | Baseline | Split (r=0.5) | Split (r=0.7) |
|------|----------|---------------|---------------|
| T_real | 98,304 | 98,204 | 98,204 |
| 每请求 slot | 8,192 | 4,096 | 2,458 |
| 并发 8K 请求数 | 12 | 23 | 39 |
| 提升 | 1× | **1.92×** | **3.25×** |

## 7. Decode 加速分析

压缩不仅省显存, decode 阶段的 per-token 延迟也下降:

```
Decode attention 计算量:
  FLOPs ∝ batch_size × num_heads × seq_len × head_dim
                                    ^^^^^^^^
                                    压缩后为 seq_len × (1-r)

理论加速 = 1 / (1-r)
  r=0.5 → decode 速度 2×
  r=0.7 → decode 速度 3.3×
```

Decode 是 **memory-bound** 操作, KV cache 的 read bandwidth 是瓶颈。
压缩后 KV cache 减小, memory read 量降低, **实际加速接近理论值**。

## 8. 设计约束与权衡

| 约束 | 说明 |
|------|------|
| GlobalKVPool 大小 | 必须 ≥ batch 内最大序列长度, 否则 OOM |
| 压缩质量 | 压缩丢弃了部分 token, 可能影响输出质量; 通过 `window_size` 保留最近 token 缓解 |
| Prefill 延迟 | 增加了压缩计算开销 (SnapKV 的 Q-K matmul + topk), 但总 prefill 时间通常仍被 FA 主导 |
| 跨层一致性 | 所有层使用 layer-0 确定的 `keep_indices` 和 `real_slots` (或每层独立计算, 本实现为每层独立) |
| CPU prefix cache | GlobalKVPool 设计天然适配: CPU 命中时, prefix KV 直接加载到 global buffer 而非 real pool |

## 9. 代码结构总览

```shell
schedule_batch.py::prepare_for_extend()
  ├── 计算 compressed_total_lens = (prefix+extend) × (1-r)
  ├── alloc_for_extend(use_compressed_len=True)  # 只分配压缩后 slot
  └── 传递 compressed_total_lens_cpu 到 ForwardBatch

model_runner_kv_cache_mixin.py::init_memory_pool()
  └── GlobalKVPool(max_tokens=context_len, 1层)

compressed_flashattention_backend.py::_forward_extend_global_real_split()
  ├── Phase 1: CPU prefix + model KV → GlobalKVPool
  ├── Phase 2: flash_attn_varlen_func(full KV from GlobalKVPool)
  └── Phase 3: compress → write to RealPool → free GlobalKVPool slots

compressed_flashattention_backend.py::forward_decode()
  └── 标准 decode (读取 Real KV Pool 中的压缩 KV)
```
