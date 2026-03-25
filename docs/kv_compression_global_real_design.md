# KV Cache 压缩扩展设计文档

## 概述

本文档描述在现有 SnapKV 风格 KV Cache 压缩方案基础上新增的两项功能：

1. **Global/Real 双池设计**：将 KV Cache 分为全量临时缓冲（`GlobalKVPool`）和压缩持久化缓冲（`RealKVPool`），使 alloc 步骤以压缩后的长度为准。
2. **CPU Prefix 缓存**：对相同前缀的多次请求，第一次 prefill 结束后将全量 KV 存入 CPU，后续请求从 CPU 加载并压缩，跳过 GPU prefill。

---

## 背景

### 现有设计痛点

现有 `CompressedFlashAttentionBackend` 的标准压缩流程：

```
Prefill:
  1. 调度器预分配 seq_len 个 token slot（全量）
  2. 每层 forward_extend：全量写入 KV → 压缩 → 就地移动 KV → 释放多余 slot
```

**问题**：调度器在决定是否接受请求时，检查的是 `seq_len`（全量）而非压缩后的长度。这导致：
- 内存使用估算偏大，排队请求更少
- `GlobalKVPool` 为空：每层处理窗口内需要同时持有全量 KV，但没有明确的临时缓冲层

---

## Feature 1：Global/Real 双池设计

### 核心思想

```
GlobalKVPool   ← 单层全量 KV 缓冲（跨层复用，逐层释放）
RealKVPool     ← 多层压缩 KV 缓冲（持久，按压缩后长度分配）
```

每层处理时：
1. 全量 k, v → `GlobalKVPool`（按全量 seq_len 分配）
2. 压缩 → 确定 `keep_indices`
3. 压缩后 k, v → `RealKVPool`（仅分配 `compressed_len` 个 slot）
4. 释放 `GlobalKVPool` slot（供下一层复用）

### 内存对比

| 设计 | GPU HBM 用量（prefill 阶段） |
|------|---------------------------|
| 旧（全量 alloc） | `seq_len × num_layers × 2 × heads × dim`（先分配后释放） |
| 新（Global/Real）| `seq_len × 1 × 2 × heads × dim`（Global，逐层复用）<br>+ `compressed_len × num_layers × 2 × heads × dim`（Real） |

当 `compression_ratio=0.5`，Global 池仅为原来单层的开销。

### 调度器改造

调度器接受请求的内存检查应修改为：

```python
# 旧逻辑
if token_to_kv_pool_allocator.available_size() < seq_len * num_layers:
    reject()

# 新逻辑（Feature 1）
compressed_len = ceil(seq_len * (1 - compression_ratio))
if real_kv_pool_allocator.available_size() < compressed_len:
    reject()
if global_kv_pool.available_size() < seq_len:   # 一层缓冲足够
    reject()
```

### 数据结构

#### `GlobalKVPool`（新增，`mem_cache/global_kv_pool.py`）

```
k_buffer: [max_tokens + 1, num_kv_heads, head_dim]   # 单层，GPU
v_buffer: [max_tokens + 1, num_kv_heads, v_head_dim]  # 单层，GPU
free_slots: [max_tokens]   # 简单 free-list，不分页
```

- `max_tokens` = 最大 batch token 数（不是 `num_layers × max_tokens`）
- slot 0 预留为 dummy（与 `MHATokenToKVPool` 约定一致）
- 每层 forward 完成后 `free(global_slots)`，全部归还

#### `RealKVPool`

复用现有 `MHATokenToKVPool`，但 size 改为 `max_compressed_tokens`（≈ `max_tokens × (1 - ratio)`）。调度器使用 `real_kv_pool_allocator.available_size()` 做内存检查。

#### `req_to_token_pool`（映射关系）

`req_to_token[req_pool_idx, 0:compressed_len]` → `real_kv_pool` 的 slot 索引。

与旧设计相同，只是 slot 指向 `RealKVPool`（更小）而不是原始全量池。

### 前向传播流程（`_forward_extend_global_real_split`）

#### 与预分配的关系

`prepare_for_extend()` 调用 `alloc_for_extend()` 时，已从 `token_to_kv_pool_allocator` 预分配了 `extend_num_tokens`（全量）个 slot，存入 `forward_batch.out_cache_loc`。forward 函数内部**不能再次分配**，否则会双重占用并泄漏预分配的 slot。

正确做法：利用预分配的 `out_cache_loc`，取其前 `compressed_len` 个位置作为 "real slots"，在 layer 0 末尾将多余的 `seq_len - compressed_len` 个 slot **归还**给 allocator。

```
对 batch 中每个 sequence seq_idx：
  seq_cache_loc = out_cache_loc[start:end]   ← 预分配的全量 slot（来自 alloc_for_extend）

  Layer 0:
    ① global_slots = GlobalKVPool.alloc(seq_len)
    ② GlobalKVPool.write_kv(global_slots, k_seq, v_seq)
    ③ [可选] 保存到 CPU（Feature 2）
    ④ keep_indices = compress(q_seq, k_seq, v_seq)
    ⑤ real_slots = seq_cache_loc[:compressed_len]       ← 复用预分配 slot 的前段
    ⑥ k_buffer[0][real_slots] = k_seq[keep_indices]
       v_buffer[0][real_slots] = v_seq[keep_indices]
    ⑦ req_to_token_pool.write(req_pool_idx, 0:compressed_len, real_slots)
    ⑧ token_to_kv_pool_allocator.free(seq_cache_loc[compressed_len:])  ← 归还多余 slot
    ⑨ GlobalKVPool.free(global_slots)

  Layer 1 ~ L-1:
    ① global_slots = GlobalKVPool.alloc(seq_len)
    ② GlobalKVPool.write_kv(global_slots, k_seq, v_seq)
    ③ keep_indices = compress(q_seq, k_seq, v_seq)     ← 各层独立压缩
    ④ real_slots 复用 Layer 0 确定的（seq_cache_loc[:compressed_len]）
    ⑤ k_buffer[L][real_slots] = k_seq[keep_indices]
       v_buffer[L][real_slots] = v_seq[keep_indices]
    ⑥ GlobalKVPool.free(global_slots)
```

**净效果**：Layer 0 结束后，每个 sequence 只占用 `compressed_len` 个永久 slot（多余的已归还）。调度器可在 `alloc_for_extend` 前检查 `available_size() >= compressed_len` 来决定是否接受请求。

> **注意**：各层的 `keep_indices` 可能不同（SnapKV 是每层独立计算的），但 `real_slots`（物理地址）相同。这意味着 `RealKVPool.k_buffer[L][real_slots[i]]` 在不同 L 存储的是不同位置的 token，但都通过 `req_to_token_pool` 的同一映射访问——这与现有设计一致（现有代码也是各层 KV 移动到 `seq_cache_loc[:num_to_keep]`，映射只在 layer 0 更新）。

### Decode 阶段

Decode 阶段无需 GlobalKVPool：
- 新 token 的 k, v 直接写入 `RealKVPool`（通过正常的 `set_kv_buffer` 路径）
- Attention 计算读取 `RealKVPool.k_buffer[L][req_to_token[req_pool_idx, :compressed_len+decode_steps]]`

---

## Feature 2：CPU Prefix KV 缓存

### 适用场景

- 多个请求共享相同的长 prefix（如系统 prompt）
- 单卡 HBM 装不下所有请求的全量 KV
- Radix cache 关闭（无法复用 GPU KV）

### 核心思想

```
第一次请求（相同 prefix）：
  Prefill → 压缩 → Decode
  Prefill 时顺手将每层的全量 k, v（压缩前）保存到 CPU RAM

第二次及以后请求（相同 prefix）：
  for each layer L:
    CPU 加载 k_cpu[L], v_cpu[L] → GPU
    压缩（使用新请求的 query）
    写入 RealKVPool[L][real_slots]
  直接进入 Decode（跳过 GPU Prefill）
```

### 为什么存全量而非压缩后？

- 不同请求的 query 不同 → importance scores 不同 → keep_indices 可能不同
- 存全量（CPU 便宜）→ GPU 端按新 query 重新压缩 → 更精确

### 数据结构（`mem_cache/prefix_cpu_cache.py`）

```python
CPUKVEntry:
    token_ids: List[int]          # prefix token IDs（用于碰撞检测）
    seq_len:   int
    kv_layers: List[(k_cpu, v_cpu)]  # [num_layers] × [seq_len, num_kv_heads, head_dim]

PrefixCPUCache:
    _cache: Dict[hash(token_ids) → CPUKVEntry]
    _lru_order: List[int]         # 最近使用顺序，用于 LRU 淘汰
    max_entries: int
    max_total_bytes: Optional[int]
```

### 接口

```python
# 第一次请求 prefill 时（每层调用一次）
cache.accumulate_layer_kv(request_id, layer_id, k_gpu, v_gpu)

# 所有层处理完毕后
cache.finalize_entry(request_id, prefix_token_ids)

# 第二次请求到来时
entry = cache.lookup(prefix_token_ids)
if entry:
    backend.prefill_from_cpu_cache(
        token_ids=prefix_token_ids,
        forward_batch=...,
        seq_idx=...,
        q_for_compression=new_query,  # 用新请求的 query 做压缩
    )
```

### CPU→GPU 传输开销

每层加载：
```
transfer_time = seq_len × num_kv_heads × head_dim × 2 (K+V) × bytes_per_element / bandwidth
```

以 Llama-2-7B（32 层，32 heads，128 dim，fp16）为例：
- 2048 token prefix：每层约 32 × 2048 × 32 × 128 × 2B ≈ 0.5 GB
- PCIe 4.0 带宽 ~16 GB/s → 每层 ~31ms，32 层共 ~1s

**优化**：可使用 NVLink 或 pinned memory（non-blocking transfer）将传输与 GPU 计算重叠。

### 与 Feature 1 集成

Feature 2 在 Feature 1 的基础上运行最为自然：
- 第一次请求：`GlobalKVPool` 写完后、free 之前，将 k, v 保存到 `_cpu_accumulator`
- 第二次请求：CPU → GPU 后直接进入压缩 → 写 `RealKVPool` 流程

---

## 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `mem_cache/global_kv_pool.py` | **新增** | GlobalKVPool 类 |
| `mem_cache/prefix_cpu_cache.py` | **新增** | PrefixCPUCache、CPUKVEntry 类 |
| `layers/attention/compressed_flashattention_backend.py` | **修改** | 新增 `global_real_split` 模式；集成 CPU 缓存接口 |
| `mem_cache/memory_pool.py` | **待修改** | 可选：扩展 `ReqToTokenPool.write_global()` |
| `model_executor/model_runner.py` | **待修改** | 初始化 GlobalKVPool；将其传入 backend |
| `managers/scheduler.py` 或等价 | **待修改** | 调度器内存检查改用 `compressed_len` |

---

## 关键约束与注意事项

### 约束 1：`num_to_keep` 必须在所有层一致

`req_to_token_pool` 为一个请求维护单一的 token 位置映射。所有层共用同一个 `real_slots` 数组，因此 `compressed_len`（即 `num_to_keep`）必须在 Layer 0 确定，并对所有后续层复用。

每层可选择不同的 token（不同的 `keep_indices`），但最终保留的数量必须相同。

### 约束 2：各层压缩结果独立

现有代码对每层分别执行 SnapKV 压缩（不同层的 attention pattern 不同）。`_forward_extend_global_real_split` 同样对每层独立压缩，保持了这一行为。

### 约束 3：Radix Cache 必须关闭

两项功能均假设 `--disable-radix-cache`。Radix Cache 的前缀共享逻辑与 Global/Real 双池设计的 slot 分配存在冲突。

### 约束 4：CPU Prefix Cache 的键碰撞

使用 `hash(tuple(token_ids))` 作为键，存在极小概率的哈希碰撞。已在 `lookup()` 中加入 token IDs 等值校验（`CPUKVEntry.token_ids == token_ids`）作为防护。

---

## 使用示例

```python
from sglang.srt.mem_cache.global_kv_pool import GlobalKVPool
from sglang.srt.mem_cache.prefix_cpu_cache import PrefixCPUCache
from sglang.srt.layers.attention.kv_compressor import CompressionConfig
from sglang.srt.layers.attention.compressed_flashattention_backend import (
    CompressedFlashAttentionBackend,
)

# 创建 GlobalKVPool（单层，max_batch_tokens 大小）
global_pool = GlobalKVPool(
    max_tokens=4096,         # 最大 batch token 数
    num_kv_heads=32,
    head_dim=128,
    v_head_dim=128,
    dtype=torch.float16,
    device="cuda:0",
)

# 创建 CPU Prefix Cache
cpu_cache = PrefixCPUCache(
    max_entries=32,
    max_total_bytes=8 * 1024**3,  # 8 GB CPU RAM
)

# 创建压缩 Backend（Feature 1 + Feature 2 同时启用）
compression_cfg = CompressionConfig(
    enabled=True,
    compression_ratio=0.5,
    compression_method="snapkv",
    window_size=64,
    min_tokens_to_keep=32,
)

backend = CompressedFlashAttentionBackend(
    runner,
    compression_config=compression_cfg,
    importance_method="snapkv",
    compression_scheme="global_real_split",   # Feature 1
    global_kv_pool=global_pool,
    real_kv_pool_allocator=runner.token_to_kv_pool_allocator,
    cpu_prefix_cache=cpu_cache,               # Feature 2
    save_prefix_to_cpu=True,
)

# 调度器内存检查（示意）
seq_len = 2048
if not backend.can_accept_request(seq_len):
    print("内存不足，拒绝请求")
```

---

## 后续工作

1. **调度器集成**：修改 `scheduler.py` 中的 `get_new_batch_prefill` / `_schedule` 逻辑，调用 `backend.can_accept_request(seq_len)` 替代原有的 slot 数量检查。

2. **GlobalKVPool 大小自动计算**：在 `model_runner.py` 初始化时，根据最大 batch size 和 chunked prefill 配置计算 `GlobalKVPool.max_tokens`。

3. **Pinned Memory 加速**：为 `PrefixCPUCache` 的 CPU tensor 使用 `pin_memory=True`，配合 `non_blocking=True` 转移，减少 CPU→GPU 等待时间。

4. **多卡支持**：当前 `GlobalKVPool` 为单设备。TP（张量并行）环境下每个 rank 维护独立 GlobalKVPool；PP（流水线并行）需协调各 stage 的 CPU 缓存。

5. **压缩 ratio 动态调整**：根据 `RealKVPool.available_size()` 实时调整 `compression_ratio`，在内存压力大时压缩更多。
