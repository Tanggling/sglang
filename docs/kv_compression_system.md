# KV Cache 压缩系统：推理流程与 Metrics 统计

本文档描述 SGLang fork 中 KV Cache 压缩系统的核心设计，包括推理时的完整压缩流程、调度器适配、以及 Metrics 统计方法。供其他上下文快速理解系统全貌。

---

## 1. 核心架构：双池分离

系统将 KV 存储分为两个池：

| 池 | 生命周期 | 层数 | 存储内容 |
|---|---|---|---|
| **GlobalKVPool** | 临时（单层处理完即释放） | 1 层 | 未压缩的完整 KV |
| **Real KV Pool** | 持久（直到请求结束） | L 层 | 压缩后的 KV |

**关键文件**：`python/sglang/srt/mem_cache/global_kv_pool.py`

```python
# GlobalKVPool 结构
k_buffer: [max_tokens+1, num_kv_heads, head_dim]  # slot 0 保留
v_buffer: [max_tokens+1, num_kv_heads, head_dim]
free_slots: 空闲 slot 列表，alloc(n)/free(slots) 管理
```

**显存公式**：
- GlobalKVPool 开销 = `S × H × D × B`（仅 1 层，S=最大序列长度）
- Real Pool 容量 = `M_rest / (L × H × D × B)`
- 有效并发容量 ≈ `Real Pool 容量 / (1 - r)`，其中 r 为压缩比

---

## 2. 推理压缩流程（三阶段流水线）

入口：`compressed_flashattention_backend.py::_forward_extend_global_real_split()`

### 2.1 Layer 0 初始化

```python
# compressed_flashattention_backend.py:255-263
if layer_id == 0:
    forward_batch._gr_real_slots = {}       # 每个序列的压缩后 real pool slots
    forward_batch._gr_compressed_lens = []  # 压缩后长度列表
    forward_batch._gr_slots_to_free = []    # 多余预分配 slots（待释放）
    forward_batch._timer_prefill = LayerAccumTimer()
    forward_batch._timer_compress = LayerAccumTimer()
    forward_batch._timer_transfer = LayerAccumTimer()
```

### 2.2 Phase 1：在 GlobalKVPool 中组装完整 KV

**每层、每个序列**执行：

1. **CPU prefix cache 命中**：从 CPU 加载 KV 到 GlobalKVPool
   ```python
   # compressed_flashattention_backend.py:310-350
   forward_batch._timer_transfer.mark_start()
   k_prefix, v_prefix = cpu_prefix_cache.load_layer_to_gpu(entry, layer_id, ...)
   forward_batch._timer_transfer.mark_end()
   global_kv_pool.write_kv(global_slots[:prefix_len], k_prefix, v_prefix)
   ```

2. **模型 extend**：将当前层计算出的新 k, v 写入 GlobalKVPool
   ```python
   global_kv_pool.write_kv(global_slots[prefix_len:], k_extend, v_extend)
   ```

3. **拼接**：`k_full = cat([k_prefix, k_extend])`，得到完整未压缩 KV

### 2.3 Phase 2：FlashAttention 全量注意力

```python
# compressed_flashattention_backend.py:394-422
k_all = torch.cat(all_k_for_attn, dim=0)  # 所有序列的完整 KV
v_all = torch.cat(all_v_for_attn, dim=0)
output = flash_attn_varlen_func(
    q=q_view, k=k_all, v=v_all,
    cu_seqlens_q=..., cu_seqlens_k=...,
    causal=True,
)
```

注意力计算使用**完整未压缩 KV**，保证输出质量不受压缩影响。

### 2.4 Phase 3：压缩并写入 Real Pool

**每层、每个序列**执行：

```python
# compressed_flashattention_backend.py:424-500
# 1. 重要性评估 → 得到保留的 token 索引
_, _, keep_indices = self._estimate_importance(
    method=self.importance_method,  # "lse" / "key_norm" / "snapkv"
    q=q_for_compress, k=k_full, v=v_full, ...
)

# 2. Layer 0 时确定 real slots 并释放多余预分配
if layer_id == 0:
    real_slots = real_slots_seq[:num_to_keep]
    excess = real_slots_seq[num_to_keep:]
    if excess.numel() > 0:
        forward_batch._gr_slots_to_free.append(excess)
    # 更新 req_to_token 映射
    req_to_token_pool.write((req_id, slice(0, num_to_keep)), real_slots)

# 3. 将压缩后的 KV 写入 Real Pool
k_buf[real_slots] = k_full[keep_indices]
v_buf[real_slots] = v_full[keep_indices]

# 4. 释放 GlobalKVPool slots（下一层复用）
global_kv_pool.free(global_slots)
```

### 2.5 最后一层收尾

```python
# compressed_flashattention_backend.py:519-553
if layer_id == num_layers - 1:
    # 释放所有多余预分配 slots
    if forward_batch._gr_slots_to_free:
        token_to_kv_pool_allocator.free(torch.cat(slots_to_free))

    # CPU cache miss 的序列：保存到 CPU prefix cache
    for seq_idx in cpu_miss_set:
        cpu_prefix_cache.finalize_entry(req_id, tokens)

    # 同步计时器并记录 per-request metrics
    _prefill_ms = forward_batch._timer_prefill.sync_and_total()
    _compress_ms = forward_batch._timer_compress.sync_and_total()
    _transfer_ms = forward_batch._timer_transfer.sync_and_total()
    for _si in range(batch_size):
        _req_id = req_pool_indices[_si].item()
        _cm.log_prefill_time(_req_id, _prefill_ms / batch_size)
        _cm.log_compress_time(_req_id, _compress_ms / batch_size)

    _cm.log_global_summary()
```

### 2.6 流程图

```
Layer 0                    Layer 1 ... Layer L-1
┌──────────────────┐      ┌──────────────────┐
│ Phase1: 组装全KV  │      │ Phase1: 组装全KV  │
│  CPU→GlobalPool  │      │  (同样流程)       │
│  Model→GlobalPool│      │                  │
├──────────────────┤      ├──────────────────┤
│ Phase2: FlashAttn│      │ Phase2: FlashAttn│
│  (完整KV,无损)   │      │                  │
├──────────────────┤      ├──────────────────┤
│ Phase3: 压缩     │      │ Phase3: 压缩     │
│  重要性评估      │      │  重要性评估       │
│  确定real_slots  │      │  复用layer0 slots│
│  写入RealPool    │      │  写入RealPool    │
│  释放GlobalPool  │      │  释放GlobalPool  │
└──────────────────┘      ├──────────────────┤
                          │ 收尾:            │
                          │  释放多余slots   │
                          │  同步计时器      │
                          │  记录metrics     │
                          └──────────────────┘
```

---

## 3. 预分配策略（Scheduler 层）

### 3.1 prepare_for_extend 中的压缩长度预计算

```python
# schedule_batch.py:1543-1577
_ratio = server_args.kv_compression_ratio  # e.g. 0.5
_min_tok = server_args.kv_compression_min_tokens
_win = server_args.kv_compression_window_size

for _pl, _el in zip(prefix_lens, extend_lens):
    _full_len = _pl + _el
    if _full_len <= _min_tok or _full_len <= _win:
        _comp_lens.append(_full_len)          # 短序列不压缩
    else:
        _comp_lens.append(max(_min_tok, int(_full_len * (1.0 - _ratio))))

# 用压缩后长度分配 real pool slots
out_cache_loc, req_pool_indices, ... = alloc_for_extend(
    self, use_compressed_len=True
)
```

### 3.2 调度器预算检查适配

`PrefillAdder`（`schedule_policy.py`）接收 `kv_compression_ratio` 参数：

```python
# schedule_policy.py:734-753 (add_one_req)
total_tokens = extend_input_len + max_new_tokens
if self.kv_compression_ratio > 0.0:
    total_tokens = int(total_tokens * (1 - self.kv_compression_ratio))
# 用压缩后的 token 数和 real pool 可用量比较
if total_tokens >= self.rem_total_tokens:
    return AddReqResult.NO_TOKEN
```

```python
# schedule_policy.py:514-525 (_update_prefill_budget)
pool_tokens = extend_input_len + max_new_tokens
if self.kv_compression_ratio > 0.0:
    pool_tokens = int(pool_tokens * (1 - self.kv_compression_ratio))
self.rem_total_token_offset += pool_tokens  # 扣减压缩后的量
```

### 3.3 最大请求长度适配

```python
# tp_worker.py:287-301
if compression_ratio > 0.0:
    effective_pool_size = int(max_token_pool_size / (1 - compression_ratio))
else:
    effective_pool_size = max_token_pool_size
max_req_len = min(context_len - 1, effective_pool_size - 1)
max_req_input_len = max_req_len - 5
```

---

## 4. Metrics 统计系统

**核心文件**：`python/sglang/srt/layers/attention/compression_metrics.py`

### 4.1 计时工具

#### CudaTimer（同步阻塞）
```python
# 用于单次测量，__exit__ 时 synchronize
with CudaTimer() as t:
    gpu_work()
print(t.elapsed_ms)
```

#### LayerAccumTimer（非阻塞累积）
```python
# 跨层累积，仅在最后一层同步一次
timer = LayerAccumTimer()
for layer in layers:
    timer.mark_start()   # 记录 CUDA event（不同步）
    gpu_work()
    timer.mark_end()     # 记录 CUDA event（不同步）
total_ms = timer.sync_and_total()  # 最后同步一次，求和
```

### 4.2 Per-request 生命周期追踪

```python
@dataclass
class RequestMetrics:
    req_id: int
    original_len: int          # 原始序列长度
    compressed_len: int        # 压缩后长度
    cpu_cache_hit: bool        # 是否命中 CPU prefix cache
    cpu_match_len: int         # CPU cache 匹配长度
    prefill_ms: float          # prefill 耗时
    cpu_transfer_ms: float     # CPU→GPU 传输耗时
    compress_ms: float         # 压缩耗时
    decode_total_ms: float     # decode 总耗时（累积所有 step）
    decode_steps: int          # decode 步数
```

### 4.3 CompressionMetrics 单例

```python
_global_metrics = CompressionMetrics()  # 通过 get_metrics() 获取

class CompressionMetrics:
    # 全局累积计数器
    total_requests, total_original_tokens, total_compressed_tokens
    cpu_cache_hits, cpu_cache_misses
    total_prefill_ms, total_compress_ms, total_decode_ms, decode_steps

    # Per-request 追踪
    _active: Dict[int, RequestMetrics]   # req_pool_idx → 活跃请求
    _history: List[RequestMetrics]       # 已完成请求（最近 256 个）
```

### 4.4 数据流

```
Prefill 阶段（backend forward_extend, 最后一层）:
  log_compression(req_id, original_len, compressed_len)
      → 创建 _active[req_id]，记录压缩信息
      → 累加全局 total_original_tokens, total_compressed_tokens
  log_prefill_time(req_id, ms)
      → _active[req_id].prefill_ms = ms
  log_compress_time(req_id, ms)
      → _active[req_id].compress_ms += ms
  log_cpu_transfer_time(req_id, ms)   [仅 CPU cache hit]
      → _active[req_id].cpu_transfer_ms += ms
  log_cpu_cache_hit/miss(req_id, ...)
      → _active[req_id].cpu_cache_hit = True/False

Decode 阶段（backend forward_decode）:
  log_decode_step(req_ids=[id1, id2, ...], batch_ms=1.5)
      → per_req_ms = batch_ms / len(req_ids)
      → 每个 _active[rid].decode_total_ms += per_req_ms
      → 每个 _active[rid].decode_steps += 1

请求完成（scheduler_output_processor_mixin, req.finished()）:
  # 注意：必须在 release_kv_cache 之前保存 req_pool_idx
  _saved_pool_idx = req.req_pool_idx.item()
  release_kv_cache(req, tree_cache)  # 会清空 req.req_pool_idx
  get_metrics().finalize_request(_saved_pool_idx)
      → 从 _active 弹出，输出单次请求日志，移入 _history
```

### 4.5 输出格式

**Per-request 日志**（`finalize_request`，请求完成时输出一次）：
```
[KV Req#12] | seq=8192→4096 (50%) | cpu_hit=7000tok | transfer=15.3ms | prefill=42.1ms | compress=30.5ms | decode=312.4ms (256steps, 1.22ms/tok) | total=400.3ms
```

**Global 平均日志**（`log_global_summary`，每次 prefill batch 后输出）：
```
[KV Compress Global] reqs: 100, avg ratio: 50.0% kept, cpu cache hit: 80/100 (80%), avg match: 5000 tokens, avg transfer: 12.5ms, avg prefill: 40.0ms, avg compress: 28.0ms, avg decode: 1.15ms/tok
```

### 4.6 调用位置汇总

| 方法 | 调用位置 | 时机 |
|------|---------|------|
| `log_compression` | `compressed_flashattention_backend.py:494` | Phase 3 每层每序列 |
| `log_cpu_cache_hit/miss` | `compressed_flashattention_backend.py:274-286` | Layer 0 初始化 |
| `log_prefill_time` | `compressed_flashattention_backend.py:548` | 最后一层收尾 |
| `log_compress_time` | `compressed_flashattention_backend.py:549` | 最后一层收尾 |
| `log_cpu_transfer_time` | `compressed_flashattention_backend.py:551` | 最后一层收尾 |
| `log_decode_step` | `compressed_flashattention_backend.py:1614-1616` | 每次 decode forward |
| `log_global_summary` | `compressed_flashattention_backend.py:553` | 每次 prefill batch 后 |
| `finalize_request` | `scheduler_output_processor_mixin.py:502-504` | 请求完成时 |

---

## 5. 关键设计约束与注意事项

1. **GlobalKVPool 容量**：必须 ≥ 单个请求的最大未压缩序列长度（`min(context_len, max_total_num_tokens)`）
2. **req_pool_idx 类型**：backend 中通过 `.item()` 转为 int 存入 `_active`，finalize 时也需要 `.item()` 转换
3. **release_kv_cache 顺序**：`finalize_request` 必须在 `release_kv_cache` 之前获取 `req_pool_idx`，因为后者会将其置为 None
4. **计时器同步策略**：`LayerAccumTimer` 在每层只记录 CUDA event 不同步，最后一层调用 `sync_and_total()` 一次性同步，避免逐层同步的性能开销
5. **Decode 时间分摊**：batch decode 时间按请求数均分（`batch_ms / len(req_ids)`），因为 batch 中所有请求共享一次 forward
