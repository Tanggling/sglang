# Pinned Memory 加速 CPU Prefix Cache 传输实验报告

> 实验平台: NVIDIA A800-SXM4-80GB, PCIe Gen4 x16
> 模型: Qwen2.5-7B-Instruct (4 kv_heads, 28 q_heads, 128 head_dim)
> 数据类型: BFloat16
> 实验脚本: `benchmark/kv_compression/bench_pinned_memory_transfer.py`

---

## 1. 背景与动机

在 KV 压缩 pipeline 中，CPU Prefix Cache 命中时需要将前缀 KV 从 CPU 传输到 GPU 的 GlobalKVPool。当前实现使用 pageable memory（普通 `torch.randn` 分配），传输带宽远低于 PCIe 理论上限。

**传输路径**:
```
CPU Prefix Cache (per layer KV) → PCIe Gen4 x16 → GPU GlobalKVPool → FA → Compress → Real Pool
```

**传输数据量（单层）**:
```
transfer_bytes = prefix_len × num_kv_heads × head_dim × 2 (K+V) × sizeof(bf16)
               = prefix_len × 4 × 128 × 2 × 2 bytes
               = prefix_len × 2048 bytes
```

---

## 2. PCIe Gen4 x16 带宽理论分析

| 参数 | 值 |
|------|-----|
| 传输速率 | 16 GT/s per lane |
| 通道数 | 16 lanes |
| 编码效率 | 128b/130b (98.46%) |
| **理论单向带宽** | **31.5 GB/s** |
| 实际可达峰值 (pinned) | ~25 GB/s |
| Pageable 典型带宽 | ~8-14 GB/s |

**Pageable vs Pinned 的本质区别**:
- **Pageable**: 数据在用户空间 → CUDA driver 先 memcpy 到 pinned staging buffer → DMA 传输到 GPU（两次拷贝）
- **Pinned**: 数据已在 DMA 可达的锁页内存 → 直接 DMA 传输到 GPU（一次拷贝）
- **Non-blocking**: DMA 引擎独立于 CPU 工作，CPU 可继续执行其他操作

---

## 3. 实验一：原始传输带宽测量

测量不同数据量下 pageable 和 pinned memory 的 CPU→GPU 传输带宽。

### 数据

| tokens | data_MB | pageable (ms) | pageable BW | pinned+nb (ms) | pinned BW | 加速比 | 达到理论% |
|--------|---------|---------------|-------------|----------------|-----------|--------|-----------|
| 512 | 1.0 | 0.150 | 6.5 GB/s | 0.059 | 16.7 GB/s | 2.57x | 53.0% |
| 1,024 | 2.0 | 0.266 | 7.3 GB/s | 0.101 | 19.3 GB/s | 2.64x | 61.3% |
| 2,048 | 4.0 | 0.393 | 9.9 GB/s | 0.186 | 21.0 GB/s | 2.11x | 66.6% |
| 4,096 | 8.0 | 0.640 | 12.2 GB/s | 0.351 | 22.2 GB/s | 1.82x | 70.6% |
| 8,192 | 16.0 | 1.120 | 14.0 GB/s | 0.702 | 22.2 GB/s | 1.59x | 70.6% |
| 16,384 | 32.0 | 2.981 | 10.5 GB/s | 1.365 | 22.9 GB/s | 2.18x | 72.7% |
| 32,768 | 64.0 | 5.599 | 11.2 GB/s | 2.724 | 22.9 GB/s | 2.06x | 72.8% |
| 65,536 | 128.0 | 12.848 | 9.7 GB/s | 5.458 | 22.9 GB/s | 2.35x | 72.7% |

### 分析

1. **Pinned memory 稳定达到 22-23 GB/s**，约为 PCIe Gen4 x16 理论带宽的 **72-73%**
2. **Pageable memory 仅 9.7-14 GB/s**，约为理论带宽的 **31-44%**
3. **加速比 1.6x-2.6x**，小数据量（<4KB）加速更明显（kernel launch overhead 占比更大）
4. **大数据量（≥16K tokens）加速稳定在 2.0-2.4x**

**为什么达不到 100% 理论带宽？**
- PCIe TLP (Transaction Layer Packet) 头部开销 (~5-10%)
- CUDA runtime 调度开销
- 内存控制器排队延迟
- 实际可达的 ~73% 已经是非常好的利用率

---

## 4. 实验二：Pipeline 场景模拟

模拟真实 CPU Prefix Cache → GlobalKVPool 传输场景，变量为 seq_len 和 prefix_ratio。

### 数据（单层传输时间）

| seq_len | prefix_ratio | prefix_tokens | data_MB | pageable (ms) | pinned (ms) | 加速比 | pinned BW |
|---------|-------------|---------------|---------|---------------|-------------|--------|-----------|
| 4,096 | 50% | 2,048 | 4.0 | 0.415 | 0.181 | 2.29x | 21.5 GB/s |
| 4,096 | 80% | 3,276 | 6.4 | 0.571 | 0.282 | 2.02x | 22.1 GB/s |
| 4,096 | 100% | 4,096 | 8.0 | 0.677 | 0.347 | 1.95x | 22.5 GB/s |
| 16,384 | 50% | 8,192 | 16.0 | 1.199 | 0.683 | 1.76x | 22.9 GB/s |
| 16,384 | 80% | 13,107 | 25.6 | 1.836 | 1.088 | 1.69x | 23.0 GB/s |
| 16,384 | 100% | 16,384 | 32.0 | 2.360 | 1.359 | 1.74x | 23.0 GB/s |
| 32,768 | 50% | 16,384 | 32.0 | 2.368 | 1.365 | 1.74x | 22.9 GB/s |
| 32,768 | 80% | 26,214 | 51.2 | 4.337 | 2.176 | 1.99x | 23.0 GB/s |
| 32,768 | 100% | 32,768 | 64.0 | 6.696 | 2.739 | 2.44x | 22.8 GB/s |

### 28 层累计传输时间

| seq_len | prefix_ratio | 28层 pageable (ms) | 28层 pinned (ms) | 节省 (ms) |
|---------|-------------|--------------------|--------------------|-----------|
| 4,096 | 80% | 16.0 | 7.9 | **8.1** |
| 16,384 | 80% | 51.4 | 30.5 | **20.9** |
| 32,768 | 50% | 66.3 | 38.2 | **28.1** |
| 32,768 | 80% | 121.4 | 60.9 | **60.5** |
| 32,768 | 100% | 187.5 | 76.7 | **110.8** |

### 关键结论

- **Pinned memory 带宽恒定在 22-23 GB/s**，不受 prefix_ratio 和 seq_len 影响
- **Pageable 带宽波动较大**（9.7-14 GB/s），可能受 OS 页面管理影响
- **32K pfx=80% 场景**：单层节省 2.16ms，28 层累计节省 60.5ms
- **compression_ratio 不影响传输时间**：压缩发生在传输之后，传输量仅由 prefix_len 决定

---

## 5. 实验三：内存分配开销

| tokens | size_MB | pageable 分配 (ms) | pinned 分配 (ms) | 比值 |
|--------|---------|--------------------|--------------------|------|
| 4,096 | 8.0 | 107.8 | 200.5 | 1.86x |
| 8,192 | 16.0 | 212.7 | 208.6 | 0.98x |
| 16,384 | 32.0 | 426.5 | 433.6 | 1.02x |
| 32,768 | 64.0 | 853.6 | 845.6 | 0.99x |
| 65,536 | 128.0 | 1705.1 | 1671.0 | 0.98x |

**结论**: 大数据量时 pinned 分配开销与 pageable 几乎相同（被随机数生成主导）。小数据量时 pinned 稍慢（OS 锁页开销），但对于 cache 场景（分配一次，传输多次）可忽略。

---

## 6. 综合结论

### Pinned Memory 能达到 PCIe 理论上限的多少？

| 指标 | 值 |
|------|-----|
| PCIe Gen4 x16 理论带宽 | 31.5 GB/s |
| Pinned memory 实测带宽 | 22-23 GB/s |
| **达到理论上限的比例** | **70-73%** |
| Pageable memory 实测带宽 | 9.7-14 GB/s |
| Pageable 达到理论上限的比例 | 31-44% |

### 剩余 27% 带宽损失来源

1. **PCIe 协议开销** (~5%): TLP header、ACK/NAK、flow control
2. **CUDA Runtime 开销** (~5-8%): kernel launch、stream synchronization
3. **内存控制器延迟** (~5%): GPU 端 GDDR/HBM 写入排队
4. **DMA 引擎调度** (~5-10%): 多个 DMA 请求的仲裁

### 对 Pipeline 的影响

以 32K seq_len, pfx=80% 为例（当前 pipeline 总时间 ~11ms from REPORT_SNAPKV_V2）：

| 阶段 | pageable (ms) | pinned (ms) | 节省 |
|------|---------------|-------------|------|
| CPU transfer (单层) | 4.73 | 2.18 | 2.55ms (54%) |
| FA (extend) | 3.92 | 3.92 | 0 |
| SnapKV compress | 1.43 | 1.43 | 0 |
| 其他 | 0.78 | 0.78 | 0 |
| **总计** | **10.86** | **8.31** | **2.55ms (23%)** |

---

## 7. 实施建议

### 修改点

1. **`prefix_cpu_cache.py`**: 存储 KV 时使用 `pin_memory=True`
   ```python
   # Before
   k_cpu = k_gpu.cpu()
   # After
   k_cpu = torch.empty_like(k_gpu, device='cpu', pin_memory=True)
   k_cpu.copy_(k_gpu)
   ```

2. **`compressed_flashattention_backend.py`**: 传输时使用 `non_blocking=True`
   ```python
   # Before
   k_gpu = k_cpu.to(device)
   # After
   k_gpu = k_cpu.to(device, non_blocking=True)
   ```

### Pinned Memory 预算

| 配置 | 单 entry 大小 | 建议 entries | 总 pinned 内存 |
|------|--------------|-------------|---------------|
| 4K tokens | 4096×4×128×2×2B×28层 = 225MB | 8 | 1.8 GB |
| 16K tokens | 16384×4×128×2×2B×28层 = 900MB | 4 | 3.6 GB |
| 32K tokens | 32768×4×128×2×2B×28层 = 1.75GB | 4 | 7.0 GB |

建议设置可配置的 pinned memory 上限（默认 8GB），超出时 fallback 到 pageable。
