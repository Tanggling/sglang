"""
Comprehensive Pinned Memory Benchmark for CPU Prefix Cache → GlobalKVPool Transfer

System: NVIDIA A800-SXM4-80GB, PCIe Gen4 x16
  - Theoretical unidirectional bandwidth: ~31.5 GB/s (GT/s × lanes × encoding)
  - Practical peak with pinned memory: ~25-26 GB/s
  - Pageable memory (with extra copy): ~8-12 GB/s

Scenario:
  CPU Prefix Cache stores full KV for prefix tokens (per layer).
  On cache hit, we transfer prefix KV from CPU → GPU GlobalKVPool.
  Transfer size per layer = prefix_len × num_kv_heads × head_dim × 2 (K+V) × sizeof(bf16)

Experiment dimensions:
  1. seq_len: [4096, 8192, 16384, 32768, 65536]
  2. prefix_ratio (match ratio): [0.5, 0.8, 0.9, 1.0]
     → prefix_len = seq_len × prefix_ratio
  3. compression_ratio: [0.3, 0.5, 0.7]
     → After transfer, only compressed_len tokens are kept in real pool
     → But transfer always moves full prefix_len (compression happens after)
  4. Memory mode: pageable vs pinned (+ non_blocking)

Key insight: Transfer size is determined by prefix_len (not compression_ratio),
because compression happens AFTER the transfer to GPU.

Usage:
    python benchmark/kv_compression/bench_pinned_memory_transfer.py
    python benchmark/kv_compression/bench_pinned_memory_transfer.py --seq-lens 32768,65536 --num-layers 28
"""

import argparse
import torch
import time
import sys


# ═══════════════════════════════════════════════════════════════════════════════
# PCIe Bandwidth Theory
# ═══════════════════════════════════════════════════════════════════════════════

PCIE_GEN4_X16_THEORETICAL_GBS = 31.508  # GB/s unidirectional (16 GT/s × 16 lanes × 128b/130b)
PCIE_GEN4_X16_PRACTICAL_GBS = 25.0      # Typical achievable with pinned memory


def print_pcie_theory():
    """Print PCIe bandwidth theory for reference."""
    print("=" * 90)
    print("PCIe Bandwidth Theory (Gen4 x16)")
    print("=" * 90)
    print(f"  Transfer rate:     16 GT/s per lane")
    print(f"  Lanes:             16")
    print(f"  Encoding:          128b/130b")
    print(f"  Theoretical BW:    {PCIE_GEN4_X16_THEORETICAL_GBS:.1f} GB/s (unidirectional)")
    print(f"  Practical peak:    ~{PCIE_GEN4_X16_PRACTICAL_GBS:.0f} GB/s (pinned, large transfers)")
    print(f"  Pageable typical:  ~8-12 GB/s (extra memcpy host→pinned staging)")
    print()
    print("  Why pinned memory is faster:")
    print("    - Pageable: CPU page → pinned staging buffer → DMA → GPU (2 copies)")
    print("    - Pinned:   pinned buffer → DMA → GPU (1 copy, DMA-capable)")
    print("    - Non-blocking: GPU DMA engine works independently of CPU")
    print("=" * 90)


# ═══════════════════════════════════════════════════════════════════════════════
# Transfer Benchmark Core
# ═══════════════════════════════════════════════════════════════════════════════

def bench_transfer(
    num_tokens: int,
    num_kv_heads: int,
    head_dim: int,
    dtype=torch.bfloat16,
    pinned: bool = False,
    non_blocking: bool = False,
    warmup: int = 5,
    repeat: int = 20,
) -> float:
    """
    Benchmark CPU→GPU transfer of K+V tensors.
    Returns average transfer time in ms.
    """
    shape = (num_tokens, num_kv_heads, head_dim)

    if pinned:
        k_cpu = torch.empty(shape, dtype=dtype, pin_memory=True)
        v_cpu = torch.empty(shape, dtype=dtype, pin_memory=True)
        k_cpu.normal_()
        v_cpu.normal_()
    else:
        k_cpu = torch.randn(shape, dtype=dtype)
        v_cpu = torch.randn(shape, dtype=dtype)

    times = []
    for i in range(warmup + repeat):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        k_gpu = k_cpu.to("cuda", non_blocking=non_blocking)
        v_gpu = v_cpu.to("cuda", non_blocking=non_blocking)

        end.record()
        torch.cuda.synchronize()

        if i >= warmup:
            times.append(start.elapsed_time(end))
        del k_gpu, v_gpu

    return sum(times) / len(times)


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 1: Raw Transfer Bandwidth (varying data size)
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_raw_bandwidth(num_kv_heads=4, head_dim=128):
    """Measure raw CPU→GPU bandwidth for different transfer sizes."""
    print("\n" + "=" * 90)
    print("Experiment 1: Raw CPU→GPU Transfer Bandwidth")
    print("=" * 90)
    print(f"  Model config: num_kv_heads={num_kv_heads}, head_dim={head_dim}, dtype=bf16")
    print(f"  Per-token data: K+V = 2 × {num_kv_heads} × {head_dim} × 2B = {2*num_kv_heads*head_dim*2} bytes")
    print(f"  PCIe Gen4 x16 theoretical: {PCIE_GEN4_X16_THEORETICAL_GBS:.1f} GB/s")
    print()

    token_counts = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

    modes = [
        ("pageable",            False, False),
        ("pageable+nb",         False, True),
        ("pinned",              True,  False),
        ("pinned+nb",           True,  True),
    ]

    # Header
    print(f"{'tokens':>8} {'data_MB':>8}", end="")
    for name, _, _ in modes:
        print(f" {'ms':>8} {'GB/s':>7}", end="")
    print(f" {'speedup':>8} {'%_theory':>9}")
    print(f"{'':>8} {'':>8}", end="")
    for name, _, _ in modes:
        print(f" {name:>16}", end="")
    print(f" {'pin/page':>8} {'pin_nb':>9}")
    print("─" * 120)

    for n_tok in token_counts:
        data_bytes = n_tok * 2 * num_kv_heads * head_dim * 2  # K+V, bf16
        data_mb = data_bytes / (1024**2)
        data_gb = data_bytes / (1024**3)

        results = {}
        for name, pinned, nb in modes:
            ms = bench_transfer(n_tok, num_kv_heads, head_dim, pinned=pinned, non_blocking=nb)
            bw = data_gb / (ms / 1000) if ms > 0 else 0
            results[name] = (ms, bw)

        print(f"{n_tok:>8} {data_mb:>7.1f}", end="")
        for name, _, _ in modes:
            ms, bw = results[name]
            print(f" {ms:>7.3f} {bw:>6.1f}", end="")

        page_ms = results["pageable"][0]
        pin_nb_ms = results["pinned+nb"][0]
        pin_nb_bw = results["pinned+nb"][1]
        speedup = page_ms / pin_nb_ms if pin_nb_ms > 0 else 0
        pct_theory = pin_nb_bw / PCIE_GEN4_X16_THEORETICAL_GBS * 100

        print(f" {speedup:>7.2f}x {pct_theory:>7.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 2: Pipeline Simulation (seq_len × prefix_ratio × compression_ratio)
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_pipeline_simulation(
    seq_lens=(4096, 8192, 16384, 32768, 65536),
    prefix_ratios=(0.5, 0.8, 0.9, 1.0),
    compression_ratios=(0.3, 0.5, 0.7),
    num_kv_heads=4,
    head_dim=128,
    num_layers=28,
):
    """
    Simulate the full CPU prefix → GPU transfer in the pipeline context.

    For each (seq_len, prefix_ratio, compression_ratio):
      - prefix_len = seq_len × prefix_ratio (tokens transferred from CPU)
      - compressed_len = seq_len × (1 - compression_ratio) (tokens kept after compress)
      - Transfer happens for ALL prefix_len tokens (compression is post-transfer)
      - We measure: pageable time, pinned time, time saved
      - We also compute: what fraction of total pipeline time is transfer
    """
    print("\n" + "=" * 90)
    print("Experiment 2: Pipeline Transfer Simulation")
    print("=" * 90)
    print(f"  Scenario: CPU Prefix Cache → GlobalKVPool (per layer)")
    print(f"  Transfer size = prefix_len × num_kv_heads × head_dim × 2 (K+V) × 2B")
    print(f"  Note: compression_ratio affects final kept tokens, NOT transfer size")
    print(f"  num_layers={num_layers} (transfer happens once per layer)")
    print()

    print(f"{'seq_len':>8} {'pfx%':>5} {'pfx_tok':>8} {'xfer_MB':>8} "
          f"{'page(ms)':>9} {'pin(ms)':>8} {'speedup':>8} {'pin_BW':>8} "
          f"{'all_layers_page':>16} {'all_layers_pin':>15} {'saved_ms':>10}")
    print("─" * 130)

    results_all = []

    for seq_len in seq_lens:
        for pfx_ratio in prefix_ratios:
            prefix_len = int(seq_len * pfx_ratio)
            if prefix_len == 0:
                continue

            # Transfer size per layer
            data_bytes = prefix_len * 2 * num_kv_heads * head_dim * 2
            data_mb = data_bytes / (1024**2)
            data_gb = data_bytes / (1024**3)

            # Benchmark single layer transfer
            t_page = bench_transfer(prefix_len, num_kv_heads, head_dim, pinned=False, non_blocking=False)
            t_pin = bench_transfer(prefix_len, num_kv_heads, head_dim, pinned=True, non_blocking=True)

            speedup = t_page / t_pin if t_pin > 0 else 0
            pin_bw = data_gb / (t_pin / 1000) if t_pin > 0 else 0

            # All layers (in current impl, transfer is per-layer sequential)
            all_page = t_page * num_layers
            all_pin = t_pin * num_layers
            saved = all_page - all_pin

            print(f"{seq_len:>8} {pfx_ratio:>4.0%} {prefix_len:>8} {data_mb:>7.1f} "
                  f"{t_page:>8.3f} {t_pin:>7.3f} {speedup:>7.2f}x {pin_bw:>6.1f} "
                  f"{all_page:>14.1f}ms {all_pin:>13.1f}ms {saved:>8.1f}ms")

            results_all.append({
                "seq_len": seq_len, "prefix_ratio": pfx_ratio,
                "prefix_len": prefix_len, "data_mb": data_mb,
                "pageable_ms": t_page, "pinned_ms": t_pin,
                "speedup": speedup, "pin_bw_gbs": pin_bw,
                "all_layers_page_ms": all_page, "all_layers_pin_ms": all_pin,
            })

        print()  # separator between seq_lens

    return results_all


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 3: End-to-End Pipeline Impact
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_pipeline_impact(
    seq_lens=(8192, 16384, 32768),
    prefix_ratios=(0.8, 0.9, 1.0),
    compression_ratios=(0.3, 0.5, 0.7),
    num_kv_heads=4,
    head_dim=128,
):
    """
    Estimate end-to-end pipeline impact of pinned memory.

    Uses measured transfer times + estimated FA/compress times from REPORT_SNAPKV_V2.md
    to show what percentage of total pipeline time is saved.
    """
    print("\n" + "=" * 90)
    print("Experiment 3: End-to-End Pipeline Impact Estimation")
    print("=" * 90)
    print("  Using measured transfer times + estimated FA/compress from report data")
    print("  FA time ≈ f(extend_len), compress time ≈ f(seq_len)")
    print()

    # Approximate FA time based on extend_len (from report data, ratio=0.5)
    # These are rough fits from the report
    def estimate_fa_ms(extend_len):
        if extend_len <= 0:
            return 0.2  # minimal overhead
        # Roughly quadratic: FA ≈ 0.62ms at 2K, 1.84ms at 4K, 6.15ms at 8K, 21ms at 16K
        return 0.62 * (extend_len / 2048) ** 1.7

    # Approximate compress time based on seq_len
    def estimate_compress_ms(seq_len):
        # From report: 0.46ms at 4K, 0.61ms at 8K, 1.13ms at 16K, 1.45ms at 32K
        return 0.46 * (seq_len / 4096) ** 0.7

    print(f"{'seq_len':>8} {'pfx%':>5} {'ratio':>6} "
          f"{'xfer_page':>10} {'xfer_pin':>9} {'FA':>6} {'comp':>6} "
          f"{'total_page':>11} {'total_pin':>10} {'saved%':>7}")
    print("─" * 100)

    for seq_len in seq_lens:
        for pfx_ratio in prefix_ratios:
            prefix_len = int(seq_len * pfx_ratio)
            extend_len = seq_len - prefix_len

            # Measure transfer
            t_page = bench_transfer(prefix_len, num_kv_heads, head_dim, pinned=False, non_blocking=False)
            t_pin = bench_transfer(prefix_len, num_kv_heads, head_dim, pinned=True, non_blocking=True)

            fa_ms = estimate_fa_ms(extend_len)
            comp_ms = estimate_compress_ms(seq_len)

            for comp_ratio in compression_ratios:
                # compression_ratio doesn't affect transfer or FA time significantly
                # It slightly affects write time but that's negligible
                total_page = t_page + fa_ms + comp_ms
                total_pin = t_pin + fa_ms + comp_ms
                saved_pct = (total_page - total_pin) / total_page * 100

                print(f"{seq_len:>8} {pfx_ratio:>4.0%} {comp_ratio:>5.1f} "
                      f"{t_page:>9.2f}ms {t_pin:>8.2f}ms {fa_ms:>5.2f} {comp_ms:>5.2f} "
                      f"{total_page:>9.2f}ms {total_pin:>8.2f}ms {saved_pct:>6.1f}%")

            print()
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 4: Pinned Memory Allocation Overhead
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_alloc_overhead(num_kv_heads=4, head_dim=128):
    """
    Measure the overhead of allocating pinned memory vs pageable.
    This is important because PrefixCPUCache allocates memory when saving entries.
    """
    print("\n" + "=" * 90)
    print("Experiment 4: Pinned vs Pageable Memory Allocation Overhead")
    print("=" * 90)
    print("  Pinned memory allocation is slower (OS must pin pages)")
    print("  But for a cache, we allocate once and reuse many times")
    print()

    token_counts = [4096, 8192, 16384, 32768, 65536]

    print(f"{'tokens':>8} {'size_MB':>8} {'pageable_alloc(ms)':>20} {'pinned_alloc(ms)':>18} {'ratio':>7}")
    print("─" * 70)

    for n_tok in token_counts:
        shape = (n_tok, num_kv_heads, head_dim)
        size_mb = n_tok * num_kv_heads * head_dim * 2 * 2 / (1024**2)  # K+V bf16

        # Pageable allocation
        times_page = []
        for _ in range(10):
            t0 = time.perf_counter()
            k = torch.randn(shape, dtype=torch.bfloat16)
            v = torch.randn(shape, dtype=torch.bfloat16)
            t1 = time.perf_counter()
            times_page.append((t1 - t0) * 1000)
            del k, v

        # Pinned allocation
        times_pin = []
        for _ in range(10):
            t0 = time.perf_counter()
            k = torch.empty(shape, dtype=torch.bfloat16, pin_memory=True)
            v = torch.empty(shape, dtype=torch.bfloat16, pin_memory=True)
            k.normal_()
            v.normal_()
            t1 = time.perf_counter()
            times_pin.append((t1 - t0) * 1000)
            del k, v

        avg_page = sum(times_page) / len(times_page)
        avg_pin = sum(times_pin) / len(times_pin)
        ratio = avg_pin / avg_page if avg_page > 0 else 0

        print(f"{n_tok:>8} {size_mb:>7.1f} {avg_page:>18.2f}ms {avg_pin:>16.2f}ms {ratio:>6.2f}x")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Pinned Memory Transfer Benchmark")
    parser.add_argument("--seq-lens", default="4096,8192,16384,32768,65536")
    parser.add_argument("--prefix-ratios", default="0.5,0.8,0.9,1.0")
    parser.add_argument("--compression-ratios", default="0.3,0.5,0.7")
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--exp", type=int, default=0, help="Run specific experiment (1-4), 0=all")
    args = parser.parse_args()

    seq_lens = tuple(int(x) for x in args.seq_lens.split(","))
    prefix_ratios = tuple(float(x) for x in args.prefix_ratios.split(","))
    compression_ratios = tuple(float(x) for x in args.compression_ratios.split(","))

    print_pcie_theory()

    if args.exp == 0 or args.exp == 1:
        experiment_raw_bandwidth(num_kv_heads=args.num_kv_heads, head_dim=args.head_dim)

    if args.exp == 0 or args.exp == 2:
        experiment_pipeline_simulation(
            seq_lens=seq_lens, prefix_ratios=prefix_ratios,
            compression_ratios=compression_ratios,
            num_kv_heads=args.num_kv_heads, head_dim=args.head_dim,
            num_layers=args.num_layers,
        )

    if args.exp == 0 or args.exp == 3:
        experiment_pipeline_impact(
            seq_lens=tuple(s for s in seq_lens if s >= 8192),
            prefix_ratios=prefix_ratios,
            compression_ratios=compression_ratios,
            num_kv_heads=args.num_kv_heads, head_dim=args.head_dim,
        )

    if args.exp == 0 or args.exp == 4:
        experiment_alloc_overhead(num_kv_heads=args.num_kv_heads, head_dim=args.head_dim)

    # Summary
    print("\n" + "=" * 90)
    print("Implementation Recommendations")
    print("=" * 90)
    print("""
1. PrefixCPUCache should store KV in pinned memory:
   - Change: torch.empty(...) → torch.empty(..., pin_memory=True)
   - Location: prefix_cpu_cache.py, in save/accumulate methods
   - Use non_blocking=True when calling .to('cuda')

2. Memory budget for pinned memory:
   - Per entry: seq_len × num_kv_heads × head_dim × 2 (K+V) × 2B × num_layers
   - Example: 32K tokens × 4 heads × 128 dim × 2 × 2B × 28 layers = 1.75 GB per entry
   - Recommend: Cap at 4-8 entries (7-14 GB pinned), configurable

3. Transfer optimization in compressed_flashattention_backend.py:
   - In _forward_extend_global_real_split, load_layer_to_gpu should use non_blocking=True
   - The .to('cuda', non_blocking=True) only works with pinned source tensors

4. Expected speedup:
   - 32K, pfx=80%: transfer from ~5ms → ~2.5ms per layer (2x speedup)
   - Total pipeline: ~11ms → ~8.5ms (23% reduction)
   - Higher prefix ratios benefit more (transfer is larger fraction of total)
""")


if __name__ == "__main__":
    main()
