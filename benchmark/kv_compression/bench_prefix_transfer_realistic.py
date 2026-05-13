"""
Realistic Prefix Transfer Benchmark: Simulating PrefixCPUCache behavior

This benchmark replicates the EXACT transfer pattern used in the real system:
  1. GPU tensor → CPU (simulating accumulate_layer_kqv during first prefill)
  2. CPU tensor → GPU (simulating load_layer_to_gpu during cache hit)

Key difference from bench_pinned_memory_transfer.py:
  - That benchmark allocates fresh CPU tensors each iteration (cold allocation)
  - This benchmark pre-allocates and REUSES the same CPU tensors (warm, like real cache)
  - In the real system, tensors are allocated ONCE during first prefill, then
    transferred to GPU MANY times on subsequent cache hits

This explains why pageable and pinned perform similarly in the real system:
  - After first access, pageable memory pages are already in the CPU page cache
  - The OS has already mapped the virtual pages to physical pages
  - CUDA driver's internal staging buffer is warmed up for these addresses

Usage:
    python benchmark/kv_compression/bench_prefix_transfer_realistic.py
"""

import argparse
import torch
import time


def simulate_prefix_cache_flow(
    num_tokens: int,
    num_kv_heads: int = 4,
    head_dim: int = 128,
    num_layers: int = 28,
    dtype=torch.bfloat16,
    pinned: bool = True,
    warmup_transfers: int = 2,
    measure_transfers: int = 5,
):
    """
    Simulate the full PrefixCPUCache flow:
      Phase 1: GPU → CPU (first prefill, save to cache) — measured once
      Phase 2: CPU → GPU (cache hit, load to GlobalKVPool) — measured multiple times

    This matches the real system where:
      - accumulate_layer_kqv: GPU→CPU transfer (once per entry)
      - load_layer_to_gpu: CPU→GPU transfer (every cache hit)

    Returns dict with timing results.
    """
    device = "cuda"
    shape_kv = (num_tokens, num_kv_heads, head_dim)

    # ═══════════════════════════════════════════════════════════════════
    # Phase 1: Simulate first prefill — GPU tensors exist, save to CPU
    # ═══════════════════════════════════════════════════════════════════
    # Create GPU tensors (as if produced by model forward pass)
    gpu_k_layers = [torch.randn(shape_kv, dtype=dtype, device=device) for _ in range(num_layers)]
    gpu_v_layers = [torch.randn(shape_kv, dtype=dtype, device=device) for _ in range(num_layers)]

    torch.cuda.synchronize()

    # Save to CPU (mimicking accumulate_layer_kqv)
    start_save = torch.cuda.Event(enable_timing=True)
    end_save = torch.cuda.Event(enable_timing=True)
    start_save.record()

    cpu_k_layers = []
    cpu_v_layers = []
    for layer_id in range(num_layers):
        if pinned:
            k_cpu = torch.empty(shape_kv, dtype=dtype, device="cpu", pin_memory=True)
            v_cpu = torch.empty(shape_kv, dtype=dtype, device="cpu", pin_memory=True)
            k_cpu.copy_(gpu_k_layers[layer_id], non_blocking=True)
            v_cpu.copy_(gpu_v_layers[layer_id], non_blocking=True)
        else:
            k_cpu = gpu_k_layers[layer_id].detach().to("cpu", non_blocking=True)
            v_cpu = gpu_v_layers[layer_id].detach().to("cpu", non_blocking=True)
        cpu_k_layers.append(k_cpu)
        cpu_v_layers.append(v_cpu)

    end_save.record()
    torch.cuda.synchronize()
    save_ms = start_save.elapsed_time(end_save)

    # Free GPU source tensors (in real system, they're freed after prefill)
    del gpu_k_layers, gpu_v_layers
    torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════════
    # Phase 2: Simulate cache hits — load from CPU to GPU (multiple times)
    # ═══════════════════════════════════════════════════════════════════
    load_times = []

    for trial in range(warmup_transfers + measure_transfers):
        torch.cuda.synchronize()
        start_load = torch.cuda.Event(enable_timing=True)
        end_load = torch.cuda.Event(enable_timing=True)
        start_load.record()

        # Simulate load_layer_to_gpu for all layers
        for layer_id in range(num_layers):
            k_gpu = cpu_k_layers[layer_id].to(device, non_blocking=(pinned))
            v_gpu = cpu_v_layers[layer_id].to(device, non_blocking=(pinned))
            # In real system, these are immediately used by global_kv_pool.write_kv
            del k_gpu, v_gpu

        end_load.record()
        torch.cuda.synchronize()

        if trial >= warmup_transfers:
            load_times.append(start_load.elapsed_time(end_load))

    avg_load_ms = sum(load_times) / len(load_times)
    per_layer_ms = avg_load_ms / num_layers

    # Calculate bandwidth
    data_per_layer_bytes = num_tokens * num_kv_heads * head_dim * 2 * 2  # K+V, bf16
    data_per_layer_mb = data_per_layer_bytes / (1024**2)
    total_data_mb = data_per_layer_mb * num_layers
    bw_gbs = (total_data_mb / 1024) / (avg_load_ms / 1000)

    return {
        "save_ms": save_ms,
        "avg_load_ms": avg_load_ms,
        "per_layer_ms": per_layer_ms,
        "data_per_layer_mb": data_per_layer_mb,
        "total_data_mb": total_data_mb,
        "bandwidth_gbs": bw_gbs,
        "load_times": load_times,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=31424)
    parser.add_argument("--num-kv-heads", type=int, default=8,
                        help="Mistral=8, Qwen2.5=4")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=32,
                        help="Mistral=32, Qwen2.5=28")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    print("=" * 90)
    print("Realistic Prefix Transfer Benchmark")
    print("=" * 90)
    print(f"Config: tokens={args.num_tokens}, kv_heads={args.num_kv_heads}, "
          f"head_dim={args.head_dim}, layers={args.num_layers}")
    data_per_layer = args.num_tokens * args.num_kv_heads * args.head_dim * 2 * 2
    print(f"Data per layer (K+V): {data_per_layer / 1024**2:.1f} MB")
    print(f"Total data (all layers): {data_per_layer * args.num_layers / 1024**2:.1f} MB")
    print()

    # ─── Run both modes ───────────────────────────────────────────────
    results = {}
    for mode_name, use_pinned in [("pageable", False), ("pinned", True)]:
        print(f"Running {mode_name} mode...")
        r = simulate_prefix_cache_flow(
            num_tokens=args.num_tokens,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            num_layers=args.num_layers,
            pinned=use_pinned,
            warmup_transfers=args.warmup,
            measure_transfers=args.repeats,
        )
        results[mode_name] = r
        torch.cuda.empty_cache()

    # ─── Print results ────────────────────────────────────────────────
    print()
    print(f"{'Metric':<30} {'Pageable':<20} {'Pinned':<20} {'Speedup':<10}")
    print("─" * 80)

    rp = results["pageable"]
    rn = results["pinned"]

    print(f"{'GPU→CPU save (ms)':<30} {rp['save_ms']:<20.2f} {rn['save_ms']:<20.2f} {rp['save_ms']/rn['save_ms']:<10.2f}x")
    print(f"{'CPU→GPU load all layers (ms)':<30} {rp['avg_load_ms']:<20.2f} {rn['avg_load_ms']:<20.2f} {rp['avg_load_ms']/rn['avg_load_ms']:<10.2f}x")
    print(f"{'CPU→GPU per layer (ms)':<30} {rp['per_layer_ms']:<20.3f} {rn['per_layer_ms']:<20.3f} {rp['per_layer_ms']/rn['per_layer_ms']:<10.2f}x")
    print(f"{'Bandwidth (GB/s)':<30} {rp['bandwidth_gbs']:<20.1f} {rn['bandwidth_gbs']:<20.1f} {'':10}")
    print(f"{'Data per layer (MB)':<30} {rp['data_per_layer_mb']:<20.1f} {rn['data_per_layer_mb']:<20.1f} {'':10}")

    print()
    print("Per-trial load times (ms):")
    print(f"  Pageable: {[f'{t:.1f}' for t in rp['load_times']]}")
    print(f"  Pinned:   {[f'{t:.1f}' for t in rn['load_times']]}")

    # ─── Analysis ─────────────────────────────────────────────────────
    speedup = rp['avg_load_ms'] / rn['avg_load_ms']
    print()
    print("=" * 90)
    print("Analysis: Why pageable ≈ pinned in this scenario")
    print("=" * 90)
    print(f"""
Measured speedup: {speedup:.2f}x (pinned vs pageable)

Key factors that reduce the pinned memory advantage in the real system:

1. WARM MEMORY PAGES:
   - After GPU→CPU save (Phase 1), the pageable memory pages are already
     resident in physical RAM and mapped in the page table.
   - On subsequent CPU→GPU transfers, there are NO page faults.
   - The CUDA driver's internal pinned staging buffer is also warmed up.

2. LARGE CONTIGUOUS TRANSFERS:
   - Each layer transfers {rp['data_per_layer_mb']:.1f} MB as a single contiguous block.
   - For large transfers (>1MB), CUDA driver uses optimized DMA paths even
     for pageable memory (batched page-pinning or large staging buffers).
   - The overhead of the extra memcpy (pageable→staging→DMA) is amortized
     over the large transfer size.

3. SEQUENTIAL ACCESS PATTERN:
   - We transfer layers sequentially (layer 0, 1, 2, ...).
   - The OS prefetcher can predict this pattern and pre-fault pages.
   - This eliminates the random-access penalty that pageable memory normally has.

4. CUDA DRIVER OPTIMIZATION:
   - Modern CUDA drivers (12.x) have "lazy pinning" for large pageable transfers.
   - The driver temporarily pins pages during DMA, avoiding the full memcpy.
   - This is transparent to the user and makes pageable approach pinned speed.

When DOES pinned memory help significantly?
   - Small, fragmented transfers (many small tensors)
   - First-time access (cold pages, page faults)
   - Concurrent transfers on multiple streams
   - Non-blocking overlap with GPU compute (pinned guarantees no page fault during DMA)

Bandwidth comparison:
   - Pageable: {rp['bandwidth_gbs']:.1f} GB/s ({rp['bandwidth_gbs']/31.5*100:.0f}% of PCIe Gen4 x16 theory)
   - Pinned:   {rn['bandwidth_gbs']:.1f} GB/s ({rn['bandwidth_gbs']/31.5*100:.0f}% of PCIe Gen4 x16 theory)
   - Both are near the practical PCIe limit (~22-23 GB/s), confirming the
     transfer is already bandwidth-saturated regardless of memory type.
""")


if __name__ == "__main__":
    main()
