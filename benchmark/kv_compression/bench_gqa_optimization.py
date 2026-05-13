"""
Benchmark: GQA Expand Elimination in SnapKV Compression

Compares:
  OLD: repeat_interleave K from [kv_heads, prefix_len, dim] → [q_heads, prefix_len, dim], then matmul
  NEW: reshape Q from [q_heads, window, dim] → [kv_heads, group*window, dim], then bmm (no K expand)

Measures:
  1. Latency (ms) per compression call
  2. Peak GPU memory (MB) during compression
  3. Memory saved by not expanding K

Usage:
    python benchmark/kv_compression/bench_gqa_optimization.py
    python benchmark/kv_compression/bench_gqa_optimization.py --model qwen  # 4 kv_heads, 28 q_heads
    python benchmark/kv_compression/bench_gqa_optimization.py --model mistral  # 8 kv_heads, 32 q_heads
"""

import argparse
import math
import torch
import torch.nn.functional as F


def snapkv_old_repeat_interleave(
    q_window, k_prefix, k_window, num_kv_heads, kv_group_num, head_dim, window_size, kernel_size=5
):
    """Original SnapKV with repeat_interleave (OLD)."""
    q_t = q_window.transpose(0, 1).contiguous()  # [q_heads, window, dim]
    k_t = k_prefix.transpose(0, 1).contiguous()  # [kv_heads, prefix_len, dim]

    # GQA expand: K from kv_heads → q_heads
    k_t_expanded = k_t.repeat_interleave(kv_group_num, dim=0)  # [q_heads, prefix_len, dim]
    attn_weights = torch.matmul(q_t, k_t_expanded.transpose(-2, -1)) / math.sqrt(head_dim)

    # Window attention with causal mask
    k_window_t = k_window.transpose(0, 1).contiguous()
    k_window_t_expanded = k_window_t.repeat_interleave(kv_group_num, dim=0)
    attn_weights_window = torch.matmul(q_t, k_window_t_expanded.transpose(-2, -1)) / math.sqrt(head_dim)

    q_window_len = q_window.shape[0]
    k_window_len = k_window.shape[0]
    causal_mask = torch.triu(
        torch.full((q_window_len, k_window_len), float('-inf'), device=q_window.device, dtype=attn_weights_window.dtype),
        diagonal=1 + k_window_len - q_window_len
    )
    attn_weights_window = attn_weights_window + causal_mask.unsqueeze(0)

    attn_weights_full = torch.cat([attn_weights, attn_weights_window], dim=-1)
    attn_weights_prefix = attn_weights_full[:, :, :k_prefix.shape[0]]

    # Sum and reduce to kv_heads
    attn_weights_sum = attn_weights_prefix.sum(dim=1)  # [q_heads, prefix_len]
    attn_weights_sum = attn_weights_sum.view(num_kv_heads, kv_group_num, -1).sum(dim=1)  # [kv_heads, prefix_len]

    # Pooling + topk
    attn_cache = F.max_pool1d(attn_weights_sum.unsqueeze(0), kernel_size=kernel_size, padding=kernel_size//2, stride=1).squeeze(0)
    return attn_cache


def snapkv_new_grouped_bmm(
    q_window, k_prefix, k_window, num_kv_heads, kv_group_num, head_dim, window_size, kernel_size=5
):
    """Optimized SnapKV with grouped bmm (NEW, no repeat_interleave)."""
    q_t = q_window.transpose(0, 1).contiguous()  # [q_heads, window, dim]
    k_t = k_prefix.transpose(0, 1).contiguous()  # [kv_heads, prefix_len, dim]

    # Reshape Q: [q_heads, window, dim] → [kv_heads, group*window, dim]
    q_grouped = q_t.view(num_kv_heads, kv_group_num * window_size, head_dim)
    attn_weights = torch.bmm(q_grouped, k_t.transpose(-2, -1)) / math.sqrt(head_dim)

    # Window attention
    k_window_t = k_window.transpose(0, 1).contiguous()
    attn_weights_window = torch.bmm(q_grouped, k_window_t.transpose(-2, -1)) / math.sqrt(head_dim)

    q_window_len = q_window.shape[0]
    k_window_len = k_window.shape[0]
    causal_mask_base = torch.triu(
        torch.full((q_window_len, k_window_len), float('-inf'), device=q_window.device, dtype=attn_weights_window.dtype),
        diagonal=1 + k_window_len - q_window_len
    )
    causal_mask = causal_mask_base.repeat(kv_group_num, 1)
    attn_weights_window = attn_weights_window + causal_mask.unsqueeze(0)

    attn_weights_full = torch.cat([attn_weights, attn_weights_window], dim=-1)
    attn_weights_prefix = attn_weights_full[:, :, :k_prefix.shape[0]]

    # Sum over group*window → [kv_heads, prefix_len]
    attn_weights_sum = attn_weights_prefix.sum(dim=1)

    # Pooling + topk
    attn_cache = F.max_pool1d(attn_weights_sum.unsqueeze(0), kernel_size=kernel_size, padding=kernel_size//2, stride=1).squeeze(0)
    return attn_cache


def measure_peak_memory_and_latency(fn, *args, warmup=5, repeat=20):
    """Measure peak GPU memory and latency for a function."""
    # Warmup
    for _ in range(warmup):
        _ = fn(*args)
    torch.cuda.synchronize()

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated()

    # Measure latency
    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    mem_peak = torch.cuda.max_memory_allocated()
    mem_used = mem_peak - mem_before

    avg_ms = sum(times) / len(times)
    return avg_ms, mem_used / (1024**2)  # ms, MB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mistral", "qwen"], default="mistral")
    parser.add_argument("--seq-lens", default="4096,8192,16384,31424")
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()

    if args.model == "mistral":
        num_kv_heads, num_q_heads, head_dim, num_layers = 8, 32, 128, 32
    else:
        num_kv_heads, num_q_heads, head_dim, num_layers = 4, 28, 128, 28

    kv_group_num = num_q_heads // num_kv_heads
    window_size = args.window_size
    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    print("=" * 100)
    print(f"GQA Optimization Benchmark: {args.model.upper()}")
    print("=" * 100)
    print(f"  num_kv_heads={num_kv_heads}, num_q_heads={num_q_heads}, "
          f"kv_group_num={kv_group_num}, head_dim={head_dim}, window={window_size}")
    print(f"  OLD: repeat_interleave K [{num_kv_heads}→{num_q_heads}] + matmul")
    print(f"  NEW: reshape Q [{num_q_heads}→{num_kv_heads}×{kv_group_num}] + bmm (no K expand)")
    print()

    print(f"{'seq_len':>8} {'prefix':>8} "
          f"{'OLD_ms':>8} {'NEW_ms':>8} {'speedup':>8} "
          f"{'OLD_mem':>10} {'NEW_mem':>10} {'mem_saved':>10} "
          f"{'K_expand_MB':>12}")
    print("─" * 100)

    for seq_len in seq_lens:
        prefix_len = seq_len - window_size
        if prefix_len <= 0:
            continue

        # Create input tensors
        q_window = torch.randn(window_size, num_q_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        k_full = torch.randn(seq_len, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        k_prefix = k_full[:-window_size]
        k_window = k_full[-window_size:]

        # Theoretical K expand memory
        k_expand_mb = num_q_heads * prefix_len * head_dim * 2 / (1024**2)

        # Measure OLD
        torch.cuda.empty_cache()
        old_ms, old_mem = measure_peak_memory_and_latency(
            snapkv_old_repeat_interleave,
            q_window, k_prefix, k_window, num_kv_heads, kv_group_num, head_dim, window_size,
            warmup=args.warmup, repeat=args.repeat,
        )

        # Measure NEW
        torch.cuda.empty_cache()
        new_ms, new_mem = measure_peak_memory_and_latency(
            snapkv_new_grouped_bmm,
            q_window, k_prefix, k_window, num_kv_heads, kv_group_num, head_dim, window_size,
            warmup=args.warmup, repeat=args.repeat,
        )

        speedup = old_ms / new_ms if new_ms > 0 else 0
        mem_saved = old_mem - new_mem

        print(f"{seq_len:>8} {prefix_len:>8} "
              f"{old_ms:>7.3f} {new_ms:>7.3f} {speedup:>7.2f}x "
              f"{old_mem:>8.1f}MB {new_mem:>8.1f}MB {mem_saved:>8.1f}MB "
              f"{k_expand_mb:>10.1f}MB")

        del q_window, k_full, k_prefix, k_window
        torch.cuda.empty_cache()

    # Summary
    print()
    print("=" * 100)
    print("Analysis")
    print("=" * 100)
    print(f"""
Memory savings explanation:
  OLD allocates: K_expanded = [{num_q_heads}, prefix_len, {head_dim}] × bf16
                 + K_window_expanded = [{num_q_heads}, {window_size}, {head_dim}] × bf16
                 + attn_weights = [{num_q_heads}, {window_size}, prefix_len+{window_size}] × bf16

  NEW allocates: q_grouped = [{num_kv_heads}, {kv_group_num}×{window_size}, {head_dim}] × bf16 (just a view, no alloc)
                 + attn_weights = [{num_kv_heads}, {kv_group_num}×{window_size}, prefix_len+{window_size}] × bf16

  Key difference: 
    - OLD: K_expanded is {kv_group_num}x larger than K (the biggest allocation)
    - NEW: No K expansion at all. Q reshape is a view (zero-copy).
    - attn_weights tensor is same size in both (same total elements, just reshaped)

Latency savings explanation:
  - Eliminates repeat_interleave (memory allocation + copy)
  - bmm with fewer batch dims can be more efficient than matmul with larger batch
  - Reduced memory pressure → better cache utilization
""")


if __name__ == "__main__":
    main()
