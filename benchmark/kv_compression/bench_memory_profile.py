"""Benchmark: Memory profiling across compression pipeline phases.

Tracks GPU memory allocation at each phase boundary.

Usage:
    python benchmark/kv_compression/bench_memory_profile.py \
        --seq-lens 4096,8192 --methods key_norm,snapkv
"""

import argparse
import math
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from benchmark.kv_compression.profiler.memory_tracker import MemoryTracker


def run_memory_profile(
    seq_len: int,
    method: str,
    compression_ratio: float,
    num_kv_heads: int = 8,
    num_heads: int = 32,
    head_dim: int = 128,
):
    dtype = torch.bfloat16
    device = "cuda"
    num_to_keep = max(1, int(seq_len * (1 - compression_ratio)))
    kv_group = num_heads // num_kv_heads
    pool_size = seq_len + 256

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    tracker = MemoryTracker()

    tracker.snapshot("baseline")

    # Allocate GlobalKVPool buffers (simulating 1 layer)
    gpool_k = torch.zeros(pool_size, num_kv_heads, head_dim, dtype=dtype, device=device)
    gpool_v = torch.zeros(pool_size, num_kv_heads, head_dim, dtype=dtype, device=device)
    tracker.snapshot("global_pool_alloc")

    # Simulate Phase 1: write KV to pool + read back
    k_data = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    v_data = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    slots = torch.arange(seq_len, device=device)
    gpool_k[slots] = k_data
    gpool_v[slots] = v_data
    k_full = gpool_k[slots].clone()
    v_full = gpool_v[slots].clone()
    tracker.snapshot("phase1_done")

    # Simulate Phase 2: attention
    q = torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device=device)
    tracker.snapshot("q_allocated")

    q_t = q.transpose(0, 1).unsqueeze(0)
    k_t = k_full.transpose(0, 1).unsqueeze(0)
    v_t = v_full.transpose(0, 1).unsqueeze(0)
    if kv_group > 1:
        k_t = k_t.repeat_interleave(kv_group, dim=1)
        v_t = v_t.repeat_interleave(kv_group, dim=1)
    tracker.snapshot("phase2_expanded")

    out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
    tracker.snapshot("phase2_done")
    del q_t, k_t, v_t, out
    torch.cuda.empty_cache()
    tracker.snapshot("phase2_cleanup")

    # Simulate Phase 3: compression
    if method == "snapkv":
        window_size = min(64, seq_len)
        q_w = q[-window_size:]
        k_pre = k_full[:-window_size]
        q_t2 = q_w.transpose(0, 1).contiguous()
        k_t2 = k_pre.transpose(0, 1).contiguous()
        if kv_group > 1:
            k_t2 = k_t2.repeat_interleave(kv_group, dim=0)
        tracker.snapshot("phase3_snapkv_expanded")

        attn = torch.matmul(q_t2, k_t2.transpose(-2, -1)) / math.sqrt(head_dim)
        tracker.snapshot("phase3_qk_matmul")

        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(dtype)
        tracker.snapshot("phase3_softmax")

        del q_t2, k_t2, attn
        torch.cuda.empty_cache()
        tracker.snapshot("phase3_cleanup")
    else:
        k_norm = k_full.norm(dim=-1).mean(dim=-1)
        v_norm = v_full.norm(dim=-1).mean(dim=-1)
        importance = (k_norm + v_norm) / 2
        tracker.snapshot("phase3_importance")

        _, sorted_idx = torch.sort(importance, descending=True)
        keep = sorted_idx[:num_to_keep]
        tracker.snapshot("phase3_sort")

    # Real pool write
    rpool_k = torch.zeros(pool_size, num_kv_heads, head_dim, dtype=dtype, device=device)
    rpool_v = torch.zeros(pool_size, num_kv_heads, head_dim, dtype=dtype, device=device)
    tracker.snapshot("real_pool_alloc")

    # Cleanup
    del gpool_k, gpool_v, k_full, v_full, k_data, v_data, q
    torch.cuda.empty_cache()
    tracker.snapshot("final_cleanup")

    return tracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens", default="4096,8192")
    parser.add_argument("--methods", default="key_norm,snapkv")
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()

    for seq_len in [int(x) for x in args.seq_lens.split(",")]:
        for method in args.methods.split(","):
            print(f"\n{'='*75}")
            print(f"Memory Profile: seq_len={seq_len}, method={method}, ratio={args.ratio}")
            print(f"{'='*75}")

            cell_mb = 2 * args.num_kv_heads * args.head_dim * 2 * seq_len / 1024**2
            print(f"Theoretical KV size (K+V, 1 layer): {cell_mb:.1f} MB")

            if method == "snapkv":
                window = min(64, seq_len)
                prefix = seq_len - window
                attn_mb = args.num_heads * window * prefix * 2 / 1024**2
                print(f"SnapKV attn_weights intermediate: {attn_mb:.1f} MB")

            tracker = run_memory_profile(
                seq_len, method, args.ratio,
                args.num_kv_heads, args.num_heads, args.head_dim,
            )
            print()
            print(tracker.report())


if __name__ == "__main__":
    main()
