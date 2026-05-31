"""
Pipeline Breakdown Benchmark v2: Full prefix-hit pipeline with optimizations

Simulates the complete CPU prefix cache hit pipeline per layer:
  1. CPU→GPU transfer (prefix KV from CPU cache to GlobalKVPool)
  2. GlobalKVPool write (scatter write KV into pool buffer)
  3. GlobalKVPool read (gather KV for FA)
  4. FlashAttention (Q=extend, K/V=full sequence)
  5. SnapKV compress (importance estimation + token selection)
  6. Scatter write to real KV pool

Compares:
  - OLD: pageable memory + repeat_interleave GQA expand
  - NEW: pinned memory + grouped bmm (no K expand)

Measures per-phase latency and peak memory for multiple (seq_len, prefix_ratio, ratio) configs.

Usage:
    python benchmark/kv_compression/bench_pipeline_breakdown_v2.py
    python benchmark/kv_compression/bench_pipeline_breakdown_v2.py --model mistral
    python benchmark/kv_compression/bench_pipeline_breakdown_v2.py --model qwen
"""

import argparse
import math
import torch
import torch.nn.functional as F
import time


# ═══════════════════════════════════════════════════════════════════════════════
# SnapKV implementations (OLD vs NEW)
# ═══════════════════════════════════════════════════════════════════════════════

def snapkv_compress_old(q_window, k_prefix, k_window, num_kv_heads, kv_group_num, head_dim,
                        window_size, num_tokens_to_keep, kernel_size=5):
    """OLD: repeat_interleave K then matmul."""
    q_t = q_window.transpose(0, 1).contiguous()
    k_t = k_prefix.transpose(0, 1).contiguous()

    k_t_expanded = k_t.repeat_interleave(kv_group_num, dim=0)
    attn_weights = torch.matmul(q_t, k_t_expanded.transpose(-2, -1)) / math.sqrt(head_dim)

    k_window_t = k_window.transpose(0, 1).contiguous()
    k_window_t_expanded = k_window_t.repeat_interleave(kv_group_num, dim=0)
    attn_weights_window = torch.matmul(q_t, k_window_t_expanded.transpose(-2, -1)) / math.sqrt(head_dim)

    q_window_len = window_size
    causal_mask = torch.triu(
        torch.full((q_window_len, q_window_len), float('-inf'), device=q_window.device, dtype=attn_weights_window.dtype),
        diagonal=1
    )
    attn_weights_window = attn_weights_window + causal_mask.unsqueeze(0)

    attn_weights_full = torch.cat([attn_weights, attn_weights_window], dim=-1)
    attn_weights_prefix = attn_weights_full[:, :, :k_prefix.shape[0]]
    attn_weights_sum = attn_weights_prefix.sum(dim=1)
    attn_weights_sum = attn_weights_sum.view(num_kv_heads, kv_group_num, -1).sum(dim=1)

    attn_cache = F.max_pool1d(attn_weights_sum.unsqueeze(0), kernel_size=kernel_size,
                              padding=kernel_size//2, stride=1).squeeze(0)
    _, indices = attn_cache.topk(num_tokens_to_keep - window_size, dim=-1)
    indices = torch.sort(indices, dim=-1).values

    seq_len = k_prefix.shape[0] + window_size
    window_indices = torch.arange(seq_len - window_size, seq_len, device=q_window.device).unsqueeze(0).expand(num_kv_heads, -1)
    keep_indices = torch.cat([indices, window_indices], dim=-1)
    return keep_indices


def snapkv_compress_new(q_window, k_prefix, k_window, num_kv_heads, kv_group_num, head_dim,
                        window_size, num_tokens_to_keep, kernel_size=5):
    """NEW: grouped bmm, no K expand."""
    q_t = q_window.transpose(0, 1).contiguous()
    k_t = k_prefix.transpose(0, 1).contiguous()

    q_grouped = q_t.view(num_kv_heads, kv_group_num * window_size, head_dim)
    attn_weights = torch.bmm(q_grouped, k_t.transpose(-2, -1)) / math.sqrt(head_dim)

    k_window_t = k_window.transpose(0, 1).contiguous()
    attn_weights_window = torch.bmm(q_grouped, k_window_t.transpose(-2, -1)) / math.sqrt(head_dim)

    q_window_len = window_size
    causal_mask_base = torch.triu(
        torch.full((q_window_len, q_window_len), float('-inf'), device=q_window.device, dtype=attn_weights_window.dtype),
        diagonal=1
    )
    causal_mask = causal_mask_base.repeat(kv_group_num, 1)
    attn_weights_window = attn_weights_window + causal_mask.unsqueeze(0)

    attn_weights_full = torch.cat([attn_weights, attn_weights_window], dim=-1)
    attn_weights_prefix = attn_weights_full[:, :, :k_prefix.shape[0]]
    attn_weights_sum = attn_weights_prefix.sum(dim=1)

    attn_cache = F.max_pool1d(attn_weights_sum.unsqueeze(0), kernel_size=kernel_size,
                              padding=kernel_size//2, stride=1).squeeze(0)
    _, indices = attn_cache.topk(num_tokens_to_keep - window_size, dim=-1)
    indices = torch.sort(indices, dim=-1).values

    seq_len = k_prefix.shape[0] + window_size
    window_indices = torch.arange(seq_len - window_size, seq_len, device=q_window.device).unsqueeze(0).expand(num_kv_heads, -1)
    keep_indices = torch.cat([indices, window_indices], dim=-1)
    return keep_indices


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline simulation
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline_single_layer(
    prefix_len, extend_len, num_kv_heads, num_q_heads, head_dim, compression_ratio,
    window_size, use_pinned, use_new_compress, warmup=3, repeat=10,
):
    """Run one layer of the pipeline and measure each phase."""
    device = "cuda"
    dtype = torch.bfloat16
    kv_group_num = num_q_heads // num_kv_heads
    seq_len = prefix_len + extend_len
    num_tokens_to_keep = max(32, int(seq_len * (1 - compression_ratio)))

    # ── Prepare CPU prefix data (simulating cached entry) ──
    shape_kv = (prefix_len, num_kv_heads, head_dim)
    if use_pinned:
        k_cpu = torch.empty(shape_kv, dtype=dtype, pin_memory=True)
        v_cpu = torch.empty(shape_kv, dtype=dtype, pin_memory=True)
        k_cpu.normal_()
        v_cpu.normal_()
    else:
        k_cpu = torch.randn(shape_kv, dtype=dtype)
        v_cpu = torch.randn(shape_kv, dtype=dtype)

    # Extend KV (from model forward, already on GPU)
    k_extend = torch.randn(extend_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_extend = torch.randn(extend_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    q_extend = torch.randn(extend_len, num_q_heads, head_dim, device=device, dtype=dtype)

    # GlobalKVPool buffer (pre-allocated)
    pool_k = torch.zeros(seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    pool_v = torch.zeros(seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)

    # Real KV pool slots
    real_k = torch.zeros(num_tokens_to_keep, num_kv_heads, head_dim, device=device, dtype=dtype)
    real_v = torch.zeros(num_tokens_to_keep, num_kv_heads, head_dim, device=device, dtype=dtype)

    torch.cuda.synchronize()

    # ── Warmup ──
    for _ in range(warmup):
        k_gpu = k_cpu.to(device, non_blocking=use_pinned)
        v_gpu = v_cpu.to(device, non_blocking=use_pinned)
        torch.cuda.synchronize()
        del k_gpu, v_gpu

    # ── Measure each phase ──
    phase_times = {"transfer": [], "pool_write": [], "pool_read": [], "fa": [], "compress": [], "write_real": []}

    for _ in range(repeat):
        torch.cuda.synchronize()

        # Phase 1: CPU→GPU transfer
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        k_gpu = k_cpu.to(device, non_blocking=use_pinned)
        v_gpu = v_cpu.to(device, non_blocking=use_pinned)
        e.record(); torch.cuda.synchronize()
        phase_times["transfer"].append(s.elapsed_time(e))

        # Phase 2: Write to GlobalKVPool
        s.record()
        pool_k[:prefix_len] = k_gpu
        pool_v[:prefix_len] = v_gpu
        pool_k[prefix_len:] = k_extend
        pool_v[prefix_len:] = v_extend
        e.record(); torch.cuda.synchronize()
        phase_times["pool_write"].append(s.elapsed_time(e))

        # Phase 3: Read from GlobalKVPool (for FA)
        s.record()
        k_full = pool_k[:seq_len].clone()
        v_full = pool_v[:seq_len].clone()
        e.record(); torch.cuda.synchronize()
        phase_times["pool_read"].append(s.elapsed_time(e))

        # Phase 4: FlashAttention (simplified as bmm for benchmark — real system uses flash_attn)
        # Q=extend, K/V=full. For short extend, this is memory-bound.
        s.record()
        if extend_len > 0:
            q_view = q_extend.view(extend_len, num_q_heads, head_dim)
            # Simplified attention: Q×K^T (no actual FA kernel, just matmul to simulate compute)
            q_t = q_view.transpose(0, 1)  # [q_heads, extend, dim]
            k_t = k_full.transpose(0, 1)  # [kv_heads, seq_len, dim]
            k_t_exp = k_t.repeat_interleave(kv_group_num, dim=0)
            _ = torch.bmm(q_t, k_t_exp.transpose(-2, -1))
        e.record(); torch.cuda.synchronize()
        phase_times["fa"].append(s.elapsed_time(e))

        # Phase 5: SnapKV compress
        s.record()
        # Always use a full window_size Q for compression (simulating Q padding from CPU cache)
        q_window_for_compress = torch.randn(window_size, num_q_heads, head_dim, device=device, dtype=dtype)
        k_prefix_part = k_full[:-window_size]
        k_window_part = k_full[-window_size:]

        if use_new_compress:
            keep_indices = snapkv_compress_new(q_window_for_compress, k_prefix_part, k_window_part,
                                              num_kv_heads, kv_group_num, head_dim,
                                              window_size, num_tokens_to_keep)
        else:
            keep_indices = snapkv_compress_old(q_window_for_compress, k_prefix_part, k_window_part,
                                              num_kv_heads, kv_group_num, head_dim,
                                              window_size, num_tokens_to_keep)
        e.record(); torch.cuda.synchronize()
        phase_times["compress"].append(s.elapsed_time(e))

        # Phase 6: Write compressed KV to real pool
        s.record()
        for head_idx in range(num_kv_heads):
            src_idx = keep_indices[head_idx]
            real_k[:num_tokens_to_keep, head_idx] = k_full[src_idx, head_idx]
            real_v[:num_tokens_to_keep, head_idx] = v_full[src_idx, head_idx]
        e.record(); torch.cuda.synchronize()
        phase_times["write_real"].append(s.elapsed_time(e))

        del k_gpu, v_gpu, k_full, v_full

    # Average times
    avg = {k: sum(v)/len(v) for k, v in phase_times.items()}
    avg["total"] = sum(avg.values())
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mistral", "qwen"], default="mistral")
    parser.add_argument("--seq-lens", default="4096,8192,16384,32768")
    parser.add_argument("--prefix-ratios", default="0.5,0.8,1.0")
    parser.add_argument("--compression-ratio", type=float, default=0.5)
    parser.add_argument("--window-size", type=int, default=64)
    args = parser.parse_args()

    if args.model == "mistral":
        num_kv_heads, num_q_heads, head_dim, num_layers = 8, 32, 128, 32
    else:
        num_kv_heads, num_q_heads, head_dim, num_layers = 4, 28, 128, 28

    kv_group_num = num_q_heads // num_kv_heads
    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    prefix_ratios = [float(x) for x in args.prefix_ratios.split(",")]

    print("=" * 120)
    print(f"Pipeline Breakdown Benchmark v2: {args.model.upper()} (ratio={args.compression_ratio})")
    print("=" * 120)
    print(f"  Config: kv_heads={num_kv_heads}, q_heads={num_q_heads}, group={kv_group_num}, "
          f"head_dim={head_dim}, layers={num_layers}, window={args.window_size}")
    print(f"  OLD: pageable + repeat_interleave")
    print(f"  NEW: pinned + grouped bmm")
    print()

    # Header
    print(f"{'seq':>6} {'pfx%':>5} {'mode':>5} "
          f"{'xfer':>7} {'pool_w':>7} {'pool_r':>7} {'FA':>7} {'comp':>7} {'write':>7} {'TOTAL':>7} "
          f"{'×layers':>8}")
    print("─" * 120)

    for seq_len in seq_lens:
        for pfx_ratio in prefix_ratios:
            prefix_len = int(seq_len * pfx_ratio)
            extend_len = seq_len - prefix_len
            if prefix_len == 0:
                continue

            # OLD pipeline
            old = run_pipeline_single_layer(
                prefix_len, extend_len, num_kv_heads, num_q_heads, head_dim,
                args.compression_ratio, args.window_size,
                use_pinned=False, use_new_compress=False,
            )

            # NEW pipeline
            new = run_pipeline_single_layer(
                prefix_len, extend_len, num_kv_heads, num_q_heads, head_dim,
                args.compression_ratio, args.window_size,
                use_pinned=True, use_new_compress=True,
            )

            for mode, r in [("OLD", old), ("NEW", new)]:
                total_layers = r["total"] * num_layers
                print(f"{seq_len:>6} {pfx_ratio:>4.0%} {mode:>5} "
                      f"{r['transfer']:>6.2f} {r['pool_write']:>6.2f} {r['pool_read']:>6.2f} "
                      f"{r['fa']:>6.2f} {r['compress']:>6.2f} {r['write_real']:>6.2f} "
                      f"{r['total']:>6.2f} {total_layers:>7.1f}ms")

            # Speedup
            sp = old["total"] / new["total"] if new["total"] > 0 else 0
            sp_comp = old["compress"] / new["compress"] if new["compress"] > 0 else 0
            print(f"{'':>6} {'':>5} {'Δ':>5} "
                  f"{'':>7} {'':>7} {'':>7} {'':>7} "
                  f"{sp_comp:>6.2f}x {'':>7} {sp:>6.2f}x")
            print()

    # Memory comparison
    print()
    print("=" * 120)
    print("Peak Memory Comparison (single layer, largest config)")
    print("=" * 120)
    largest_seq = max(seq_lens)
    prefix_len = int(largest_seq * 0.8)
    prefix_mb = prefix_len * num_kv_heads * head_dim * 2 * 2 / (1024**2)
    k_expand_mb = num_q_heads * (prefix_len - args.window_size) * head_dim * 2 / (1024**2)

    print(f"  Prefix KV (K+V): {prefix_mb:.1f} MB")
    print(f"  K expanded (OLD): {k_expand_mb:.1f} MB (eliminated in NEW)")
    print(f"  Savings per layer: {k_expand_mb:.1f} MB")
    print(f"  Savings all layers (peak): {k_expand_mb:.1f} MB (only 1 layer active at a time)")


if __name__ == "__main__":
    main()
