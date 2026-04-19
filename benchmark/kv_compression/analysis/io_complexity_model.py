"""IO Complexity Model for KV Compression Pipeline.

Computes theoretical HBM traffic, roofline analysis, and optimization
potential for the three-phase compression pipeline.

Usage:
    python benchmark/kv_compression/analysis/io_complexity_model.py \
        --model llama-3-8b --seq-len 8192 --ratio 0.5
"""

import argparse
import sys

sys.path.insert(0, ".")
from benchmark.kv_compression.profiler.hbm_model import (
    PRESETS, ModelConfig, kv_cell_bytes, theoretical_hbm_bytes, print_analysis,
)


def optimization_potential(
    seq_len: int,
    cfg: ModelConfig,
    compression_ratio: float = 0.5,
    cpu_prefix_ratio: float = 0.8,
):
    """Analyze optimization potential for each proposed direction."""
    cell = kv_cell_bytes(cfg)
    prefix_len = int(seq_len * cpu_prefix_ratio)
    extend_len = seq_len - prefix_len
    keep_len = max(1, int(seq_len * (1 - compression_ratio)))

    current = theoretical_hbm_bytes(seq_len, cfg, compression_ratio, cpu_prefix_ratio, "key_norm")
    current_total = current["total_per_layer"]["total"]

    print(f"\n{'='*70}")
    print(f"Optimization Potential Analysis")
    print(f"Model: {cfg.name}, seq_len={seq_len}, ratio={compression_ratio}, "
          f"cpu_prefix={cpu_prefix_ratio:.0%}")
    print(f"Current total HBM per layer: {current_total / 1024**2:.2f} MB")
    print(f"{'='*70}")

    optimizations = []

    # Opt 1: LSE-based importance (eliminate Phase 3 re-read)
    saved = seq_len * cell  # Phase 3 importance read eliminated
    new_total = current_total - saved
    optimizations.append((
        "LSE-based importance (eliminate Phase 3 K,V re-read)",
        saved, new_total, saved / current_total * 100,
    ))

    # Opt 2: Skip GlobalKVPool for extend tokens
    # Extend tokens: save 1 write + 1 read to GlobalKVPool
    saved = extend_len * cell * 2  # write + read
    new_total = current_total - saved
    optimizations.append((
        f"Skip GlobalKVPool for extend tokens ({extend_len} tok)",
        saved, new_total, saved / current_total * 100,
    ))

    # Opt 3: Async CPU transfer (doesn't reduce HBM, but hides latency)
    pcie_bytes = prefix_len * cell
    optimizations.append((
        f"Async pinned CPU transfer ({prefix_len} tok, {pcie_bytes/1024**2:.1f} MB)",
        0, current_total, 0,  # HBM unchanged, but latency hidden
    ))

    # Opt 4: Combined (LSE + skip GlobalKVPool for extend)
    saved_lse = seq_len * cell
    saved_skip = extend_len * cell * 2
    total_saved = saved_lse + saved_skip
    new_total = current_total - total_saved
    optimizations.append((
        "Combined: LSE + skip extend GlobalKVPool",
        total_saved, new_total, total_saved / current_total * 100,
    ))

    # Theoretical minimum: each token read once (FA) + write once (real pool)
    theoretical_min = seq_len * cell + keep_len * cell  # 1 read + 1 write(compressed)
    optimizations.append((
        f"Theoretical minimum (1 read + 1 compressed write)",
        current_total - theoretical_min, theoretical_min,
        (current_total - theoretical_min) / current_total * 100,
    ))

    print(f"\n{'Optimization':<55} {'Saved':>10} {'New Total':>10} {'Reduction':>10}")
    print("─" * 90)
    for name, saved_bytes, new_bytes, pct in optimizations:
        saved_mb = saved_bytes / 1024**2
        new_mb = new_bytes / 1024**2
        print(f"{name:<55} {saved_mb:>8.1f}MB {new_mb:>8.1f}MB {pct:>8.1f}%")

    # All-layers summary
    print(f"\n--- Across all {cfg.num_layers} layers ---")
    current_gb = current_total * cfg.num_layers / 1024**3
    min_gb = theoretical_min * cfg.num_layers / 1024**3
    print(f"Current:  {current_gb:.2f} GB")
    print(f"Minimum:  {min_gb:.2f} GB")
    print(f"Potential reduction: {(1 - min_gb/current_gb)*100:.0f}%")


def snapkv_intermediate_analysis(cfg: ModelConfig, seq_len: int, window_size: int = 64):
    """Analyze SnapKV's intermediate tensor memory."""
    prefix_len = seq_len - window_size
    kv_group = cfg.num_heads // cfg.num_kv_heads

    print(f"\n{'='*70}")
    print(f"SnapKV Intermediate Tensor Analysis")
    print(f"Model: {cfg.name}, seq_len={seq_len}, window={window_size}")
    print(f"GQA group: {kv_group} (num_heads={cfg.num_heads}, num_kv_heads={cfg.num_kv_heads})")
    print(f"{'='*70}")

    # Q@K^T intermediate: [num_heads, window_size, prefix_len]
    attn_elements = cfg.num_heads * window_size * prefix_len
    attn_bytes_bf16 = attn_elements * 2
    attn_bytes_fp32 = attn_elements * 4  # after softmax cast to fp32

    # repeat_interleave: K expanded by kv_group
    k_expanded_elements = cfg.num_heads * prefix_len * cfg.head_dim
    k_expanded_bytes = k_expanded_elements * 2

    # Original K (before expansion)
    k_original_bytes = cfg.num_kv_heads * prefix_len * cfg.head_dim * 2

    print(f"\n{'Tensor':<40} {'Shape':<30} {'Size':>10}")
    print("─" * 85)
    print(f"{'K prefix (original)':<40} {'['+str(cfg.num_kv_heads)+','+str(prefix_len)+','+str(cfg.head_dim)+']':<30} {k_original_bytes/1024**2:>8.1f}MB")
    print(f"{'K prefix (after repeat_interleave)':<40} {'['+str(cfg.num_heads)+','+str(prefix_len)+','+str(cfg.head_dim)+']':<30} {k_expanded_bytes/1024**2:>8.1f}MB")
    print(f"{'attn_weights (bf16)':<40} {'['+str(cfg.num_heads)+','+str(window_size)+','+str(prefix_len)+']':<30} {attn_bytes_bf16/1024**2:>8.1f}MB")
    print(f"{'attn_weights (fp32 after softmax)':<40} {'['+str(cfg.num_heads)+','+str(window_size)+','+str(prefix_len)+']':<30} {attn_bytes_fp32/1024**2:>8.1f}MB")

    peak = k_expanded_bytes + attn_bytes_fp32
    print(f"\n{'Peak intermediate (K_expanded + attn_fp32)':<40} {'':30} {peak/1024**2:>8.1f}MB")
    print(f"{'Savings from eliminating repeat_interleave':<40} {'':30} {(k_expanded_bytes - k_original_bytes)/1024**2:>8.1f}MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama-3-8b", choices=list(PRESETS.keys()))
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--cpu-prefix-ratio", type=float, default=0.8)
    parser.add_argument("--window-size", type=int, default=64)
    args = parser.parse_args()

    cfg = PRESETS[args.model]

    # 1. Current HBM analysis (key_norm vs snapkv)
    print_analysis(args.seq_len, cfg, args.ratio, args.cpu_prefix_ratio, "key_norm")
    print_analysis(args.seq_len, cfg, args.ratio, args.cpu_prefix_ratio, "snapkv")

    # 2. Optimization potential
    optimization_potential(args.seq_len, cfg, args.ratio, args.cpu_prefix_ratio)

    # 3. SnapKV intermediate analysis
    snapkv_intermediate_analysis(cfg, args.seq_len, args.window_size)


if __name__ == "__main__":
    main()
