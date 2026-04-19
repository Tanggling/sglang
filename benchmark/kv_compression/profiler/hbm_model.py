"""Theoretical HBM access model for KV compression pipeline.

Computes the expected number of bytes read/written per token per layer
for each phase of the three-phase compression pipeline.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class ModelConfig:
    name: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    num_heads: int  # Q heads (for GQA ratio)
    dtype_bytes: int = 2  # bf16


PRESETS: Dict[str, ModelConfig] = {
    "llama-3-8b": ModelConfig("Llama-3-8B", 32, 8, 128, 32, 2),
    "llama-3-70b": ModelConfig("Llama-3-70B", 80, 8, 128, 64, 2),
    "qwen2-7b": ModelConfig("Qwen2-7B", 28, 4, 128, 28, 2),
}


def kv_cell_bytes(cfg: ModelConfig) -> int:
    """Bytes per token per layer for K+V combined."""
    return 2 * cfg.num_kv_heads * cfg.head_dim * cfg.dtype_bytes


def theoretical_hbm_bytes(
    seq_len: int,
    cfg: ModelConfig,
    compression_ratio: float = 0.5,
    cpu_prefix_ratio: float = 0.0,
    method: str = "key_norm",
) -> Dict[str, Dict[str, float]]:
    """Compute theoretical HBM bytes per phase per layer.

    Args:
        seq_len: Total sequence length (prefix + extend).
        cfg: Model configuration.
        compression_ratio: Fraction of tokens to DROP (0.5 = keep 50%).
        cpu_prefix_ratio: Fraction of seq_len from CPU prefix cache.
        method: "key_norm" or "snapkv".

    Returns:
        Dict with per-phase breakdown: {phase: {read_bytes, write_bytes, total}}.
    """
    cell = kv_cell_bytes(cfg)  # bytes per token (K+V)
    prefix_len = int(seq_len * cpu_prefix_ratio)
    extend_len = seq_len - prefix_len
    keep_len = max(1, int(seq_len * (1 - compression_ratio)))

    result = {}

    # Phase 1: Assemble full KV in GlobalKVPool
    # - CPU prefix: write to GlobalKVPool (prefix_len tokens)
    # - Extend: write to GlobalKVPool (extend_len tokens)
    # - Read all from GlobalKVPool for FA input concat
    p1_write = seq_len * cell  # all tokens written to GlobalKVPool
    p1_read = seq_len * cell   # all tokens read back for concat
    result["phase1_assemble"] = {
        "read_bytes": p1_read,
        "write_bytes": p1_write,
        "total": p1_read + p1_write,
        "note": f"write {seq_len} tok + read {seq_len} tok (GlobalKVPool)",
    }

    # Phase 2: FlashAttention (IO-optimal)
    # FA reads K,V once in tiled fashion: O(N) HBM reads
    p2_read = seq_len * cell  # K,V read once
    p2_write = 0  # output is in registers/SRAM, written to output buffer (not KV)
    result["phase2_flashattn"] = {
        "read_bytes": p2_read,
        "write_bytes": p2_write,
        "total": p2_read + p2_write,
        "note": f"FA reads {seq_len} tok K,V (IO-optimal tiling)",
    }

    # Phase 3: Compress + scatter write
    # - Importance estimation: reads K,V again
    if method == "snapkv":
        window_size = min(64, seq_len)
        # SnapKV reads K for Q@K^T, plus window K for causal
        # Also reads Q (window_size tokens)
        q_bytes = window_size * cfg.num_heads * cfg.head_dim * cfg.dtype_bytes
        p3_importance_read = seq_len * cell + q_bytes
        # Intermediate: attn_weights [num_heads, window_size, seq_len]
        attn_intermediate = cfg.num_heads * window_size * seq_len * cfg.dtype_bytes
        result["phase3_snapkv_intermediate_bytes"] = attn_intermediate
    else:
        # key_norm: reads K,V for norm computation
        p3_importance_read = seq_len * cell
        attn_intermediate = 0

    # Scatter write to Real Pool: only keep_len tokens
    p3_write = keep_len * cell
    # Free GlobalKVPool: metadata only, negligible
    result["phase3_compress"] = {
        "read_bytes": p3_importance_read,
        "write_bytes": p3_write,
        "total": p3_importance_read + p3_write,
        "note": f"importance reads {seq_len} tok, writes {keep_len} tok to real pool",
    }

    # CPU→GPU PCIe transfer (not HBM, but important)
    if prefix_len > 0:
        pcie_bytes = prefix_len * cell
        result["pcie_cpu_transfer"] = {
            "read_bytes": 0,
            "write_bytes": 0,
            "pcie_bytes": pcie_bytes,
            "note": f"CPU→GPU {prefix_len} tok ({pcie_bytes / 1024**2:.1f} MB)",
        }

    # Totals
    total_read = p1_read + p2_read + p3_importance_read
    total_write = p1_write + p3_write
    result["total_per_layer"] = {
        "read_bytes": total_read,
        "write_bytes": total_write,
        "total": total_read + total_write,
        "reads_per_token": total_read / (seq_len * cell) if seq_len > 0 else 0,
        "writes_per_token": total_write / (seq_len * cell) if seq_len > 0 else 0,
    }

    # Across all layers
    result["total_all_layers"] = {
        "total_bytes": (total_read + total_write) * cfg.num_layers,
        "total_gb": (total_read + total_write) * cfg.num_layers / 1024**3,
    }

    return result


def print_analysis(
    seq_len: int,
    cfg: ModelConfig,
    compression_ratio: float = 0.5,
    cpu_prefix_ratio: float = 0.0,
    method: str = "key_norm",
):
    """Print formatted HBM analysis."""
    r = theoretical_hbm_bytes(seq_len, cfg, compression_ratio, cpu_prefix_ratio, method)
    cell = kv_cell_bytes(cfg)

    print(f"\n{'='*70}")
    print(f"HBM Analysis: {cfg.name}, seq_len={seq_len}, ratio={compression_ratio}, method={method}")
    print(f"KV cell size: {cell} bytes/token/layer (K+V, {cfg.num_kv_heads}h × {cfg.head_dim}d × {cfg.dtype_bytes}B × 2)")
    print(f"{'='*70}")

    for phase, data in r.items():
        if phase.startswith("total"):
            continue
        if not isinstance(data, dict):
            print(f"\n  {phase}: {data / 1024**2:.2f} MB")
            continue
        print(f"\n  {phase}:")
        for k, v in data.items():
            if k == "note":
                print(f"    {v}")
            elif "bytes" in k and isinstance(v, (int, float)):
                print(f"    {k}: {v / 1024**2:.2f} MB")
            else:
                print(f"    {k}: {v}")

    t = r["total_per_layer"]
    print(f"\n  --- Per Layer Total ---")
    print(f"    HBM read:  {t['read_bytes'] / 1024**2:.2f} MB ({t['reads_per_token']:.1f}× per token)")
    print(f"    HBM write: {t['write_bytes'] / 1024**2:.2f} MB ({t['writes_per_token']:.1f}× per token)")
    print(f"    Total:     {t['total'] / 1024**2:.2f} MB")

    a = r["total_all_layers"]
    print(f"\n  --- All {cfg.num_layers} Layers ---")
    print(f"    Total HBM traffic: {a['total_gb']:.2f} GB")
    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama-3-8b", choices=list(PRESETS.keys()))
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--cpu-prefix-ratio", type=float, default=0.8)
    parser.add_argument("--method", default="key_norm", choices=["key_norm", "snapkv"])
    args = parser.parse_args()

    cfg = PRESETS[args.model]
    print_analysis(args.seq_len, cfg, args.ratio, args.cpu_prefix_ratio, args.method)
    print()
    print_analysis(args.seq_len, cfg, args.ratio, args.cpu_prefix_ratio, "snapkv")
