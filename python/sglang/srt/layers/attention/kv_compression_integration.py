"""
KV Cache Compression Integration for SGLang

This module provides a complete integration example for using KV cache compression
with radix cache and chunked prefill disabled.

Usage:
    # Method 1: Using command line arguments
    python -m sglang.launch_server \
        --model-path meta-llama/Llama-2-7b-hf \
        --disable-radix-cache \
        --chunked-prefill-size -1 \
        --enable-kv-compression \
        --kv-compression-ratio 0.5
    
    # Method 2: Using Python API
    from sglang.srt.layers.attention.kv_compression_integration import (
        create_server_args_with_compression,
        setup_compression_backend,
    )
    
    server_args = create_server_args_with_compression(
        model_path="meta-llama/Llama-2-7b-hf",
        compression_ratio=0.5,
        disable_radix_cache=True,
        disable_chunked_prefill=True,
    )
"""

import dataclasses
from typing import List, Optional

from sglang.srt.server_args import ServerArgs


@dataclasses.dataclass
class KVCompressionArgs:
    """Arguments for KV cache compression."""
    
    enable_kv_compression: bool = False
    kv_compression_ratio: float = 0.5
    kv_compression_method: str = "importance"
    kv_compression_window_size: int = 64
    kv_compression_min_tokens: int = 32
    kv_compression_layers: Optional[str] = None
    kv_compression_retain_first: int = 16


def add_kv_compression_args(parser):
    """Add KV compression arguments to argument parser."""
    group = parser.add_argument_group("KV Cache Compression")
    
    group.add_argument(
        "--enable-kv-compression",
        action="store_true",
        help="Enable KV cache compression during prefill",
    )
    group.add_argument(
        "--kv-compression-ratio",
        type=float,
        default=0.5,
        help="Ratio of KV cache to compress (0.5 means keep 50%%)",
    )
    group.add_argument(
        "--kv-compression-method",
        type=str,
        default="importance",
        choices=["importance", "clustering", "snapkv"],
        help="Compression method to use",
    )
    group.add_argument(
        "--kv-compression-window-size",
        type=int,
        default=64,
        help="Window size for recent tokens to always keep",
    )
    group.add_argument(
        "--kv-compression-min-tokens",
        type=int,
        default=32,
        help="Minimum number of tokens to keep after compression",
    )
    group.add_argument(
        "--kv-compression-layers",
        type=str,
        default=None,
        help="Comma-separated list of layer indices to compress (default: all layers)",
    )
    group.add_argument(
        "--kv-compression-retain-first",
        type=int,
        default=16,
        help="Number of first tokens to always retain",
    )
    
    return parser


def create_server_args_with_compression(
    model_path: str,
    compression_ratio: float = 0.5,
    compression_method: str = "importance",
    disable_radix_cache: bool = True,
    disable_chunked_prefill: bool = True,
    **kwargs,
) -> ServerArgs:
    """
    Create ServerArgs with KV compression enabled.
    
    Args:
        model_path: Path to the model
        compression_ratio: Ratio of KV cache to compress
        compression_method: Compression method to use
        disable_radix_cache: Whether to disable radix cache
        disable_chunked_prefill: Whether to disable chunked prefill
        **kwargs: Additional ServerArgs parameters
    
    Returns:
        ServerArgs instance with compression settings
    """
    chunked_prefill_size = kwargs.pop("chunked_prefill_size", None)
    if disable_chunked_prefill:
        chunked_prefill_size = -1
    
    server_args = ServerArgs(
        model_path=model_path,
        disable_radix_cache=disable_radix_cache,
        chunked_prefill_size=chunked_prefill_size,
        **kwargs,
    )
    
    server_args.kv_compression_args = KVCompressionArgs(
        enable_kv_compression=True,
        kv_compression_ratio=compression_ratio,
        kv_compression_method=compression_method,
    )
    
    return server_args


def setup_compression_backend(model_runner):
    """
    Setup compression backend for model runner.
    
    This function should be called after model_runner initialization
    to replace the default attention backend with the compression-enabled one.
    
    Args:
        model_runner: ModelRunner instance
    """
    from sglang.srt.layers.attention.kv_compressor import CompressionConfig
    from sglang.srt.layers.attention.compressed_flashattention_backend import (
        CompressedFlashAttentionBackend,
    )
    
    server_args = model_runner.server_args
    
    if not hasattr(server_args, "kv_compression_args"):
        return
    
    kv_args = server_args.kv_compression_args
    if not kv_args.enable_kv_compression:
        return
    
    layers_to_compress = None
    if kv_args.kv_compression_layers:
        layers_to_compress = [
            int(x.strip()) for x in kv_args.kv_compression_layers.split(",")
        ]
    
    compression_config = CompressionConfig(
        enabled=True,
        compression_ratio=kv_args.kv_compression_ratio,
        compression_method=kv_args.kv_compression_method,
        window_size=kv_args.kv_compression_window_size,
        min_tokens_to_keep=kv_args.kv_compression_min_tokens,
        layers_to_compress=layers_to_compress,
        retain_first_n_tokens=kv_args.kv_compression_retain_first,
    )
    
    model_runner.attn_backend = CompressedFlashAttentionBackend(
        model_runner,
        compression_config=compression_config,
    )
    
    print(f"[KV Compression] Enabled with ratio={compression_config.compression_ratio}, "
          f"method={compression_config.compression_method}")


class CompressionMonitor:
    """
    Monitor for tracking compression statistics.
    
    Usage:
        monitor = CompressionMonitor(model_runner)
        
        # After inference
        stats = monitor.get_stats()
        monitor.print_summary()
    """
    
    def __init__(self, model_runner):
        self.model_runner = model_runner
        self.backend = model_runner.attn_backend
    
    def get_stats(self) -> dict:
        """Get compression statistics."""
        if hasattr(self.backend, "get_compression_stats"):
            return self.backend.get_compression_stats()
        return {}
    
    def reset_stats(self):
        """Reset compression statistics."""
        if hasattr(self.backend, "reset_compression_stats"):
            self.backend.reset_compression_stats()
    
    def print_summary(self):
        """Print compression summary."""
        stats = self.get_stats()
        if not stats:
            print("No compression statistics available")
            return
        
        print("\n" + "=" * 60)
        print("KV Cache Compression Summary")
        print("=" * 60)
        print(f"Total compression operations: {stats.get('total_compressed', 0)}")
        print(f"Total tokens freed: {stats.get('total_freed', 0)}")
        
        layer_stats = stats.get("layer_stats", {})
        if layer_stats:
            print("\nPer-layer statistics:")
            for layer_id, layer_stat in sorted(layer_stats.items()):
                avg_original = layer_stat["total_original"] / layer_stat["count"]
                avg_compressed = layer_stat["total_compressed"] / layer_stat["count"]
                ratio = 1 - (avg_compressed / avg_original)
                print(f"  Layer {layer_id}: avg compression ratio = {ratio:.2%}")
        
        print("=" * 60 + "\n")


def patch_model_runner_for_compression():
    """
    Monkey-patch ModelRunner to automatically setup compression backend.
    
    This should be called before creating any ModelRunner instances.
    """
    from sglang.srt.model_executor.model_runner import ModelRunner
    original_init = ModelRunner.__init__
    
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        setup_compression_backend(self)
    
    ModelRunner.__init__ = patched_init
    print("[KV Compression] ModelRunner patched for automatic compression setup")


def create_chunk_cache_instead_of_radix(tree_cache_class):
    """
    Factory function to create ChunkCache instead of RadixCache.
    
    When radix cache is disabled, SGLang uses ChunkCache which doesn't
    do prefix matching and simply allocates contiguous memory blocks.
    
    Args:
        tree_cache_class: The original tree cache class
    
    Returns:
        ChunkCache class
    """
    from sglang.srt.mem_cache.chunk_cache import ChunkCache
    return ChunkCache
