"""
Complete Example: KV Cache Compression with SGLang

This example demonstrates how to use KV cache compression during prefill
with radix cache and chunked prefill disabled.

Files created:
1. kv_compressor.py - Compression algorithms
2. compressed_flashattention_backend.py - Attention backend with compression
3. kv_compression_integration.py - Integration utilities
4. kv_compression_example.py - This file (usage examples)
"""

# =============================================================================
# Example 1: Basic Usage with Python API
# =============================================================================

def example_basic_usage():
    """
    Basic usage example with Python API.
    """
    from sglang.srt.layers.attention.kv_compressor import (
        CompressionConfig,
        create_compressor,
    )
    import torch
    
    config = CompressionConfig(
        enabled=True,
        compression_ratio=0.5,
        compression_method="importance",
        window_size=64,
        min_tokens_to_keep=32,
        retain_first_n_tokens=16,
    )
    
    compressor = create_compressor(config)
    
    seq_len = 256
    num_heads = 32
    head_dim = 128
    
    k = torch.randn(seq_len, num_heads, head_dim)
    v = torch.randn(seq_len, num_heads, head_dim)
    
    compressed_k, compressed_v, keep_indices = compressor.compress(
        k=k, v=v, layer_id=0
    )
    
    print(f"Original sequence length: {seq_len}")
    print(f"Compressed sequence length: {compressed_k.shape[0]}")
    print(f"Compression ratio: {1 - compressed_k.shape[0] / seq_len:.2%}")
    print(f"Kept indices: {keep_indices[:10]}...")


# =============================================================================
# Example 2: Integration with SGLang Server
# =============================================================================

def example_server_integration():
    """
    Example of integrating compression with SGLang server.
    """
    code = '''
# Method 1: Command line
python -m sglang.launch_server \\
    --model-path meta-llama/Llama-2-7b-hf \\
    --disable-radix-cache \\
    --chunked-prefill-size -1 \\
    --enable-kv-compression \\
    --kv-compression-ratio 0.5 \\
    --kv-compression-method importance

# Method 2: Python script
import sglang as sgl
from sglang.srt.layers.attention.kv_compression_integration import (
    create_server_args_with_compression,
    patch_model_runner_for_compression,
)

# Patch ModelRunner before starting server
patch_model_runner_for_compression()

# Create server args with compression
server_args = create_server_args_with_compression(
    model_path="meta-llama/Llama-2-7b-hf",
    compression_ratio=0.5,
    compression_method="importance",
    disable_radix_cache=True,
    disable_chunked_prefill=True,
)

# Launch server
runtime = sgl.Runtime(server_args=server_args)
'''
    print(code)


# =============================================================================
# Example 3: Custom Compression Algorithm
# =============================================================================

def example_custom_compressor():
    """
    Example of implementing a custom compression algorithm.
    """
    code = '''
from sglang.srt.layers.attention.kv_compressor import (
    BaseKVCompressor,
    CompressionConfig,
)
import torch
from typing import Tuple, Optional

class MyCustomCompressor(BaseKVCompressor):
    """Custom compressor that keeps every N-th token."""
    
    def __init__(self, config: CompressionConfig):
        super().__init__(config)
        self.stride = 2  # Keep every 2nd token
    
    def compress(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None,
        layer_id: int = 0,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = k.shape[0]
        
        # Select every N-th token
        keep_indices = torch.arange(0, seq_len, self.stride, device=k.device)
        
        # Always keep the last window_size tokens
        if self.config.window_size > 0:
            window_start = max(0, seq_len - self.config.window_size)
            window_indices = torch.arange(window_start, seq_len, device=k.device)
            keep_indices = torch.unique(torch.cat([keep_indices, window_indices]))
        
        # Always keep first N tokens
        if self.config.retain_first_n_tokens > 0:
            first_n = torch.arange(
                min(self.config.retain_first_n_tokens, seq_len),
                device=k.device
            )
            keep_indices = torch.unique(torch.cat([first_n, keep_indices]))
        
        keep_indices = torch.sort(keep_indices)[0]
        
        compressed_k = k[keep_indices]
        compressed_v = v[keep_indices]
        
        return compressed_k, compressed_v, keep_indices


# Register custom compressor
from sglang.srt.layers.attention.kv_compressor import create_compressor

# Monkey-patch the factory function
original_create = create_compressor

def patched_create_compressor(config: CompressionConfig):
    if config.compression_method == "custom":
        return MyCustomCompressor(config)
    return original_create(config)

import sglang.srt.layers.attention.kv_compressor as compressor_module
compressor_module.create_compressor = patched_create_compressor
'''
    print(code)


# =============================================================================
# Example 4: Monitoring Compression Statistics
# =============================================================================

def example_monitoring():
    """
    Example of monitoring compression statistics.
    """
    code = '''
from sglang.srt.layers.attention.kv_compression_integration import CompressionMonitor

# After running inference
monitor = CompressionMonitor(model_runner)

# Get statistics
stats = monitor.get_stats()
print(f"Total tokens freed: {stats['total_freed']}")

# Print detailed summary
monitor.print_summary()

# Reset statistics
monitor.reset_stats()
'''
    print(code)


# =============================================================================
# Example 5: Different Compression Methods Comparison
# =============================================================================

def example_compare_methods():
    """
    Compare different compression methods.
    """
    import torch
    from sglang.srt.layers.attention.kv_compressor import (
        CompressionConfig,
        create_compressor,
    )
    
    seq_len = 512
    num_heads = 32
    head_dim = 128
    
    k = torch.randn(seq_len, num_heads, head_dim)
    v = torch.randn(seq_len, num_heads, head_dim)
    
    methods = ["importance", "clustering", "snapkv"]
    
    print("=" * 70)
    print("Compression Method Comparison")
    print("=" * 70)
    print(f"Original sequence length: {seq_len}")
    print()
    
    for method in methods:
        config = CompressionConfig(
            enabled=True,
            compression_ratio=0.5,
            compression_method=method,
            window_size=64,
            min_tokens_to_keep=32,
        )
        
        compressor = create_compressor(config)
        
        import time
        start = time.time()
        compressed_k, compressed_v, keep_indices = compressor.compress(k=k, v=v, layer_id=0)
        elapsed = time.time() - start
        
        print(f"Method: {method}")
        print(f"  Compressed length: {compressed_k.shape[0]}")
        print(f"  Compression ratio: {1 - compressed_k.shape[0] / seq_len:.2%}")
        print(f"  Time: {elapsed*1000:.2f} ms")
        print()


# =============================================================================
# Example 6: Integration with ModelRunner
# =============================================================================

def example_model_runner_integration():
    """
    Example of integrating compression directly with ModelRunner.
    """
    code = '''
# In your custom model_runner.py or as a patch:

from sglang.srt.layers.attention.kv_compressor import CompressionConfig
from sglang.srt.layers.attention.compressed_flashattention_backend import (
    CompressedFlashAttentionBackend,
)

class ModelRunnerWithCompression:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Create compression config
        compression_config = CompressionConfig(
            enabled=True,
            compression_ratio=0.5,
            compression_method="importance",
            window_size=64,
            min_tokens_to_keep=32,
            layers_to_compress=[0, 4, 8, 12, 16, 20, 24, 28],  # Every 4th layer
        )
        
        # Replace attention backend
        self.attn_backend = CompressedFlashAttentionBackend(
            self,
            compression_config=compression_config,
        )
        
        print(f"[Compression] Initialized with ratio={compression_config.compression_ratio}")
    
    def forward_extend(self, forward_batch, *args, **kwargs):
        # The compression happens automatically in the attention backend
        result = super().forward_extend(forward_batch, *args, **kwargs)
        
        # Optionally log compression stats
        if hasattr(self.attn_backend, 'get_compression_stats'):
            stats = self.attn_backend.get_compression_stats()
            # Log or monitor stats here
        
        return result
'''
    print(code)


# =============================================================================
# Example 7: Disable Radix Cache and Chunked Prefill
# =============================================================================

def example_disable_features():
    """
    Example of disabling radix cache and chunked prefill.
    """
    code = '''
# Method 1: ServerArgs
from sglang.srt.server_args import ServerArgs

server_args = ServerArgs(
    model_path="meta-llama/Llama-2-7b-hf",
    
    # Disable radix cache (use ChunkCache instead)
    disable_radix_cache=True,
    
    # Disable chunked prefill (process entire sequence at once)
    chunked_prefill_size=-1,  # -1 means disabled
)

# Method 2: Command line
# --disable-radix-cache
# --chunked-prefill-size -1

# What happens when disabled:

# Radix Cache Disabled:
# - Uses ChunkCache instead of RadixCache
# - No prefix matching between requests
# - Simpler memory allocation
# - Better for single-request scenarios

# Chunked Prefill Disabled:
# - Entire prompt processed in one forward pass
# - No splitting of long sequences
# - Simpler execution flow
# - May require more memory for long prompts
'''
    print(code)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("KV Cache Compression Examples for SGLang")
    print("=" * 70)
    
    print("\n" + "-" * 70)
    print("Example 1: Basic Usage")
    print("-" * 70)
    example_basic_usage()
    
    print("\n" + "-" * 70)
    print("Example 2: Server Integration")
    print("-" * 70)
    example_server_integration()
    
    print("\n" + "-" * 70)
    print("Example 3: Custom Compression Algorithm")
    print("-" * 70)
    example_custom_compressor()
    
    print("\n" + "-" * 70)
    print("Example 4: Monitoring Compression Statistics")
    print("-" * 70)
    example_monitoring()
    
    print("\n" + "-" * 70)
    print("Example 5: Compare Compression Methods")
    print("-" * 70)
    example_compare_methods()
    
    print("\n" + "-" * 70)
    print("Example 6: ModelRunner Integration")
    print("-" * 70)
    example_model_runner_integration()
    
    print("\n" + "-" * 70)
    print("Example 7: Disable Radix Cache and Chunked Prefill")
    print("-" * 70)
    example_disable_features()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
