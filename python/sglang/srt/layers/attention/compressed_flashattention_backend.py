"""
Compressed FlashAttention Backend for SGLang

This backend extends FlashAttentionBackend to support KV cache compression
during the prefill (extend) phase.

Key Features:
1. Uses FlashAttention for efficient attention computation
2. Computes full attention output first (no quality loss)
3. Estimates token importance using LSE or key norms
4. Applies compression algorithm to select important KV entries
5. Releases unused KV cache slots after attention computation
6. Updates req_to_token mapping for compressed sequences
7. Works with radix cache disabled mode

Importance Estimation Methods:
- "lse": Use FlashAttention's softmax LSE (fast, approximate)
- "key_norm": Use key vector norms (fast, no attention needed)
- "snapkv": Compute full attention scores (slow, accurate)
"""

from typing import TYPE_CHECKING, Optional, Tuple, Literal

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend

if TYPE_CHECKING:
    from sglang.srt.layers.attention.kv_compressor import CompressionConfig
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


ImportanceMethod = Literal["lse", "key_norm", "snapkv"]


class CompressedFlashAttentionBackend(FlashAttentionBackend):
    """
    FlashAttention backend with KV cache compression support.
    
    This backend first computes full attention output, then compresses KV cache
    after each attention layer's computation during the prefill phase, immediately
    releasing unused memory. The returned output is from full attention computation.
    
    Importance Estimation Methods:
        - "lse": Use FlashAttention's softmax LSE (fast, memory efficient)
        - "key_norm": Use key vector norms (fastest, no extra computation)
        - "snapkv": Compute full attention scores (accurate but slow)
    
    Usage:
        from sglang.srt.layers.attention.kv_compressor import CompressionConfig
        
        compression_config = CompressionConfig(
            enabled=True,
            compression_ratio=0.5,
            compression_method="importance",
            window_size=64,
        )
        
        self.attn_backend = CompressedFlashAttentionBackend(
            runner,
            compression_config=compression_config,
            importance_method="key_norm",  # or "lse" or "snapkv"
        )
    """
    
    def __init__(
        self,
        runner,
        compression_config: Optional["CompressionConfig"] = None,
        fa_impl_ver: int = 3, 
        importance_method: ImportanceMethod = "snapkv",
    ):
        super().__init__(runner, fa_impl_ver=fa_impl_ver)
        
        from sglang.srt.layers.attention.kv_compressor import (
            CompressionConfig,
            create_compressor,
        )
        
        self.compression_config = compression_config or CompressionConfig()
        self.compressor = create_compressor(self.compression_config)
        self.importance_method = importance_method
        
        self._compression_stats = {
            "total_compressed": 0,
            "total_freed": 0,
            "layer_stats": {},
        }
    
    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for extend mode with KV cache compression.
        
        This method:
        1. First computes full attention output using parent's forward_extend
        2. Then estimates token importance using selected method
        3. Applies compression to select important KV entries
        4. Updates req_to_token mapping
        5. Releases unused KV cache slots
        6. Returns the full attention output (no quality loss)
        
        Note: q, k, v have shape [total_tokens, num_heads, head_dim] where
        total_tokens is the sum of all tokens across all sequences in the batch.
        """
        output = super().forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )
        
        layer_id = layer.layer_id
        
        total_tokens = q.shape[0]
        should_compress = self.compressor.should_compress(layer_id, total_tokens)
        
        if should_compress and save_kv_cache:
            self._compress_kv_cache_after_attention(
                q, k, v, layer, forward_batch, **kwargs
            )
        
        return output
    
    def _compress_kv_cache_after_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        **kwargs,
    ):
        """
        Compress KV cache after attention computation.
        
        This method processes each sequence in the batch separately:
        1. Estimates importance for each sequence's tokens
        2. Determines which tokens to keep per sequence
        3. Updates req_to_token mapping for each request
        4. Frees unused cache slots
        
        Note: q, k, v have shape [total_tokens, num_heads, head_dim]
        """
        layer_id = layer.layer_id
        num_heads = layer.tp_q_head_num
        num_kv_heads = layer.tp_k_head_num
        head_dim = layer.head_dim
        
        metadata = self.forward_metadata
        cu_seqlens_q = metadata.cu_seqlens_q
        
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )
        
        batch_size = cu_seqlens_q.shape[0] - 1
        req_pool_indices = forward_batch.req_pool_indices
        
        all_keep_local_indices = []
        all_num_to_keep = []
        all_seq_cache_locs = []
        
        for seq_idx in range(batch_size):
            start_idx = cu_seqlens_q[seq_idx].item()
            end_idx = cu_seqlens_q[seq_idx + 1].item()
            seq_len = end_idx - start_idx
            
            if seq_len == 0:
                all_keep_local_indices.append(None)
                all_num_to_keep.append(0)
                all_seq_cache_locs.append(None)
                continue
            
            q_seq = q[start_idx:end_idx]
            k_seq = k[start_idx:end_idx]
            v_seq = v[start_idx:end_idx]
            
            compressed_k_seq, compressed_v_seq, keep_indices = self._estimate_importance(
                method=self.importance_method,
                q=q_seq,
                k=k_seq,
                v=v_seq,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                scaling=layer.scaling,
            )
            
            seq_cache_loc = cache_loc[start_idx:end_idx]
            
            all_keep_local_indices.append(keep_indices)
            all_num_to_keep.append(keep_indices.shape[0] if keep_indices is not None else 0)
            all_seq_cache_locs.append(seq_cache_loc)
        
        compressed_lens = self._reorganize_kv_cache_and_update_mapping(
            forward_batch=forward_batch,
            req_pool_indices=req_pool_indices,
            cu_seqlens_q=cu_seqlens_q,
            all_keep_local_indices=all_keep_local_indices,
            all_num_to_keep=all_num_to_keep,
            all_seq_cache_locs=all_seq_cache_locs,
            batch_size=batch_size,
        )
        
        forward_batch.kv_compressed_lens = compressed_lens
        
        total_original = cu_seqlens_q[-1].item()
        total_kept = sum(n for n in all_num_to_keep if n > 0)
        total_freed = total_original - total_kept
        
        if total_freed > 0:
            self._update_compression_stats(layer_id, total_original, total_kept)
    
    def _reorganize_kv_cache_and_update_mapping(
        self,
        forward_batch: "ForwardBatch",
        req_pool_indices: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        all_keep_local_indices: list,
        all_num_to_keep: list,
        all_seq_cache_locs: list,
        batch_size: int,
    ) -> list:
        """
        Reorganize KV cache by keeping only important tokens for each sequence.
        
        This method:
        1. Updates req_to_token mapping for each request
        2. Frees unused cache slots
        3. Returns the compressed lengths for each sequence
        
        For page_size == 1, we directly update req_to_token mapping without
        needing to move KV cache data. The mapping points to the kept physical
        locations which can be non-contiguous.
        """
        all_indices_to_free = []
        compressed_lens = []
        
        for seq_idx in range(batch_size):
            keep_local_indices = all_keep_local_indices[seq_idx]
            num_to_keep = all_num_to_keep[seq_idx]
            seq_cache_loc = all_seq_cache_locs[seq_idx]
            
            if keep_local_indices is None or seq_cache_loc is None:
                compressed_lens.append(0)
                continue
            
            seq_len = seq_cache_loc.shape[0]
            if num_to_keep >= seq_len:
                compressed_lens.append(seq_len)
                continue
            
            req_pool_idx = req_pool_indices[seq_idx].item()
            
            kept_cache_locs = seq_cache_loc[keep_local_indices]
            
            discard_mask = torch.ones(seq_len, dtype=torch.bool, device=seq_cache_loc.device)
            discard_mask[keep_local_indices] = False
            freed_cache_locs = seq_cache_loc[discard_mask]
            
            forward_batch.req_to_token_pool.write(
                (req_pool_idx, slice(0, num_to_keep)),
                kept_cache_locs,
            )
            
            compressed_lens.append(num_to_keep)
            all_indices_to_free.append(freed_cache_locs)
        
        if all_indices_to_free:
            all_freed = torch.cat(all_indices_to_free)
            forward_batch.token_to_kv_pool_allocator.free(all_freed)
        
        return compressed_lens
    
    def _estimate_importance(
        self,
        method: ImportanceMethod,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scaling: float,
    ):
        """
        Estimate token importance using selected method and compress KV cache.
        
        Args:
            method: Importance estimation method
            q: Query tensor [seq_len, num_heads, head_dim] for a single sequence
            k: Key tensor [seq_len, num_kv_heads, head_dim] for a single sequence
            v: Value tensor [seq_len, num_kv_heads, head_dim] for a single sequence
            num_heads: Number of query heads
            num_kv_heads: Number of key/value heads
            head_dim: Head dimension
            scaling: Softmax scale
        
        Returns:
            compressed_k: Compressed key tensor
            compressed_v: Compressed value tensor
            keep_indices: Indices of retained tokens
        """
        compressed_k, compressed_v, keep_indices = self.compressor.compress(
            k, v, query=q
        )
        
        return compressed_k, compressed_v, keep_indices
    
    def _update_compression_stats(
        self,
        layer_id: int,
        original_len: int,
        compressed_len: int,
    ):
        """Update compression statistics."""
        self._compression_stats["total_compressed"] += 1
        self._compression_stats["total_freed"] += original_len - compressed_len
        
        if layer_id not in self._compression_stats["layer_stats"]:
            self._compression_stats["layer_stats"][layer_id] = {
                "count": 0,
                "total_original": 0,
                "total_compressed": 0,
            }
        
        stats = self._compression_stats["layer_stats"][layer_id]
        stats["count"] += 1
        stats["total_original"] += original_len
        stats["total_compressed"] += compressed_len
    
    def get_compression_stats(self) -> dict:
        """Get compression statistics."""
        return self._compression_stats.copy()
    
    def reset_compression_stats(self):
        """Reset compression statistics."""
        self._compression_stats = {
            "total_compressed": 0,
            "total_freed": 0,
            "layer_stats": {},
        }


def create_compressed_backend(
    runner,
    compression_config: Optional["CompressionConfig"] = None,
    importance_method: ImportanceMethod = "key_norm",
    **kwargs,
) -> CompressedFlashAttentionBackend:
    """
    Factory function to create a compressed attention backend.
    
    Args:
        runner: ModelRunner instance
        compression_config: Configuration for KV cache compression
        importance_method: Method for estimating token importance
            - "key_norm": Use key norms (fastest, recommended)
            - "lse": Use FlashAttention LSE (fast, approximate)
            - "snapkv": Compute full attention (slow, accurate)
        **kwargs: Additional arguments for FlashAttentionBackend
    
    Returns:
        CompressedFlashAttentionBackend instance
    """
    return CompressedFlashAttentionBackend(
        runner,
        compression_config=compression_config,
        importance_method=importance_method,
        **kwargs,
    )
