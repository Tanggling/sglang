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
CompressionScheme = Literal["standard", "single_layer_zero_out"]


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
    
    Compression Schemes:
        - "standard": Standard compression with memory release and mapping update
        - "single_layer_zero_out": Only compress first layer, zero out non-kept tokens
          without releasing memory or updating mappings
    
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
            compression_scheme="standard",  # or "single_layer_zero_out"
        )
    """
    
    def __init__(
        self,
        runner,
        compression_config: Optional["CompressionConfig"] = None,
        fa_impl_ver: int = 3, 
        importance_method: ImportanceMethod = "snapkv",
        compression_scheme: CompressionScheme = "standard",
    ):
        super().__init__(runner, fa_impl_ver=fa_impl_ver)
        
        from sglang.srt.layers.attention.kv_compressor import (
            CompressionConfig,
            create_compressor,
        )
        
        self.compression_config = compression_config or CompressionConfig()
        self.compressor = create_compressor(self.compression_config)
        self.importance_method = importance_method
        self.compression_scheme = compression_scheme
                    
        # Store reference to token_to_kv_pool_allocator from runner
        self.token_to_kv_pool_allocator = runner.token_to_kv_pool_allocator
        
        # Store references to k_buffer and v_buffer for each layer
        # These are used to move KV cache data after compression
        self.k_buffer = runner.token_to_kv_pool.k_buffer
        self.v_buffer = runner.token_to_kv_pool.v_buffer
    
        self._compression_stats = {
            "total_compressed": 0,
            "total_freed": 0,
            "layer_stats": {},
        }
        
        # Store original KV values for verification in decode phase
        # Format: {layer_id: {seq_idx: {head_idx: {"k": tensor, "v": tensor, "src_locs": tensor}}}}
        self._verification_data = {}
        self._verification_enabled = True  # Set to False to disable verification overhead
    
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
        
        Note: 
        - q has shape [total_tokens, num_heads * head_dim] (flattened)
        - k, v have shape [total_tokens, num_kv_heads, head_dim]
        """
        layer_id = layer.layer_id
        num_heads = layer.tp_q_head_num
        num_kv_heads = layer.tp_k_head_num
        head_dim = layer.head_dim
        
        # Reshape q from [total_tokens, num_heads * head_dim] to [total_tokens, num_heads, head_dim]
        q = q.view(-1, num_heads, head_dim)
        
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
            # For per-head compression, keep_indices shape is [num_kv_heads, num_tokens_to_keep]
            if keep_indices is not None and keep_indices.dim() == 2:
                num_to_keep = keep_indices.shape[1]
                all_num_to_keep.append(num_to_keep)
            else:
                all_num_to_keep.append(keep_indices.shape[0] if keep_indices is not None else 0)
            all_seq_cache_locs.append(seq_cache_loc)
        
        # Print compression info (same format as SnapKV)
        total_original = cu_seqlens_q[-1].item()
        total_kept = sum(n for n in all_num_to_keep if n > 0)
        print(f"[SnapKV] Compression: {total_original} -> {total_kept} tokens (freed {total_original - total_kept})")
        
        # Print all KV heads' keep indices for first sequence (sorted from small to large)
        if batch_size > 0 and all_keep_local_indices[0] is not None and all_keep_local_indices[0].dim() == 2:
            first_seq_indices = all_keep_local_indices[0]  # [num_kv_heads, num_tokens_to_keep]
            num_kv_heads = first_seq_indices.shape[0]
            for i in range(num_kv_heads):
                head_indices = sorted(first_seq_indices[i].tolist())
                print(f"[SnapKV] {i} KV head keep indices ({len(head_indices)} tokens): {head_indices}")
        
        # Choose compression scheme
        if self.compression_scheme == "single_layer_zero_out":
            # Single-layer compression: only compress first layer, zero out non-kept tokens
            compressed_lens = self._single_layer_compression_zero_out(
                forward_batch=forward_batch,
                req_pool_indices=req_pool_indices,
                cu_seqlens_q=cu_seqlens_q,
                all_keep_local_indices=all_keep_local_indices,
                all_num_to_keep=all_num_to_keep,
                all_seq_cache_locs=all_seq_cache_locs,
                batch_size=batch_size,
                layer_id=layer_id,
            )
        else:
            # Standard compression: reorganize KV cache and update mapping
            compressed_lens = self._reorganize_kv_cache_and_update_mapping(
                forward_batch=forward_batch,
                req_pool_indices=req_pool_indices,
                cu_seqlens_q=cu_seqlens_q,
                all_keep_local_indices=all_keep_local_indices,
                all_num_to_keep=all_num_to_keep,
                all_seq_cache_locs=all_seq_cache_locs,
                batch_size=batch_size,
                layer_id=layer_id,
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
        layer_id: int = None,
    ) -> list:
        """
        Reorganize KV cache by keeping only important tokens for each sequence.
        
        This method:
        1. Moves KV cache data from kept positions to the beginning of seq_cache_loc
        2. Updates req_to_token mapping for each request
        3. Frees unused cache slots
        4. Returns the compressed lengths for each sequence
        
        For per-head compression:
        - keep_local_indices shape: [num_kv_heads, num_tokens_to_keep]
        - Each head has its own compression indices
        - KV cache data is moved per-head
        
        For each sequence and each head, we move:
        - k_buffer[layer_id][head_idx][kept_cache_locs] -> k_buffer[layer_id][head_idx][seq_cache_loc[:num_to_keep]]
        - v_buffer[layer_id][head_idx][kept_cache_locs] -> v_buffer[layer_id][head_idx][seq_cache_loc[:num_to_keep]]
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
            
            # Check if per-head compression (keep_local_indices is 2D)
            is_per_head = keep_local_indices.dim() == 2
            
            if is_per_head:
                # Per-head compression: each head has its own indices
                num_kv_heads = keep_local_indices.shape[0]
                
                # Move KV cache data per-head
                if layer_id is not None:
                    self._move_kv_cache_data_per_head(
                        layer_id=layer_id,
                        keep_local_indices=keep_local_indices,
                        seq_cache_loc=seq_cache_loc,
                        num_to_keep=num_to_keep,
                        seq_idx=seq_idx,
                    )
                
                # Update mapping to point to the new contiguous positions
                if layer_id == 0:
                    forward_batch.req_to_token_pool.write(
                        (req_pool_idx, slice(0, num_to_keep)),
                        seq_cache_loc[:num_to_keep],
                    )
                
                # After moving KV cache data to seq_cache_loc[:num_to_keep],
                # the positions to free are seq_cache_loc[num_to_keep:]
                # because all heads share the same seq_cache_loc
                freed_cache_locs = seq_cache_loc[num_to_keep:]
            else:
                # Original single-index compression
                kept_cache_locs = seq_cache_loc[keep_local_indices]
                
                # Move KV cache data from kept positions to the beginning of seq_cache_loc
                if layer_id is not None:
                    self._move_kv_cache_data(
                        layer_id=layer_id,
                        src_loc=kept_cache_locs,
                        dst_loc=seq_cache_loc[:num_to_keep],
                    )
                
                # After moving, free the positions after num_to_keep
                freed_cache_locs = seq_cache_loc[num_to_keep:]
                
                # Update mapping to point to the new contiguous positions
                if layer_id == 0:
                    forward_batch.req_to_token_pool.write(
                        (req_pool_idx, slice(0, num_to_keep)),
                        seq_cache_loc[:num_to_keep],
                    )
            
            compressed_lens.append(num_to_keep)
            if freed_cache_locs.numel() > 0:
                all_indices_to_free.append(freed_cache_locs)
        
        if all_indices_to_free and layer_id == 0:
            all_freed = torch.cat(all_indices_to_free)
            self.token_to_kv_pool_allocator.free(all_freed)
        
        return compressed_lens
    
    def _move_kv_cache_data_per_head(
        self,
        layer_id: int,
        keep_local_indices: torch.Tensor,
        seq_cache_loc: torch.Tensor,
        num_to_keep: int,
        seq_idx: int = 0,
    ) -> None:
        """
        Move KV cache data per-head for per-head compression.
        
        Args:
            layer_id: The layer ID
            keep_local_indices: Indices to keep per head [num_kv_heads, num_tokens_to_keep]
            seq_cache_loc: Cache locations for this sequence [seq_len]
            num_to_keep: Number of tokens to keep per head
            seq_idx: seq_idx in the batch
        """
        k_buffer = self.k_buffer[layer_id]
        v_buffer = self.v_buffer[layer_id]
        num_kv_heads = keep_local_indices.shape[0]
        
        # Initialize verification data structure for this layer and sequence
        if self._verification_enabled:
            if layer_id not in self._verification_data:
                self._verification_data[layer_id] = {}
            if seq_idx not in self._verification_data[layer_id]:
                self._verification_data[layer_id][seq_idx] = {}
        
        # Track verification results for this layer/seq
        all_passed = True
        
        # For each head, move its KV cache data
        for head_idx in range(num_kv_heads):
            head_keep_indices = keep_local_indices[head_idx]
            src_locs = seq_cache_loc[head_keep_indices]
            dst_locs = seq_cache_loc[:num_to_keep]
            
            # === VERIFICATION: Save original values before moving ===
            original_k_values = k_buffer[src_locs, head_idx].clone()
            original_v_values = v_buffer[src_locs, head_idx].clone()
            
            # Store for decode phase verification
            if self._verification_enabled:
                self._verification_data[layer_id][seq_idx][head_idx] = {
                    "k": original_k_values,
                    "v": original_v_values,
                    "src_locs": src_locs.clone(),
                    "dst_locs": dst_locs.clone(),
                    "num_to_keep": num_to_keep,
                }
            
            # Move key data for this head
            k_buffer[dst_locs, head_idx] = original_k_values
            # Move value data for this head
            v_buffer[dst_locs, head_idx] = original_v_values
            
            # === VERIFICATION: Check if moved values match original ===
            moved_k_values = k_buffer[dst_locs, head_idx]
            moved_v_values = v_buffer[dst_locs, head_idx]
            
            k_match = torch.allclose(original_k_values, moved_k_values, rtol=1e-5, atol=1e-5)
            v_match = torch.allclose(original_v_values, moved_v_values, rtol=1e-5, atol=1e-5)
            
            if not (k_match and v_match):
                all_passed = False
        
        # Print summary for this layer/seq
        status = "PASSED" if all_passed else "FAILED"
        print(f"[PREFILL] Layer {layer_id}, Seq {seq_idx}: {status} (num_to_keep={num_to_keep}, heads={num_kv_heads})")
        
        # Print per-head token selection for debugging
        if seq_idx == 0:  # Only print for first sequence to avoid too much output
            print(f"[PER-HEAD COMPRESSION] Layer {layer_id}:")
            for head_idx in range(min(2, num_kv_heads)):  # Only print first 2 heads
                head_keep_indices = keep_local_indices[head_idx]
                sorted_indices = sorted(head_keep_indices.tolist())
                print(f"  Head {head_idx} keeps tokens: {sorted_indices[:min(10, len(sorted_indices))]}")
    
    def _move_kv_cache_data(
        self,
        layer_id: int,
        src_loc: torch.Tensor,
        dst_loc: torch.Tensor,
    ) -> None:
        """
        Move KV cache data from src_loc to dst_loc for a specific layer.
        
        Args:
            layer_id: The layer ID
            src_loc: Source token locations [num_tokens]
            dst_loc: Destination token locations [num_tokens]
        """
        # Get the KV buffer for this layer
        k_buffer = self.k_buffer[layer_id]
        v_buffer = self.v_buffer[layer_id]
        
        # Move key data
        k_buffer[dst_loc] = k_buffer[src_loc]
        
        # Move value data
        v_buffer[dst_loc] = v_buffer[src_loc]
    
    def _single_layer_compression_zero_out(
        self,
        forward_batch: "ForwardBatch",
        req_pool_indices: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        all_keep_local_indices: list,
        all_num_to_keep: list,
        all_seq_cache_locs: list,
        batch_size: int,
        layer_id: int = None,
    ) -> list:
        """
        Single-layer compression: only compress the first layer.
        
        This method:
        1. For layer_id == 0: zero out the KV cache for non-kept tokens
        2. Does NOT free any cache slots
        3. Does NOT update req_to_token mapping
        4. Does NOT modify sequence lengths
        5. Returns None (no compression information)
        
        Args:
            forward_batch: Forward batch information
            req_pool_indices: Request pool indices
            cu_seqlens_q: Cumulative sequence lengths for queries
            all_keep_local_indices: Indices to keep for each sequence
            all_num_to_keep: Number of tokens to keep for each sequence
            all_seq_cache_locs: Cache locations for each sequence
            batch_size: Batch size
            layer_id: Layer ID
        
        Returns:
            None (no compression information)
        """
        # Only process layer 0
        if layer_id != 0:
            return None
        
        # Process each sequence in the batch
        for seq_idx in range(batch_size):
            keep_local_indices = all_keep_local_indices[seq_idx]
            num_to_keep = all_num_to_keep[seq_idx]
            seq_cache_loc = all_seq_cache_locs[seq_idx]
            
            if keep_local_indices is None or seq_cache_loc is None:
                continue
            
            seq_len = seq_cache_loc.shape[0]
            if num_to_keep >= seq_len:
                continue
            
            # Check if per-head compression
            is_per_head = keep_local_indices.dim() == 2
            
            if is_per_head:
                # Per-head compression: zero out non-kept tokens for each head
                num_kv_heads = keep_local_indices.shape[0]
                k_buffer = self.k_buffer[layer_id]
                v_buffer = self.v_buffer[layer_id]
                
                for head_idx in range(num_kv_heads):
                    head_keep_indices = keep_local_indices[head_idx]
                    
                    # Create a mask for tokens to zero out
                    zero_out_mask = torch.ones(seq_len, dtype=torch.bool, device=seq_cache_loc.device)
                    zero_out_mask[head_keep_indices] = False
                    
                    # Get cache locations to zero out
                    zero_out_cache_locs = seq_cache_loc[zero_out_mask]
                    
                    # Zero out the KV cache for this head
                    k_buffer[zero_out_cache_locs, head_idx] = 0
                    v_buffer[zero_out_cache_locs, head_idx] = 0
                
                print(f"[SINGLE_LAYER_COMPRESSION] Layer {layer_id}, Seq {seq_idx}: zeroed out {seq_len - num_to_keep} tokens per head (per-head compression)")
            else:
                # Single-index compression: zero out non-kept tokens
                # Create a mask for tokens to zero out
                zero_out_mask = torch.ones(seq_len, dtype=torch.bool, device=seq_cache_loc.device)
                zero_out_mask[keep_local_indices] = False
                
                # Get cache locations to zero out
                zero_out_cache_locs = seq_cache_loc[zero_out_mask]
                
                # Zero out the KV cache
                k_buffer = self.k_buffer[layer_id]
                v_buffer = self.v_buffer[layer_id]
                k_buffer[zero_out_cache_locs] = 0
                v_buffer[zero_out_cache_locs] = 0
                
                print(f"[SINGLE_LAYER_COMPRESSION] Layer {layer_id}, Seq {seq_idx}: zeroed out {len(zero_out_cache_locs)} tokens (single-index compression)")
        
        return None
    
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
    
    def verify_decode_kv_cache(
        self,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
    ):
        """
        Verify that KV cache values in decode phase match the original compressed values.
        
        This method is called at the beginning of each decode forward pass to verify
        that the KV cache values accessed through page_table match the values that
        were saved during compression.
        
        Args:
            layer: The attention layer
            forward_batch: The forward batch containing metadata
        """
        if not self._verification_enabled:
            return
        
        layer_id = layer.layer_id
        if layer_id not in self._verification_data:
            return
        
        metadata = self.forward_metadata
        cache_seqlens = metadata.cache_seqlens_int32
        
        # Get KV buffer for this layer
        k_buffer = self.k_buffer[layer_id]
        v_buffer = self.v_buffer[layer_id]
        
        all_passed = True
        
        # Verify each sequence
        for seq_idx, seq_data in self._verification_data[layer_id].items():
            seq_len = cache_seqlens[seq_idx].item()
            req_pool_idx = forward_batch.req_pool_indices[seq_idx].item()
            
            # Read the mapping from req_to_token_pool
            cache_locs = forward_batch.req_to_token_pool.req_to_token[req_pool_idx, :seq_len]
            
            # Verify each head
            for head_idx, head_data in seq_data.items():
                original_k = head_data["k"]
                original_v = head_data["v"]
                expected_num_to_keep = head_data["num_to_keep"]
                
                # The first num_to_keep cache_locs should be used
                actual_cache_locs = cache_locs[:expected_num_to_keep]
                
                # Read KV values from these cache locations
                actual_k = k_buffer[actual_cache_locs, head_idx]
                actual_v = v_buffer[actual_cache_locs, head_idx]
                
                # Compare with original values
                k_match = torch.allclose(original_k, actual_k, rtol=1e-5, atol=1e-5)
                v_match = torch.allclose(original_v, actual_v, rtol=1e-5, atol=1e-5)
                
                if not (k_match and v_match):
                    all_passed = False
        
        # Print summary for this layer
        status = "PASSED" if all_passed else "FAILED"
        print(f"[DECODE] Layer {layer_id}: {status}")
        
        # Clear verification data after last layer's decode to avoid memory overhead
        # Only clear after all layers have been verified
        max_layer_id = max(self._verification_data.keys()) if self._verification_data else 0
        if layer_id == max_layer_id:
            self._verification_data = {}
    
    def forward_decode(
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
        Forward pass for decode mode with KV cache verification.
        
        This method:
        1. Verifies that KV cache values match the original compressed values
        2. Calls parent's forward_decode for actual computation
        """
        # Verify KV cache values before computation
        self.verify_decode_kv_cache(layer, forward_batch)
        
        # Call parent's forward_decode
        return super().forward_decode(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )


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
