"""
KV Cache Compression Module for SGLang

This module provides various compression algorithms for KV cache during prefill phase.
The compression is performed immediately after each attention layer's computation,
selectively retaining important KV cache entries and releasing the rest.

Key Features:
1. Multiple compression algorithms (importance-based, clustering, random sampling)
2. Integration with SGLang's attention backend
3. Configurable compression ratio and retention strategy
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

import logging

logger = logging.getLogger(__name__)



@dataclass
class CompressionConfig:
    """Configuration for KV cache compression"""
    
    enabled: bool = True
    
    compression_ratio: float = 0.5
    
    compression_method: str = "snapkv"
    
    window_size: int = 64
    
    min_tokens_to_keep: int = 32
    
    compress_every_layer: bool = True
    
    layers_to_compress: Optional[List[int]] = None
    
    retain_first_n_tokens: int = 0
    
    importance_metric: str = "attention_score"
    
    num_clusters: Optional[int] = None
    
    temperature: float = 1.0
    
    random_seed: Optional[int] = None


class BaseKVCompressor(ABC):
    """Base class for KV cache compressors"""
    
    def __init__(self, config: CompressionConfig):
        self.config = config
    
    @abstractmethod
    def compress(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None,
        layer_id: int = 0,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compress KV cache.
        
        Args:
            k: Key tensor [seq_len, num_kv_heads, head_dim]
            v: Value tensor [seq_len, num_kv_heads, head_dim]
            attention_scores: Attention scores [num_heads, seq_len, seq_len] (optional)
            layer_id: Layer index
        
        Returns:
            compressed_k: Compressed key tensor
            compressed_v: Compressed value tensor
            keep_indices: Indices of retained tokens
        """
        raise NotImplementedError
    
    def should_compress(self, layer_id: int, seq_len: int) -> bool:
        """Check if compression should be applied"""
        if not self.config.enabled:
            return False
        
        if seq_len <= self.config.min_tokens_to_keep:
            return False
        
        if self.config.compress_every_layer:
            return True
        
        if self.config.layers_to_compress is not None:
            return layer_id in self.config.layers_to_compress
        
        return layer_id == 0


class ImportanceBasedCompressor(BaseKVCompressor):
    """
    Importance-based KV cache compressor.
    
    Selects tokens to retain based on their importance scores,
    which can be derived from attention scores or other metrics.
    """
    
    def __init__(self, config: CompressionConfig):
        super().__init__(config)
        logger.info(f"use ImportanceBasedCompressor with importance_metric={self.config.importance_metric}")
    
    def compute_importance_scores(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute importance scores for each token.
        
        Args:
            k: Key tensor [seq_len, num_kv_heads, head_dim]
            v: Value tensor [seq_len, num_kv_heads, head_dim]
            attention_scores: Pre-computed attention scores [num_heads, seq_len, seq_len]
        
        Returns:
            importance: Importance scores [seq_len]
        """
        seq_len = k.shape[0]
        
        if attention_scores is not None:
            if attention_scores.dim() == 3:
                importance = attention_scores.mean(dim=(0, 1))
                importance = importance.mean(dim=0)
            else:
                importance = attention_scores.mean(dim=-1)
        else:
            k_norm = k.norm(dim=-1).mean(dim=-1)
            v_norm = v.norm(dim=-1).mean(dim=-1)
            importance = (k_norm + v_norm) / 2
        
        return importance
    
    def compress(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None,
        layer_id: int = 0,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compress KV cache based on importance scores."""
        seq_len = k.shape[0]
        num_tokens_to_keep = max(
            self.config.min_tokens_to_keep,
            int(seq_len * (1 - self.config.compression_ratio))
        )
        
        importance = self.compute_importance_scores(k, v, attention_scores)
        
        _, sorted_indices = torch.sort(importance, descending=True)
        
        keep_indices = sorted_indices[:num_tokens_to_keep]
        keep_indices = torch.sort(keep_indices)[0]
        
        if self.config.retain_first_n_tokens > 0:
            first_n = torch.arange(
                min(self.config.retain_first_n_tokens, seq_len),
                device=k.device
            )
            keep_indices = torch.unique(torch.cat([first_n, keep_indices]))
        
        if self.config.window_size > 0:
            window_start = max(0, seq_len - self.config.window_size)
            window_indices = torch.arange(window_start, seq_len, device=k.device)
            keep_indices = torch.unique(torch.cat([keep_indices, window_indices]))
        
        compressed_k = k[keep_indices]
        compressed_v = v[keep_indices]
        
        return compressed_k, compressed_v, keep_indices


class ClusteringCompressor(BaseKVCompressor):
    """
    Clustering-based KV cache compressor.
    
    Groups similar tokens into clusters and retains representative tokens
    from each cluster.
    """
    
    def __init__(self, config: CompressionConfig):
        super().__init__(config)
        logger.info(f"use ClusteringCompressor with num_clusters={self.config.num_clusters}")
    
    def compress(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None,
        layer_id: int = 0,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compress KV cache using clustering."""
        seq_len = k.shape[0]
        num_clusters = self.config.num_clusters or max(
            self.config.min_tokens_to_keep,
            int(seq_len * (1 - self.config.compression_ratio))
        )
        
        kv_combined = torch.cat([k, v], dim=-1)
        kv_flat = kv_combined.view(seq_len, -1)
        
        cluster_centers = self._kmeans_plusplus_init(kv_flat, num_clusters)
        
        for _ in range(10):
            distances = torch.cdist(kv_flat, cluster_centers)
            cluster_assignments = distances.argmin(dim=1)
            
            new_centers = torch.zeros_like(cluster_centers)
            for i in range(num_clusters):
                mask = cluster_assignments == i
                if mask.sum() > 0:
                    new_centers[i] = kv_flat[mask].mean(dim=0)
                else:
                    new_centers[i] = cluster_centers[i]
            cluster_centers = new_centers
        
        distances = torch.cdist(kv_flat, cluster_centers)
        cluster_assignments = distances.argmin(dim=1)
        
        keep_indices = []
        for i in range(num_clusters):
            mask = cluster_assignments == i
            if mask.sum() > 0:
                indices = torch.where(mask)[0]
                cluster_k = k[indices]
                center = cluster_centers[i]
                center_k = center[:k.shape[-1]]
                dists = ((cluster_k - center_k) ** 2).sum(dim=-1).sum(dim=-1)
                closest_idx = indices[dists.argmin()]
                keep_indices.append(closest_idx)
        
        keep_indices = torch.stack(keep_indices)
        keep_indices = torch.sort(keep_indices)[0]
        
        if self.config.retain_first_n_tokens > 0:
            first_n = torch.arange(
                min(self.config.retain_first_n_tokens, seq_len),
                device=k.device
            )
            keep_indices = torch.unique(torch.cat([first_n, keep_indices]))
        
        if self.config.window_size > 0:
            window_start = max(0, seq_len - self.config.window_size)
            window_indices = torch.arange(window_start, seq_len, device=k.device)
            keep_indices = torch.unique(torch.cat([keep_indices, window_indices]))
        
        compressed_k = k[keep_indices]
        compressed_v = v[keep_indices]
        
        return compressed_k, compressed_v, keep_indices
    
    def _kmeans_plusplus_init(self, data: torch.Tensor, k: int) -> torch.Tensor:
        """Initialize cluster centers using k-means++ algorithm."""
        n = data.shape[0]
        centers = torch.zeros(k, data.shape[1], device=data.device)
        
        idx = torch.randint(0, n, (1,), device=data.device)
        centers[0] = data[idx]
        
        for i in range(1, k):
            distances = torch.cdist(data, centers[:i])
            min_distances = distances.min(dim=1)[0]
            probabilities = min_distances / min_distances.sum()
            idx = torch.multinomial(probabilities, 1)
            centers[i] = data[idx]
        
        return centers


class SnapKVStyleCompressor(BaseKVCompressor):
    """
    SnapKV-style KV cache compressor with per-head compression.
    
    Implements the compression strategy from SnapKV paper:
    - Compute attention scores for recent window
    - Select important tokens based on attention patterns for each head
    - Apply maxpool to reduce noise
    - Each attention head has its own compression indices
    
    Query is expected to contain per-query-head activations for the current
    observation window. The input query shape is
    [min(window_size, extend_len), num_heads, head_dim].
    """
    
    def __init__(self, config: CompressionConfig):
        super().__init__(config)
        self.kernel_size = 5
        self.pooling = "maxpool"
        self.apply_causal_mask = True
        logger.info(f"use SnapKVStyleCompressor with kernel_size={self.kernel_size}, pooling={self.pooling}")
    
    def compress(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None,
        layer_id: int = 0,
        query: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compress KV cache using SnapKV-style algorithm with per-head compression.
        
        Args:
            k: Key tensor [seq_len, num_kv_heads, head_dim]
            v: Value tensor [seq_len, num_kv_heads, head_dim]
            query: Query tensor [window_tokens, num_heads, head_dim] for the
                   current extend observation window.
        
        Returns:
            compressed_k: None (not used, gather done externally)
            compressed_v: None (not used, gather done externally)
            keep_indices: Indices to keep per head [num_kv_heads, num_tokens_to_keep]
        """
        seq_len = k.shape[0]
        num_kv_heads = k.shape[1]
        kv_head_dim = k.shape[2]
        
        num_tokens_to_keep = max(
            self.config.min_tokens_to_keep,
            int(seq_len * (1 - self.config.compression_ratio))
        )

        num_tokens_to_keep = min(num_tokens_to_keep, seq_len)
        
        if query is not None and query.shape[0] < 1:
            raise ValueError("SnapKV compression requires at least one query token")

        window_size = min(self.config.window_size, seq_len)

        # Ensure window fits within num_tokens_to_keep
        if window_size > num_tokens_to_keep:
            window_size = num_tokens_to_keep

        if attention_scores is None and query is not None:
            # query is [window_tokens, num_heads, head_dim]
            q_head_dim = query.shape[2]
            num_q_heads = query.shape[1]
            if num_q_heads % num_kv_heads != 0:
                raise ValueError(
                    f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
                )
            kv_group_size = num_q_heads // num_kv_heads
            
            # Use the provided window or the query length (whichever is smaller)
            actual_window = min(window_size, query.shape[0])
            q_window = query[-actual_window:]
            k_prefix = k[:-actual_window] if actual_window < seq_len else k[:0]
            
            if k_prefix.shape[0] == 0:
                keep_indices = torch.arange(seq_len, device=k.device).unsqueeze(0).expand(num_kv_heads, -1)
            else:
                q_grouped = q_window.view(actual_window, num_kv_heads, kv_group_size, q_head_dim)
                q_grouped = q_grouped.permute(1, 2, 0, 3).contiguous()

                # k_t: [num_kv_heads, prefix_len, head_dim]
                k_t = k_prefix.transpose(0, 1).contiguous()

                # Compute per-query-head logits before any reduction.
                # Result: [num_kv_heads, kv_group_size, window_size, prefix_len]
                attn_weights = torch.einsum(
                    "hgtd,hpd->hgtp", q_grouped, k_t
                ) / math.sqrt(q_head_dim)

                if self.apply_causal_mask:
                    k_window = k[-actual_window:]
                    # k_window_t: [num_kv_heads, window_size, head_dim]
                    k_window_t = k_window.transpose(0, 1).contiguous()
                    
                    # Window attention per query head.
                    attn_weights_window = torch.einsum(
                        "hgtd,hwd->hgtw", q_grouped, k_window_t
                    ) / math.sqrt(q_head_dim)

                    q_window_len = actual_window
                    k_window_len = actual_window
                    # Causal mask for [window_size, window_size]
                    causal_mask_base = torch.triu(
                        torch.full((q_window_len, k_window_len), float('-inf'), device=k.device, dtype=attn_weights_window.dtype),
                        diagonal=1 + k_window_len - q_window_len
                    )
                    attn_weights_window = attn_weights_window + causal_mask_base.unsqueeze(0).unsqueeze(0)
                    
                    attn_weights_full = torch.cat([attn_weights, attn_weights_window], dim=-1)
                    attention_scores = F.softmax(attn_weights_full, dim=-1)
                    attn_weights_prefix = attention_scores[:, :, :, :k_prefix.shape[0]]
                else:
                    attention_scores = F.softmax(attn_weights, dim=-1)
                    attn_weights_prefix = attention_scores
                
                # attn_weights_prefix: [num_kv_heads, kv_group_size, window_size, prefix_len]
                # Sum softmaxed per-token/per-query-head scores into KV-head scores.
                attn_weights_sum = attn_weights_prefix.sum(dim=(1, 2))
                
                if attn_weights_sum.shape[-1] > self.kernel_size:
                    if self.pooling == 'maxpool':
                        attn_cache = F.max_pool1d(
                            attn_weights_sum.unsqueeze(0),
                            kernel_size=self.kernel_size,
                            padding=self.kernel_size // 2,
                            stride=1
                        ).squeeze(0)
                    else:
                        attn_cache = F.avg_pool1d(
                            attn_weights_sum.unsqueeze(0),
                            kernel_size=self.kernel_size,
                            padding=self.kernel_size // 2,
                            stride=1
                        ).squeeze(0)
                else:
                    attn_cache = attn_weights_sum
                
                _, indices = attn_cache.topk(num_tokens_to_keep - actual_window, dim=-1)
                indices = torch.sort(indices, dim=-1).values
                
                window_indices = torch.arange(
                    seq_len - actual_window, seq_len,
                    device=k.device
                ).unsqueeze(0).expand(num_kv_heads, -1)
                
                keep_indices = torch.cat([indices, window_indices], dim=-1)
        else:
            keep_indices = torch.arange(seq_len, device=k.device).unsqueeze(0).expand(num_kv_heads, -1)
        
        return None, None, keep_indices


def create_compressor(config: CompressionConfig) -> BaseKVCompressor:
    """Factory function to create a compressor based on configuration."""
    compressors = {
        "importance": ImportanceBasedCompressor,
        "clustering": ClusteringCompressor,
        "snapkv": SnapKVStyleCompressor,
    }
    
    compressor_class = compressors.get(config.compression_method)
    if compressor_class is None:
        raise ValueError(f"Unknown compression method: {config.compression_method}")
    
    return compressor_class(config)
