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
    
    retain_first_n_tokens: int = 16
    
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
    SnapKV-style KV cache compressor.
    
    Implements the compression strategy from SnapKV paper:
    - Compute attention scores for recent window
    - Select important tokens based on attention patterns
    - Apply pooling to reduce noise
    """
    
    def __init__(self, config: CompressionConfig):
        super().__init__(config)
        self.kernel_size = 5
        logger.info(f"use SnapKVStyleCompressor with kernel_size={self.kernel_size}")
    
    def compress(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None,
        layer_id: int = 0,
        query: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compress KV cache using SnapKV-style algorithm."""
        seq_len = k.shape[0]
        num_kv_heads = k.shape[1]
        kv_head_dim = k.shape[2]
        
        num_tokens_to_keep = max(
            self.config.min_tokens_to_keep,
            int(seq_len * (1 - self.config.compression_ratio))
        )
        
        window_size = min(self.config.window_size, seq_len)
        
        if attention_scores is None and query is not None:
            print(f"Query shape: {query.shape}, K shape: {k.shape}")
            num_heads = query.shape[1]
            q_head_dim = query.shape[2]
            kv_group_num = num_heads // num_kv_heads
            
            q_window = query[-window_size:]
            k_prefix = k[:-window_size]
            
            if k_prefix.shape[0] == 0:
                importance_pooled = torch.ones(seq_len, device=k.device)
            else:
                q_t = q_window.transpose(0, 1)
                k_t = k_prefix.transpose(0, 1)
                
                if kv_group_num > 1:
                    k_t = k_t.repeat_interleave(kv_group_num, dim=0)
                
                attn_weights = torch.matmul(q_t, k_t.transpose(-2, -1)) / math.sqrt(q_head_dim)
                
                attention_scores = F.softmax(attn_weights, dim=-1)
                
                if attention_scores.dim() == 3:
                    importance = attention_scores.mean(dim=0)
                else:
                    importance = attention_scores
                
                importance = importance.mean(dim=0)
                
                if importance.shape[0] > self.kernel_size:
                    importance_pooled = F.avg_pool1d(
                        importance.unsqueeze(0).unsqueeze(0),
                        kernel_size=self.kernel_size,
                        stride=1,
                        padding=self.kernel_size // 2
                    ).squeeze()
                else:
                    importance_pooled = importance
                
                full_importance = torch.zeros(seq_len - window_size, device=k.device)
                full_importance[:importance_pooled.shape[0]] = importance_pooled
                importance_pooled = torch.cat([
                    full_importance,
                    torch.zeros(window_size, device=k.device)
                ])
        else:
            importance_pooled = torch.ones(seq_len, device=k.device)
        
        window_importance = torch.zeros(seq_len, device=k.device)
        window_importance[-window_size:] = float('inf')
        
        combined_importance = importance_pooled + window_importance
        
        _, sorted_indices = torch.sort(combined_importance, descending=True)
        keep_indices = sorted_indices[:num_tokens_to_keep]
        keep_indices = torch.sort(keep_indices)[0]
        
        if self.config.retain_first_n_tokens > 0:
            first_n = torch.arange(
                min(self.config.retain_first_n_tokens, seq_len),
                device=k.device
            )
            keep_indices = torch.unique(torch.cat([first_n, keep_indices]))
        
        compressed_k = k[keep_indices]
        compressed_v = v[keep_indices]
        
        return compressed_k, compressed_v, keep_indices


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
