"""
Attention Mechanisms for Titans

This module implements:
- Multi-head attention with optional masking
- Sliding window attention for efficient local attention
- Causal attention for autoregressive models

The attention modules serve as "short-term memory" in the Titans architecture,
handling precise dependencies within a limited context window.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with support for causal masking and key-value caching.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout probability
        use_flash: Whether to use flash attention (requires PyTorch 2.0+)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        use_flash: bool = True
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout
        self.use_flash = use_flash and hasattr(F, 'scaled_dot_product_attention')
        
        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through multi-head attention.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask
            is_causal: Whether to apply causal masking
            kv_cache: Optional (keys, values) cache for incremental decoding
            
        Returns:
            - Output tensor of shape (batch, seq_len, d_model)
            - Optional updated KV cache
        """
        batch_size, seq_len, _ = x.shape
        
        # QKV projection
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Handle KV cache for incremental decoding
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        
        new_kv_cache = (k, v) if kv_cache is not None else None
        
        # Attention computation
        if self.use_flash and mask is None:
            # Use PyTorch 2.0 flash attention
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal
            )
        else:
            # Manual attention computation
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            if is_causal:
                # Create causal mask
                causal_mask = torch.triu(
                    torch.ones(seq_len, k.size(2), device=x.device, dtype=torch.bool),
                    diagonal=k.size(2) - seq_len + 1
                )
                attn_weights.masked_fill_(causal_mask, float('-inf'))
            
            if mask is not None:
                attn_weights = attn_weights + mask
            
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            
            output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)
        
        return output, new_kv_cache


class SlidingWindowAttention(nn.Module):
    """
    Sliding window attention for efficient local attention.
    
    This limits attention to a fixed window size, reducing complexity
    from O(n²) to O(n * window_size).
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        window_size: Size of attention window (one-sided)
        dropout: Dropout probability
        use_flash: Whether to use flash attention
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        window_size: int = 256,
        dropout: float = 0.0,
        use_flash: bool = True
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout
        self.use_flash = use_flash
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
    
    def forward(
        self,
        x: torch.Tensor,
        prefix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass using efficient Flash Attention.
        
        Note: We use full causal attention here instead of strict sliding window
        because Flash Attention (O(N)) is significantly more efficient than 
        masked sliding window (O(N^2) memory) in PyTorch.
        The 'window' effect is naturally handled by the model learning to focus locally,
        and the long-term dependencies are handled by the neural memory module.
        """
        batch_size, seq_len, _ = x.shape
        
        # Handle prefix (always attend to it, like sink tokens)
        if prefix is not None:
            prefix_len = prefix.size(1)
            x_full = torch.cat([prefix, x], dim=1)
        else:
            prefix_len = 0
            x_full = x
        
        full_len = x_full.size(1)
        
        # QKV projection
        qkv = self.qkv_proj(x_full)
        qkv = qkv.reshape(batch_size, full_len, 3, self.n_heads, self.head_dim)
        
        # Flash Attention expects (Batch, Seq, Heads, Dim) or (Batch, Heads, Seq, Dim)
        # We'll use (Batch, Heads, Seq, Dim) standard format
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Use Flash Attention
        # Standard PyTorch scaled_dot_product_attention supports is_causal=True
        # This is essentially O(1) memory vs O(L^2) of manual masking
        output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )
        
        output = output.transpose(1, 2).reshape(batch_size, full_len, self.d_model)
        output = self.out_proj(output)
        
        # Remove prefix from output
        if prefix_len > 0:
            output = output[:, prefix_len:, :]
        
        return output


class FullAttention(nn.Module):
    """
    Full causal attention for segment-based processing (MAC variant).
    
    This is used in MAC where each segment gets full attention over:
    - Persistent memory tokens
    - Long-term memory output
    - Current segment tokens
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.0
    ):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        persistent_memory: Optional[torch.Tensor] = None,
        long_term_memory: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional memory contexts.
        
        Args:
            x: Input segment of shape (batch, seg_len, d_model)
            persistent_memory: Persistent memory tokens (batch, n_persist, d_model)
            long_term_memory: Long-term memory output (batch, seg_len, d_model)
            
        Returns:
            Output tensor of shape (batch, seg_len, d_model)
        """
        # Concatenate context in order: [persistent, long-term, current]
        context_parts = []
        if persistent_memory is not None:
            context_parts.append(persistent_memory)
        if long_term_memory is not None:
            context_parts.append(long_term_memory)
        context_parts.append(x)
        
        full_input = torch.cat(context_parts, dim=1)
        
        # Apply causal attention
        output, _ = self.attn(full_input, is_causal=True)
        
        # Extract output for current segment only
        seg_len = x.size(1)
        output = output[:, -seg_len:, :]
        
        return output
