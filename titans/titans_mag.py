"""
Titans MAG (Memory as Gate) Architecture

Memory and attention branches combined via gating mechanism.
Now using ultra-fast vectorized memory with analytic gradients.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

# Use fast memory module
from .memory_fast import NeuralMemory, MemoryState
from .attention import SlidingWindowAttention
from .persistent import PersistentMemory


class TitansMAGBlock(nn.Module):
    """Single block with parallel attention and memory branches."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_persistent: int = 8,
        window_size: int = 256,
        memory_depth: int = 2,  # Ignored in fast mode
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        
        # Use fast memory with larger chunks for speed
        self.long_term_memory = NeuralMemory(d_model, chunk_size=64)
        self.persistent_memory = PersistentMemory(n_persistent, d_model)
        self.attention = SlidingWindowAttention(d_model, n_heads, window_size, dropout)
        
        self.attn_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.gate_proj = nn.Linear(d_model * 2, d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(d_model * 4, d_model), nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, memory_state: Optional[MemoryState] = None):
        batch_size = x.shape[0]
        if memory_state is None:
            memory_state = self.long_term_memory.get_initial_state(batch_size, x.device)
        
        persistent = self.persistent_memory(batch_size)
        attn_output = self.attn_norm(self.attention(x, prefix=persistent))
        memory_output, new_state = self.long_term_memory(x, memory_state, return_state=True)
        memory_output = self.memory_norm(memory_output)
        
        gate = torch.sigmoid(self.gate_proj(torch.cat([attn_output, memory_output], dim=-1)))
        combined = gate * attn_output + (1 - gate) * memory_output
        
        x = self.norm1(x + combined)
        x = self.norm2(x + self.ffn(x))
        return x, new_state


class TitansMAG(nn.Module):
    """Full Titans MAG Model."""
    
    def __init__(self, d_model: int, n_layers: int = 12, n_heads: int = 8,
                 n_persistent: int = 8, window_size: int = 256,
                 memory_depth: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TitansMAGBlock(d_model, n_heads, n_persistent, window_size, memory_depth, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, memory_states: Optional[List[MemoryState]] = None):
        if memory_states is None:
            memory_states = [None] * len(self.layers)
        new_states = []
        for i, layer in enumerate(self.layers):
            x, state = layer(x, memory_states[i])
            new_states.append(state)
        return self.final_norm(x), new_states
