import math
import torch
import torch.nn as nn
from jimmylabs.model.config import GPTConfig

class MultiHeadCausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention block.
    Built explicitly from primitives to demonstrate math: softmax(QK^T / sqrt(d_k)) * V
    Follows SPEC.md §4 and docs/08_ATTENTION.md.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Combined projection for queries, keys, and values
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask to ensure that attention is only applied to the left in the input sequence
        # Shape: (1, 1, T, T) where T is block_size. We use lower triangular matrix.
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor, use_cache: bool = False, past_key_value: tuple = None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        
        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape to (B, n_head, T, C // n_head)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)
            
        present_key_value = (k, v) if use_cache else None
        
        # Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T_k) -> (B, nh, T, T_k)
        # Store internal scores for tests/inspection
        # Q * K^T / sqrt(d_k)
        self._attn_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Apply causal mask: mask out upper triangle (future tokens) with -inf
        T_k = k.size(-2)
        past_T = T_k - T
        self._attn_scores = self._attn_scores.masked_fill(self.bias[:, :, past_T : past_T + T, : T_k] == 0, float('-inf'))
        
        # Softmax over keys (last dimension)
        self._attn_weights = torch.softmax(self._attn_scores, dim=-1)
        self._attn_weights = self.attn_dropout(self._attn_weights)
        
        # Weighted sum of values: (B, nh, T, T_k) x (B, nh, T_k, hs) -> (B, nh, T, hs)
        y = self._attn_weights @ v
        
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection and residual dropout
        y = self.resid_dropout(self.c_proj(y))
        
        if use_cache:
            return y, present_key_value
        return y
