import torch
import torch.nn as nn
from jimmylabs.model.config import GPTConfig
from jimmylabs.model.attention import MultiHeadCausalSelfAttention
from jimmylabs.model.feedforward import FeedForward

class TransformerBlock(nn.Module):
    """
    Transformer pre-norm block:
    x = x + Attn(LN(x))
    x = x + FFN(LN(x))
    Follows SPEC.md §2 and docs/09_TRANSFORMER.md.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadCausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)
        
    def forward(self, x: torch.Tensor, use_cache: bool = False, past_key_value: tuple = None):
        """
        Args:
            x: Tensor of shape (B, T, C)
            use_cache: whether to return KV cache
            past_key_value: tuple of previous (K, V)
        Returns:
            Tensor of shape (B, T, C) or (Tensor, present_key_value)
        """
        attn_out = self.attn(self.ln_1(x), use_cache=use_cache, past_key_value=past_key_value)
        if use_cache:
            attn_out, present_key_value = attn_out
            
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        
        if use_cache:
            return x, present_key_value
        return x
