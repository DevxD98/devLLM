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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T, C)
        Returns:
            Tensor of shape (B, T, C)
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
