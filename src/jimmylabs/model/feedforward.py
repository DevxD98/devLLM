import torch
import torch.nn as nn
from jimmylabs.model.config import GPTConfig

class FeedForward(nn.Module):
    """
    A simple linear feed-forward network with GELU activation.
    Linear(C, 4C) -> GELU -> Linear(4C, C) -> Dropout
    Aligns with SPEC.md §4 and docs/09_TRANSFORMER.md.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T, C)
        Returns:
            Tensor of shape (B, T, C)
        """
        return self.net(x)
