import torch
import torch.nn as nn
from jimmylabs.model.config import GPTConfig

class TokenAndPositionEmbedding(nn.Module):
    """
    Token and learned positional embeddings combined.
    (B, T) int64 -> (B, T, C).
    Aligns with SPEC.md §4 and docs/07_EMBEDDINGS.md.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.block_size = config.block_size
        
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.size()
        
        # Ensure sequence length is within block_size bounds
        if t > self.block_size:
            raise ValueError(f"Cannot forward sequence of length {t}, block size is only {self.block_size}")
            
        # Token embeddings: (B, T) -> (B, T, C)
        tok_emb = self.token_embedding(idx)
        
        # Position embeddings: (T) -> (T, C)
        # We generate positions [0, 1, ..., T-1] and get their embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        pos_emb = self.position_embedding(pos)
        
        # Sum them up and apply dropout
        # Broadcasts pos_emb (T, C) -> (B, T, C) over the batch dimension
        x = self.drop(tok_emb + pos_emb)
        
        return x
