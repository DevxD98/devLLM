import torch
import torch.nn as nn
from jimmylabs.model.config import GPTConfig
from jimmylabs.model.embedding import TokenAndPositionEmbedding
from jimmylabs.model.block import TransformerBlock
import math

class GPT(nn.Module):
    """
    The full JimmyLabs GPT model.
    Follows SPEC.md §2 and docs/10_GPT_ARCHITECTURE.md.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = TokenAndPositionEmbedding(config),
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying (ADR-0003)
        if config.weight_tying:
            self.lm_head.weight = self.transformer.wte.token_embedding.weight
            
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
                
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        Args:
            idx: (B, T) tensor of integer token indices
            targets: (B, T) tensor of integer token indices (optional)
        Returns:
            logits: (B, T, V) tensor
            loss: cross entropy loss (scalar) if targets provided, else None
        """
        # Embeddings
        x = self.transformer.wte(idx)
        
        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)
            
        # Final LayerNorm
        x = self.transformer.ln_f(x)
        
        # LM Head
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # Flatten for cross_entropy: (B*T, V) and (B*T)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss
