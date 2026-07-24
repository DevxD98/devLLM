import pytest
import torch
import torch.nn as nn
import math
from jimmylabs.model.config import GPTConfig
from jimmylabs.model.attention import MultiHeadCausalSelfAttention

@pytest.fixture
def dummy_config():
    return GPTConfig(
        vocab_size=65,
        n_layer=4,
        n_head=4,
        n_embd=128,
        block_size=16,
        dropout=0.0, # Disable dropout for deterministic testing
        weight_tying=True
    )

def test_attention_shape_and_internal_scores(dummy_config):
    """
    Test shape (B,T,C) -> (B,T,C) and internal scores (B,h,T,T).
    """
    attn = MultiHeadCausalSelfAttention(dummy_config)
    
    B, T, C = 2, 10, dummy_config.n_embd
    x = torch.randn(B, T, C)
    
    out = attn(x)
    
    assert out.shape == (B, T, C)
    
    # Internal scores before softmax should be exposed for this test
    # (they are temporarily bound to self._attn_scores)
    assert hasattr(attn, '_attn_scores')
    scores = attn._attn_scores
    h = dummy_config.n_head
    assert scores.shape == (B, h, T, T)

def test_attention_softmax_rows_sum_to_one(dummy_config):
    """
    Test that softmax rows sum to 1.0 (within tol).
    """
    attn = MultiHeadCausalSelfAttention(dummy_config)
    
    B, T, C = 2, 10, dummy_config.n_embd
    x = torch.randn(B, T, C)
    
    attn(x)
    
    weights = attn._attn_weights
    
    # Sum over keys (last dimension)
    row_sums = weights.sum(dim=-1)
    
    # Check that all row sums are close to 1.0
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

def test_attention_causal_mask(dummy_config):
    """
    Test CAUSAL MASK: position i has exactly zero attention weight on any j > i.
    """
    attn = MultiHeadCausalSelfAttention(dummy_config)
    
    B, T, C = 2, 10, dummy_config.n_embd
    x = torch.randn(B, T, C)
    
    attn(x)
    
    weights = attn._attn_weights # shape (B, h, T, T)
    
    for b in range(B):
        for head in range(dummy_config.n_head):
            for i in range(T):
                for j in range(T):
                    if j > i:
                        assert weights[b, head, i, j].item() == 0.0, f"Leaked information! pos {i} attends to future pos {j}"
                    else:
                        # Should be generally non-zero unless numerical issues
                        pass

def test_attention_scaling(dummy_config):
    """
    Test that √d_k scaling is applied.
    """
    # Create an attention module and mock its linear projection so we know exact Q and K
    attn = MultiHeadCausalSelfAttention(dummy_config)
    
    B, T, C = 1, 2, dummy_config.n_embd
    h = dummy_config.n_head
    d_k = C // h
    
    x = torch.randn(B, T, C)
    
    # Overwrite the c_attn weights to be identity-like so we can predictably compute QK^T
    # We want Q=x, K=x, V=x for the first head.
    nn.init.normal_(attn.c_attn.weight, mean=0.0, std=1.0)
    nn.init.zeros_(attn.c_attn.bias)
    
    out = attn(x)
    
    # The scores are QK^T / sqrt(d_k)
    # Check that variance of scores is roughly what it should be
    # or just manually extract Q and K and verify the division
    
    q, k, _ = attn.c_attn(x).split(C, dim=2)
    q = q.view(B, T, h, d_k).transpose(1, 2)
    k = k.view(B, T, h, d_k).transpose(1, 2)
    
    manual_raw_scores = q @ k.transpose(-2, -1)
    manual_scaled_scores = manual_raw_scores / math.sqrt(d_k)
    
    # We must mask it to match what the forward pass did
    mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T) == 0
    manual_scaled_scores = manual_scaled_scores.masked_fill(mask, float('-inf'))
    
    assert torch.allclose(attn._attn_scores, manual_scaled_scores, atol=1e-5)
