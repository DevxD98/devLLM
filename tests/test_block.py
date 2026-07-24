import pytest
import torch
import torch.nn as nn
from jimmylabs.model.config import GPTConfig
from jimmylabs.model.block import TransformerBlock

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

def test_block_shape(dummy_config):
    """
    Test shape-invariant (B,T,C) -> (B,T,C).
    """
    block = TransformerBlock(dummy_config)
    
    B, T, C = 2, 10, dummy_config.n_embd
    x = torch.randn(B, T, C)
    
    out = block(x)
    
    assert out.shape == (B, T, C)

def test_block_residual_paths(dummy_config):
    """
    Test residual paths present. With sublayers zeroed, output ~= input.
    """
    block = TransformerBlock(dummy_config)
    
    # Zero out attention output projection weight & bias
    nn.init.zeros_(block.attn.c_proj.weight)
    if block.attn.c_proj.bias is not None:
        nn.init.zeros_(block.attn.c_proj.bias)
        
    # Zero out MLP final layer weight & bias
    nn.init.zeros_(block.mlp.net[2].weight)
    if block.mlp.net[2].bias is not None:
        nn.init.zeros_(block.mlp.net[2].bias)
        
    B, T, C = 2, 10, dummy_config.n_embd
    x = torch.randn(B, T, C)
    
    out = block(x)
    
    # Since Attn(LN(x)) is zeroed out and FFN(LN(x)) is zeroed out,
    # x = x + 0 + 0 = x
    assert torch.allclose(out, x, atol=1e-5), "Residual paths are broken or not bypassing sublayers!"

def test_block_norm_before_sublayer(dummy_config):
    """
    Test norm applied before each sublayer, not on the residual highway.
    If norm was applied on the residual highway (post-norm), then even with zeroed
    sublayers, the output would be LayerNorm(x) != x.
    Since we already tested out == x in `test_block_residual_paths`, we proved pre-norm.
    To be explicit, we can mock the layer norm to add a massive offset and ensure it 
    doesn't appear in the output when sublayers are zeroed.
    """
    block = TransformerBlock(dummy_config)
    
    # Zero out sublayers again
    nn.init.zeros_(block.attn.c_proj.weight)
    if block.attn.c_proj.bias is not None:
        nn.init.zeros_(block.attn.c_proj.bias)
    nn.init.zeros_(block.mlp.net[2].weight)
    if block.mlp.net[2].bias is not None:
        nn.init.zeros_(block.mlp.net[2].bias)
        
    # Add a massive offset to the LayerNorm biases
    nn.init.constant_(block.ln_1.bias, 1000.0)
    nn.init.constant_(block.ln_2.bias, 1000.0)
    
    B, T, C = 2, 10, dummy_config.n_embd
    x = torch.randn(B, T, C)
    
    out = block(x)
    
    # The output should still exactly equal x, because the 1000.0 offset went 
    # into the sublayers which then got zeroed out.
    # If this was post-norm: x = LN(x + Attn(...)), out would have the 1000.0 offset.
    assert torch.allclose(out, x, atol=1e-5), "Norm is incorrectly applied on the residual path!"
