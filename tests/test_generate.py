import pytest
import torch
from jimmylabs.model.config import GPTConfig
from jimmylabs.model.gpt import GPT
from jimmylabs.inference.generate import generate

@pytest.fixture
def dummy_model():
    config = GPTConfig(
        vocab_size=65, n_layer=2, n_head=2, n_embd=32, block_size=16
    )
    model = GPT(config)
    model.eval()
    return model

def test_generate_output_shape(dummy_model):
    """
    Check that output length is exactly initial_length + max_new_tokens.
    """
    B, T = 2, 4
    idx = torch.randint(0, 65, (B, T))
    
    max_new_tokens = 5
    out = generate(dummy_model, idx, max_new_tokens)
    
    assert out.shape == (B, T + max_new_tokens)
    
def test_generate_temperature_zero_is_deterministic(dummy_model):
    """
    Check that temperature -> 0 is strictly deterministic (greedy decoding).
    """
    B, T = 1, 4
    idx = torch.randint(0, 65, (B, T))
    
    # Generate 5 tokens
    out1 = generate(dummy_model, idx, 5, temperature=0.0)
    out2 = generate(dummy_model, idx, 5, temperature=0.0)
    
    # Should be exactly identical
    assert torch.equal(out1, out2)

def test_generate_top_k_1_is_deterministic(dummy_model):
    """
    Check that top_k=1 is equivalent to greedy decoding.
    """
    B, T = 1, 4
    idx = torch.randint(0, 65, (B, T))
    
    out_greedy = generate(dummy_model, idx, 5, temperature=0.0)
    out_top1 = generate(dummy_model, idx, 5, top_k=1)
    
    assert torch.equal(out_greedy, out_top1)

def test_generate_extremes_dont_crash(dummy_model):
    """
    Check that extreme parameters (temp=100.0, top_p=1.0) don't crash or NaN.
    """
    B, T = 1, 4
    idx = torch.randint(0, 65, (B, T))
    
    out_extreme = generate(dummy_model, idx, 5, temperature=100.0, top_p=1.0)
    assert not torch.isnan(out_extreme.float()).any()
    assert out_extreme.shape == (1, 9)
