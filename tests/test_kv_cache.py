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

def test_kv_cache_equivalence(dummy_model):
    """
    Generate with and without KV cache should produce EXACTLY the same tokens.
    """
    B, T = 1, 4
    idx = torch.randint(0, 65, (B, T))
    
    # Use greedy decoding to avoid sampling noise
    max_new_tokens = 20
    
    out_naive = generate(dummy_model, idx, max_new_tokens, temperature=0.0, use_cache=False)
    out_cached = generate(dummy_model, idx, max_new_tokens, temperature=0.0, use_cache=True)
    
    assert torch.equal(out_naive, out_cached), "KV cache generation does not match naive generation"
