import pytest
import torch
from jimmylabs.model.config import GPTConfig
from jimmylabs.model.feedforward import FeedForward

@pytest.fixture
def dummy_config():
    return GPTConfig(
        vocab_size=65,
        n_layer=4,
        n_head=4,
        n_embd=128,
        block_size=128,
        dropout=0.0, # Disable dropout for deterministic testing
        weight_tying=True
    )

def test_feedforward_shape(dummy_config):
    """
    Test that (B, T, C) -> (B, T, C) shape is preserved.
    """
    ffn = FeedForward(dummy_config)
    
    batch_size = 4
    seq_len = 10
    
    x = torch.randn(batch_size, seq_len, dummy_config.n_embd)
    out = ffn(x)
    
    assert out.shape == (batch_size, seq_len, dummy_config.n_embd)

def test_feedforward_nonlinearity(dummy_config):
    """
    Test that a NONLINEARITY is present.
    Output is not an affine function of input: FFN(2x) != 2 * FFN(x)
    """
    ffn = FeedForward(dummy_config)
    
    # Needs to be in eval mode if we had dropout, but we set it to 0.0 anyway
    ffn.eval()
    
    x = torch.randn(2, 5, dummy_config.n_embd)
    
    out1 = ffn(2 * x)
    out2 = 2 * ffn(x)
    
    # If the network was purely linear (without biases), out1 == out2
    # If it was linear with biases, out1 = W(2x) + b = 2Wx + b, out2 = 2(Wx + b) = 2Wx + 2b
    # So out1 != out2 in both linear (with bias) and nonlinear cases.
    # However, GELU is definitely nonlinear.
    
    # Let's ensure they are significantly different to prove nonlinearity 
    # (since linear with bias might differ by just `b`)
    # To truly isolate non-linearity from bias, we check:
    # F(x + y) != F(x) + F(y) - F(0)
    
    out_x = ffn(x)
    y = torch.randn_like(x)
    out_y = ffn(y)
    out_x_plus_y = ffn(x + y)
    out_0 = ffn(torch.zeros_like(x))
    
    linear_pred = out_x + out_y - out_0
    
    assert not torch.allclose(out_x_plus_y, linear_pred, atol=1e-5), "FeedForward network appears to be linear!"
