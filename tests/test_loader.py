import torch
from jimmylabs.data.loader import get_batch

def test_get_batch_shapes_and_types():
    """
    Test that batches have shape (B, T), dtype int64, targets are inputs shifted by one,
    and no id >= vocab_size.
    """
    vocab_size = 65
    data_length = 1000
    
    # Create dummy data tensor of int64
    data = torch.randint(0, vocab_size, (data_length,), dtype=torch.int64)
    
    batch_size = 4
    block_size = 10
    
    x, y = get_batch(data, block_size, batch_size, device='cpu')
    
    # Check shapes
    assert x.shape == (batch_size, block_size)
    assert y.shape == (batch_size, block_size)
    
    # Check dtypes
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64
    
    # Check bounds
    assert torch.all(x >= 0) and torch.all(x < vocab_size)
    assert torch.all(y >= 0) and torch.all(y < vocab_size)
    
    # Check that y is x shifted by one (for the overlap)
    # x: [token_i, token_i+1, ..., token_i+T-1]
    # y: [token_i+1, token_i+2, ..., token_i+T]
    # Therefore x[:, 1:] should equal y[:, :-1]
    assert torch.equal(x[:, 1:], y[:, :-1]), "Targets are not correctly shifted relative to inputs"
