import pytest
import torch
from jimmylabs.model.config import GPTConfig
from jimmylabs.model.embedding import TokenAndPositionEmbedding

@pytest.fixture
def dummy_config():
    return GPTConfig(
        vocab_size=65,
        n_layer=4,
        n_head=4,
        n_embd=128,
        block_size=128,
        dropout=0.0, # disable dropout for deterministic testing
        weight_tying=True
    )

def test_embedding_shape_and_type(dummy_config):
    """
    Test that (B, T) input -> (B, T, C) output.
    """
    emb = TokenAndPositionEmbedding(dummy_config)
    
    batch_size = 4
    seq_len = 10
    
    # Random integers in [0, vocab_size)
    idx = torch.randint(0, dummy_config.vocab_size, (batch_size, seq_len))
    
    out = emb(idx)
    
    assert out.shape == (batch_size, seq_len, dummy_config.n_embd)
    assert out.dtype == torch.float32

def test_positional_embeddings_effect(dummy_config):
    """
    Test that the same token at different positions produces different outputs,
    verifying that the positional embeddings are functioning.
    """
    emb = TokenAndPositionEmbedding(dummy_config)
    
    # [ [0, 0, 0, ..., 0] ]
    idx = torch.zeros(1, 5, dtype=torch.long)
    
    out = emb(idx)
    
    # Since tokens are the same, if pos_emb was zero, all outputs would be identical
    # We check if out[0, 0] != out[0, 1]
    assert not torch.allclose(out[0, 0], out[0, 1])

def test_embedding_out_of_range(dummy_config):
    """
    Test that an IndexError is raised if token ids are out of vocabulary bounds.
    """
    emb = TokenAndPositionEmbedding(dummy_config)
    
    # Create invalid ids (vocab_size is 65, so 65 is out of bounds)
    idx = torch.tensor([[65, 10]])
    
    with pytest.raises(IndexError):
        emb(idx)

def test_embedding_exceeds_block_size(dummy_config):
    """
    Test that ValueError is raised if sequence length exceeds block size.
    """
    emb = TokenAndPositionEmbedding(dummy_config)
    
    # block_size is 128, sequence is 130
    idx = torch.zeros(1, dummy_config.block_size + 2, dtype=torch.long)
    
    with pytest.raises(ValueError, match="Cannot forward sequence"):
        emb(idx)
