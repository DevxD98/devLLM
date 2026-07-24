import pytest
import torch
import torch.optim as optim
from jimmylabs.model.config import GPTConfig
from jimmylabs.model.gpt import GPT
from jimmylabs.utils.seed import seed_everything

@pytest.fixture
def v0_1_config():
    """Returns the exact v0.1 config to test parameter counts."""
    return GPTConfig(
        vocab_size=65,
        n_layer=4,
        n_head=4,
        n_embd=128,
        block_size=128,
        dropout=0.1,
        weight_tying=True
    )

def test_gpt_forward_shape(v0_1_config):
    """
    Test forward shape (B,T) -> (B,T,V) logits.
    """
    model = GPT(v0_1_config)
    
    B, T = 2, 10
    idx = torch.randint(0, v0_1_config.vocab_size, (B, T))
    
    logits, loss = model(idx)
    
    assert logits.shape == (B, T, v0_1_config.vocab_size)
    assert loss is None
    
    # Test with targets
    targets = torch.randint(0, v0_1_config.vocab_size, (B, T))
    logits, loss = model(idx, targets)
    
    assert logits.shape == (B, T, v0_1_config.vocab_size)
    assert loss is not None
    assert loss.dim() == 0  # scalar

def test_gpt_parameter_count(v0_1_config):
    """
    Check parameter count is exactly within a tight tolerance of 818,048
    for the model_v0_1_char_100k.yaml config (matches SPEC.md §5).
    """
    model = GPT(v0_1_config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Assert it exactly matches the SPEC calculation
    assert total_params == 818048, f"Parameter count mismatch! Expected 818,048 but got {total_params}"

def test_gpt_weight_tying(v0_1_config):
    """
    Verify that lm_head.weight IS the token_embedding.weight.
    """
    model = GPT(v0_1_config)
    
    # They should be the exact same object in memory
    assert model.lm_head.weight is model.transformer.wte.token_embedding.weight
    
    # Disabling weight tying
    untied_config = GPTConfig(
        vocab_size=65, n_layer=4, n_head=4, n_embd=128, block_size=128, dropout=0.1, weight_tying=False
    )
    model_untied = GPT(untied_config)
    assert model_untied.lm_head.weight is not model_untied.transformer.wte.token_embedding.weight

def test_gpt_overfit_a_batch(v0_1_config):
    """
    OVERFIT-A-BATCH: on one tiny fixed batch, loss falls to near-zero in N steps.
    """
    seed_everything(1337)
    
    # Turn off dropout for overfitting test
    v0_1_config.dropout = 0.0
    model = GPT(v0_1_config)
    model.train()
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    B, T = 2, 8
    # Dummy data
    idx = torch.randint(0, v0_1_config.vocab_size, (B, T))
    targets = torch.randint(0, v0_1_config.vocab_size, (B, T))
    
    # Train for a few steps
    epochs = 60
    for _ in range(epochs):
        optimizer.zero_grad()
        logits, loss = model(idx, targets)
        loss.backward()
        optimizer.step()
        
    assert loss.item() < 0.1, f"Failed to overfit a batch, final loss: {loss.item()}"
