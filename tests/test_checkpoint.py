import pytest
import torch
import torch.optim as optim
import os
from jimmylabs.model.config import GPTConfig
from jimmylabs.model.gpt import GPT
from jimmylabs.training.checkpoint import save_checkpoint, load_checkpoint

@pytest.fixture
def dummy_model_setup():
    config = GPTConfig(
        vocab_size=65, n_layer=2, n_head=2, n_embd=32, block_size=16
    )
    model = GPT(config)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    return config, model, optimizer

def test_checkpoint_roundtrip(dummy_model_setup, tmp_path):
    """
    Test ROUND-TRIP: save then load reproduces identical weights + optimizer state + step.
    Resumed model gives identical logits.
    """
    config, model, optimizer = dummy_model_setup
    
    # 1. Take a small training step so the optimizer has state
    idx = torch.randint(0, config.vocab_size, (2, 4))
    targets = torch.randint(0, config.vocab_size, (2, 4))
    _, loss = model(idx, targets)
    loss.backward()
    optimizer.step()
    
    # Grab initial output for reproducibility test
    model.eval()
    with torch.no_grad():
        test_idx = torch.randint(0, config.vocab_size, (1, 4))
        original_logits, _ = model(test_idx)
        
    # 2. Save the checkpoint
    ckpt_path = os.path.join(tmp_path, "test_ckpt.pt")
    config_dict = {'vocab_size': config.vocab_size, 'n_layer': config.n_layer} # mock config dump
    save_checkpoint(ckpt_path, model, optimizer, config_dict, step=10, val_loss=0.5)
    
    # 3. Create a fresh model and optimizer
    fresh_model = GPT(config)
    fresh_optimizer = optim.AdamW(fresh_model.parameters(), lr=1e-3)
    
    # Verify they are initially different (weights mismatch)
    with torch.no_grad():
        fresh_logits, _ = fresh_model(test_idx)
    assert not torch.allclose(original_logits, fresh_logits)
    
    # 4. Load the checkpoint
    step, val_loss, loaded_config = load_checkpoint(ckpt_path, fresh_model, fresh_optimizer)
    
    # 5. Verify states
    assert step == 10
    assert val_loss == 0.5
    assert loaded_config == config_dict
    
    # Verify logits are identical now
    fresh_model.eval()
    with torch.no_grad():
        restored_logits, _ = fresh_model(test_idx)
        
    assert torch.allclose(original_logits, restored_logits, atol=1e-6)
    
    # Verify optimizer state is restored
    # Check that it has state keys
    assert len(fresh_optimizer.state_dict()['state']) > 0
