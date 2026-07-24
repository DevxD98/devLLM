import pytest
import torch
from jimmylabs.model.config import GPTConfig
from jimmylabs.model.gpt import GPT

@pytest.fixture
def dummy_model():
    config = GPTConfig(
        vocab_size=65, n_layer=2, n_head=2, n_embd=32, block_size=16
    )
    model = GPT(config)
    return model

def test_gradient_accumulation_equivalence(dummy_model):
    """
    K micro-batches of size B with grad_accum should produce the same gradients
    as 1 batch of size K*B.
    """
    B, T = 2, 8
    K = 4 # grad_accum_steps
    
    # Generate data
    X = torch.randint(0, 65, (K * B, T))
    Y = torch.randint(0, 65, (K * B, T))
    
    # 1. Single large batch
    model_large = GPT(dummy_model.config)
    model_large.load_state_dict(dummy_model.state_dict())
    model_large.eval()
    
    _, loss_large = model_large(X, Y)
    loss_large.backward()
    grads_large = [p.grad.clone() for p in model_large.parameters()]
    model_large.zero_grad()
    
    # 2. Accumulated micro-batches
    model_accum = GPT(dummy_model.config)
    model_accum.load_state_dict(dummy_model.state_dict())
    model_accum.eval()
    
    for i in range(K):
        X_micro = X[i*B:(i+1)*B]
        Y_micro = Y[i*B:(i+1)*B]
        
        _, loss_micro = model_accum(X_micro, Y_micro)
        # Scale loss
        loss_micro = loss_micro / K
        loss_micro.backward()
        
    grads_accum = [p.grad.clone() for p in model_accum.parameters()]
    model_accum.zero_grad()
    
    # Compare
    for g1, g2 in zip(grads_large, grads_accum):
        assert torch.allclose(g1, g2, atol=1e-5), "Gradients do not match between accumulation and large batch"
