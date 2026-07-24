import math
import torch
import torch.nn as nn

def get_lr(step: int, max_steps: int, warmup_steps: int, lr_max: float) -> float:
    """
    Computes the learning rate for a given step using linear warmup and cosine decay.
    Follows SPEC.md §7.
    """
    # 1. Linear warmup
    if step < warmup_steps:
        # Avoid division by zero if warmup_steps is 0
        if warmup_steps == 0:
            return lr_max
        return lr_max * (step / warmup_steps)
        
    # 2. Min LR floor
    lr_min = 0.1 * lr_max
        
    # 3. Floor after max_steps
    if step > max_steps:
        return lr_min
        
    # 4. Cosine decay
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    
    return lr_min + coeff * (lr_max - lr_min)

def clip_gradients(model: nn.Module, max_norm: float = 1.0):
    """
    Clips gradients of the model to a maximum global norm.
    Protects stability on Apple Silicon MPS hardware.
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
