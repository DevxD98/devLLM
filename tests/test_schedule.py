import pytest
import math
from jimmylabs.training.schedule import get_lr

def test_lr_schedule_warmup():
    """
    Verify LR at step=0 is appropriately small (0.0).
    Verify LR reaches lr_max exactly at warmup_steps.
    """
    lr_max = 1e-3
    warmup_steps = 100
    max_steps = 1000
    
    assert get_lr(0, max_steps, warmup_steps, lr_max) == 0.0
    
    # Mid-warmup
    mid_lr = get_lr(50, max_steps, warmup_steps, lr_max)
    assert math.isclose(mid_lr, 0.5 * lr_max, rel_tol=1e-5)
    
    # Peak
    assert get_lr(warmup_steps, max_steps, warmup_steps, lr_max) == lr_max

def test_lr_schedule_decay():
    """
    Verify LR decays monotonically after warmup and never drops below 0.1 * lr_max.
    """
    lr_max = 1e-3
    warmup_steps = 100
    max_steps = 1000
    lr_min = 0.1 * lr_max
    
    prev_lr = lr_max
    for step in range(warmup_steps + 1, max_steps + 1):
        lr = get_lr(step, max_steps, warmup_steps, lr_max)
        assert lr <= prev_lr, f"LR increased at step {step}: {lr} > {prev_lr}"
        assert lr >= lr_min, f"LR fell below floor at step {step}: {lr} < {lr_min}"
        prev_lr = lr
        
    # At exactly max_steps, should be lr_min
    assert math.isclose(get_lr(max_steps, max_steps, warmup_steps, lr_max), lr_min, rel_tol=1e-5)
    
    # After max_steps, should stay at lr_min
    assert get_lr(max_steps + 100, max_steps, warmup_steps, lr_max) == lr_min
