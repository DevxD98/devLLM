import torch
import numpy as np
from jimmylabs.utils.seed import seed_everything

def test_seed_repeatability():
    """
    Test that calling seed_everything with the same seed multiple times 
    yields the exact same random outputs from torch and numpy, ensuring
    same-device repeatability.
    """
    # First run
    seed_everything(42)
    t1 = torch.rand(5)
    n1 = np.random.rand(5)
    
    # Second run
    seed_everything(42)
    t2 = torch.rand(5)
    n2 = np.random.rand(5)
    
    assert torch.equal(t1, t2), "Torch outputs differ despite same seed."
    assert np.array_equal(n1, n2), "Numpy outputs differ despite same seed."
