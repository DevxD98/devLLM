import random
import numpy as np
import torch

def seed_everything(seed: int) -> None:
    """
    Sets the seed for python random, numpy, and torch for determinism.
    
    Caveat (from SPEC.md §11): Exact bitwise reproducibility on MPS can differ
    from CPU and across macOS/torch versions. This guarantees same-device,
    same-seed repeatability.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # If MPS is available, seed it as well
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
