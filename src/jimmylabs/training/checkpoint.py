import torch
import torch.nn as nn
import torch.optim as optim

def save_checkpoint(path: str, model: nn.Module, optimizer: optim.Optimizer, config: dict, step: int, val_loss: float):
    """
    Saves a checkpoint matching the checkpoints/README.md contract.
    """
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'config': config,
        'step': step,
        'val_loss': val_loss,
        'rng_state': torch.get_rng_state()
    }
    torch.save(checkpoint, path)

def load_checkpoint(path: str, model: nn.Module, optimizer: optim.Optimizer = None) -> tuple[int, float, dict]:
    """
    Loads a checkpoint, restoring the model and optimizer states.
    Returns (step, val_loss, config).
    """
    # map_location='cpu' ensures we can load an MPS/CUDA checkpoint on CPU safely
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    
    # Strict loading ensures no missing or unexpected keys
    model.load_state_dict(checkpoint['model_state'], strict=True)
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        
    if 'rng_state' in checkpoint:
        torch.set_rng_state(checkpoint['rng_state'])
        
    return checkpoint['step'], checkpoint.get('val_loss', float('inf')), checkpoint['config']
