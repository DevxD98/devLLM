import torch
import os
import argparse
from jimmylabs.model.gpt import GPT
from jimmylabs.tokenizer.char import CharTokenizer
from jimmylabs.training.checkpoint import load_checkpoint

def val_to_char(val):
    chars = [" ", " ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    # clamp val between 0 and 1
    val = max(0.0, min(1.0, float(val)))
    idx = int(val * (len(chars) - 1))
    return chars[idx]

def visualize(prompt="O Romeo, Romeo!", ckpt_path="checkpoints/best_model.pt", vocab_path="datasets/shakespeare/meta.json"):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    tokenizer = CharTokenizer.load(vocab_path)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    from jimmylabs.model.config import GPTConfig
    valid_keys = {"vocab_size", "n_layer", "n_head", "n_embd", "block_size", "dropout", "weight_tying"}
    filtered_config = {k: v for k, v in checkpoint['config'].items() if k in valid_keys}
    config = GPTConfig(**filtered_config)
    model = GPT(config)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    
    # encode prompt
    input_ids = tokenizer.encode(prompt)
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    # forward pass
    with torch.no_grad():
        _ = model(idx)
        
    T = idx.size(1)
    
    for layer_idx, block in enumerate(model.transformer.h):
        # block.attn is MultiHeadCausalSelfAttention
        # it saved _attn_weights of shape (B, nh, T, T)
        if not hasattr(block.attn, '_attn_weights'):
            print("Error: Model does not seem to have saved _attn_weights during forward pass.")
            return
            
        attn_weights = block.attn._attn_weights[0].cpu().numpy() # (nh, T, T)
        nh = attn_weights.shape[0]
        
        print(f"\n{'='*50}")
        print(f"Layer {layer_idx}")
        print(f"{'='*50}")
        
        for head_idx in range(nh):
            print(f"\nHead {head_idx}:\n")
            
            # Print X-axis labels (header)
            header = "    "
            for t in range(T):
                char = tokenizer.decode([input_ids[t]])
                char = repr(char)[1:-1] if char == '\n' else char
                header += f"{char:^3}"
            print(header)
            
            for row in range(T):
                char = tokenizer.decode([input_ids[row]])
                char = repr(char)[1:-1] if char == '\n' else char
                row_str = f"{char:>2} |"
                
                for col in range(T):
                    weight = attn_weights[head_idx, row, col]
                    if col > row: # Causal mask
                        row_str += "   "
                    else:
                        row_str += f" {val_to_char(weight)} "
                print(row_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Self-Attention")
    parser.add_argument("--prompt", type=str, default="O Romeo, Romeo!")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--vocab", type=str, default="datasets/shakespeare/meta.json")
    args = parser.parse_args()
    
    if not os.path.exists(args.ckpt):
        print(f"Error: {args.ckpt} not found. Run training first.")
        exit(1)
        
    visualize(args.prompt, args.ckpt, args.vocab)
