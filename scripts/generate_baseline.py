import os
import torch
from jimmylabs.model.config import GPTConfig
from jimmylabs.model.gpt import GPT
from jimmylabs.inference.generate import generate
from jimmylabs.tokenizer.char import CharTokenizer
from jimmylabs.utils.seed import seed_everything

def main():
    # 1. Configuration (use v0.1 model size)
    config = GPTConfig(
        vocab_size=65,
        n_layer=4,
        n_head=4,
        n_embd=128,
        block_size=128,
        dropout=0.1,
        weight_tying=True
    )
    
    # 2. Setup
    seed_everything(42)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    print("Initializing fresh untrained GPT model (v0.1)...")
    model = GPT(config)
    model.to(device)
    model.eval()
    
    # Load tokenizer to decode output
    # (assuming prepare_data.py has been run and meta.json exists)
    meta_path = os.path.join('datasets', 'shakespeare', 'meta.json')
    if os.path.exists(meta_path):
        tokenizer = CharTokenizer.load(meta_path)
    else:
        print("Warning: datasets/shakespeare/meta.json not found. Make sure prepare_data.py was run.")
        print("Falling back to random characters for demonstration.")
        # Fallback dummy vocab for length 65 just to let the script run
        chars = list(" \nabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!?.,")
        chars = chars[:65] # pad or trim to 65
        tokenizer = CharTokenizer(vocab=chars)
        
    # Start with a single newline character to kick off generation
    context = "\n"
    idx = torch.tensor([tokenizer.encode(context)], dtype=torch.long, device=device)
    
    print("Generating baseline output (200 tokens)...")
    # 3. Generate (naive autoregressive)
    out_idx = generate(model, idx, max_new_tokens=200, temperature=1.0, top_k=10)
    
    # Decode
    out_text = tokenizer.decode(out_idx[0].tolist())
    
    # 4. Save to outputs
    os.makedirs('outputs', exist_ok=True)
    out_path = os.path.join('outputs', 'untrained_baseline.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(out_text)
        
    print(f"\n--- Untrained Baseline Output ---\n{out_text}\n---------------------------------")
    print(f"Saved to {out_path}")
    
if __name__ == '__main__':
    main()
