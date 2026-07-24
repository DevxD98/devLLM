import os
import urllib.request
import torch
import datetime
from jimmylabs.tokenizer.char import CharTokenizer

def main():
    # Setup paths
    dataset_dir = os.path.join('datasets', 'shakespeare')
    os.makedirs(dataset_dir, exist_ok=True)
    
    input_path = os.path.join(dataset_dir, 'input.txt')
    train_path = os.path.join(dataset_dir, 'train.pt')
    val_path = os.path.join(dataset_dir, 'val.pt')
    meta_path = os.path.join(dataset_dir, 'meta.json')
    
    # 1. Download
    if not os.path.exists(input_path):
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        print(f"Downloading Tiny Shakespeare from {url}...")
        urllib.request.urlretrieve(url, input_path)
        
    print(f"Reading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = f.read()
        
    print(f"Length of dataset in characters: {len(data):,}")
    
    # 2. Tokenize
    tokenizer = CharTokenizer(corpus=data)
    tokenizer.save(meta_path)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # 3. Encode
    print("Encoding dataset...")
    ids = tokenizer.encode(data)
    
    # 4. Split 90/10
    n = len(ids)
    train_data = ids[:int(n * 0.9)]
    val_data = ids[int(n * 0.9):]
    
    print(f"Train split: {len(train_data):,} tokens (90%)")
    print(f"Val split: {len(val_data):,} tokens (10%)")
    
    # 5. Save as PyTorch tensors
    train_tensor = torch.tensor(train_data, dtype=torch.long)
    val_tensor = torch.tensor(val_data, dtype=torch.long)
    
    print(f"Saving to {train_path} and {val_path}...")
    torch.save(train_tensor, train_path)
    torch.save(val_tensor, val_path)
    print("Done!")

if __name__ == '__main__':
    main()
