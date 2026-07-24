import pytest
import os
import torch
import json
from jimmylabs.tokenizer.char import CharTokenizer

def test_prepare_data_logic(tmp_path):
    """
    Simulate prepare_data.py logic on a dummy text.
    Verifies:
    - split ratio ~90/10
    - every id in [0, vocab)
    - tokenizer round-trips
    """
    # Dummy text of length 100
    text = "a" * 40 + "b" * 30 + "c" * 20 + "d" * 10
    
    # 1. Build tokenizer
    tokenizer = CharTokenizer(corpus=text)
    
    # 2. Encode
    ids = tokenizer.encode(text)
    
    # 3. Split 90/10
    n = len(ids)
    train_data = ids[:int(n * 0.9)]
    val_data = ids[int(n * 0.9):]
    
    # Verify split ratio
    assert len(train_data) == 90
    assert len(val_data) == 10
    
    # Verify bounds
    assert all(0 <= idx < tokenizer.vocab_size for idx in ids)
    
    # Verify round-trips
    assert tokenizer.decode(ids) == text
    assert tokenizer.decode(train_data) == text[:90]
    
    # Verify save logic
    train_tensor = torch.tensor(train_data, dtype=torch.long)
    val_tensor = torch.tensor(val_data, dtype=torch.long)
    
    train_path = tmp_path / "train.pt"
    val_path = tmp_path / "val.pt"
    
    torch.save(train_tensor, train_path)
    torch.save(val_tensor, val_path)
    
    loaded_train = torch.load(train_path, weights_only=True)
    assert torch.equal(loaded_train, train_tensor)
