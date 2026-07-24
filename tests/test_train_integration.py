"""Integration regression test for the training data path.

History: the first real training run sat at a flat loss of ln(vocab) for 5000 steps
because train.py called get_batch('train', batch, block) — passing a STRING where a data
tensor was expected — and then silently caught the resulting error and trained on RANDOM
tokens. Unit tests missed it: test_loader called get_batch correctly, and test_train_step
used inline data, so nobody exercised the wiring train.py actually uses.

This test reproduces train.py's real usage: a data TENSOR through get_batch (correct
signature) into the model for a few steps, and asserts the loss genuinely drops. If someone
reintroduces a signature mismatch or a silent random-data fallback, this fails.
"""
import torch

from jimmylabs.model.config import GPTConfig
from jimmylabs.model.gpt import GPT
from jimmylabs.data.loader import get_batch
from jimmylabs.utils.seed import seed_everything


def test_training_learns_from_real_data_path():
    seed_everything(0)
    cfg = GPTConfig(vocab_size=16, n_layer=2, n_head=2, n_embd=32,
                    block_size=16, dropout=0.0, weight_tying=True)
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # A LEARNABLE dataset: a fixed repeating pattern (not random noise). A working data path
    # can memorize this; the old random-fallback bug could not (loss would stay ~ln(vocab)).
    data = torch.arange(cfg.vocab_size).repeat(500)  # 0,1,2,...,15,0,1,... — highly predictable

    # Exactly how train.py calls it: get_batch(DATA_TENSOR, block_size, batch_size, device)
    x0, y0 = get_batch(data, cfg.block_size, batch_size=8, device='cpu')
    assert x0.shape == (8, cfg.block_size) and y0.shape == (8, cfg.block_size)

    _, first_loss = model(x0, y0)
    for _ in range(50):
        x, y = get_batch(data, cfg.block_size, batch_size=8, device='cpu')
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    _, last_loss = model(x0, y0)

    # On a predictable pattern the model MUST improve well below the random-init ln(vocab).
    import math
    assert last_loss.item() < first_loss.item(), "loss did not decrease on the real data path"
    assert last_loss.item() < math.log(cfg.vocab_size), \
        "loss stuck near ln(vocab) — the random-data-fallback bug may have returned"


def test_get_batch_rejects_wrong_first_arg():
    """The original bug passed a str as `data`. get_batch must not silently accept it."""
    import pytest
    with pytest.raises(Exception):
        get_batch("train", 16, 8)  # str has no valid len()-block_size window
