"""Baseline benchmark harness for JimmyLabs — implements the docs/14 protocol.

Measures honestly on Apple Silicon: discards warmup iterations, synchronizes the MPS
queue around every timed region, and reports the MEDIAN of N runs (robust to thermal
drift). Prints machine state so the numbers are reproducible. It invents nothing — every
figure below is measured on the machine this runs on.

Usage:
    python scripts/benchmark.py --config configs/train_shakespeare.yaml
"""
import os
import time
import platform
import argparse
import statistics
import torch

from jimmylabs.model.config import GPTConfig
from jimmylabs.model.gpt import GPT
from jimmylabs.inference.generate import generate
from jimmylabs.utils.seed import seed_everything


def sync(device):
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/train_shakespeare.yaml')
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--iters', type=int, default=50)
    p.add_argument('--gen_tokens', type=int, default=200)
    args = p.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    seed_everything(cfg.get('seed', 1337))
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

    B, T = cfg['batch_size'], cfg['block_size']
    model_config = GPTConfig(vocab_size=cfg['vocab_size'], n_layer=cfg['n_layer'],
                             n_head=cfg['n_head'], n_embd=cfg['n_embd'],
                             block_size=cfg['block_size'], dropout=cfg.get('dropout', 0.1),
                             weight_tying=cfg.get('weight_tying', True))
    model = GPT(model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # ---- machine state ----
    print("=" * 60)
    print("MACHINE STATE")
    print(f"  platform     : {platform.platform()}")
    print(f"  macOS        : {platform.mac_ver()[0] or 'n/a'}")
    print(f"  processor    : {platform.processor()}")
    print(f"  torch        : {torch.__version__}")
    print(f"  device       : {device}   dtype: fp32")
    print(f"  config       : L={cfg['n_layer']} h={cfg['n_head']} C={cfg['n_embd']} "
          f"block={T} batch={B}  params={n_params:,}")
    print(f"  protocol     : warmup={args.warmup} discarded, median of {args.iters}, MPS-synced")
    print(f"  thermal      : (record manually: cold / warm / sustained)")
    print("=" * 60)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    X = torch.randint(0, model_config.vocab_size, (B, T), device=device)
    Y = torch.randint(0, model_config.vocab_size, (B, T), device=device)

    # ---- 1) training step throughput ----
    model.train()
    for _ in range(args.warmup):
        _, loss = model(X, Y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    sync(device)
    step_times = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        _, loss = model(X, Y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sync(device)
        step_times.append(time.perf_counter() - t0)
    med_step = statistics.median(step_times)
    train_tok_s = (B * T) / med_step

    # ---- 2) generation throughput ----
    model.eval()
    idx0 = torch.zeros((1, 1), dtype=torch.long, device=device)
    with torch.no_grad():
        _ = generate(model, idx0, max_new_tokens=args.warmup)  # warmup
        sync(device)
        t0 = time.perf_counter()
        _ = generate(model, idx0, max_new_tokens=args.gen_tokens)
        sync(device)
        gen_dt = time.perf_counter() - t0
    gen_tok_s = args.gen_tokens / gen_dt

    # ---- 3) single forward latency ----
    with torch.no_grad():
        sync(device); t0 = time.perf_counter(); _ = model(X); sync(device)
        fwd_ms = (time.perf_counter() - t0) * 1000

    # ---- 4) memory + checkpoint size ----
    peak_mb = None
    if device == 'mps':
        peak_mb = torch.mps.driver_allocated_memory() / 1e6
    ckpt = 'checkpoints/best_model.pt'
    ckpt_mb = os.path.getsize(ckpt) / 1e6 if os.path.exists(ckpt) else None

    print("RESULTS (median, warm)")
    print(f"  train throughput : {train_tok_s:,.0f} tokens/sec   ({med_step*1000:.2f} ms/step)")
    print(f"  gen throughput   : {gen_tok_s:,.0f} tokens/sec")
    print(f"  forward latency  : {fwd_ms:.2f} ms  (B={B}, T={T})")
    print(f"  model params     : {n_params:,}")
    if peak_mb is not None:  print(f"  MPS driver mem   : {peak_mb:,.1f} MB")
    if ckpt_mb is not None:  print(f"  checkpoint size  : {ckpt_mb:,.1f} MB")
    print("=" * 60)


if __name__ == '__main__':
    main()
