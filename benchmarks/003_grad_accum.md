# Benchmark 003 — Gradient Accumulation (Optimization C2)

| Field | Value |
|-------|-------|
| ID | 003 |
| Date | 2026-07-25 |
| Type | optimization |
| Baseline | `benchmarks/001_baseline.md` |

## 1. What & why
Implementation of Gradient Accumulation (`scripts/train.py`). This allows us to accumulate gradients over multiple smaller "micro-batches" before stepping the optimizer. It strictly decouples the *mathematical* batch size from the *hardware memory-constrained* batch size, letting us train on large effective batches without OOM crashes.

## 2. Machine state
- Mac / chip: MacBook Air M1
- macOS: **26.5.2** (arm64)
- Python / PyTorch: **torch 2.13.0**
- Device: **mps** · dtype: **fp32**
- Protocol: warmup=10 discarded, **median of 50**, `torch.mps.synchronize()` around each timed region

## 3. Configuration
- Model: L=4, h=4, C=128, block=128 → **818,048 params** (v0.1)
- Batch size: 16 · sequence length: 128
- Accumulation: `--grad_accum_steps=4` (Effective Batch Size = 64)
- Naive Generation: 200 tokens

## 4. Results (median, warm — same-session accum=1 vs accum=4)

Measured with `scripts/benchmark.py` (same session, both warm):

| Metric | accum=1 (Batch=16) | accum=4 (Eff. Batch=64) | Delta |
|--------|-------|-------|-------|
| **Train throughput** | 64,944 tok/s | **69,478 tok/s** | +7.0% |
| **Train step time** | 31.5 ms/step | 117.9 ms/step | 4× work/step (4 micro-batches) |
| **MPS driver memory** | 232.5 MB | 299.6 MB | +67 MB |

Correctness gate: `tests/test_grad_accum.py` asserts accumulating 4 micro-batches produces
gradients matching one 4×-larger batch (`torch.allclose`, atol 1e-5) — the math is
equivalent, only the memory profile differs.

## 5. Interpretation
- **Memory footprint decoupling** (the real point): accum=1 at batch 16 used 232.5 MB; an
  *effective* batch of 64 via accumulation rose only to 299.6 MB (+67 MB) — the cost of
  keeping the accumulated `.grad` tensors alive across 4 micro-batches. Pushing the *actual*
  batch to 64 would instead multiply activation memory (linear in batch, and the $O(B·T^2)$
  attention term), which is what OOMs a real training run on 8 GB.
- **Throughput bonus**: training throughput rose ~7% (64.9k → 69.5k tok/s) because
  `optimizer.step()`/`zero_grad()` run once per 4 forward/backward passes instead of every
  one — 75% less optimizer overhead. Note the step *time* is ~4× (117.9 ms) because a step
  now does 4× the work; throughput (tokens/sec) is the fair comparison.
- **Honesty note**: at v0.1 the model is tiny (232 MB of 8 GB), so grad accumulation is not
  yet *needed* — it's validated here so it's ready when v0.2/v0.3 sizes approach the ceiling.
  An earlier draft compared against 001's memory figure; the table now uses same-session
  measurements from the fixed `scripts/benchmark.py`.

## 6. Reproduce
```bash
python scripts/benchmark.py --grad_accum_steps=4
```
