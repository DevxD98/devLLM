# Benchmark 001 — v0.1 baseline (pre-optimization)

> The reference every Phase 4 optimization is measured against. Produced by
> `scripts/benchmark.py` following the [`docs/14`](../docs/14_BENCHMARKING.md) protocol:
> warmup discarded, MPS-synced timed regions, median of N. No number here is invented —
> all measured on the machine below. Re-run: `python scripts/benchmark.py`.

| Field | Value |
|-------|-------|
| ID | 001 |
| Date | 2026-07-25 |
| Type | baseline |

## 1. What & why
Establish the v0.1 reference for training throughput, generation throughput, memory, and
checkpoint size — so KV cache, bf16, and gradient accumulation (see
[`OPTIMIZATION_BACKLOG.md`](../research/OPTIMIZATION_BACKLOG.md)) can be judged by a real
before/after, not a feeling.

## 2. Machine state
- Mac / chip: MacBook Air M1
- macOS: **26.5.2** (arm64)
- Python / PyTorch: **torch 2.13.0**
- Device: **mps** · dtype: **fp32**
- Thermal state: warm (short run; not sustained — see note)
- Protocol: warmup=10 discarded, **median of 50**, `torch.mps.synchronize()` around each timed region

## 3. Configuration
- Model: L=4, h=4, C=128, block=128 → **818,048 params** (v0.1, `configs/train_shakespeare.yaml`)
- Batch size: 16 · sequence length: 128
- Generation: 200 new tokens from a 1-token prompt, naive (no KV cache)

## 4. Results (median, warm)

| Metric | Value | Notes |
|--------|-------|-------|
| **Train throughput** | **65,027 tokens/sec** | 31.49 ms/step (fwd+bwd+opt, B·T=2048 tok/step) |
| **Generate throughput** | **103 tokens/sec** | naive O(T²) path — the KV-cache target |
| **Forward latency** | 45.46 ms | B=16, T=128, single forward |
| **Model params** | 818,048 | matches SPEC §5 exactly |
| **MPS driver memory** | 257.6 MB | vs 8 GB — v0.1 is *comfortable* (confirms SPEC §6) |
| **Checkpoint size** | 10.1 MB | weights (~3.3 MB) + AdamW moments + config/rng |

## 5. Interpretation
- **Memory is a non-issue at v0.1** (258 MB of 8 GB). The 8 GB constraint only bites at
  larger `block_size`/batch (the `B·T²` attention term, SPEC §6) — so v0.2/v0.3 sizing has
  real headroom.
- **Generation (103 tok/s) is the obvious optimization target.** It's the naive path that
  recomputes all keys/values every step; **KV cache** (`OPTIMIZATION_BACKLOG` #2) should
  move this materially and is the highest-value Phase 4 item.
- **Training throughput (65K tok/s)** is healthy for a fanless M1 at this size; bf16 (#6)
  and gradient accumulation (#3) are the levers, each to be benchmarked here as 002, 003…

## 6. Caveats
- **Thermal:** measured warm but not under sustained multi-hour load; a long training run
  will settle lower as the fanless chip throttles (docs/13). Treat these as *short-run
  warm* figures, not sustained.
- Generation throughput uses random-init-ish weights; sampling cost is weight-independent,
  so the figure is representative.

## 7. Reproduce
```
python scripts/benchmark.py --config configs/train_shakespeare.yaml
```
