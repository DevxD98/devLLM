# Benchmark NNN — <what is being measured>

> Copy to `benchmarks/NNN_short-name.md`. A number without its conditions is noise —
> fill every field. For optimizations, record **before AND after** with one variable changed.

| Field | Value |
|-------|-------|
| ID | NNN |
| Date | YYYY-MM-DD |
| Author | |
| Type | baseline / optimization / regression-check |

## 1. What & why
_What are we measuring, and what decision does the number inform?_

## 2. Machine state
- Model of Mac / chip: MacBook Air M1
- Total unified memory: 8 GB
- macOS version:
- Python / PyTorch version:
- **Thermal state:** cold (idle ≥10 min) / warm / after sustained load
  _(fanless M1 throttles — this field is not optional)_
- Other apps running / memory pressure:

## 3. Configuration
- Device: mps / cpu · dtype: fp32 / fp16 / bf16
- Model config: n_layer / n_head / n_embd / block_size / params
- Batch size · sequence length:
- Warmup iterations discarded: _(first MPS calls compile kernels — always discard some)_
- Iterations measured:

## 4. Metric definition
_Exactly how measured. E.g. "median tokens/sec over 200 generated tokens, 5 runs, warmup
of 20 tokens discarded, seed fixed." Median over mean if there's thermal drift._

## 5. Results

| Variant | Metric | Value | Peak memory |
|---------|--------|-------|-------------|
| before | | | |
| after | | | |
| Δ | | | |

## 6. Interpretation
_What does this mean? Is the win real and repeatable, or within noise? Did memory or
thermals cap it? What's the next bottleneck?_

## 7. Reproduce
_Exact command / config to regenerate this number._
