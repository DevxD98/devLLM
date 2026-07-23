# `benchmarks/` — reproducible performance measurements

## Why this folder exists

Design commitment ③: *measure every optimization — before and after, or it didn't happen.*
This folder is where that rule is enforced. Every performance claim in the docs
("weight tying saved N MB", "KV cache made generation 3× faster") must point to a
benchmark file here that anyone can re-run.

## Convention

```
benchmarks/
├── 001_baseline-forward-pass.md
├── 002_kv-cache-generation.md      ← records before AND after
└── results/                        ← raw csv/json, plots
```

Each file is a filled-in copy of
[`research/benchmark_templates/BENCHMARK_TEMPLATE.md`](../research/benchmark_templates/).

## What makes a benchmark trustworthy here

- **Machine state recorded** — including thermal state on a fanless M1.
- **One variable changes** between before and after.
- **Seed fixed**, dtype stated, batch/seq-len stated.
- **Metric defined** — tokens/second measured how, over how many tokens, warm or cold.

See [`docs/14_BENCHMARKING.md`](../docs/14_BENCHMARKING.md) and
[`docs/13_OPTIMIZATION_FOR_APPLE_SILICON.md`](../docs/13_OPTIMIZATION_FOR_APPLE_SILICON.md).
