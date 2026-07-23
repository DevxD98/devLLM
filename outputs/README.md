# `outputs/` — generated text samples

## Why this folder exists

The whole point of a language model is the text it produces. This folder keeps generation
samples so progress is **visible and comparable over time**: the garbled characters of an
undertrained v0.1 next to the almost-coherent sentences of v1.0 tell the project's story
better than any loss curve.

## What belongs here

- Sample generations tagged with the checkpoint and sampling settings that produced them
- Before/after generations when an optimization or architecture change lands
- Curated "greatest hits" and "worst misses" — both are instructive

## Convention

```
outputs/
├── v0_1_step1000_temp0.8.md
└── v1_0_best_topk40.md
```

Each sample file records the **prompt, checkpoint, temperature, top-k/top-p, and seed** at
the top — a sample you can't reproduce is an anecdote, not a result.

## What does NOT belong here

- Quantitative metrics → [`benchmarks/`](../benchmarks/) and `experiments/*/results/`

See [`docs/12_INFERENCE.md`](../docs/12_INFERENCE.md) for how sampling settings shape these
outputs.
