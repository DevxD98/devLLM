# `configs/` — annotated model & training configurations

## Why this folder exists

A model is a config plus a seed. If experiments are to be reproducible (design commitment
⑤), the config must be a **first-class, version-controlled artifact** — not a pile of
arguments buried in a script. This folder holds the configurations that define each model
size and training run, with every field annotated so the numbers are understood, not
copied.

## What belongs here

- Model configs: `vocab_size`, `n_layer`, `n_head`, `n_embd`, `block_size`, `dropout`
- Training configs: batch size, learning rate + schedule, grad-clip, steps, eval interval
- One config per named model version (v0.1 … v1.0 from `02_ARCHITECTURE.md`)

## Convention

```
configs/
├── model_v0_1_char_100k.yaml     ← ~100K params, character tokenizer
├── model_v1_0_2m.yaml            ← ~2M params, Apple-Silicon tuned
└── train_shakespeare.yaml
```

Every field carries a comment explaining **why** that value, and its memory cost on 8 GB.
See the fully-annotated example in this folder and
[`docs/16_MODEL_CONFIGURATION.md`](../docs/16_MODEL_CONFIGURATION.md).

## What does NOT belong here

- Code that reads configs → [`src/`](../src/)
- Filled-in experiment records → [`experiments/`](../experiments/)

> Note: files here are illustrative annotated configs, not yet wired to running code —
> DevLLM is documentation-first.
