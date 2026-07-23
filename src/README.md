# `src/` — the model itself

## Why this folder exists

This is where DevLLM actually lives as code. Everything else in the repository exists to
explain, test, configure, or measure what is in here. The organizing principle: **each
module does one thing, exposes a small interface, and can be understood and tested in
isolation** (design commitment ④).

## Planned module map

This mirrors [`architecture/`](../architecture/) one-to-one — concept in `docs/`, wiring
in `architecture/`, implementation here.

```
src/
├── tokenizer/      text ↔ token ids            → docs/06, architecture/tokenizer_architecture.md
├── data/           corpus → batches            → docs/17, architecture/training_pipeline.md
├── model/
│   ├── embedding   token + positional          → docs/07
│   ├── attention   multi-head self-attention   → docs/08, architecture/attention_flow.md
│   ├── feedforward MLP block                    → docs/09
│   ├── block       one transformer block        → docs/09
│   └── gpt         the full model               → docs/10, architecture/model_architecture.md
├── training/       loop, loss, checkpoint       → docs/11
└── inference/      autoregressive generation     → docs/12, architecture/inference_pipeline.md
```

## Contracts

- **No side effects on import.** Modules define; scripts run. Running lives in
  [`scripts/`](../scripts/).
- **Every module has a test** in [`tests/`](../tests/) proving its interface.
- **PyTorch is a dependency, not a crutch.** We use tensors, autograd, optimizers, and MPS.
  We do **not** import prebuilt transformers, attention, or tokenizers.

## Status

Documentation-first. This README is the contract the code will fulfill in roadmap phases
1–3. No implementation is committed yet.
