# `architecture/` — the wiring diagrams

## Why this folder exists

`docs/` explains **concepts** ("what is attention, and why"). `architecture/` explains
**wiring** ("how the tensors actually flow through *this* codebase, with these shapes,
in this order"). The split keeps each concept defined in exactly one place: the concept
doc owns the intuition and the math; the architecture doc owns the concrete data path,
tensor shapes, and module boundaries.

If you are learning an idea for the first time, start in `docs/`. If you already
understand the idea and want to know how JimmyLabs implements it, come here.

## What belongs here

- End-to-end data-flow diagrams with concrete tensor shapes `(B, T, C)`
- Module boundary definitions — what each `src/` unit takes and returns
- Memory-layout reasoning specific to 8 GB unified memory
- Apple-Silicon execution strategy (MPS placement, dtype choices)
- Forward-looking architecture sketches that are not yet built

## What does NOT belong here

- First-principles concept teaching → that lives in `docs/`
- Experiment results or benchmarks → `research/` and `benchmarks/`
- Actual implementation code → `src/`

## Contents

| File | Owns |
|------|------|
| `model_architecture.md` | full forward pass, layer-by-layer, with shapes |
| `attention_flow.md` | Q/K/V projection → scores → softmax → weighted sum |
| `training_pipeline.md` | data → loss → backward → optimizer step → checkpoint |
| `inference_pipeline.md` | prompt → autoregressive loop → sampling → decode |
| `memory_layout.md` | where every large tensor lives in 8 GB |
| `apple_silicon_strategy.md` | MPS, unified memory, dtype, thermal reality |
| `tokenizer_architecture.md` | text ↔ id boundary, vocab storage |
| `future_architecture.md` | KV cache, RoPE, GQA — what a v2 would change |

## See also

- [`docs/02_ARCHITECTURE.md`](../docs/02_ARCHITECTURE.md) — the high-level anchor
- [`src/README.md`](../src/README.md) — the module map these diagrams describe
