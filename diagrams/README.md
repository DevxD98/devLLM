# `diagrams/` — the ASCII source of truth

## Why this folder exists

Several diagrams appear in more than one place (the forward pass shows up in the README,
in `02_ARCHITECTURE.md`, and in `architecture/model_architecture.md`). To avoid three
slightly-different copies drifting apart, the **canonical** version of a reused diagram
lives here as a plain `.txt`, and docs quote it. When the architecture changes, you edit
one file.

## What belongs here

- Any ASCII diagram referenced by **two or more** documents
- The "master" versions of: repository tree, forward pass, attention flow, training loop,
  inference loop, memory layout, tensor-shape charts

## Convention

```
diagrams/
├── forward_pass.txt
├── attention_flow.txt
├── training_loop.txt
├── inference_loop.txt
└── memory_layout.txt
```

- Keep to a **72-column** width so diagrams render inside code fences everywhere.
- Use a consistent glyph vocabulary: `→ ↓ ▼ │ ┌─┐ └─┘` for flow; `(B, T, C)` for shapes.

## What does NOT belong here

- A diagram used in exactly one doc — just inline it there.
- Raster figures → [`assets/`](../assets/).

> Single-use diagrams live where they're used. Shared diagrams live here. That's the whole
> rule, and it exists to prevent the most common documentation rot: three diagrams that
> were once the same.
