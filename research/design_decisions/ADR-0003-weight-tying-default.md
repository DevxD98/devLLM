# ADR-0003 — Enable Weight Tying as v0.1 Default Baseline

| Field | Value |
|-------|-------|
| Status | accepted |
| Date | 2026-07-24 |
| Deciders | JimmyLabs Core Team |
| Related | SPEC.md §5 · research/OPTIMIZATION_BACKLOG.md #1 · docs/09_TRANSFORMER.md · docs/16_MODEL_CONFIGURATION.md |

## Context
In a language model, the token embedding layer maps token indices into a continuous hidden space (`(B, T) → (B, T, C)` using a weight matrix of shape `V × C`), while the output language model head projects hidden states back to logit scores over the vocabulary (`(B, T, C) → (B, T, V)` using a weight matrix of shape `C × V`). 

In early Transformer implementations, these two matrices were initialized and trained independently. However, Press & Wolf (2017) demonstrated that sharing (tying) the weight matrix between the input token embeddings and the output linear projection (`W_out = W_embᵀ`) yields equal or superior language modeling performance while significantly reducing parameter count. 

For JimmyLabs's v0.1 baseline (`V=65, C=128`), untied weights would add an extra `V · C = 8,320` parameters to the head. At larger vocabulary sizes (e.g. `V=512`), untied weights add over 65,000 parameters. Architecture Review item R2 noted that framing weight tying as a "Phase 4 optimization" undersells a fundamental design choice: weight tying is a free parameter and memory win that should be active by default in v0.1.

## Options considered

### Option A — Separate (Untied) Embedding and LM Head Weights
- Pros:
  - Allows independent parameterization of input lookup vs output projection.
  - Permits distinct embedding dimension sizes for input vs output if non-linear head projections are used.
- Cons:
  - Consumes extra parameters (`V · C` additional parameters) and extra memory for weights, gradients, and Adam optimizer states (which store 2 moments per weight).
  - Misses geometric regularization benefits where input and output spaces share dual representation geometry.

### Option B — Tied Embedding and LM Head Weights (v0.1 Default Baseline)
- Pros:
  - Eliminates `V · C` parameters entirely from the model footprint (0 extra parameters for LM head).
  - Saves weight memory, gradient memory, and optimizer memory on 8 GB Apple Silicon.
  - Enforces joint representation geometry between input token embeddings and output output logits (standard in GPT-2 and modern small GPTs).
  - Recorded as an applied default in [`OPTIMIZATION_BACKLOG.md`](../OPTIMIZATION_BACKLOG.md#1).
- Cons:
  - Requires token embedding width `n_embd` to exactly match LM head input dimension.
  - Requires computing transposed matrix multiplication (`x @ W_emb.T`) in the forward pass rather than calling a standard `nn.Linear`.

## Decision
We chose **Option B (Tied Weights by Default)** as the v0.1 baseline architecture. Weight tying is enabled by default in model configurations (`weight_tying: true`) and documented in [`SPEC.md` §5](../../SPEC.md#5-parameter-count-worked-for-v01).

## Consequences
- **Easier now:** Saves `V · C` parameters (8,320 params at v0.1 baseline, more at larger vocabs); lowers total memory footprint; simplifies baseline model architecture.
- **Harder now:** LM head implementation must explicitly reference `W_emb.weight` and execute transposed matmul.
- **To watch:** If future architectural experiments explore custom non-linear projection heads or asymmetric embedding dimensions, weight tying can be explicitly toggled via the config flag `weight_tying: false`.

## Notes
- See [`SPEC.md` §5](../../SPEC.md#5-parameter-count-worked-for-v01) for the full parameter arithmetic showing `lm head tied → 0 extra`.
- Tracked as Item #1 (Status: applied default) in [`research/OPTIMIZATION_BACKLOG.md`](../OPTIMIZATION_BACKLOG.md#1).
