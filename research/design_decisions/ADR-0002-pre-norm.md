# ADR-0002 — Use Pre-Layer Normalization by Default

| Field | Value |
|-------|-------|
| Status | accepted |
| Date | 2026-07-24 |
| Deciders | DevLLM Core Team |
| Related | docs/09_TRANSFORMER.md · docs/02_ARCHITECTURE.md · SPEC.md §2 |

## Context
When stacking multiple transformer blocks, activation scales and gradient magnitudes can become unstable during training. The original 2017 Transformer architecture (Vaswani et al.) applied Layer Normalization after each sublayer's residual addition (Post-Layer Normalization, or Post-Norm): `x_{l+1} = LayerNorm(x_l + Sublayer(x_l))`. In Post-Norm, gradients flowing back through deep stacks must pass through the normalization layer at every block. This places severe bounds on gradient scale, requiring careful learning-rate warmup schedules and precise initialization to prevent vanishing or exploding gradients.

For DevLLM, training on a MacBook Air M1 (8 GB unified memory, MPS backend), training stability is critical. We need an architecture that trains reliably from step zero without requiring complex warmup tuning or extra memory overhead for gradient stabilization. Modern GPT architectures (GPT-2, GPT-3, LLaMA) place Layer Normalization on the input of each sublayer before the attention and feed-forward operations (Pre-Layer Normalization, or Pre-Norm): `x_{l+1} = x_l + Sublayer(LayerNorm(x_l))`.

## Options considered

### Option A — Post-Layer Normalization (Post-Norm)
- Pros:
  - Matches original 2017 "Attention Is All You Need" transformer formulation.
  - Keeps activations at the output boundary of each block strictly normalized (zero mean, unit variance).
- Cons:
  - Normalization sits directly on the residual skip path, modifying the gradient highway at every layer.
  - Requires steep learning-rate warmup schedules to avoid early gradient explosion or collapse.
  - Harder to train stably as depth increases, increasing iteration failure risk on resource-constrained hardware.

### Option B — Pre-Layer Normalization (Pre-Norm)
- Pros:
  - Leaves the residual skip path (`+ x`) completely unnormalized, creating an unhindered "identity highway" for gradients.
  - Enables stable backpropagation across deep layer stacks without strict reliance on learning-rate warmup.
  - Significantly improves early training stability and convergence speed on small-scale hardware like Apple Silicon (MPS).
  - Aligns with modern decoder-only GPT standards (GPT-2, nanoGPT).
- Cons:
  - Activation variance can accumulate along the unnormalized residual stream as layer depth grows like O(√L).
  - Requires an explicit final LayerNorm operation before the language model head (`logits = LMHead(LayerNorm(x))`).

## Decision
We chose **Option B (Pre-Layer Normalization)** as the default baseline for DevLLM. Pre-Norm guarantees a clean residual gradient highway (`x + Sublayer(LayerNorm(x))`), preventing gradient collapse and enabling robust, stable training without fragile warmup requirements on 8 GB Apple Silicon hardware.

## Consequences
- **Easier now:** Training is significantly more stable across learning rates; gradients flow directly through skip connections; hyperparameter tuning is simpler.
- **Harder now:** Must maintain an explicit final LayerNorm layer prior to feeding hidden representations into the language model head.
- **To watch:** Potential activation scale growth along the residual stream if scaling to deeper network depths (n_layer ≥ 12) in future model versions.

## Notes
Detailed conceptual explanation, block flow diagrams, and residual highway math live in [`docs/09_TRANSFORMER.md`](../../docs/09_TRANSFORMER.md#pre-norm-vs-post-norm--a-real-design-decision).
