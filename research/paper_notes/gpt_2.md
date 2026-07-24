# Paper Note — Language Models are Unsupervised Multitask Learners (GPT-2)

| Field | Value |
|-------|-------|
| Authors | Radford, Wu, Child, Luan, Amodei, Sutskever |
| Year | 2019 |
| Venue | OpenAI |
| Read on | 2026-07-24 |
| Reused in JimmyLabs | pre-norm, weight tying, vocabulary size |

> A note is not a summary. It records what **we** understood and what **we** borrowed —
> in our own words. The teaching version lives in
> [`docs/10_GPT_ARCHITECTURE.md`](../../docs/10_GPT_ARCHITECTURE.md); this is the lab's reading record.

## Why we read it

GPT-2 proved that language models begin to learn tasks without explicit supervision if they are large enough and trained on enough data. More importantly for JimmyLabs, it introduced critical architectural tweaks (like pre-normalization) that are now standard for training stability, and solidified the use of weight tying.

## Core idea in plain words

If you make GPT-1 much larger and train it on a much bigger, more diverse dataset (WebText), it starts doing zero-shot task transfer. It doesn't need the fine-tuning step for many tasks; you can just prompt it. To make this larger model train stably, the Layer Normalization needs to be moved to the *input* of each sub-block, ensuring a clean residual path from the first to the last layer.

## Mechanisms we reuse

- **Pre-normalization (Pre-norm)** — Layer normalization is applied *before* the self-attention and MLP blocks, rather than after. We adopt this for training stability.
  → [`docs/09_TRANSFORMER.md`](../../docs/09_TRANSFORMER.md) and `ADR-0002`.
- **Weight tying** — The embedding matrix used to turn tokens into vectors is transposed and reused as the final projection matrix (LM head) that turns vectors back into vocabulary logits. This saves massive amounts of parameters.
  → [`docs/07_EMBEDDINGS.md`](../../docs/07_EMBEDDINGS.md) and `ADR-0003`.
- **Additional LayerNorm** — A final LayerNorm is applied after the final transformer block, before the LM head.
  → [`docs/10_GPT_ARCHITECTURE.md`](../../docs/10_GPT_ARCHITECTURE.md).

## What we deliberately ignore (and why)

- **Scaled residual initialization.** GPT-2 scaled the weights of residual layers at initialization by `1/√N` (where N is the number of residual layers). Given our very small depth (1-4M params implies maybe 2-4 layers), this specific scaling is less critical for us and adds complexity.
- **The 1.5B parameter scale.** Our absolute maximum is 4M parameters due to hardware constraints (M1, 8GB, no CUDA).

## Open questions this raised

- At our 1-4M parameter scale, does pre-norm actually show a measurable difference in training stability compared to post-norm, or is it only necessary for deeper networks? → candidate ablation, [`research/experiment_templates/`](../experiment_templates/).

## One-line takeaway

> Bigger is better, and if you move LayerNorm to the beginning of the sublayers (pre-norm), the gradients flow much better allowing you to actually train that bigger model.
