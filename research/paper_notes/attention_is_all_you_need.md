# Paper Note — Attention Is All You Need

| Field | Value |
|-------|-------|
| Authors | Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin |
| Year | 2017 |
| Venue | NeurIPS |
| Read on | 2026-07-24 |
| Reused in DevLLM | attention, transformer block, positional encoding idea |

> A note is not a summary. It records what **we** understood and what **we** borrowed —
> in our own words. The teaching version lives in
> [`docs/08_ATTENTION.md`](../../docs/08_ATTENTION.md); this is the lab's reading record.

## Why we read it

It is the origin of the transformer — the architecture every modern LLM, including DevLLM,
descends from. You cannot claim to build a GPT "from scratch, with understanding" without
reading the paper that defines the core operation.

## Core idea in plain words

Before this paper, sequence models processed tokens **in order** (RNNs/LSTMs), which is slow
and dilutes long-range information. The paper's claim, in the title, is radical: you can drop
the recurrence entirely and rely on **attention** — letting every position look directly at
every other position — as the *only* mechanism for moving information between positions.
Doing so makes the computation highly parallel (a big deal for hardware) and gives every
token a direct path to every other token (a big deal for long-range dependencies).

## Mechanisms we reuse

- **Scaled dot-product attention** — `softmax(QKᵀ/√d_k)·V`. The `√d_k` scaling (their
  Section 3.2.1) is the detail we make sure not to drop; without it the softmax saturates.
  → [`docs/08_ATTENTION.md`](../../docs/08_ATTENTION.md).
- **Multi-head attention** — run several attention operations in parallel over splits of the
  embedding so different heads capture different relationships. → doc 08.
- **The sublayer pattern** — `Sublayer + residual + normalization`, stacked. DevLLM keeps
  the pattern but moves the norm to *pre-norm* (a later refinement, see below). → doc 09.
- **Position must be injected explicitly** — since attention is order-agnostic, position
  information has to be added. They use fixed sinusoids; we start with learned positional
  embeddings. → [`docs/07_EMBEDDINGS.md`](../../docs/07_EMBEDDINGS.md).

## What we deliberately ignore (and why)

- **The encoder–decoder structure and cross-attention.** The paper targets machine
  translation, which needs an encoder (reads source) and a decoder (writes target). A GPT is
  **decoder-only**: one stack, causal self-attention, next-token prediction. We take only the
  decoder half. → [`docs/10_GPT_ARCHITECTURE.md`](../../docs/10_GPT_ARCHITECTURE.md).
- **Sinusoidal positional encodings.** Elegant, but learned positional embeddings are simpler
  to reason about at our scale; RoPE is a later option
  ([`architecture/future_architecture.md`](../../architecture/future_architecture.md)).
- **Post-norm placement.** The paper normalizes *after* each sublayer; GPT-2+ and DevLLM use
  pre-norm for training stability. Recorded as a design decision. → doc 09.
- **Label smoothing, big-batch training schedules, BLEU tuning** — translation-specific
  machinery irrelevant to a tiny character-level LM.

## Open questions this raised

- How much does the `√d_k` scaling actually matter at our tiny `d_k`? → candidate ablation,
  [`research/experiment_templates/`](../experiment_templates/).
- Learned vs sinusoidal positions at `block_size` ≤ 256 — measurable difference? → experiment.

## One-line takeaway

> Replace recurrence with "every token attends to every token," wrap it in residual +
> norm + a per-token MLP, stack it — and that stack, decoder-only and causally masked, is a
> GPT.
