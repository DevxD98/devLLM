# Paper Note — Language Models are Few-Shot Learners (GPT-3)

| Field | Value |
|-------|-------|
| Authors | Brown, Mann, Ryder, et al. |
| Year | 2020 |
| Venue | NeurIPS |
| Read on | 2026-07-24 |
| Reused in JimmyLabs | scaling philosophy (in spirit) |

> A note is not a summary. It records what **we** understood and what **we** borrowed —
> in our own words. The teaching version lives in
> [`docs/10_GPT_ARCHITECTURE.md`](../../docs/10_GPT_ARCHITECTURE.md); this is the lab's reading record.

## Why we read it

GPT-3 is the paper that introduced the concept of "in-context learning" to the world at large. It demonstrated that at a massive scale (175B parameters), a language model doesn't even need fine-tuning—you can just give it a few examples in the prompt, and it learns the task on the fly. We read it to understand the ceiling of the architecture we are building.

## Core idea in plain words

If you take the exact same architecture as GPT-2 (with a minor tweak to sparse attention) but scale it up by 100x and train it on a massive amount of data, the model develops a new capability: meta-learning. It can recognize patterns in its immediate context window and perform tasks it wasn't explicitly trained for, just by predicting what comes next based on a few examples (few-shot prompting).

## Mechanisms we reuse

- **The core architecture remains the same.** GPT-3 proved that the basic decoder-only transformer with pre-norm is remarkably robust. We are building this same core engine.
  → [`docs/10_GPT_ARCHITECTURE.md`](../../docs/10_GPT_ARCHITECTURE.md).

## What we deliberately ignore (and why)

- **Sparse Attention.** GPT-3 used alternating dense and locally banded sparse attention patterns in the layers of the transformer to handle longer contexts efficiently. At our 1-4M parameter scale with short contexts (e.g., 256 tokens), full dense attention is perfectly fine and much easier to implement.
- **In-context learning expectations.** The paper notes that these emergent meta-learning capabilities require significant scale. Our 1-4M parameter model will **not** exhibit in-context learning. It will struggle to generate coherent paragraphs, let alone solve few-shot translation tasks. We accept this limitation.

## Open questions this raised

- What is the absolute minimum parameter count where *any* form of in-context learning begins to emerge? Is it possible to demonstrate a microscopic version of it on a highly constrained, synthetic dataset?

## One-line takeaway

> The transformer architecture scales incredibly well; throw enough parameters and compute at it, and it learns how to learn from its context window.
