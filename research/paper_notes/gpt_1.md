# Paper Note — Improving Language Understanding by Generative Pre-Training (GPT-1)

| Field | Value |
|-------|-------|
| Authors | Radford, Narasimhan, Salimans, Sutskever |
| Year | 2018 |
| Venue | OpenAI (Preprint) |
| Read on | 2026-07-24 |
| Reused in JimmyLabs | decoder-only architecture, pre-training objective |

> A note is not a summary. It records what **we** understood and what **we** borrowed —
> in our own words. The teaching version lives in
> [`docs/10_GPT_ARCHITECTURE.md`](../../docs/10_GPT_ARCHITECTURE.md); this is the lab's reading record.

## Why we read it

This paper introduced the Generative Pre-trained Transformer (GPT) paradigm. It demonstrated that a decoder-only transformer, trained simply to predict the next word on a large unlabeled text corpus, could learn robust representations that transfer well to various downstream tasks with minimal fine-tuning. It forms the architectural baseline for JimmyLabs.

## Core idea in plain words

Instead of building complex, task-specific architectures for every NLP problem, we can do two steps: 1) Train a massive capacity language model on huge amounts of raw text to just predict the next word (unsupervised pre-training). 2) Tweak this model slightly with supervised data for specific tasks like classification or question answering. The decoder-only transformer is perfectly suited for step 1 because the causal mask prevents it from "cheating" by looking ahead.

## Mechanisms we reuse

- **Decoder-only architecture** — We discard the encoder from the original Transformer. The model is just a stack of masked self-attention blocks. This simplifies the architecture and aligns perfectly with autoregressive generation.
  → [`docs/10_GPT_ARCHITECTURE.md`](../../docs/10_GPT_ARCHITECTURE.md).
- **Generative pre-training objective** — Our loss function is standard cross-entropy on next-token prediction.
  → [`docs/11_TRAINING_PIPELINE.md`](../../docs/11_TRAINING_PIPELINE.md).

## What we deliberately ignore (and why)

- **The supervised fine-tuning stage.** GPT-1 heavily emphasized step 2 (fine-tuning for specific tasks). In JimmyLabs, we focus purely on the unsupervised pre-training foundation to understand how the representations form.
- **Task-specific input transformations.** GPT-1 used special tokens (Start, Extract, Delimiter) to format various NLP tasks into sequences. We focus on raw language modeling.

## Open questions this raised

- GPT-1 used a relatively small dataset (BooksCorpus) and model (117M params). Can our 1-4M parameter JimmyLabs model learn anything meaningful, or is there a hard capability floor? → empirical evaluation needed.

## One-line takeaway

> Strip the Transformer down to just its decoder, train it to predict the next token on a massive corpus, and you have a general-purpose language understanding engine.
