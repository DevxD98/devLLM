# ADR-0001 — Start with a character-level tokenizer

| Field | Value |
|-------|-------|
| Status | accepted |
| Date | 2026-07-24 |
| Deciders | DevLLM project |
| Related | [`docs/06_TOKENIZER.md`](../../docs/06_TOKENIZER.md) · [`docs/03_LEARNING_PATH.md`](../../docs/03_LEARNING_PATH.md) |

## Context

DevLLM needs to turn text into integer token IDs before anything else can happen
([`docs/06_TOKENIZER.md`](../../docs/06_TOKENIZER.md)). The two realistic choices for an
educational, from-scratch project on 8 GB are a **character-level** tokenizer (one ID per
character) or a **subword/BPE** tokenizer (one ID per frequent character-chunk, like
GPT-2's).

The decision matters because the tokenizer sets the vocabulary size, which in turn sizes the
embedding table and the output layer — both meaningful memory costs at our scale — and
because it's the *first* thing a learner meets. A confusing first step poisons the whole
learning path.

## Options considered

### Option A — Character-level tokenizer
- **Pros:** trivially simple to implement and understand (vocab = the set of unique
  characters, often <100); no external dependencies; the encode/decode round-trip is
  obviously correct; tiny vocabulary → tiny embedding and output matrices → less memory;
  perfect for the Phase-2 "get a forward pass working" milestone.
- **Cons:** sequences are long (one token per character), so a given `block_size` covers
  less actual text, and the model must spend capacity learning to spell before it can learn
  words.

### Option B — Byte-Pair Encoding (BPE) / subword
- **Pros:** far shorter sequences for the same text (better use of a small `block_size`);
  closer to how real GPTs work; better sample quality per training step.
- **Cons:** materially more complex to implement correctly from scratch (merge rules,
  training the merges, handling unknown bytes); larger vocabulary (thousands) → larger
  embedding/output matrices → more memory; a much steeper *first* step for a beginner; more
  surface area for silent bugs before the model even runs.

## Decision

**Start with a character-level tokenizer (Option A)** for v0.1–v0.3, and revisit BPE as a
deliberate, benchmarked upgrade later (a future ADR + experiment).

The single most important reason: this is an **educational** project whose first goal is a
*correct, understood* forward pass. A character tokenizer removes an entire category of
complexity and bugs from the critical early path, so the learner's attention stays on
embeddings, attention, and training — the actual subject matter — rather than on merge
tables.

## Consequences

- **Easier now:** the tokenizer is ~30 lines and obviously correct; the vocabulary is small
  so the model comfortably fits the 1M-parameter / 8 GB target; the learning path stays
  gentle through [`docs/06`–`10`](../../docs/06_TOKENIZER.md).
- **Harder now:** each `block_size` window holds less text, so long-range structure is
  harder to learn and generated text will look more "spelling-first" early on.
- **To watch:** when sample quality plateaus and the bottleneck is clearly context length
  rather than model size, that's the signal to run the BPE experiment and, if it wins, write
  the superseding ADR.

## Notes

This ADR is intentionally the project's first, and doubles as the worked example for the ADR
format (see [`ADR-TEMPLATE.md`](ADR-TEMPLATE.md)). The concept behind the trade-off is taught
in full in [`docs/06_TOKENIZER.md`](../../docs/06_TOKENIZER.md).
