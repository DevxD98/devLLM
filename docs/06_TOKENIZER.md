# 06 — Tokenizer

> **Prerequisites:** [`03_LEARNING_PATH.md`](03_LEARNING_PATH.md) and [`05_NEURAL_NETWORKS.md`](05_NEURAL_NETWORKS.md).
>
> **Next:** [`07_EMBEDDINGS.md`](07_EMBEDDINGS.md)

---

## Purpose

Neural networks cannot process raw text strings like `"Hello world"`. They only operate on numbers and tensors. A **tokenizer** is the translation bridge between human-readable text and integer token ID sequences:

$$\text{Text String} \quad \mathop{\rightleftarrows}_{\text{decode}}^{\text{encode}} \quad \text{Integer Token IDs}$$

This document explains tokenization strategies, vocabulary construction, encoding/decoding workflows, and alignment with [`ADR-0001`](../research/design_decisions/ADR-0001-character-tokenizer-first.md). It is the **canonical home** in JimmyLabs for tokenizer concepts.

---

## Background

### The tokenization spectrum

Text can be broken down into discrete units at three primary granularity levels:

```
  CHARACTER-LEVEL              BYTE-PAIR ENCODING (BPE)        WORD-LEVEL
  ['h','e','l','l','o']  ──►   ['hel', 'lo']             ──►   ['hello']
  Small vocab (~65)            Medium vocab (~32K-100K)        Huge vocab (1M+)
  Long sequences               Balanced length                 Out-of-vocab issues
```

1. **Word-level:** Splits on whitespace/punctuation. Vocabulary explosion; cannot handle unseen words (Out-Of-Vocabulary / OOV).
2. **Character-level (JimmyLabs v0.1 Default):** Treats every character (`a-z`, `A-Z`, punctuation, spaces) as a token. Vocabulary is tiny (~65–100), OOV errors are impossible, but sequence length $T$ is longer per sentence.
3. **Subword / BPE (Byte-Pair Encoding):** Iteratively merges frequent character pairs into subwords (e.g. `"ing"`, `"th"`). Balances sequence length and vocabulary size.

### Alignment with ADR-0001

As documented in [`ADR-0001`](../research/design_decisions/ADR-0001-character-tokenizer-first.md), JimmyLabs starts explicitly with a **character-level tokenizer**. 

Starting with a character tokenizer removes an entire class of preprocessing dependencies and edge-case bugs (unk tokens, byte merges, vocabulary files) from the critical early path, letting us focus on transformer mechanics.

---

## Concepts

Core tokenization concepts:

- **Vocabulary (`vocab`):** The fixed set of all known tokens mapped to unique integer indices ($0$ to $V-1$).
- **Vocabulary Size (`vocab_size` / $V$):** The total number of unique tokens in the dictionary.
- **Encode (`encode`):** Converting a raw string into a list or 1-D tensor of integer token IDs.
- **Decode (`decode`):** Converting a list or tensor of integer token IDs back into a human-readable string.
- **Out-of-Vocabulary (OOV):** Characters or subwords not present in the vocabulary. In character-level tokenizers built from a complete dataset alphabet, OOV rate is $0\%$.
- **Round-Trip Contract:** The strict requirement that $\text{decode}(\text{encode}(x)) \equiv x$ for all valid corpus text.

---

## Detailed Explanation

### 1. Character Tokenizer Mechanics

A character tokenizer builds its vocabulary directly from the unique characters present in the training text corpus.

#### Vocabulary Construction
Given a corpus (such as Shakespeare's plays):
1. Extract all unique characters: `sorted(list(set(corpus)))`.
2. Map each character to an integer index from $0$ to $V-1$:
   ```python
   stoi = {ch: i for i, ch in enumerate(chars)}
   itos = {i: ch for i, ch in enumerate(chars)}
   ```
3. For Shakespeare, $V \approx 65$ (includes uppercase, lowercase, numbers, space, newline, punctuation).

#### Encoding Example
Text: `"hello"`
- `'h' -> 7`, `'e' -> 4`, `'l' -> 11`, `'o' -> 14`
- `encode("hello") -> [7, 4, 11, 11, 14]`

#### Decoding Example
Token IDs: `[7, 4, 11, 11, 14]`
- `7 -> 'h'`, `4 -> 'e'`, `11 -> 'l'`, `14 -> 'o'`
- `decode([7, 4, 11, 11, 14]) -> "hello"`

---

### 2. Byte-Pair Encoding (BPE) Overview

While v0.1 uses a character tokenizer, BPE is the standard for larger models (GPT-2, GPT-4) and is planned as a future candidate ([`SPEC.md` §3](../SPEC.md#3-model-versions)).

#### BPE Algorithm
1. Start with character-level vocabulary.
2. Count frequencies of all adjacent token pairs in corpus.
3. Merge the most frequent pair into a new single token (e.g. `'t' + 'h' -> 'th'`).
4. Repeat until reaching target vocabulary size (e.g. 32,000).

#### Trade-offs: Character vs BPE

| Metric / Aspect | Character Tokenizer (v0.1) | BPE Subword Tokenizer |
|-----------------|----------------------------|------------------------|
| Vocabulary Size $V$ | Tiny (~65–100) | Large (32,000–100,000) |
| Sequence Length $T$ | Long (~5x vs BPE) | Short / Compact |
| OOV Rate | 0% | 0% (if byte-fallback used) |
| Implementation | ~10 lines of Python | Complex (merges table, regex) |
| Embedding Memory | Trivial ($65 \cdot C$) | Substantial ($32,000 \cdot C$) |

---

### 3. Tensor Shape Contracts at Tokenizer Boundary

The tokenizer forms the boundary between raw input strings and PyTorch tensor processing:

```
   Raw String:  "to be or not to be"
                      │
                      ▼  encode()
   Integer List: [19, 14, 0, 1, 4, 0, 14, 17, 0, ...]  (Length T)
                      │
                      ▼  torch.tensor()
   Tensor Shape: (B, T)  int64 tensor of token IDs
```

Downstream layers (token embeddings in [`07_EMBEDDINGS.md`](07_EMBEDDINGS.md)) accept integer tensors of shape `(B, T)` and map each integer ID to a hidden vector of shape `(B, T, C)`.

---

## Visual Diagrams

### Tokenizer Encoding / Decoding Loop

```
   RAW TEXT STRING
   "Hello"
      │
      │  encode() ──► Lookup char in `stoi` mapping
      ▼
   INTEGER TENSOR
   [20, 4, 11, 11, 14]   shape: (1, 5)  dtype: int64
      │
      │  (Model Processing & Next-Token Sampling)
      ▼
   GENERATED TENSOR
   [20, 4, 11, 11, 14, 1]
      │
      │  decode() ──► Lookup ID in `itos` mapping
      ▼
   OUTPUT TEXT STRING
   "Hello!"
```

---

## Common Mistakes

- **Failing Round-Trip Contract:** Modifying text during tokenization (e.g. stripping spaces or lowercasing without recording original mappings), causing `decode(encode(text)) != text`.
- **Vocabulary Mismatch Between Train and Test:** Building `stoi` on training text and encountering unseen characters during inference, leading to `KeyError` crashes.
- **Off-By-One Indexing:** Indexing token IDs from $1$ to $V$ instead of $0$ to $V-1$, causing out-of-bounds CUDA/MPS index errors during embedding lookup.
- **Ignoring Newline Characters:** Omitting `\n` from character set when training on text datasets like Shakespeare, losing line structure during generation.

---

## Future Improvements

- Add standalone BPE tokenizer implementation in `src/tokenizer/bpe.py` for v1.0 experiments.
- Add serialization utilities to save/load tokenizer mappings as JSON artifacts.

---

## Glossary

Canonical definitions live in [`19_GLOSSARY.md`](19_GLOSSARY.md). Tokenizer terms:

| Term | Definition |
|------|------------|
| Token | The atomic text unit (character or subword) processed by the model |
| `vocab_size` ($V$) | Total number of unique tokens in vocabulary dictionary |
| `encode` | Converting raw text string into integer token ID list |
| `decode` | Converting integer token ID list back into raw text string |
| Round-Trip Contract | Guarantee that $\text{decode}(\text{encode}(x)) \equiv x$ |

---

## Learning Checklist

You master tokenization principles when you can:

- [ ] Explain why neural networks require tokenizers.
- [ ] Build a character-level `stoi` and `itos` mapping in Python.
- [ ] State the pros and cons of character vs BPE tokenization.
- [ ] Verify the round-trip contract $\text{decode}(\text{encode}(x)) \equiv x$ on sample text.
- [ ] State the tensor shape `(B, T)` produced by batching token ID sequences.

---

## References

- [`ADR-0001`](../research/design_decisions/ADR-0001-character-tokenizer-first.md) — Rationale for character tokenizer first.
- Sennrich et al. (2016), *Neural Machine Translation of Rare Words with Subword Units* — Original BPE paper.
- [`SPEC.md` §3](../SPEC.md#3-model-versions) — Baseline model vocabulary configurations.

## Further Reading

- [`07_EMBEDDINGS.md`](07_EMBEDDINGS.md) — **Next:** Mapping token IDs to continuous vector embeddings.
- [`architecture/tokenizer_architecture.md`](../architecture/tokenizer_architecture.md) — Concrete data flow and module boundaries for tokenization in JimmyLabs.

> **Next:** [`07_EMBEDDINGS.md`](07_EMBEDDINGS.md)
