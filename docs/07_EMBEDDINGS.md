# 07 — Embeddings

> **Prerequisites:** [`04_MATHEMATICS.md`](04_MATHEMATICS.md) (vectors and dot products) and [`06_TOKENIZER.md`](06_TOKENIZER.md) (token IDs).
>
> **Next:** [`08_ATTENTION.md`](08_ATTENTION.md)

---

## Purpose

Token IDs (like `[7, 4, 11, 11, 14]`) are discrete integers. Integers carry ordinal assumptions ($11 > 4$) that do not reflect linguistic relationships. Neural networks require continuous vector representations where mathematical distance corresponds to semantic similarity.

This document explains how DevLLM converts discrete token IDs into continuous hidden vectors using two complementary embedding layers:
1. **Token Embeddings** — mapping *what* a token is into vector space.
2. **Positional Embeddings** — mapping *where* a token occurs in the sequence.
3. **Weight Tying Introduction** — the concept of sharing weights between input token embeddings and output language model heads ([`ADR-0003`](../research/design_decisions/ADR-0003-weight-tying-default.md)).

This document is the **canonical home** in DevLLM for embedding representations and the conceptual introduction to weight tying.

---

## Background

### Why integers are not enough

If we passed raw token IDs directly into a neural network layer:
- Token $4$ (`'e'`) and Token $5$ (`'f'`) would be treated as numerically closer than Token $4$ (`'e'`) and Token $20$ (`'E'`).
- Integer values force a 1-dimensional ordering on concepts that require high-dimensional representation spaces.

### The Embedding Solution

An **embedding** maps each discrete token ID to a continuous vector of width $C = \text{n\_embd}$ (e.g. $C = 128$). Instead of hardcoding these vectors, the network **learns** them during training via backpropagation ([`05_NEURAL_NETWORKS.md`](05_NEURAL_NETWORKS.md)). Over training, tokens used in similar contexts evolve vectors pointing in similar directions in $C$-dimensional space.

```
   Token ID 7 ('h')  ──► [ Lookup Matrix W_token ] ──► Vector [0.12, -0.45, ..., 0.89]
                                                        (Shape: C dimensions)
```

---

## Concepts

Core embedding concepts:

- **Token Embedding Matrix ($W_{\text{token}}$):** A weight matrix of shape $(V \times C)$ storing a $C$-dimensional vector for each of the $V$ vocabulary tokens.
- **Positional Embedding Matrix ($W_{\text{pos}}$):** A weight matrix of shape $(T \times C)$ storing a $C$-dimensional vector for each position index $0, 1, \dots, T-1$ up to context window $T = \text{block\_size}$.
- **Learned Absolute Position Embeddings:** Position representations trained via gradient descent alongside token embeddings (standard in GPT-2 and DevLLM).
- **Weight Tying Concept:** Reusing the token embedding matrix $W_{\text{token}}$ as the linear projection weights for the output Language Model head ($W_{\text{out}} = W_{\text{token}}^T$), reducing parameter count and memory footprint.

---

## Detailed Explanation

### 1. Token Embeddings ($W_{\text{token}}$)

The token embedding layer acts as a high-speed matrix lookup table.

#### Operation
Given an integer tensor of token IDs $X$ of shape $(B, T)$:
1. For every index $x_{b,t} \in \{0, 1, \dots, V-1\}$, retrieve row $x_{b,t}$ from $W_{\text{token}}$ (shape $V \times C$).
2. Output tensor shape: $(B, T, C)$.

Mathematically, retrieving row $k$ from $W_{\text{token}}$ is equivalent to multiplying a one-hot vector $e_k$ (length $V$) by $W_{\text{token}}$:

$$v_{\text{token}} = e_k \cdot W_{\text{token}}$$

In PyTorch, `nn.Embedding(vocab_size, n_embd)` performs this lookup in $O(1)$ time without constructing one-hot matrices.

---

### 2. Positional Embeddings ($W_{\text{pos}}$)

Self-attention ([`08_ATTENTION.md`](08_ATTENTION.md)) is **permutation-equivariant**: if you scramble the order of input vectors, self-attention produces the exact same output vectors, scrambled in the same order. Attention has no inherent concept of word order.

To make position matter, we inject positional information into each token vector.

#### Learned Absolute Positional Embeddings
DevLLM uses learned positional embeddings:
1. Construct a positional index tensor $P = [0, 1, 2, \dots, T-1]$ of shape $(1, T)$.
2. Lookup corresponding row vectors from positional matrix $W_{\text{pos}}$ of shape $(T \times C)$.
3. Add token embeddings and positional embeddings element-wise:

$$\text{Embedding}(b, t) = W_{\text{token}}[X_{b,t}] + W_{\text{pos}}[t]$$

```
   Token Vector  (B, T, C)  "cat"
   + Pos Vector   (1, T, C)  "position 1"
   ─────────────────────────────────────
   Input Tensor  (B, T, C)  "cat at position 1"
```

Both matrices share dimension $C$, so element-wise addition $(+)$ preserves shape $(B, T, C)$ for downstream transformer blocks ([`09_TRANSFORMER.md`](09_TRANSFORMER.md)).

---

### 3. Introduction to Weight Tying

In a language model, the network performs two token-to-vector transitions:
1. **Input:** Discrete Token IDs $\to$ Continuous Vectors (using $W_{\text{token}}$, shape $V \times C$).
2. **Output:** Continuous Hidden Vectors $\to$ Discrete Token Logits (using LM Head $W_{\text{out}}$, shape $C \times V$).

#### The Weight Tying Insight
Press & Wolf (2017) demonstrated that the input lookup matrix $W_{\text{token}}$ and the output projection matrix $W_{\text{out}}$ perform dual geometric roles in the same representation space. By setting:

$$W_{\text{out}} = W_{\text{token}}^T$$

the model reuses the exact same weight matrix for both input and output operations.

```
   INPUT:  Token ID  ──► Lookup W_token[ID] ──────► Hidden Vector (C)
                                                           │
                                                           ▼ (Transformer Blocks)
   OUTPUT: Logits    ◄── Matmul x @ W_token.T ◄── Hidden Vector (C)
```

#### Why Weight Tying is the v0.1 Baseline Default
As established in [`ADR-0003`](../research/design_decisions/ADR-0003-weight-tying-default.md) and [`SPEC.md` §5](../SPEC.md#5-parameter-count-worked-for-v01):
- Saves $V \cdot C$ parameters (8,320 params for v0.1 baseline, over 65,000 for larger vocabs).
- Halves weight and optimizer memory for vocabulary projections on 8 GB Apple Silicon.
- Enforces geometric alignment between input and output representations.

---

## Visual Diagrams

### Combined Embedding Tensor Flow

```
   Token IDs X          Position Indices P
   [19, 4, 11] (1, T)   [0, 1, 2] (1, T)
        │                    │
        ▼                    ▼
   Lookup W_token       Lookup W_pos
   (1, T, C)            (1, T, C)
        │                    │
        └───────────┬────────┘
                    ▼
          Element-wise Addition (+)
                    │
                    ▼
            Input Tensor (B, T, C)  ──► to Block 1
```

---

## Common Mistakes

- **Concatenating instead of Adding Embeddings:** Concatenating token and position vectors expands hidden dimension to $2C$, doubling parameters and breaking block shape invariance. Embeddings must be added element-wise ($+$).
- **Out-of-Bounds Sequence Length:** Passing sequence length $T > \text{block\_size}$, attempting to index past rows of $W_{\text{pos}}$ and causing runtime bounds crashes.
- **Instantiating Standalone LM Head when Weight Tying:** Creating `nn.Linear(n_embd, vocab_size)` separately without sharing `W_token.weight`, wasting $V \cdot C$ parameters and violating [`ADR-0003`](../research/design_decisions/ADR-0003-weight-tying-default.md).

---

## Future Improvements

- **Rotary Position Embeddings (RoPE):** Explore rotating query and key vectors in attention space rather than adding absolute positional embeddings at the input layer.
- **Sinusoidal Position Embeddings:** Compare fixed non-learnable sine/cosine positional encodings against learned positional embeddings.

---

## Glossary

Canonical definitions live in [`19_GLOSSARY.md`](19_GLOSSARY.md). Core embedding terms:

| Term | Definition |
|------|------------|
| Token Embedding | $V \times C$ lookup matrix mapping token IDs to continuous hidden vectors |
| Position Embedding | $T \times C$ matrix mapping sequence positions to positional vectors |
| Permutation Equivariance | Property where rearranging input vectors yields identically rearranged outputs |
| Weight Tying | Reusing input token embedding matrix as the output LM head projection matrix |

---

## Learning Checklist

You master embeddings when you can:

- [ ] Explain why discrete integer IDs cannot be fed directly to neural networks.
- [ ] State the tensor shapes of $W_{\text{token}}$, $W_{\text{pos}}$, and the combined output tensor $(B, T, C)$.
- [ ] Explain why self-attention requires positional embeddings.
- [ ] State why token and position embeddings are added element-wise rather than concatenated.
- [ ] Explain how weight tying saves $V \cdot C$ parameters and memory.

---

## References

- [`ADR-0003`](../research/design_decisions/ADR-0003-weight-tying-default.md) — Weight tying as v0.1 default baseline.
- [`SPEC.md` §5](../SPEC.md#5-parameter-count-worked-for-v01) — Worked parameter count arithmetic.
- Press & Wolf (2017), *Using the Output Embedding to Improve Language Models*.
- Vaswani et al. (2017), *Attention Is All You Need* — Section 3.5 Positional Encoding.

## Further Reading

- [`08_ATTENTION.md`](08_ATTENTION.md) — **Next:** How multi-head self-attention processes context-aware embedded vectors.
- [`16_MODEL_CONFIGURATION.md`](16_MODEL_CONFIGURATION.md) — Hyperparameter specs for `vocab_size`, `n_embd`, `block_size`, and `weight_tying`.

> **Next:** [`08_ATTENTION.md`](08_ATTENTION.md)
