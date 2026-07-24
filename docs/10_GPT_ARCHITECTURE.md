# 10 — GPT Architecture

> **Prerequisites:** [`07_EMBEDDINGS.md`](07_EMBEDDINGS.md) (token and position embeddings), [`08_ATTENTION.md`](08_ATTENTION.md) (multi-head self-attention), [`09_TRANSFORMER.md`](09_TRANSFORMER.md) (the pre-norm transformer block), and [`SPEC.md`](../SPEC.md).
>
> **Next:** [`11_TRAINING_PIPELINE.md`](11_TRAINING_PIPELINE.md)

---

## Purpose

A single transformer block ([`09_TRANSFORMER.md`](09_TRANSFORMER.md)) processes context-aware representations, but one block alone is insufficient for language modeling. A **Generative Pre-trained Transformer (GPT)** is a decoder-only model created by stacking $L = \text{n\_layer}$ pre-norm transformer blocks, preceded by an embedding layer, and followed by a final LayerNorm and language model head.

This document explains how individual components assemble into a complete, shape-preserving GPT model, works through the exact parameter arithmetic matching [`SPEC.md` §5](../SPEC.md#5-parameter-count-worked-for-v01), and describes the full forward pass from input token IDs to next-token logit predictions.

---

## Background

### Decoder-only vs Encoder-Decoder

Early transformer architectures (Vaswani et al., 2017) used an encoder-decoder structure for translation. Modern autoregressive language models (GPT-2, GPT-3, LLaMA) use a **decoder-only** architecture:
- Every position attends only to current and previous positions via causal masking ([`08_ATTENTION.md`](08_ATTENTION.md)).
- Cross-attention layers are omitted.
- The model computes $P(x_t \mid x_{<t})$ for every token in the sequence simultaneously during training.

### Shape Invariance across the Stack

Because token embeddings ([`07_EMBEDDINGS.md`](07_EMBEDDINGS.md)) map inputs to shape $(B, T, C)$, and every pre-norm transformer block takes $(B, T, C)$ and returns $(B, T, C)$, stacking $L$ blocks requires zero shape adapter logic:

```
   (B, T) ─► Embeddings ─► (B, T, C) ─► [ Block × L ] ─► (B, T, C) ─► Final LN ─► LM Head ─► (B, T, V)
```

---

## Concepts

Core GPT architectural units:

- **Token & Position Embeddings:** Map integer token IDs and positional indices into initial hidden representations $(B, T, C)$.
- **Transformer Block Stack:** $L$ identical pre-norm blocks running attention and feed-forward sublayers in sequence.
- **Final LayerNorm:** Normalizes hidden representations from the unnormalized residual highway prior to logit projection ([`ADR-0002`](../research/design_decisions/ADR-0002-pre-norm.md)).
- **Language Model (LM) Head:** Linear projection from model dimension $C$ to vocabulary size $V$, producing unnormalized logit scores $(B, T, V)$.
- **Tied LM Head:** Reusing the token embedding matrix transpose $W_{\text{token}}^T$ for the LM head projection ([`ADR-0003`](../research/design_decisions/ADR-0003-weight-tying-default.md)).

---

## Detailed Explanation

### 1. End-to-End Forward Pass Dataflow

For a batch of token ID sequences $X$ of shape $(B, T)$:

1. **Embedding Stage:**
   $$H_0 = \text{Embedding}(X) = W_{\text{token}}[X] + W_{\text{pos}}[0 \dots T-1] \quad \text{shape: } (B, T, C)$$

2. **Block Stacking Stage (Pre-Norm):**
   For $l = 1 \dots L$:
   $$H_l = \text{Block}_l(H_{l-1}) \quad \text{shape: } (B, T, C)$$

3. **Final Normalization:**
   $$\tilde{H} = \text{LayerNorm}_{\text{final}}(H_L) \quad \text{shape: } (B, T, C)$$

4. **LM Head Projection (Weight Tied):**
   $$\text{Logits} = \tilde{H} \cdot W_{\text{token}}^T \quad \text{shape: } (B, T, V)$$

The resulting logits at position $t$ represent raw unnormalized scores predicting token $t+1$.

---

### 2. Parameter Count Arithmetic (Worked for v0.1 Baseline)

Let $V = 65$, $L = 4$, $h = 4$, $C = 128$, $T = 128$, with weight tying active ([`SPEC.md` §5](../SPEC.md#5-parameter-count-worked-for-v01)).

```
   1. Token Embedding (W_token):
      V × C = 65 × 128                                    =     8,320
   2. Position Embedding (W_pos):
      T × C = 128 × 128                                   =    16,384

   3. Per Transformer Block (~12 C^2):
      Attention Wq, Wk, Wv, Wo:  4 × C^2 = 4 × 16,384      =    65,536
      Attention Biases:          4 × C                     =       512
      LayerNorms (2 × 2C):       4 × C                     =       512
      FFN W1 (C × 4C):           4 × C^2                   =    65,536
      FFN W2 (4C × C):           4 × C^2                   =    65,536
      FFN Biases:                4C + C                    =       640
      ─────────────────────────────────────────────────────────────
      Block Total                                          ≈   198,272

   4. Stack of L=4 Blocks:
      4 × 198,272                                          =   793,088

   5. Final LayerNorm:
      2 × C                                                =       256

   6. Output LM Head (Tied to W_token):
      Weight tied to W_token (0 additional parameters)     =         0
   ────────────────────────────────────────────────────────────────
   TOTAL PARAMETERS (v0.1 baseline)                        ≈   818,048  (~0.82M)
```

#### Key Architectural Rule
The $L$ transformer blocks dominate model capacity, with FFN layers holding $\approx 2/3$ of each block's parameters. Untying the head would add an unnecessary $V \cdot C = 8,320$ parameters.

---

## Visual Diagrams

### Full Decoder-Only GPT Network Architecture

```
   Token IDs (B, T) ──────► Token Emb (V × C) ──┐
                                                 (+) ──► H0 (B, T, C)
   Pos Indices (1, T) ────► Pos Emb   (T × C) ──┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ Pre-Norm Block 1      │
                     │  x = x + MHSA(LN(x))  │
                     │  x = x + FFN(LN(x))   │
                     └───────────┬───────────┘
                                 │  H1 (B, T, C)
                                 ▼
                                ...
                                 │  H_(L-1) (B, T, C)
                                 ▼
                     ┌───────────────────────┐
                     │ Pre-Norm Block L      │
                     └───────────┬───────────┘
                                 │  HL (B, T, C)
                                 ▼
                     ┌───────────────────────┐
                     │ Final LayerNorm       │
                     └───────────┬───────────┘
                                 │  H_final (B, T, C)
                                 ▼
                     ┌───────────────────────┐
                     │ LM Head (@ W_token.T) │ ◄── Tied Weights
                     └───────────┬───────────┘
                                 │
                                 ▼
                           Logits (B, T, V)
```

---

## Common Mistakes

- **Forgetting Final LayerNorm:** Omitting `final_ln` before the LM head in pre-norm architectures, passing unnormalized residual activations directly to logit projection.
- **Untying LM Head Weights:** Creating a standalone output linear layer without sharing `W_token`, violating [`ADR-0003`](../research/design_decisions/ADR-0003-weight-tying-default.md) and inflating weight/optimizer memory.
- **Flawed Causal Masking:** Allowing attention tokens to peek at future positions, causing loss to drop artificially fast during training while producing gibberish during generation.
- **Shape Mismatches in Residual Stream:** Modifying hidden dimension width inside a sublayer without restoring it to $C$ before the residual addition `$x + \text{Sublayer}(x)$`.

---

## Future Improvements

- Add support for Mixture of Experts (MoE) block variants in Phase 5 research.
- Implement SwiGLU gated activations as an alternative to GELU in the FFN.

---

## Glossary

Canonical definitions live in [`19_GLOSSARY.md`](19_GLOSSARY.md). GPT architecture terms:

| Term | Definition |
|------|------------|
| Decoder-Only | Autoregressive transformer architecture using causal masking without cross-attention |
| Final LayerNorm | Normalization layer applied after the block stack before logit projection |
| LM Head | Linear layer projecting hidden state $(B, T, C)$ to token logits $(B, T, V)$ |
| $12 \cdot C^2$ Rule | Rule of thumb estimating total parameter count of one transformer block |

---

## Learning Checklist

You master GPT architecture when you can:

- [ ] Draw the complete dataflow from token IDs $(B, T)$ to logits $(B, T, V)$.
- [ ] Explain why pre-norm architectures require a final LayerNorm layer.
- [ ] Derive the $\approx 12 \cdot C^2$ per-block parameter approximation.
- [ ] Calculate total parameter count for $V=65, L=4, C=128, T=128$ tied.
- [ ] State why cross-attention is absent in decoder-only models.

---

## References

- [`SPEC.md` §2–5](../SPEC.md#2-architecture) — Canonical architectural specification and parameter count math.
- [`ADR-0002`](../research/design_decisions/ADR-0002-pre-norm.md) — Pre-Layer Normalization decision.
- [`ADR-0003`](../research/design_decisions/ADR-0003-weight-tying-default.md) — Weight Tying decision.
- Radford et al. (2019), *Language Models are Unsupervised Multitask Learners* (GPT-2).

## Further Reading

- [`11_TRAINING_PIPELINE.md`](11_TRAINING_PIPELINE.md) — **Next:** Training the assembled GPT model with loss, AdamW, and schedulers.
- [`architecture/model_architecture.md`](../architecture/model_architecture.md) — Tensor shape contracts for all modules in DevLLM.

> **Next:** [`11_TRAINING_PIPELINE.md`](11_TRAINING_PIPELINE.md)
