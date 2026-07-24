# 19 — Glossary

> **Prerequisites:** None. This is the canonical definitions hub for the entire repository.
>
> **Next:** [`20_TODO.md`](20_TODO.md)

---

## Purpose
This document is the **single canonical source of truth** for definitions across the JimmyLabs project. Whenever a term is referenced in `docs/`, `research/`, or `architecture/`, its single canonical definition resides here to maintain strict cross-reference discipline and prevent conceptual drift.

---

## Background
N/A — This is a reference document.

---

## Concepts
N/A — This is a reference document.

---

## Detailed Explanation

| Term | Canonical Definition | Owning Doc |
|---|---|---|
| **AdamW** | Variant of Adam optimizer incorporating decoupled weight decay applied directly to weight parameters. | [`11_TRAINING_PIPELINE.md`](11_TRAINING_PIPELINE.md) |
| **attention** | Mechanism letting each position's vector be replaced by a relevance-weighted average of information from all positions. | [`08_ATTENTION.md`](08_ATTENTION.md) |
| **autoregressive** | Sequential text generation where each predicted token is appended to the input sequence to predict the next token. | [`12_INFERENCE.md`](12_INFERENCE.md) |
| **block_size** | Maximum context sequence length ($T$) processed in a single forward pass. | [`16_MODEL_CONFIGURATION.md`](16_MODEL_CONFIGURATION.md) |
| **BPE** | Byte-Pair Encoding; a subword tokenization algorithm iteratively merging frequent character pairs. | [`06_TOKENIZER.md`](06_TOKENIZER.md) |
| **cross-entropy** | Negative log probability loss measuring accuracy of next-token predictions against target labels. | [`05_NEURAL_NETWORKS.md`](05_NEURAL_NETWORKS.md) |
| **d_k** | Per-head key/query dimension, calculated as total embedding dimension $C$ divided by number of heads $h$. | [`08_ATTENTION.md`](08_ATTENTION.md) |
| **embedding** | Continuous vector representation of a discrete token or position ID. | [`07_EMBEDDINGS.md`](07_EMBEDDINGS.md) |
| **GELU** | Gaussian Error Linear Unit; a smooth activation function weighting inputs by their value. | [`09_TRANSFORMER.md`](09_TRANSFORMER.md) |
| **gradient clipping** | Rescaling global gradient norm to threshold $g_{\text{max}} = 1.0$ to prevent exploding gradients. | [`11_TRAINING_PIPELINE.md`](11_TRAINING_PIPELINE.md) |
| **KV cache** | Cache storing past Key and Value tensors during generation to lower step complexity from $O(T^2)$ to $O(T)$. | [`12_INFERENCE.md`](12_INFERENCE.md) |
| **LayerNorm** | Per-token normalization standardizing feature activations to zero mean and unit variance. | [`09_TRANSFORMER.md`](09_TRANSFORMER.md) |
| **logits** | Raw, unnormalized prediction scores output by the final language model head. | [`04_MATHEMATICS.md`](04_MATHEMATICS.md) |
| **MPS** | Metal Performance Shaders; Apple Silicon GPU acceleration framework in PyTorch. | [`13_OPTIMIZATION_FOR_APPLE_SILICON.md`](13_OPTIMIZATION_FOR_APPLE_SILICON.md) |
| **O(T²)** | Quadratic computational and memory scaling of self-attention w.r.t. sequence length $T$. | [`08_ATTENTION.md`](08_ATTENTION.md) |
| **perplexity** | Exponentiated cross-entropy loss, representing how confused the model is when predicting the next token. | [`11_TRAINING_PIPELINE.md`](11_TRAINING_PIPELINE.md) |
| **pre-norm** | Applying LayerNorm to sublayer inputs (`x + Sublayer(LN(x))`) for clean residual flow. | [`09_TRANSFORMER.md`](09_TRANSFORMER.md) |
| **Q/K/V** | Query, Key, and Value; the three learned projections representing a token's question, label, and content. | [`08_ATTENTION.md`](08_ATTENTION.md) |
| **residual** | Skip connection adding input directly to sublayer output (`x + Sublayer(x)`), maintaining clean gradient flow. | [`09_TRANSFORMER.md`](09_TRANSFORMER.md) |
| **softmax** | Function converting raw logits into a normalized probability distribution summing to 1.0. | [`04_MATHEMATICS.md`](04_MATHEMATICS.md) |
| **temperature** | Logit scaling knob controlling sampling randomness. | [`12_INFERENCE.md`](12_INFERENCE.md) |
| **tokenizer** | Algorithm responsible for converting raw text into discrete sequences of integer IDs. | [`06_TOKENIZER.md`](06_TOKENIZER.md) |
| **top-k** | Sampling truncation method restricting selection to the $k$ most likely vocabulary candidates. | [`12_INFERENCE.md`](12_INFERENCE.md) |
| **top-p** | Sampling truncation method restricting selection to the smallest set of candidates whose cumulative probability exceeds $p$. | [`12_INFERENCE.md`](12_INFERENCE.md) |
| **unified memory** | Apple Silicon architecture where CPU and GPU share the same physical RAM pool. | [`13_OPTIMIZATION_FOR_APPLE_SILICON.md`](13_OPTIMIZATION_FOR_APPLE_SILICON.md) |
| **weight tying** | Reusing input token embedding matrix as the output LM head projection matrix ($W_{\text{out}} = W_{\text{token}}^T$). | [`07_EMBEDDINGS.md`](07_EMBEDDINGS.md) |

---

## Visual Diagrams (ASCII)
N/A — This is a reference document.

---

## Common Mistakes
N/A — This is a reference document.

---

## Future Improvements
N/A — This is a reference document.

---

## Glossary
(This entire document serves as the central glossary.)

---

## Learning Checklist
N/A — This is a reference document.

---

## References
N/A — This is a reference document.

---

## Further Reading
N/A — This is a reference document.

> **Next:** [`20_TODO.md`](20_TODO.md)
