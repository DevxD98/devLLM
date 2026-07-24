# 18 — Research Papers

> **Prerequisites:** [`03_LEARNING_PATH.md`](03_LEARNING_PATH.md) and [`08_ATTENTION.md`](08_ATTENTION.md).
>
> **Next:** [`19_GLOSSARY.md`](19_GLOSSARY.md)

---

## Purpose

JimmyLabs is a research platform as well as an educational codebase. Reading primary literature is essential for understanding *why* transformer components were designed the way they are. 

This document serves as the curated reading list of foundational papers that inspired JimmyLabs. Each paper entry outlines **what it introduced**, **why it matters**, and links to deep-dive notes in [`research/paper_notes/`](../research/paper_notes/).

---

## The Core Reading List

### 1. Transformer Foundations

#### *Attention Is All You Need* (Vaswani et al., 2017)
- **What it introduced:** The original Transformer architecture replacing recurrence/RNNs with scaled dot-product attention and multi-head self-attention.
- **Why it matters:** Introduced the $Q, K, V$ attention formulation ([`08_ATTENTION.md`](08_ATTENTION.md)) and feed-forward block structure ([`09_TRANSFORMER.md`](09_TRANSFORMER.md)).
- **Paper Note:** [`research/paper_notes/attention_is_all_you_need.md`](../research/paper_notes/attention_is_all_you_need.md).

---

### 2. Decoder-Only GPT Lineage

#### *Improving Language Understanding by Generative Pre-Training* (GPT-1, Radford et al., 2018)
- **What it introduced:** Decoder-only transformer architecture trained with autoregressive language modeling followed by fine-tuning.
- **Why it matters:** Demonstrated that self-supervised pre-training on unlabeled text produces rich transferable representations.

#### *Language Models are Unsupervised Multitask Learners* (GPT-2, Radford et al., 2019)
- **What it introduced:** Moving Layer Normalization to the input of each sublayer (Pre-Layer Normalization / Pre-Norm) and scaling model capacity zero-shot.
- **Why it matters:** Formed the structural foundation for JimmyLabs's pre-norm architecture ([`ADR-0002`](../research/design_decisions/ADR-0002-pre-norm.md)) and weight-tying default ([`ADR-0003`](../research/design_decisions/ADR-0003-weight-tying-default.md)).

#### *Language Models are Few-Shot Learners* (GPT-3, Brown et al., 2020)
- **What it introduced:** Scaling decoder-only transformers to 175B parameters, showing emergent in-context learning capabilities.
- **Why it matters:** Proved that scaling model parameters and dataset size predictably improves capabilities (Scaling Laws).

---

### 3. Optimization, Normalization & Embeddings

#### *Using the Output Embedding to Improve Language Models* (Press & Wolf, 2017)
- **What it introduced:** Weight tying between input token embeddings and output linear projection layers ($W_{\text{out}} = W_{\text{token}}^T$).
- **Why it matters:** Directly incorporated as JimmyLabs's v0.1 baseline default ([`ADR-0003`](../research/design_decisions/ADR-0003-weight-tying-default.md)), saving $V \cdot C$ parameters and memory.

#### *Decoupled Weight Decay Regularization* (AdamW, Loshchilov & Hutter, 2019)
- **What it introduced:** Decoupling weight decay ($L_2$ penalty) from gradient updates in Adam optimization.
- **Why it matters:** JimmyLabs's standard optimizer ([`11_TRAINING_PIPELINE.md`](11_TRAINING_PIPELINE.md)).

#### *Gaussian Error Linear Units (GELUs)* (Hendrycks & Gimpel, 2016)
- **What it introduced:** Smooth non-linear activation function weighting inputs by their value rather than gating strictly at zero.
- **Why it matters:** Modern activation choice in JimmyLabs FFN sublayers ([`09_TRANSFORMER.md`](09_TRANSFORMER.md)).

---

### 4. Apple Silicon & Efficiency Papers

#### *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (Dao et al., 2022)
- **What it introduced:** Tiled attention computation avoiding materialization of the $O(T^2)$ attention score matrix in memory.
- **Why it matters:** Key candidate for context length scaling within 8 GB memory limits ([`research/OPTIMIZATION_BACKLOG.md` #11](../research/OPTIMIZATION_BACKLOG.md#11)).

---

## Glossary & References

- [`research/paper_notes/`](../research/paper_notes/) — Deep-dive notes and worked equations for papers.
- [`SPEC.md`](../SPEC.md) — Technical specification built on these primary sources.

> **Next:** [`19_GLOSSARY.md`](19_GLOSSARY.md)
