# 19 — Glossary

> **Prerequisites:** None. This is the canonical definitions hub for the entire repository.
>
> **Next:** [`20_TODO.md`](20_TODO.md)

---

## Purpose

This document is the **single canonical source of truth** for definitions across the DevLLM project. Whenever a term is referenced in `docs/`, `research/`, or `architecture/`, its single canonical definition resides here to maintain strict cross-reference discipline and prevent conceptual drift.

---

## Terms & Canonical Definitions

### A
- **Activation Function:** Non-linear mathematical function (e.g. GELU, ReLU) applied to layer outputs to enable neural networks to learn non-linear functions ([`05_NEURAL_NETWORKS.md`](05_NEURAL_NETWORKS.md)).
- **AdamW:** Variant of Adam optimizer incorporating decoupled weight decay applied directly to matmul weight parameters ([`11_TRAINING_PIPELINE.md`](11_TRAINING_PIPELINE.md)).
- **Architecture Decision Record (ADR):** Append-only document capturing a critical design decision, context, rejected alternatives, and consequences ([`research/design_decisions/`](../research/design_decisions/)).
- **Attention Score Matrix ($QK^T / \sqrt{d_k}$):** The $T \times T$ (or $B \cdot h \cdot T \cdot T$) tensor storing pairwise relevance scores between queries and keys ([`08_ATTENTION.md`](08_ATTENTION.md)).
- **Autoregressive Generation:** Sequential text generation where each predicted token is appended to the input sequence to predict the next token ([`12_INFERENCE.md`](12_INFERENCE.md)).

### B
- **Backpropagation:** Algorithm applying the calculus chain rule backward through the computation graph to calculate loss gradients w.r.t. parameters ([`05_NEURAL_NETWORKS.md`](05_NEURAL_NETWORKS.md)).
- **Batch Size ($B$):** Number of independent sequence samples processed in parallel during one optimization step ([`16_MODEL_CONFIGURATION.md`](16_MODEL_CONFIGURATION.md)).
- **Block Size ($T$):** Maximum context sequence length processed in a single forward pass ([`16_MODEL_CONFIGURATION.md`](16_MODEL_CONFIGURATION.md)).
- **Byte-Pair Encoding (BPE):** Subword tokenization algorithm iteratively merging frequent character pairs into vocabulary tokens ([`06_TOKENIZER.md`](06_TOKENIZER.md)).

### C
- **Causal Mask:** Triangular matrix setting future positions ($j > i$) to $-\infty$ before softmax, ensuring tokens attend only to past and current positions ([`08_ATTENTION.md`](08_ATTENTION.md)).
- **Checkpoint:** Saved dictionary file containing model parameters, optimizer state, step count, and validation loss ([`11_TRAINING_PIPELINE.md`](11_TRAINING_PIPELINE.md)).
- **Cosine Decay:** Learning rate schedule decaying $\eta$ smoothly along a cosine curve from $\eta_{\text{max}}$ down to $\eta_{\text{min}}$ ([`11_TRAINING_PIPELINE.md`](11_TRAINING_PIPELINE.md)).
- **Cross-Entropy Loss:** Negative log probability loss measuring accuracy of next-token predictions against target labels ([`05_NEURAL_NETWORKS.md`](05_NEURAL_NETWORKS.md)).

### D
- **Decoder-Only:** Transformer architecture using causal self-attention without cross-attention or an encoder stack (e.g. GPT-2, DevLLM) ([`10_GPT_ARCHITECTURE.md`](10_GPT_ARCHITECTURE.md)).
- **Dot Product:** Operation taking two equal-length vectors and producing a scalar measuring directional alignment ([`04_MATHEMATICS.md`](04_MATHEMATICS.md)).
- **Dropout:** Regularization technique zeroing random activations with probability $p$ during training ([`16_MODEL_CONFIGURATION.md`](16_MODEL_CONFIGURATION.md)).

### E
- **Embedding:** Continuous vector representation of a discrete token or position ID ([`07_EMBEDDINGS.md`](07_EMBEDDINGS.md)).
- **Embedding Dimension ($n\_embd$ / $C$):** Hidden vector width across model blocks ([`16_MODEL_CONFIGURATION.md`](16_MODEL_CONFIGURATION.md)).

### F
- **Feed-Forward Network (FFN):** Per-position MLP sublayer ($W_2 \cdot \text{GELU}(W_1 \cdot x)$) expanding width to $4C$ before projecting back to $C$ ([`09_TRANSFORMER.md`](09_TRANSFORMER.md)).
- **Forward Pass:** Computation flow passing input tensors through model layers to generate predictions ([`05_NEURAL_NETWORKS.md`](05_NEURAL_NETWORKS.md)).

### G
- **GELU (Gaussian Error Linear Unit):** Smooth activation function used in transformer FFN sublayers ([`09_TRANSFORMER.md`](09_TRANSFORMER.md)).
- **Gradient:** Vector of partial derivatives $\frac{\partial L}{\partial w}$ indicating direction of steepest loss increase ([`04_MATHEMATICS.md`](04_MATHEMATICS.md)).
- **Gradient Clipping:** Rescaling global gradient norm to threshold $g_{\text{max}} = 1.0$ to prevent exploding gradients ([`11_TRAINING_PIPELINE.md`](11_TRAINING_PIPELINE.md)).

### H
- **Head ($h$ / $n\_head$):** One parallel self-attention operation operating on sub-dimension $d_k = C / h$ ([`08_ATTENTION.md`](08_ATTENTION.md)).

### K
- **Key ($K$):** Learned projection vector representing what a token advertises to queries ([`08_ATTENTION.md`](08_ATTENTION.md)).
- **KV Cache:** Cache storing past Key and Value tensors during generation to lower step complexity from $O(T^2)$ to $O(T)$ ([`12_INFERENCE.md`](12_INFERENCE.md)).

### L
- **Layer Normalization (LayerNorm):** Per-token normalization standardizing feature activations to zero mean and unit variance ([`09_TRANSFORMER.md`](09_TRANSFORMER.md)).
- **Learning Rate ($\eta$):** Scalar step size multiplier used during gradient descent updates ([`04_MATHEMATICS.md`](04_MATHEMATICS.md)).
- **Logits:** Raw, unnormalized prediction scores output by the final language model head ([`04_MATHEMATICS.md`](04_MATHEMATICS.md)).

### M
- **Matrix Multiplication (Matmul / $@$):** Fundamental linear algebra operation combining two 2-D grids via row-column dot products ([`04_MATHEMATICS.md`](04_MATHEMATICS.md)).
- **MPS (Metal Performance Shaders):** Apple Silicon GPU acceleration framework in PyTorch ([`13_OPTIMIZATION_FOR_APPLE_SILICON.md`](13_OPTIMIZATION_FOR_APPLE_SILICON.md)).
- **Multi-Head Self-Attention (MHSA):** Running $h$ parallel self-attention heads and concatenating outputs ([`08_ATTENTION.md`](08_ATTENTION.md)).

### O
- **Overfit-a-Batch:** Debugging procedure training a model on 2–10 samples until loss reaches near-zero ([`05_NEURAL_NETWORKS.md`](05_NEURAL_NETWORKS.md)).
- **$O(T^2)$ Complexity:** Quadratic computational and memory scaling of self-attention w.r.t. sequence length $T$ ([`08_ATTENTION.md`](08_ATTENTION.md)).

### P
- **Pre-Layer Normalization (Pre-Norm):** Applying LayerNorm to sublayer inputs (`x + Sublayer(LN(x))`) for clean residual flow ([`ADR-0002`](../research/design_decisions/ADR-0002-pre-norm.md)).
- **Query ($Q$):** Learned projection vector representing what a position is looking for in attention ([`08_ATTENTION.md`](08_ATTENTION.md)).

### R
- **Residual Connection:** Skip connection adding input directly to sublayer output (`x + Sublayer(x)`), maintaining clean gradient flow ([`09_TRANSFORMER.md`](09_TRANSFORMER.md)).

### S
- **Scaled Dot-Product Attention:** Core attention formula $\text{softmax}(QK^T / \sqrt{d_k}) V$ ([`08_ATTENTION.md`](08_ATTENTION.md)).
- **Softmax:** Function converting raw logits into a normalized probability distribution summing to 1.0 ([`04_MATHEMATICS.md`](04_MATHEMATICS.md)).

### T
- **Temperature ($T_{\text{temp}}$):** Logit scaling knob controlling sampling randomness ([`12_INFERENCE.md`](12_INFERENCE.md)).
- **Top-$K$ / Top-$P$:** Sampling truncation methods restricting vocabulary candidate selection ([`12_INFERENCE.md`](12_INFERENCE.md)).

### V
- **Value ($V$):** Learned projection vector representing the content retrieved by attention ([`08_ATTENTION.md`](08_ATTENTION.md)).
- **Vocabulary Size ($V$ / $vocab\_size$):** Total number of unique tokens in vocabulary ([`06_TOKENIZER.md`](06_TOKENIZER.md)).

### W
- **Weight Tying:** Reusing input token embedding matrix as the output LM head projection matrix ($W_{\text{out}} = W_{\text{token}}^T$) ([`ADR-0003`](../research/design_decisions/ADR-0003-weight-tying-default.md)).

---

> **Next:** [`20_TODO.md`](20_TODO.md)
