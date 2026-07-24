# TinyLLM-Mac — Roadmap

## Phase 0 — Foundations (Weeks 1–3)

### Learn
- Linear algebra required for deep learning
- Probability basics
- Gradient descent
- Backpropagation
- Neural networks
- PyTorch fundamentals

### Build
- Simple neuron
- MLP
- XOR
- MNIST classifier

Deliverable:
A solid understanding of how learning works before building transformers.

---

## Phase 1 — NLP Foundations (Weeks 4–5)

Build:
- Character tokenizer
- Vocabulary builder
- Dataset loader
- Batch generator

Study:
- Character tokenization
- BPE
- Embeddings

Deliverable:
Raw text -> token IDs.

---

## Phase 2 — TinyGPT (Weeks 6–8)

Implement:

- Embedding
- Positional embedding
- Self-attention
- Multi-head attention
- Feed-forward
- LayerNorm
- Residual connections
- Transformer block
- GPT model (with weight tying as v0.1 default; see SPEC §5)

Target:
~1M parameters.

Deliverable:
Working forward pass.

---

## Phase 3 — Training (Weeks 9–10)

Implement:

- Training loop
- Validation
- Checkpointing
- Learning-rate scheduler
- Gradient clipping
- Logging

Train on:
- TinyStories
- Shakespeare
- Small code corpus

Deliverable:
First coherent text generation.

---

## Phase 4 — Optimization (Weeks 11–14)

Profile before optimizing.

Measure:
- Tokens/second
- RAM
- MPS utilization
- Training throughput

Optimize:
- Better batching
- Mixed precision (where supported)
- KV cache
- Efficient sampling
- Quantization experiments
- Memory usage

Deliverable:
Fast Apple Silicon inference.

---

## Phase 5 — Research

Read:
- Attention Is All You Need
- GPT-1
- GPT-2
- GPT-3
- Chinchilla
- LoRA
- FlashAttention

Experiment with architecture changes and document every result.
