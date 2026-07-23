# TinyLLM-Mac — Architecture

## High-Level Pipeline

Raw Text
↓
Tokenizer
↓
Token IDs
↓
Embeddings
↓
Positional Embeddings
↓
Transformer Blocks
↓
Language Modeling Head
↓
Next Token Prediction

---

## Modules

### tokenizer/
Responsible for:
- Vocabulary
- Encoding
- Decoding

### dataset/
Responsible for:
- Reading text
- Sequence creation
- Batching

### model/

Contains:

Embedding
↓
Attention
↓
Feed Forward
↓
Transformer Block
↓
TinyGPT

Each module is independent and testable.

### training/

Responsibilities:

- Forward pass
- Loss computation
- Backpropagation
- Optimizer
- Checkpointing
- Validation

### inference/

Responsibilities:

- Prompt encoding
- Autoregressive generation
- Temperature
- Top-k
- Top-p
- KV cache (later)

---

## Apple Silicon Strategy

Target hardware:
- MacBook Air M1
- 8 GB unified memory

Constraints:
- Low memory footprint
- Fast startup
- Local execution
- MPS acceleration

Optimization philosophy:

1. Correctness
2. Profiling
3. Bottleneck identification
4. Targeted optimization

Never optimize blindly.

---

## Model Evolution

Version 0.1
- Character tokenizer
- ~100K parameters

Version 0.2
- 500K parameters

Version 0.3
- 1M parameters

Version 1.0
- 2–4M parameters
- Apple Silicon optimized
- Fully documented
- Benchmark suite
- Research notebook

## Repository Standards

- Every module has unit tests.
- Every experiment is logged.
- Every benchmark is reproducible.
- Every optimization includes before/after metrics.
- Every architectural change has written reasoning.

The repository should read like a miniature research lab rather than a coding tutorial.
