# TinyLLM-Mac — Project Vision

## Mission

Build a tiny GPT-style language model from scratch on a **MacBook Air M1 (8 GB)** to deeply understand how modern language models work.

This is **not** a project to compete with ChatGPT or other frontier models. It is an educational engineering project with an emphasis on understanding, correctness, reproducibility, and optimization.

## Core Vision

The goal is to prove that a carefully engineered language model with only **1–4 million parameters** can be:

- Built from scratch
- Trained locally
- Fast enough for interactive use
- Well documented
- Optimized specifically for Apple Silicon

## What "from scratch" means

We will implement ourselves:

- Tokenizer
- Embedding layer
- Positional embeddings
- Multi-head self-attention
- Feed-forward network
- Layer normalization
- Residual connections
- Transformer blocks
- GPT architecture
- Training loop
- Text generation
- Checkpointing
- Evaluation pipeline

We will **not** use prebuilt Transformer implementations or pretrained models.

PyTorch will only provide:
- Tensor operations
- Automatic differentiation
- Optimizers
- MPS acceleration

## Success Criteria

By the end of the project I should be able to:

1. Explain every major component of a GPT model.
2. Implement the model from a blank project.
3. Train it locally on my M1 Mac.
4. Measure and optimize performance.
5. Document every engineering decision.

## Guiding Principles

1. Understanding over speed.
2. Correctness before optimization.
3. Measure every optimization.
4. Small, elegant, maintainable code.
5. Every experiment is reproducible.
