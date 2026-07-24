# 23 — Curated Resources

> **Prerequisites:** [`22_FUTURE_VERSIONS.md`](22_FUTURE_VERSIONS.md).

---

## Purpose

A curated catalog of high-quality courses, repositories, visual guides, and reference implementations that complement JimmyLabs's educational journey. Each resource includes a one-line description and timing indicating when to consume it relative to our curriculum.

---

## Courses

- **[Andrej Karpathy — *Neural Networks: Zero to Hero*](https://karpathy.ai/zero-to-hero.html)**
  - *Description:* The gold standard video series for building neural networks, autograd, tokenizers, and GPTs from scratch in Python.
  - *Timing:* Watch continuously from [Module 0 (Week 3)](CURRICULUM.md#week-3--neurons-mlps-backprop-the-training-loop) through Module 3.

---

## Videos & Visual Guides

- **[3Blue1Brown — *Neural Networks* & *Essence of Linear Algebra*](https://www.3blue1brown.com/)**
  - *Description:* Geometric visual intuitions for matrix transformations, vectors, dot products, and backpropagation.
  - *Timing:* Watch before or during [Module 0 (Weeks 1–2)](CURRICULUM.md#week-1--linear-algebra-you-actually-need).

- **[Jay Alammar — *The Illustrated Transformer*](http://jalammar.github.io/illustrated-transformer/)**
  - *Description:* Excellent visual breakdown of attention matrices, encoder-decoder blocks, and token vectors.
  - *Timing:* Read before starting [Module 2 (Week 6)](CURRICULUM.md#week-6--attention-budget-extra-time).

- **[bbycroft — *LLM Visualization*](https://bbycroft.net/llm)**
  - *Description:* Interactive 3D visualization of tensor flows through a transformer model.
  - *Timing:* Explore during [Module 2 (Week 8)](CURRICULUM.md#week-8--assemble-the-gpt-forward-pass) to build intuition for tensor shapes.

---

## Repositories

- **[nanoGPT (Karpathy)](https://github.com/karpathy/nanoGPT)**
  - *Description:* Minimal, readable PyTorch implementation of GPT-2 training and inference.
  - *Timing:* Review during [Module 3 (Week 9)](CURRICULUM.md#week-9--training) as a reference for your training pipeline.

- **[minGPT (Karpathy)](https://github.com/karpathy/minGPT)**
  - *Description:* Clean, educational re-implementation of GPT, emphasizing readability over extreme performance.
  - *Timing:* Review alongside [Module 2 (Week 8)](CURRICULUM.md#week-8--assemble-the-gpt-forward-pass) to see the architectural wiring in code.

- **[micrograd (Karpathy)](https://github.com/karpathy/micrograd)**
  - *Description:* Tiny scalar-valued autograd engine demonstrating backpropagation from first principles.
  - *Timing:* Study and rebuild alongside [Module 0 (Week 3)](CURRICULUM.md#week-3--neurons-mlps-backprop-the-training-loop).

---

## Papers

- **[Primary Literature Reading List](18_RESEARCH_PAPERS.md)**
  - *Description:* The comprehensive guide to foundational papers (Attention is All You Need, GPT-1/2/3, Chinchilla) underpinning JimmyLabs.
  - *Timing:* Read incrementally from [Module 2 (Week 6)](CURRICULUM.md#ongoing--research-module-5--roadmap-phase-5) onward.

---

## Tools

- **[PyTorch](https://pytorch.org/)**
  - *Description:* The deep learning framework powering JimmyLabs, providing tensor operations and autograd.
  - *Timing:* Fundamental dependency introduced in [Module 0 (Week 3)](CURRICULUM.md#week-3--neurons-mlps-backprop-the-training-loop).

- **[MLX (Apple)](https://github.com/ml-explore/mlx)**
  - *Description:* Efficient machine learning framework designed specifically for Apple Silicon unified memory.
  - *Timing:* Optional advanced reading during [Module 4 (Week 11)](CURRICULUM.md#week-11--the-machine).

---

## Related JimmyLabs Documents

- [`CURRICULUM.md`](CURRICULUM.md) — The week-by-week syllabus.
- [`18_RESEARCH_PAPERS.md`](18_RESEARCH_PAPERS.md) — Primary literature reading list.
- [`research/tiny_gpt_landscape.md`](../research/tiny_gpt_landscape.md) — Architectural breakdown of the tiny GPT ecosystem.
