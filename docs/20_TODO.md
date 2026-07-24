# 20 — TODO & Task Tracker

> **Prerequisites:** [`19_GLOSSARY.md`](19_GLOSSARY.md).
>
> **Next:** [`21_DEVLOG.md`](21_DEVLOG.md)

---

## Purpose

This document tracks active development tasks, milestone goals, and architecture review action items across the DevLLM project. It ensures every folder earns its keep and all implementation milestones are completed systematically.

---

## Architecture Review Action Items (v0.1 Milestone)

- [x] **R1:** Write `SPEC.md` with explicit parameter and memory arithmetic.
- [x] **R2:** Establish weight tying as v0.1 default baseline (`ADR-0003`).
- [x] **R3:** Folder audit — verify every top-level folder holds active artifacts by v0.1.
- [x] **R5:** Write evaluation contract into `SPEC.md`.
- [x] **ADR-0002:** Document Pre-Layer Normalization in `research/design_decisions/ADR-0002-pre-norm.md`.
- [x] **ADR-0003:** Document Weight Tying in `research/design_decisions/ADR-0003-weight-tying-default.md`.

---

## Phase Milestones & Core Documentation

### Phase 0 & 1 — Foundations & Tokenization
- [x] Write `docs/04_MATHEMATICS.md` (vectors, matmul, derivatives, softmax).
- [x] Write `docs/05_NEURAL_NETWORKS.md` (neuron, MLP, backprop, overfit-a-batch).
- [x] Write `docs/06_TOKENIZER.md` (character tokenizer, BPE, round-trip contract).
- [x] Write `docs/07_EMBEDDINGS.md` (token + positional embeddings, weight tying intro).

### Phase 2 — TinyGPT Model Architecture
- [x] Write `docs/08_ATTENTION.md` (scaled dot-product attention, multi-head, causal mask).
- [x] Write `docs/09_TRANSFORMER.md` (pre-norm transformer block).
- [x] Write `docs/10_GPT_ARCHITECTURE.md` (full decoder-only model assembly).

### Phase 3 & 4 — Training & Optimization
- [x] Write `docs/11_TRAINING_PIPELINE.md` (loss, AdamW, LR schedules, grad clip).
- [x] Write `docs/12_INFERENCE.md` (autoregressive loop, sampling knobs, KV cache concept).
- [x] Write `docs/15_EXPERIMENT_GUIDE.md` (reproducible scientific method).
- [x] Write `docs/16_MODEL_CONFIGURATION.md` & `configs/model_v0_1_char_100k.yaml`.

---

## Code Implementation Tasks (`src/` & `tests/`)

- [ ] Implement character tokenizer in `src/tokenizer/char_tokenizer.py`.
- [ ] Implement shape tests in `tests/test_shapes.py`.
- [ ] Implement overfit-a-batch harness in `tests/test_overfit.py`.
- [ ] Implement pre-norm transformer block in `src/model/transformer.py`.
- [ ] Implement training loop in `src/train.py`.

---

> **Next:** [`21_DEVLOG.md`](21_DEVLOG.md)
