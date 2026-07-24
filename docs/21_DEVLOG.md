# 21 — Devlog

> **Prerequisites:** [`20_TODO.md`](20_TODO.md).
>
> **Next:** [`22_FUTURE_VERSIONS.md`](22_FUTURE_VERSIONS.md)

---

## Purpose

The Devlog is a chronological record of major project milestones, architectural decisions, and lab notes. It records *how* JimmyLabs evolved from initial design scaffolds to a working 1–4M parameter model.

---

## Chronological Entries

### 2026-07-24 — Phase 0–4 Core Documentation Completion
- **Milestone:** Completed foundational, model architecture, training, inference, and experiment documentation across `docs/04` through `docs/16`.
- **Key Deliverables:**
  - `docs/04_MATHEMATICS.md`: Established canonical definitions for matrix multiplication and softmax.
  - `docs/05_NEURAL_NETWORKS.md`: Documented the overfit-a-batch golden rule for ML debugging.
  - `docs/06_TOKENIZER.md` & `docs/07_EMBEDDINGS.md`: Tokenization contracts, positional embeddings, and weight tying concepts.
  - `docs/10_GPT_ARCHITECTURE.md`: Decoder-only model assembly matching `SPEC.md` §5 param math.
  - `docs/11_TRAINING_PIPELINE.md` & `docs/12_INFERENCE.md`: Full training optimization loop and autoregressive sampling mechanics.
  - `docs/15_EXPERIMENT_GUIDE.md` & `docs/16_MODEL_CONFIGURATION.md`: Scientific lab method and annotated `configs/model_v0_1_char_100k.yaml`.
- **Architectural Fixes Actioned:**
  - Added `ADR-0002` (Pre-Layer Normalization by Default).
  - Added `ADR-0003` (Weight Tying as v0.1 Default Baseline).
  - Updated `docs/01_ROADMAP.md` and `research/ARCHITECTURE_REVIEW.md` checkboxes.

---

### 2026-07-24 — Research Platform & Constitution Infrastructure
- **Milestone:** Established repository constitution and specification source of truth.
- **Key Deliverables:**
  - `ENGINEERING_PRINCIPLES.md`: 10 core principles governing performance profiling, reproducibility, and configuration separation.
  - `SPEC.md`: Living technical specification with worked parameter and 8 GB memory arithmetic.
  - `research/ARCHITECTURE_REVIEW.md`: Comprehensive Staff Engineer architecture review.
  - `research/OPTIMIZATION_BACKLOG.md`: Prioritized backlog for Apple Silicon MPS optimization.

---

### 2026-07-24 — Initial Repository Scaffold
- **Milestone:** Created repository structure, documentation dependency graph, and initial flagship documents (`docs/03`, `docs/08`, `docs/09`).
- **Key Rationale:** Documentation-first architecture—designing and understanding components before writing code.

---

> **Next:** [`22_FUTURE_VERSIONS.md`](22_FUTURE_VERSIONS.md)
