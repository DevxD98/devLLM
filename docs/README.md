# `docs/` — the DevLLM textbook

This folder is a **book**, not a wiki. The files are numbered because they are meant to be
read roughly in order: each builds on vocabulary and intuition established earlier. You can
jump around, but the numbers tell you what you're assumed to already know.

> **Audience:** someone with basic Python and *almost no* machine-learning background.
> Advanced mathematics is **not** assumed — where math appears, it is built up from the
> ground in [`04_MATHEMATICS.md`](04_MATHEMATICS.md).

---

## The reading path

```
                        ┌─────────────────────────────┐
   THE WHY              │ 00 VISION → 01 ROADMAP →     │
   (motivation)        │ 02 ARCHITECTURE → 03 LEARNING │
                        └─────────────┬───────────────┘
                                      ▼
                        ┌─────────────────────────────┐
   THE FOUNDATIONS      │ 04 MATH → 05 NEURAL NETS     │
   (learning machinery) └─────────────┬───────────────┘
                                      ▼
                        ┌─────────────────────────────┐
   FROM TEXT TO MODEL   │ 06 TOKENIZER → 07 EMBEDDINGS │
   (representing language)             │ → 08 ATTENTION │
                        │ → 09 TRANSFORMER → 10 GPT     │
                        └─────────────┬───────────────┘
                                      ▼
                        ┌─────────────────────────────┐
   MAKING IT RUN        │ 11 TRAINING → 12 INFERENCE → │
   (on an M1 Air)       │ 13 APPLE SILICON → 14 BENCH  │
                        └─────────────┬───────────────┘
                                      ▼
                        ┌─────────────────────────────┐
   HOW WE WORK          │ 15 EXPERIMENTS · 16 CONFIG · │
   (the lab's method)   │ 17 DATASETS · 18 PAPERS ·    │
                        │ 19 GLOSSARY · 20 TODO ·       │
                        │ 21 DEVLOG · 22 FUTURE ·       │
                        │ 23 RESOURCES                  │
                        └─────────────────────────────┘
```

## Dependency graph (what you must read first)

An arrow `A → B` means "understanding B properly requires A."

```
   04 MATH ─────────────┬────────────► 08 ATTENTION ──► 09 TRANSFORMER ──► 10 GPT
       │                │                    ▲                                │
       ▼                │                    │                                ▼
   05 NEURAL NETS ──────┴──► 07 EMBEDDINGS ──┘                          11 TRAINING
       │                          ▲                                          │
       │                          │                                         ▼
       └──────────► 06 TOKENIZER ─┘                                    12 INFERENCE
                                                                            │
   13 APPLE SILICON  ◄───────────── informs ──────────── 11, 12, 14 ◄──────┘
```

Read top-to-bottom and you never hit a term that wasn't defined earlier. Every doc also
opens with an explicit **Prerequisites** line and closes with a **Next** pointer.

---

## Full contents

| # | File | One-line purpose |
|---|------|------------------|
| 00 | [`00_PROJECT_VISION.md`](00_PROJECT_VISION.md) | why this project exists, and what "from scratch" means |
| 01 | [`01_ROADMAP.md`](01_ROADMAP.md) | the 3–4 month phase plan |
| 02 | [`02_ARCHITECTURE.md`](02_ARCHITECTURE.md) | high-level module map and standards |
| 03 | [`03_LEARNING_PATH.md`](03_LEARNING_PATH.md) | how to learn everything here, in order, from near-zero |
| 04 | `04_MATHEMATICS.md` | vectors, matrices, gradients, softmax — only what we need |
| 05 | `05_NEURAL_NETWORKS.md` | neurons, MLPs, backprop, training loop intuition |
| 06 | `06_TOKENIZER.md` | turning text into integers; char vs BPE |
| 07 | `07_EMBEDDINGS.md` | integers into vectors; token + positional embeddings |
| 08 | [`08_ATTENTION.md`](08_ATTENTION.md) | self-attention from first principles |
| 09 | [`09_TRANSFORMER.md`](09_TRANSFORMER.md) | assembling a transformer block |
| 10 | `10_GPT_ARCHITECTURE.md` | stacking blocks into a GPT; the LM head |
| 11 | `11_TRAINING_PIPELINE.md` | loss, backprop, optimizer, scheduling, checkpoints |
| 12 | `12_INFERENCE.md` | autoregressive generation, temperature, top-k/p, KV cache |
| 13 | `13_OPTIMIZATION_FOR_APPLE_SILICON.md` | MPS, unified memory, dtype, thermal reality |
| 14 | `14_BENCHMARKING.md` | measuring tokens/sec, memory, throughput honestly |
| 15 | `15_EXPERIMENT_GUIDE.md` | how to run and record an experiment |
| 16 | `16_MODEL_CONFIGURATION.md` | every config field and its cost on 8 GB |
| 17 | `17_DATASET_GUIDE.md` | choosing, preparing, and streaming data |
| 18 | `18_RESEARCH_PAPERS.md` | the reading list and why each paper matters |
| 19 | `19_GLOSSARY.md` | every term, one place (the canonical definitions) |
| 20 | `20_TODO.md` | live task backlog |
| 21 | `21_DEVLOG.md` | dated journal of what happened and what was learned |
| 22 | `22_FUTURE_VERSIONS.md` | what a v2+ would explore |
| 23 | `23_RESOURCES.md` | courses, videos, repos, tools worth your time |

> **Cross-reference rule:** every term is *defined* in exactly one canonical doc (usually
> `04` or `19`) and *linked* everywhere else. If you find the same concept explained twice,
> that's a bug — open a note in [`notes/`](../notes/).

Files 03, 08, and 09 are written to full depth as the **quality bar** for the rest;
04–07 and 10–23 are being filled in to match.
