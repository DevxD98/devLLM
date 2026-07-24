# 15 — Experiment Guide

> **Prerequisites:** [`ENGINEERING_PRINCIPLES.md`](../ENGINEERING_PRINCIPLES.md) (Principles 4 & 7) and [`14_BENCHMARKING.md`](14_BENCHMARKING.md).
>
> **Next:** [`16_MODEL_CONFIGURATION.md`](16_MODEL_CONFIGURATION.md)

---

## Purpose

This document details the scientific methodology governing experimentation in DevLLM. It transforms machine learning from an ad-hoc process of trial-and-error into a disciplined, hypothesis-driven lab practice.

By standardizing experimental design through [`research/experiment_templates/EXPERIMENT_TEMPLATE.md`](../research/experiment_templates/EXPERIMENT_TEMPLATE.md), DevLLM enforces **Principle 4** (*Every experiment is logged and reproducible*) and **Principle 7** (*One change per experiment*) from [`ENGINEERING_PRINCIPLES.md`](../ENGINEERING_PRINCIPLES.md). Following this guide ensures that every training run tests a single falsifiable hypothesis, produces verifiable measurements on the target machine, and leaves an immutable audit trail of learning.

---

## Background

### The problem with ad-hoc machine learning

In typical hobby or unstructured ML projects, experiments suffer from critical methodological flaws:
1. **Confounded variables:** Changing learning rate, batch size, and architecture simultaneously makes it impossible to attribute performance gains or regressions to a specific root cause.
2. **Hindsight bias (post-hoc explanation):** Writing explanations after observing results leads to self-deception and false conclusions about model behavior.
3. **Unreproducible setups:** Omitting random seeds, environment details, or exact config snapshots prevents future researchers (or future-you) from verifying the outcome.
4. **Discarded negative results:** Abandoning "failed" runs without logging forfeits valuable negative knowledge, causing repeated mistakes.

### The lab notebook philosophy

DevLLM treats the codebase as a research platform rather than a simple model script. Every experiment is a formal record stored in `experiments/NNN_short-name/experiment.md`. 

A hypothesis is formulated **before** initiating execution. If a run refutes the hypothesis, the experiment is not a failure—it is a successful scientific result that prevents future re-litigation of settled choices.

---

## Concepts

Scientific experimentation in DevLLM rests on five foundational concepts:

- **Hypothesis-Before-Run Contract:** Stating expected outcome and theoretical reasoning prior to launching a training script. Editing the hypothesis post-run is strictly forbidden.
- **Single-Variable Isolation (Principle 7):** Varying exactly one independent variable relative to a control baseline to establish clear cause-and-effect attribution.
- **Deterministic Seed & Environment (Principle 4):** Fixing all random number generators across PyTorch, NumPy, Python `random`, and Apple Silicon MPS hardware to guarantee 100% bitwise or statistical reproducibility.
- **Baseline Delta ($\Delta$):** Measuring metric changes relative to an established checkpoint baseline.
- **ADR Promotion:** Converting validated structural or architectural discoveries into binding Architecture Decision Records in [`research/design_decisions/`](../research/design_decisions/).

```
  ┌─────────────────────────────────────────────────────────────┐
  │                 1. Formulate Hypothesis                     │
  │     "Increasing block_size from 128 to 256 will..."        │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                 2. Lock Setup & Seed                        │
  │     Vary ONLY block_size · Fixed seed 1337 · Target M1      │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                 3. Measure on Target Machine                │
  │     Track val loss & memory; compare Δ against baseline     │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                 4. Document & Promote                       │
  │     Log result in experiments/ · Promote win to ADR         │
  └─────────────────────────────────────────────────────────────┘
```

---

## Detailed Explanation

Every experiment follows the 7-stage protocol defined in [`research/experiment_templates/EXPERIMENT_TEMPLATE.md`](../research/experiment_templates/EXPERIMENT_TEMPLATE.md).

### Section 1 — Question
Formulate a single, concise question. 
- *Bad:* "How can we make the model better?"
- *Good:* "Does replacing GELU with ReLU in the FFN increase validation perplexity on Shakespeare?"

### Section 2 — Hypothesis
State what you expect to observe and **why**, grounded in transformer theory ([`08_ATTENTION.md`](08_ATTENTION.md), [`09_TRANSFORMER.md`](09_TRANSFORMER.md)). 
- *Rule:* The hypothesis must be written before running the script. A refuted hypothesis is a successful experiment because it eliminates an incorrect assumption.

### Section 3 — Setup (The Reproducibility Contract)
Specify the complete execution environment:
- **Config:** Path to the version-controlled `configs/*.yaml` snapshot ([`16_MODEL_CONFIGURATION.md`](16_MODEL_CONFIGURATION.md)).
- **Model:** Architectural parameter summary (`n_layer`, `n_head`, `n_embd`, `block_size`).
- **Data:** Dataset split name and preparation details ([`17_DATASET_GUIDE.md`](17_DATASET_GUIDE.md)).
- **Seed:** Explicit integer seed for all RNGs.
- **Device & Precision:** Hardware backend (`mps` on Apple Silicon) and numeric precision (`fp32`, `fp16`, `bf16`).
- **Isolated Variable:** The exact parameter being modified.
- **Baseline:** The prior experiment ID or baseline checkpoint used for comparison.

### Section 4 — Metric Definition
Define exact success criteria before execution:
- Primary metric: Held-out validation cross-entropy loss and perplexity.
- Secondary metric: Memory footprint or throughput (tokens/sec) measured on the target machine.
- Protocol: Evaluation interval frequency and sample generation prompt set.

### Section 5 — Results
Record empirical data gathered during the run.
- *Rule:* Never invent empirical numbers (tokens/sec, training times, latency). Always state "measure on the target machine" or report exact empirical logs produced by execution on the target hardware.
- Log metrics in a clear comparison table:

| Metric | Baseline | This run | $\Delta$ |
|--------|----------|----------|----------|
| val loss | measured | measured | calculated |
| tokens/sec | measure on target machine | measure on target machine | $\Delta$ |

### Section 6 — Conclusion
Evaluate whether the empirical results confirm or refute the hypothesis. Synthesize technical learnings without bias.

### Section 7 — Follow-ups & ADR Promotion
If an experiment uncovers a permanent architectural improvement (such as weight tying by default in [`ADR-0003`](../research/design_decisions/ADR-0003-weight-tying-default.md)), promote the finding to an ADR in [`research/design_decisions/`](../research/design_decisions/) and update the optimization backlog in [`research/OPTIMIZATION_BACKLOG.md`](../research/OPTIMIZATION_BACKLOG.md).

---

## Visual Diagrams

### Single-Variable Experimentation (Principle 7)

```
  VALID EXPERIMENT (Isolated Variable)
  Baseline Config ──► [ n_layer: 4, n_embd: 128, lr: 1e-3 ]
  Run A           ──► [ n_layer: 6, n_embd: 128, lr: 1e-3 ]  (Δ = n_layer)
  Result          ──► Change in loss is strictly attributable to depth.

  CONFOUNDED EXPERIMENT (Violates Principle 7)
  Baseline Config ──► [ n_layer: 4, n_embd: 128, lr: 1e-3 ]
  Run B           ──► [ n_layer: 6, n_embd: 256, lr: 5e-4 ]  (3 variables!)
  Result          ──► Unattributable noise; experiment teaches nothing.
```

### Reproducibility Loop (Principle 4)

```
    [ Config YAML ] ──┐
                      ├──► [ Seeded Execution ] ──► Exact Bitwise /
    [ Seed 1337   ] ──┘      (MPS Backend)          Statistical Repro
```

---

## Common Mistakes

- **Post-Hoc Hypothesis Modification:** Editing the hypothesis section after observing run results to make the outcome look predicted. This defeats the scientific purpose of the lab notebook.
- **Multi-Variable Sweeps without Controls:** Changing hidden dimension `n_embd` and learning rate `lr` in a single run, making performance attribution impossible (violates Principle 7).
- **Unseeded MPS Execution:** Omitting seed specification across PyTorch MPS backends, leading to non-deterministic run variance.
- **Reporting Unmeasured Numbers:** Inventing synthetic performance numbers instead of directing readers to measure on their target hardware (violates repo principle 2).
- **Discarding Negative Results:** Failing to commit experiment files when an idea does not improve loss. Negative results prevent redundant work.

---

## Future Improvements

- **Automated Experiment Scaffold Script:** A utility script (`scripts/new_experiment.py`) to auto-generate timestamped `experiments/NNN_name/experiment.md` files pre-filled from template schema.
- **Config Auto-Diff Tool:** Automated CLI tool to compare an experiment's YAML config against the baseline config and highlight changed keys.
- **Git Commit Hash Tagging:** Automatically embedding the active git commit hash into experiment headers for auditability.

---

## Glossary

Canonical definitions live in [`19_GLOSSARY.md`](19_GLOSSARY.md). Experimentation terms:

| Term | Meaning |
|------|---------|
| Hypothesis | A pre-execution prediction of model behavior grounded in theory |
| Baseline | The reference control run against which deltas are measured |
| Confounded Experiment | A flawed run where multiple variables change simultaneously |
| Seed | Deterministic initializer for random number generators |
| ADR Promotion | Documenting a validated experiment win as a permanent ADR |

---

## Learning Checklist

You master the experiment method when you can:

- [ ] Explain why hypotheses must be written prior to execution (Principle 4).
- [ ] Demonstrate single-variable isolation when designing a model sweep (Principle 7).
- [ ] Fill out all 7 sections of `EXPERIMENT_TEMPLATE.md` for a hypothetical run.
- [ ] Explain why negative results are valuable scientific findings.
- [ ] Trace an experiment win to its promotion in `research/design_decisions/`.

---

## References

- [`ENGINEERING_PRINCIPLES.md`](../ENGINEERING_PRINCIPLES.md) — Principles 4 & 7 (The Reproducibility & Isolation Constitution).
- [`research/experiment_templates/EXPERIMENT_TEMPLATE.md`](../research/experiment_templates/EXPERIMENT_TEMPLATE.md) — Canonical template.
- [`14_BENCHMARKING.md`](14_BENCHMARKING.md) — Performance measurement protocols.
- [`SPEC.md`](../SPEC.md) — Technical specification and baseline standards.

## Further Reading

- [`16_MODEL_CONFIGURATION.md`](16_MODEL_CONFIGURATION.md) — Model hyperparameter specs.
- [`research/OPTIMIZATION_BACKLOG.md`](../research/OPTIMIZATION_BACKLOG.md) — Prioritized optimization experiments.
- [`research/design_decisions/README.md`](../research/design_decisions/README.md) — Index of Architecture Decision Records.

> **Next:** [`16_MODEL_CONFIGURATION.md`](16_MODEL_CONFIGURATION.md)
