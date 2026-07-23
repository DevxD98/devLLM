# Experiment NNN — <short descriptive title>

> Copy this file to `experiments/NNN_short-name/experiment.md`.
> Fill **Hypothesis → Setup** *before* running. Fill **Results → Conclusion** *after*.
> Never edit the hypothesis to match the result — that's how you lie to your future self.

| Field | Value |
|-------|-------|
| ID | NNN |
| Date started | YYYY-MM-DD |
| Author | |
| Status | planned / running / done / abandoned |
| Related | ADR-XXXX · docs/… · benchmark NNN |

## 1. Question
_What single question does this experiment answer? One sentence._

## 2. Hypothesis
_What do you expect to happen, and **why**? State it before running so the result can
surprise you._

## 3. Setup (the reproducibility contract)
- **Config:** `configs/....yaml` (or paste the exact fields)
- **Model:** n_layer / n_head / n_embd / block_size / params
- **Data:** which corpus, which split, prepared how
- **Seed:** (fixed — required)
- **Device / dtype:** mps / cpu · fp32 / fp16 / bf16
- **What changes vs baseline:** _exactly one variable, ideally_
- **Baseline:** which prior experiment/checkpoint this is compared against

## 4. Metric definition
_How is success measured, precisely? (e.g. "validation cross-entropy over the held-out
10% split, evaluated every 500 steps.") A vague metric makes the result unfalsifiable._

## 5. Results
_Numbers, curves, and sample generations. Link plots in `results/`._

| Metric | Baseline | This run | Δ |
|--------|----------|----------|---|
| val loss | | | |
| tokens/sec | | | |

## 6. Conclusion
_Did the hypothesis hold? What did you actually learn? Be honest — a refuted hypothesis is
a successful experiment._

## 7. Follow-ups
_New questions this raised. Promote real decisions to an ADR in
`research/design_decisions/`._
