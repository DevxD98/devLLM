# Engineering Principles — the JimmyLabs constitution

> These are the rules the project holds itself to. They are deliberately few, deliberately
> strict, and deliberately about **process**, not code style. When a decision is unclear,
> re-read these; the answer is usually here. Every principle has a *why* — a rule you don't
> understand is a rule you'll break under pressure.

This document is normative. `docs/` teaches; this governs. If code, a doc, or an experiment
violates a principle here, the principle wins (or the principle changes — by ADR).

---

## The ten principles

### 1. Never optimize before profiling.
No optimization is written until a measurement shows it's needed and where.
**Why:** intuition about performance is wrong more often than right, and doubly so on
Apple Silicon's unified-memory/MPS model. Guessing wastes effort and adds complexity that
buys nothing. → [`docs/14_BENCHMARKING.md`](docs/14_BENCHMARKING.md).

### 2. Measure every optimization — before and after.
An optimization without a before/after benchmark, one variable changed, **did not happen**
and does not get committed as an improvement.
**Why:** this is the only defense against "it feels faster." Numbers or it isn't real.
→ [`benchmarks/`](benchmarks/).

### 3. Every architectural decision requires written reasoning.
Non-trivial or hard-to-reverse choices get an ADR before or at the moment they're made.
**Why:** month-4 you will not remember why month-1 you chose pre-norm. The rejected
options are the most valuable record. → [`research/design_decisions/`](research/design_decisions/).

### 4. Every experiment is logged and reproducible.
Copy the template, fix a seed, record the config, write the hypothesis **before** the run.
**Why:** a result you can't reproduce is an anecdote. Reproducibility from `config + seed`
is what separates a lab from a pile of scripts. → [`docs/15_EXPERIMENT_GUIDE.md`](docs/15_EXPERIMENT_GUIDE.md).

### 5. No unexplained magic numbers.
Every constant — `√d_k`, the ×4 FFN expansion, a learning rate, `block_size` — carries a
comment or a doc link explaining where it came from and its cost.
**Why:** a magic number is a decision someone forgot to write down. At our scale, most
magic numbers are also *memory* decisions on 8 GB.

### 6. Every module has tests, and shape tests come first.
No `src/` module is "done" until [`tests/`](tests/) proves its interface — starting with
tensor shapes.
**Why:** the #1 bug in a from-scratch transformer is a wrong shape. Tests are also the
clearest executable statement of what a component *is*. → [`tests/TESTING_STRATEGY.md`](tests/TESTING_STRATEGY.md).

### 7. One change per experiment.
Change one variable at a time. If you must change several, you're running several
experiments, so log several.
**Why:** confounded experiments teach nothing. You can't attribute a result to a cause you
didn't isolate.

### 8. Configuration is separate from implementation.
Model and training hyperparameters live in [`configs/`](configs/), not hard-coded in
scripts. A model is `config + seed`.
**Why:** it makes runs reproducible, comparable, and diffable, and keeps `src/` clean and
testable. → [`docs/16_MODEL_CONFIGURATION.md`](docs/16_MODEL_CONFIGURATION.md).

### 9. Prefer readability over cleverness.
The code is a teaching artifact. If a clever one-liner and a clear five-liner do the same
thing, the five-liner wins.
**Why:** the entire project exists to be *understood*. Cleverness that obscures the idea is
a net negative here, even when it's faster. (When it's meaningfully faster, principle 2
applies: prove it, then comment *why*.)

### 10. Document trade-offs, not just decisions.
Every design doc states what a choice makes *harder*, not only what it makes easier.
**Why:** honest engineering names its costs. A doc that only lists benefits is marketing,
and future-you will trust it less for it.

---

## How the principles show up in the repo

```
   principle                          enforced by
   ─────────────────────────────────  ─────────────────────────────────────
   1,2  profile / measure             benchmarks/ + docs/14
   3    written decisions             research/design_decisions/ (ADRs)
   4,7  reproducible experiments      research/experiment_templates/ + experiments/
   5    no magic numbers              configs/ comments + doc cross-links
   6    tests, shapes first           tests/ + tests/TESTING_STRATEGY.md
   8    config ≠ implementation       configs/ vs src/
   9,10 readable, honest              code review + docs/ trade-off sections
```

The repository's structure is not decoration — it exists to make these principles the path
of least resistance.

---

## Amending the constitution

These principles can change, but only the way anything important changes here: by an ADR in
[`research/design_decisions/`](research/design_decisions/) that states what rule is changing
and why. The constitution is version-controlled like everything else.

## See also

- [`docs/00_PROJECT_VISION.md`](docs/00_PROJECT_VISION.md) — the *why* behind the whole project
- [`docs/03_LEARNING_PATH.md`](docs/03_LEARNING_PATH.md) — how to learn what these principles protect
- [`README.md`](README.md) — the five design commitments (a distilled subset of the above)
