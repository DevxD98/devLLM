# `experiments/` — the run log

## Why this folder exists

This is where hypotheses meet reality. Each experiment gets its own numbered folder
containing a filled-in copy of the experiment template, the exact config used, and the
result. The folder is the unit of reproducibility: given `NNN_name/`, anyone should be
able to re-run and get the same numbers.

## Convention

```
experiments/
├── 001_char-tokenizer-shakespeare/
│   ├── experiment.md        ← copied from research/experiment_templates/
│   ├── config.yaml          ← the exact config (or a link to configs/)
│   └── results/             ← loss curves, samples, logs
├── 002_...
```

- Number monotonically. Never renumber — links depend on it.
- The hypothesis is written **before** the run. The result is written **after**.
- A failed experiment is kept, not deleted. Null results are results.

## What does NOT belong here

- Performance timings for their own sake → [`benchmarks/`](../benchmarks/)
- The reusable blank form → [`research/experiment_templates/`](../research/experiment_templates/)

See [`docs/15_EXPERIMENT_GUIDE.md`](../docs/15_EXPERIMENT_GUIDE.md).
