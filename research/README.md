# `research/` — the lab notebook

## Why this folder exists

A model is only half of an educational project. The other half is the **record of why
it looks the way it does**: which papers shaped it, which experiments were run, what was
measured, and which decisions were made (and later reversed). This folder is the memory
of the project's reasoning. If JimmyLabs is a small research lab, `research/` is its bound
notebook — nothing here is throwaway.

## Structure

```
research/
├── paper_notes/           one note per paper, in our own words
│   └── attention_is_all_you_need.md   (worked example)
├── experiment_templates/  the fill-in-the-blank form for any experiment
├── benchmark_templates/   the fill-in-the-blank form for any measurement
└── design_decisions/      ADRs — one per irreversible-ish choice
```

## The four sub-folders

- **`paper_notes/`** — Reading a paper is not the same as understanding it. Each note
  restates the paper's core idea in plain language, extracts the one or two mechanisms
  we actually reuse, and records what we deliberately ignore and why.

- **`experiment_templates/`** — Every experiment starts by copying the template. This
  guarantees each run records its hypothesis, config, seed, and result in the same shape,
  so results are comparable months apart. Filled-in runs live in [`experiments/`](../experiments/).

- **`benchmark_templates/`** — Same discipline for performance. A benchmark that doesn't
  record the machine state, dtype, and batch size is not reproducible and does not count.

- **`design_decisions/`** — Architecture Decision Records. Each ADR captures the context,
  the options considered, the decision, and the consequences — so a choice made in month
  1 is still explainable in month 4.

## The rule

> No result without a record. No decision without a reason. No optimization without a
> before-and-after.

## See also

- [`experiments/`](../experiments/) · [`benchmarks/`](../benchmarks/)
- [`docs/15_EXPERIMENT_GUIDE.md`](../docs/15_EXPERIMENT_GUIDE.md)
- [`docs/18_RESEARCH_PAPERS.md`](../docs/18_RESEARCH_PAPERS.md)
