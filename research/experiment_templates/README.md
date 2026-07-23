# `research/experiment_templates/`

The blank forms every experiment copies before it runs. Standardizing the *shape* of an
experiment is what makes two runs — possibly months apart — comparable.

- `EXPERIMENT_TEMPLATE.md` — copy this into [`experiments/`](../../experiments/) as
  `NNN_short-name/experiment.md`, fill the hypothesis **before** running, the result
  **after**.

## Why a template at all?

Un-templated experiments rot. You forget the seed, the exact config, whether validation
loss was measured the same way. The template forces the five things that make a run
reproducible: **hypothesis, config, seed, metric definition, and outcome** — recorded in
that order, so the conclusion can never be quietly written before the data.

See [`docs/15_EXPERIMENT_GUIDE.md`](../../docs/15_EXPERIMENT_GUIDE.md) for the full method.
