# Paper Note — Training Compute-Optimal Large Language Models (Chinchilla)

| Field | Value |
|-------|-------|
| Authors | Hoffmann, Borgeaud, Mensch, et al. |
| Year | 2022 |
| Venue | DeepMind / NeurIPS |
| Read on | 2026-07-24 |
| Reused in JimmyLabs | token budget planning, parameter scaling |

> A note is not a summary. It records what **we** understood and what **we** borrowed —
> in our own words. The teaching version lives in
> [`docs/17_DATASET_GUIDE.md`](../../docs/17_DATASET_GUIDE.md); this is the lab's reading record.

## Why we read it

Before this paper, the trend was to train massive models (like GPT-3) on relatively small amounts of data (300B tokens for a 175B model). The Chinchilla paper proved mathematically and empirically that to get the best model for a given compute budget, you should scale the model size and the training data size equally. It dictates how we budget tokens for our training runs.

## Core idea in plain words

Most large language models are significantly undertrained. If you have a fixed compute budget (FLOPs), you get a better model by training a smaller model on much more data, rather than a huge model on less data. The optimal ratio is roughly 20 tokens of training data for every 1 parameter in the model.

## Mechanisms we reuse

- **Compute-optimal parameter-to-token ratio.** We use the Chinchilla ratio (~20:1) to plan our training runs. If we build a 2M parameter model, we aim to train it on roughly 40M tokens to reach its compute-optimal state.
  → [`docs/17_DATASET_GUIDE.md`](../../docs/17_DATASET_GUIDE.md).
- **Learning rate schedule alignment.** The paper emphasizes matching the learning rate cosine decay schedule to the total number of training tokens. We implement this in our training loop to ensure the learning rate decays perfectly by the end of our token budget.
  → [`docs/11_TRAINING_PIPELINE.md`](../../docs/11_TRAINING_PIPELINE.md).

## What we deliberately ignore (and why)

- **The absolute scale.** The paper deals with models in the billions of parameters and datasets in the trillions of tokens. We are operating in the millions, but the underlying power-law relationship is assumed to roughly hold, or at least provide a principled baseline for our tiny experiments.

## Open questions this raised

- Do the Chinchilla scaling coefficients hold at the extreme micro-scale (1-4M parameters)? Can we reproduce the isoFLOP curves on a MacBook over a weekend? → candidate experiment.

## One-line takeaway

> For optimal performance, model size and training data size should be scaled equally; aim for about 20 training tokens per parameter.
