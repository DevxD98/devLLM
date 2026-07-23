# 03 — Learning Path

> **Prerequisites:** basic Python (variables, functions, loops, lists), comfort running a
> script in a terminal. *No* machine-learning background assumed. *No* advanced math
> assumed — we build the math you need as we go.
>
> **Next:** [`04_MATHEMATICS.md`](04_MATHEMATICS.md)

---

## Purpose

This document is the **map** for everything else in `docs/`. Its job is to take a reader
who knows a little Python and almost no machine learning, and route them — in a sane order,
without dead ends — all the way to understanding and building a small GPT.

Most "build a GPT" tutorials fail one of two ways. Either they assume you already know
what a gradient, an embedding, and a softmax are (so beginners drown), or they hand you
working code and narrate it line-by-line (so you can *run* a transformer but couldn't
*rebuild* one). This path is designed to avoid both: every concept is introduced only
after the concepts it depends on, and every concept is explained **why-first** — what
problem it solves — before **how** it works.

---

## Background

### What you are actually going to learn

A GPT is not one big idea. It is a **stack of small, individually understandable ideas**:

```
   turning text into numbers        (tokenizer)
   turning numbers into vectors     (embeddings)
   letting positions matter         (positional embeddings)
   letting words look at each other (attention)
   mixing information per-position  (feed-forward)
   stacking that many times         (transformer blocks)
   predicting the next token        (language-model head)
   nudging weights toward better    (training / backprop)
   generating text one token at a   (inference / sampling)
     time
```

Every line above is a document in this folder. The skill this path teaches is not any
single idea — it's seeing how they **compose**.

### Why "from scratch" is the right way to learn

You can call `transformers.GPT2Model()` in one line. You will learn almost nothing from
it. Understanding comes from friction: from getting a tensor shape wrong, from watching a
loss refuse to go down, from realizing *why* attention needs a causal mask. This project
deliberately reimplements the pieces so the friction happens — that friction is the
learning. See [`00_PROJECT_VISION.md`](00_PROJECT_VISION.md) for the full philosophy.

### The one mental model to keep

Hold this picture the entire way through. Everything you learn slots into one of these
boxes:

```
   text  ──►  [ represent ]  ──►  [ process ]  ──►  [ predict ]  ──►  text
              tokenizer          transformer         lm head +
              + embeddings        blocks              sampling
```

---

## Concepts

A few words you'll meet immediately, defined here in one sentence each so the path reads
smoothly. Full definitions live in [`19_GLOSSARY.md`](19_GLOSSARY.md).

- **Tensor** — an n-dimensional array of numbers (a batch of sequences of vectors is a
  3-D tensor). The only data structure a neural network really has.
- **Parameter / weight** — a number the model *learns*. DevLLM has 1–4 million of them.
- **Gradient** — for each weight, the direction that would reduce the error; training
  follows gradients downhill.
- **Loss** — a single number measuring how wrong the model currently is. Training =
  making it smaller.
- **Forward pass** — running data through the model to get a prediction.
- **Backward pass (backprop)** — computing gradients of the loss w.r.t. every weight.
- **Token** — the atomic unit of text the model sees (a character, or a sub-word).
- **Embedding** — the vector a token gets mapped to; its learned "meaning."

If those felt fast, good — that's the point of the path. Each gets a whole document.

---

## Detailed Explanation — the path itself

The path has **five stages**. Each stage lists what to read, what to *understand* before
moving on (the gate), and roughly how it maps onto the project [`01_ROADMAP.md`](01_ROADMAP.md).

### Stage 1 — The Why (orientation)

**Read:** `00_PROJECT_VISION` → `01_ROADMAP` → `02_ARCHITECTURE` → this file.

**Goal:** understand *what* you're building and *why*, and be able to name the pieces of a
GPT even if you can't yet explain them. Don't try to understand attention yet. Just get
the shape of the journey.

**Gate:** you can draw the `text → represent → process → predict → text` diagram from
memory and name which module does each part.

### Stage 2 — The Foundations (how learning works at all)

**Read:** [`04_MATHEMATICS.md`](04_MATHEMATICS.md) → [`05_NEURAL_NETWORKS.md`](05_NEURAL_NETWORKS.md).

This is the stage beginners are tempted to skip. **Don't.** Attention is just matrix
multiplies and a softmax; if those are shaky, attention will feel like magic instead of
mechanics.

- **04 Mathematics** teaches *only* what we need: vectors and dot products (how similarity
  is measured), matrices and matrix multiplication (how a whole layer computes at once),
  the derivative/gradient (the direction of steepest descent), and the softmax (turning
  scores into probabilities). Nothing more.
- **05 Neural Networks** builds a single neuron → a layer → a multi-layer perceptron (MLP),
  then explains **backpropagation** and the **training loop** in plain terms. It ends with
  the single most useful debugging technique in the whole project: *overfit one tiny
  batch* — if your model can't memorize 10 examples, it's broken, and you've found out in
  seconds instead of hours.

**Maps to:** roadmap Phase 0. Build the neuron, the MLP, XOR, and an MNIST classifier
here — small wins that prove the machinery before transformers enter the picture.

**Gate:** you can explain, without notes, what a gradient is and what one step of gradient
descent does to a weight. You can state why a nonlinearity (like ReLU/GELU) is necessary.

### Stage 3 — From Text to Model (the heart)

**Read, strictly in order:**
`06_TOKENIZER` → `07_EMBEDDINGS` → [`08_ATTENTION`](08_ATTENTION.md) →
[`09_TRANSFORMER`](09_TRANSFORMER.md) → `10_GPT_ARCHITECTURE`.

This is where a GPT actually takes shape. The order is not negotiable — each doc uses the
previous one's output as its input:

```
   06 text ──► token ids        (integers)
   07 token ids ──► vectors     (embeddings + positions)
   08 vectors ──► context-aware vectors   (attention)
   09 wrap 08 + an MLP + norms + residuals ──► a block
   10 stack N blocks + a head ──► a GPT that predicts next tokens
```

**08 Attention** is the conceptual summit of the project. Budget real time for it; it is
written to full depth for exactly this reason. When it clicks — when you see attention as
"every position asks a question (query), every position offers a key, and answers are
mixed by how well keys match queries" — the rest of the transformer is assembly.

**Maps to:** roadmap Phase 2 (TinyGPT). By the end you should have a working **forward
pass**: random-weight model in, sensible-shaped logits out.

**Gate:** you can trace a single token's vector from embedding, through one attention head,
through the feed-forward, and out — naming the tensor shape `(B, T, C)` at each step.

### Stage 4 — Making It Run (training, inference, Apple Silicon)

**Read:** `11_TRAINING_PIPELINE` → `12_INFERENCE` →
`13_OPTIMIZATION_FOR_APPLE_SILICON` → `14_BENCHMARKING`.

Now the model *learns* and *speaks*.

- **11 Training** turns the forward pass into a loop that improves: compute loss, backprop,
  step the optimizer, clip gradients, schedule the learning rate, checkpoint.
- **12 Inference** generates text one token at a time and introduces the sampling knobs —
  temperature, top-k, top-p — that turn probabilities into words.
- **13 Apple Silicon** is where this project is distinctive: how to fit and run all of the
  above inside **8 GB of unified memory** using the **MPS (Metal)** backend, why batch
  size and dtype are memory decisions, and how a fanless MacBook Air's **thermal** behavior
  affects real throughput.
- **14 Benchmarking** makes the optimization honest: measure before and after, or it
  didn't happen.

**Maps to:** roadmap Phases 3–4. First coherent text; then a faster, leaner model.

**Gate:** you can explain why generation is `O(T²)` without a KV cache, and what unified
memory changes about how you think about batch size.

### Stage 5 — How We Work (the lab's method)

**Read as needed:** `15_EXPERIMENT_GUIDE`, `16_MODEL_CONFIGURATION`, `17_DATASET_GUIDE`,
`18_RESEARCH_PAPERS`, plus the living docs `19_GLOSSARY`, `20_TODO`, `21_DEVLOG`,
`22_FUTURE_VERSIONS`, `23_RESOURCES`.

These are reference and practice docs, not a linear read. They teach the *discipline* that
makes the project a lab and not a pile of scripts: reproducible experiments, annotated
configs, honest data provenance, and reading primary sources. Dip in when you need them.

**Maps to:** roadmap Phase 5 and everything ongoing.

---

## Visual Diagrams

### The path as a dependency tree

```
   ┌────────────────────── STAGE 1: THE WHY ──────────────────────┐
   │  00 ──► 01 ──► 02 ──► 03(you are here)                        │
   └───────────────────────────┬──────────────────────────────────┘
                               ▼
   ┌────────────────── STAGE 2: FOUNDATIONS ──────────────────────┐
   │  04 MATH ──► 05 NEURAL NETWORKS                              │
   └───────────────────────────┬──────────────────────────────────┘
                               ▼
   ┌────────────── STAGE 3: FROM TEXT TO MODEL ───────────────────┐
   │  06 ──► 07 ──► 08 ATTENTION ──► 09 ──► 10 GPT                 │
   └───────────────────────────┬──────────────────────────────────┘
                               ▼
   ┌──────────────── STAGE 4: MAKING IT RUN ──────────────────────┐
   │  11 TRAIN ──► 12 INFER ──► 13 APPLE SILICON ──► 14 BENCH      │
   └───────────────────────────┬──────────────────────────────────┘
                               ▼
   ┌──────────────── STAGE 5: HOW WE WORK ────────────────────────┐
   │  15 16 17 18   +   living: 19 20 21 22 23                     │
   └──────────────────────────────────────────────────────────────┘
```

### A realistic time budget (3–4 months, part-time)

```
   weeks 1–3    ▓▓▓  Stages 1–2   foundations + Phase 0 builds
   weeks 4–5    ▓▓   Stage 3a     tokenizer, embeddings
   weeks 6–8    ▓▓▓  Stage 3b     attention → transformer → GPT
   weeks 9–10   ▓▓   Stage 4a     training, first coherent text
   weeks 11–14  ▓▓▓▓ Stage 4b     inference + Apple-Silicon optimization
   ongoing      ▓▓   Stage 5      papers, experiments, write-ups
```

This mirrors [`01_ROADMAP.md`](01_ROADMAP.md); the roadmap is the project view, this is the
learning view of the same timeline.

---

## Common Mistakes

- **Skipping Stage 2 to "get to the transformer."** The single most common way to end up
  with a GPT you can run but not understand. Attention *is* Stage-2 math wearing a costume.
- **Reading 08 Attention once and moving on.** Almost nobody gets attention on the first
  pass. Expect to read it two or three times, and to re-read it after you've seen the code.
- **Learning by only reading.** Each stage has *builds* (a neuron, XOR, MNIST, a forward
  pass). Do them. Reading about gradient descent is not the same as watching a loss curve
  bend on your own screen.
- **Optimizing before it works.** Stage 4 has an order for a reason: correctness (11–12)
  before speed (13–14). A fast wrong model is just wrong, faster.
- **Trying to memorize.** You don't need to memorize the attention formula. You need to be
  able to *rederive* it from "queries match keys, values get mixed." Understanding beats
  recall here every time.
- **Comparing to GPT-4.** This is a 1–4M-parameter model. It will produce charmingly
  wobbly text, not essays. Measuring it against frontier models misses the entire point
  (see [`00_PROJECT_VISION.md`](00_PROJECT_VISION.md)).

---

## Future Improvements

- Add a companion set of **runnable notebooks**, one per stage, so each gate can be checked
  interactively rather than only on paper.
- Add **self-check quizzes** at each gate (5 questions that, if you can answer them, mean
  you're ready to proceed).
- Record a **DEVLOG walkthrough** ([`21_DEVLOG.md`](21_DEVLOG.md)) that follows one reader
  through the whole path, so future readers can see where others got stuck.

---

## Glossary

Terms used above, defined once here; the canonical list is [`19_GLOSSARY.md`](19_GLOSSARY.md).

| Term | Meaning (short) |
|------|-----------------|
| Gate | a checkpoint: what you must be able to explain before moving to the next stage |
| Forward pass | running input through the model to produce an output |
| Overfit-a-batch | deliberately memorizing a tiny dataset to prove the model can learn at all |
| Logits | the raw, pre-softmax scores the model outputs for each possible next token |
| `(B, T, C)` | the standard tensor shape: Batch, Time (sequence length), Channels (embedding dim) |

---

## Learning Checklist

You are ready to leave this document when you can:

- [ ] Draw the `text → represent → process → predict → text` diagram from memory.
- [ ] Name every `docs/` file and place it in one of the five stages.
- [ ] Explain why Stage 2 (math + neural nets) must come before Stage 3 (attention).
- [ ] State the gate for each stage in your own words.
- [ ] Commit to *doing the builds*, not just reading.

---

## References

- Andrej Karpathy, *Neural Networks: Zero to Hero* — the closest thing to this path in
  video form; the spiritual model for our Stage 2–3 ordering.
- 3Blue1Brown, *Neural Networks* and *Essence of Linear Algebra* — the intuition for
  Stage 2, visually.
- [`00_PROJECT_VISION.md`](00_PROJECT_VISION.md), [`01_ROADMAP.md`](01_ROADMAP.md),
  [`02_ARCHITECTURE.md`](02_ARCHITECTURE.md) — the three anchor docs this path organizes.

## Further Reading

- [`18_RESEARCH_PAPERS.md`](18_RESEARCH_PAPERS.md) — when you're ready for primary sources.
- [`23_RESOURCES.md`](23_RESOURCES.md) — the full curated list of courses, videos, and repos.
- nanoGPT (Karpathy) — after Stage 3, reading its ~300 lines is deeply clarifying; you'll
  recognize every piece.

> **Next:** begin Stage 2 with [`04_MATHEMATICS.md`](04_MATHEMATICS.md). Don't skip it.
