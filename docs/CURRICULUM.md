# Curriculum — the DevLLM mini-university

> A structured, week-by-week course that turns [`03_LEARNING_PATH.md`](03_LEARNING_PATH.md)
> (the *map*) into a *syllabus* (the schedule). Where the learning path says "understand
> attention before transformers," this says "in week 6, read X, watch Y, build Z, then quiz
> yourself with these questions."
>
> **Audience:** basic Python, almost no ML. **Pace:** ~10–12 hrs/week, ~14 weeks, matching
> [`01_ROADMAP.md`](01_ROADMAP.md). Slower is fine; skipping the *Build* and *Quiz* steps is
> not — passive reading does not teach this material.

Each week follows the same loop:

```
   MATH ─► READ ─► WATCH ─► IMPLEMENT ─► QUIZ ─► (pass?) ─► next week
                                          ▲                    │
                                          └──── no ────────────┘
```

Resources named are real and freely available; see [`23_RESOURCES.md`](23_RESOURCES.md) for
the full annotated list and links.

---

## Module 0 — Foundations (Weeks 1–3) · roadmap Phase 0

### Week 1 — Linear algebra you actually need
- **Math:** vectors, dot products (similarity), matrices, matrix–vector and matrix–matrix
  multiplication, shapes. → [`04_MATHEMATICS.md`](04_MATHEMATICS.md) §1–3.
- **Read:** `04_MATHEMATICS.md` (vectors → matmul).
- **Watch:** 3Blue1Brown, *Essence of Linear Algebra* (ch. 1–4).
- **Implement:** a dot product, a matmul, and a softmax **by hand in NumPy** (no `@` for the
  first matmul — loops — then replace with `@` and confirm they match).
- **Quiz:** What does a dot product measure? What shape is `(3×4)·(4×2)`? Why must inner
  dims match? Rederive softmax and prove each output is in (0,1) and sums to 1.

### Week 2 — Calculus, gradients, gradient descent
- **Math:** derivative as slope, partial derivatives, the gradient, the chain rule.
  → `04_MATHEMATICS.md` §4.
- **Read:** `04_MATHEMATICS.md` (derivatives → chain rule).
- **Watch:** 3Blue1Brown, *Essence of Calculus* (ch. 1–4); Karpathy, *micrograd* video.
- **Implement:** minimize `f(x)=x²` with hand-written gradient descent; print `x`
  converging to 0. Then a 2-variable bowl.
- **Quiz:** What does one gradient-descent step do to a weight? What is the chain rule and
  why does backprop need it? What happens if the learning rate is too big? Too small?

### Week 3 — Neurons, MLPs, backprop, the training loop
- **Math:** a neuron as `activation(w·x + b)`; why a nonlinearity is required.
- **Read:** [`05_NEURAL_NETWORKS.md`](05_NEURAL_NETWORKS.md) (full).
- **Watch:** Karpathy, *Neural Networks: Zero to Hero* (micrograd → makemore intro).
- **Implement:** (1) build **micrograd**-style scalar autograd, ~150 lines; (2) train an MLP
  to solve **XOR**; (3) train a tiny **MNIST** classifier in PyTorch.
- **Quiz:** Why can't a network without nonlinearities solve XOR? Name the 5 steps of a
  training loop. What is the "overfit-a-batch" test and why is it the fastest debugger you
  have?

> **Module 0 gate:** you can explain gradient descent and backprop without notes, and you
> have *built* something that learns. Do not proceed until XOR trains.

---

## Module 1 — Representing language (Weeks 4–5) · roadmap Phase 1

### Week 4 — Tokenization
- **Read:** [`06_TOKENIZER.md`](06_TOKENIZER.md); ADR
  [`ADR-0001`](../research/design_decisions/ADR-0001-character-tokenizer-first.md).
- **Watch:** Karpathy, *Let's build the GPT Tokenizer* (for BPE context; we start simpler).
- **Implement:** a **character tokenizer** — build vocab from a corpus, `encode`/`decode`,
  and a **round-trip test** `decode(encode(x)) == x`. Then a dataset loader that yields
  random `(B, T)` windows.
- **Quiz:** Why character-level first? What's the trade-off vs BPE (sequence length, vocab
  size, memory)? Why must decode(encode(x)) be exact?

### Week 5 — Embeddings & positions
- **Math:** a lookup table as a matrix; why order needs to be injected.
- **Read:** [`07_EMBEDDINGS.md`](07_EMBEDDINGS.md).
- **Watch:** Karpathy, *makemore* (embedding sections).
- **Implement:** a token-embedding lookup `(B,T)→(B,T,C)` and a learned positional embedding;
  add them. Confirm shapes.
- **Quiz:** Why does attention need explicit positional information? What shape is an
  embedding table for vocab V, width C? What does "weight tying" share, and why is it free?

> **Module 1 gate:** raw text → token IDs → summed token+positional vectors, tested.

---

## Module 2 — The transformer (Weeks 6–8) · roadmap Phase 2

### Week 6 — Attention (budget extra time)
- **Math:** `softmax(QKᵀ/√d_k)·V`, rederived from the four steps.
- **Read:** [`08_ATTENTION.md`](08_ATTENTION.md) — **read it twice.**
- **Watch:** Karpathy, *Let's build GPT: from scratch*; Jay Alammar, *Illustrated
  Transformer*.
- **Implement:** single-head causal self-attention on paper's tiny example, then in code;
  **assert** the causal mask zeroes the upper triangle and softmax rows sum to 1.
- **Quiz:** the full [`08` checklist](08_ATTENTION.md#learning-checklist). If you can't
  rederive the formula and explain √d_k and the mask, repeat the week.

### Week 7 — Multi-head + the block
- **Read:** finish [`08`](08_ATTENTION.md) (multi-head) → [`09_TRANSFORMER.md`](09_TRANSFORMER.md).
- **Implement:** multi-head attention (split C across heads, concat, project); then a full
  **pre-norm block**: `x + MHSA(LN(x))`, `x + FFN(LN(x))`. Assert shape in == shape out.
- **Quiz:** the [`09` checklist](09_TRANSFORMER.md#learning-checklist). Why is multi-head not
  extra compute? Why does the residual help both forward and backward?

### Week 8 — Assemble the GPT (forward pass)
- **Read:** [`10_GPT_ARCHITECTURE.md`](10_GPT_ARCHITECTURE.md); [`SPEC.md`](../SPEC.md) §4–5.
- **Implement:** stack N blocks + final LN + tied LM head; a random-weight **forward pass**
  producing `(B,T,V)` logits; count params and match [`SPEC.md`](../SPEC.md).
- **Quiz:** Trace one token's vector through the whole model naming shapes. Why does a block
  preserve `(B,T,C)`? What's your model's param count and does it match the arithmetic?

> **Module 2 gate:** a working forward pass whose parameter count you can *derive*.

---

## Module 3 — Make it learn & speak (Weeks 9–10) · roadmap Phase 3

### Week 9 — Training
- **Read:** [`11_TRAINING_PIPELINE.md`](11_TRAINING_PIPELINE.md); [`SPEC.md`](../SPEC.md) §7.
- **Implement:** cross-entropy loss vs shifted targets, backward, AdamW, grad clip, LR
  warmup+cosine, checkpointing. **First: overfit one batch to ~0 loss.** Then train on
  Shakespeare.
- **Quiz:** Why overfit-a-batch before real training? What does grad clipping prevent? Why
  warmup then decay?

### Week 10 — Inference
- **Read:** [`12_INFERENCE.md`](12_INFERENCE.md); [`SPEC.md`](../SPEC.md) §8.
- **Implement:** autoregressive generation; temperature, top-k, top-p. Save a
  **before-training** sample and an **after-training** sample to [`outputs/`](../outputs/).
- **Quiz:** Why is naive generation O(T²)? What does temperature do to the distribution?
  top-k vs top-p?

> **Module 3 gate:** first coherent(ish) generated text, and you can explain every sampling
> knob.

---

## Module 4 — Apple Silicon & measurement (Weeks 11–14) · roadmap Phase 4

### Week 11 — The machine
- **Read:** [`13_OPTIMIZATION_FOR_APPLE_SILICON.md`](13_OPTIMIZATION_FOR_APPLE_SILICON.md);
  [`architecture/memory_layout.md`](../architecture/memory_layout.md).
- **Implement:** move training to `mps`; a seed-everything utility; confirm determinism.
- **Quiz:** Why watch the O(T²) matrix over param count? Name three calls that force an MPS
  sync. Why are cold benchmarks misleading on an Air?

### Week 12 — Benchmark honestly
- **Read:** [`14_BENCHMARKING.md`](14_BENCHMARKING.md).
- **Implement:** a benchmark harness (warmup discarded, warm/cold recorded, median of N)
  measuring tokens/sec, peak memory, checkpoint size. File your first
  [`benchmarks/`](../benchmarks/) record.
- **Quiz:** What must a benchmark record to be reproducible? Why median over mean here?

### Weeks 13–14 — Optimize, one change at a time
- **Read:** [`OPTIMIZATION_BACKLOG.md`](../research/OPTIMIZATION_BACKLOG.md).
- **Implement:** pick items top-down (weight tying already default; then KV cache, gradient
  accumulation, bf16 *if it benchmarks well*). **Each is one experiment** with before/after.
- **Quiz:** For each optimization: what did the benchmark say, and was the win real or noise?

> **Module 4 gate:** a faster/leaner model where every improvement has a before/after number.

---

## Ongoing — Research (Module 5) · roadmap Phase 5

Interleave from Week 6 onward: read one paper from
[`18_RESEARCH_PAPERS.md`](18_RESEARCH_PAPERS.md), write a
[paper note](../research/paper_notes/), and log any architecture experiment as an
[ADR](../research/design_decisions/). Start with *Attention Is All You Need* (worked note
already in the repo).

---

## How to know you're done

You've completed the course when you can, from a blank file, **rebuild DevLLM's forward
pass from memory, train it, generate from it, and explain every design choice** — and when
your [`21_DEVLOG.md`](21_DEVLOG.md) tells the honest story of where you got stuck and how you
got unstuck. That devlog is the real diploma.

## See also

- [`03_LEARNING_PATH.md`](03_LEARNING_PATH.md) (the map) · [`01_ROADMAP.md`](01_ROADMAP.md)
  (the project timeline) · [`23_RESOURCES.md`](23_RESOURCES.md) (links).
