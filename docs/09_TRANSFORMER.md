# 09 — The Transformer Block

> **Prerequisites:** [`08_ATTENTION.md`](08_ATTENTION.md) (multi-head self-attention) and
> [`05_NEURAL_NETWORKS.md`](05_NEURAL_NETWORKS.md) (MLPs, activations). This document
> assembles the block; it assumes you can already explain what attention *does*.
>
> **Next:** [`10_GPT_ARCHITECTURE.md`](10_GPT_ARCHITECTURE.md)

---

## Purpose

Attention alone is not a model. A **transformer block** is the reusable unit that wraps
attention together with a small neural network, two normalization layers, and two residual
connections — and it is designed so that you can **stack it N times** to make a deep model
that still trains stably. This document explains each part of the block, *why* it's there,
and how they fit, so that the leap from "one block" to "a GPT" in
[`10_GPT_ARCHITECTURE.md`](10_GPT_ARCHITECTURE.md) is trivial.

The key insight: attention **moves information between positions** (tokens talk to each
other); the feed-forward network **processes information within a position** (each token
thinks on its own). A block does both, and the wrapping (norm + residual) is what makes
stacking many blocks *possible*.

---

## Background

### Why attention needs a partner

Attention is a *mixing* operation: every output vector is a weighted average of value
vectors. Averages, even learned ones, are limited — stacking pure attention layers can't
easily represent rich per-token transformations, and repeated averaging tends to make
token representations collapse toward each other. We need a component that operates on each
position **independently** and can apply a nonlinear transformation. That component is a
small MLP — the **feed-forward network (FFN)**.

### Why we can't just stack layers naively

Deep networks are hard to train. Stack twenty raw layers and gradients either vanish or
explode on the way back, and the signal degrades on the way forward. Two techniques, both
built into every transformer block, make depth trainable:

- **Residual (skip) connections** — add the input back to the output, giving gradients a
  clean highway to flow through.
- **Layer normalization** — keep the scale of activations stable across positions and
  depth.

A transformer block is best understood as **"attention and an MLP, each made safe to stack
by a norm and a residual."**

---

## Concepts

Four ingredients, then the recipe.

- **Multi-head self-attention (MHSA)** — from [`08_ATTENTION.md`](08_ATTENTION.md). The
  "talk to other tokens" step.
- **Feed-forward network (FFN / MLP)** — two linear layers with a nonlinearity between
  them, applied identically to every position:
  `FFN(x) = W₂ · GELU(W₁ · x)`. It expands the dimension (typically ×4), applies a
  nonlinearity, then projects back. This is where most of a small model's *parameters*
  actually live.
- **Residual connection** — `output = x + Sublayer(x)`. The sublayer only has to learn a
  *correction* to its input, not reproduce it.
- **Layer normalization (LayerNorm)** — for each token vector independently, subtract its
  mean and divide by its standard deviation (then scale/shift by learned parameters). It
  stabilizes training by keeping activation scales consistent.

### Pre-norm vs post-norm — a real design decision

The original 2017 transformer applied LayerNorm *after* each sublayer ("post-norm").
Modern GPTs, including DevLLM, apply it *before* each sublayer ("**pre-norm**"):

```
   POST-NORM (original)          PRE-NORM (DevLLM / GPT-2+)
   x → sublayer → +x → norm      x → norm → sublayer → +x
```

Pre-norm keeps the residual highway completely clean (nothing is normalized *on* the
skip path), which makes deep stacks dramatically more stable to train and far less
sensitive to learning-rate warmup. This choice is recorded as an ADR in
[`research/design_decisions/`](../research/design_decisions/) — it's exactly the kind of
"small decision, big consequence" the lab notebook exists to capture.

---

## Detailed Explanation

### The block, sublayer by sublayer

A DevLLM (pre-norm) transformer block does, in order:

```
   1.  a = x + MHSA( LayerNorm(x) )        # tokens exchange information
   2.  y = a + FFN(  LayerNorm(a) )        # each token is transformed
```

That's the whole block. Two sublayers, each of the form `input + Sublayer(LayerNorm(input))`.
Read it as two sentences:

1. **Attention sublayer** — normalize, let every token gather context from the others, and
   *add that back* to the original. After this line, each token's vector is enriched with
   information from the tokens it attended to.
2. **Feed-forward sublayer** — normalize, let each token independently transform itself
   through a small nonlinear network, and *add that back*. After this line, each token has
   "thought about" the context it just gathered.

### Why the residual `x +` matters so much

Consider the attention sublayer `a = x + MHSA(LayerNorm(x))`. Because of the `x +`, if
attention learned to output all zeros, the block would just pass `x` through unchanged — a
perfectly good default. The sublayer therefore only needs to learn a **useful adjustment**
to the representation, not rebuild it from scratch. During backprop, the `+` also sends the
gradient straight through to `x` untouched (the derivative of `x + f(x)` w.r.t. `x`
includes a clean `1`), which is what prevents vanishing gradients in deep stacks. Residuals
are the reason a 12-, 48-, or 96-layer transformer trains at all.

### The feed-forward network, concretely

```
   x  (…, C)
     │
   × W₁   ─►  (…, 4C)      # expand: give the token room to compute
     │
   GELU    ─►  (…, 4C)      # nonlinearity: without it, two linears collapse into one
     │
   × W₂   ─►  (…, C)       # project back to model width
```

The ×4 expansion is a convention (from the original paper) that balances capacity against
cost. **GELU** (Gaussian Error Linear Unit) is the activation modern GPTs use; it's a
smooth version of ReLU. The nonlinearity is essential: two matrix multiplies with nothing
between them are mathematically just one matrix multiply, so without GELU the FFN could
represent only linear functions. In a 1–4M-parameter DevLLM, these two matrices are where
**most of the parameters sit** — see [`16_MODEL_CONFIGURATION.md`](16_MODEL_CONFIGURATION.md)
for the parameter-count arithmetic.

### Shapes are preserved — that's the point

A transformer block takes `(B, T, C)` in and returns `(B, T, C)` out. **The shape is
invariant.** This is deliberate and powerful: because output shape equals input shape, you
can feed one block's output straight into the next, and stack as many as memory allows
without any shape bookkeeping. The stack is just:

```
   x → block₁ → block₂ → … → block_N → x'      every arrow is (B, T, C)
```

That shape-invariance is the entire reason "stack N blocks" is a one-line idea in
[`10_GPT_ARCHITECTURE.md`](10_GPT_ARCHITECTURE.md).

### Parameter count of one block (rough)

For model width `C`, one block holds approximately:

```
   attention:  4 · C²        (Wq, Wk, Wv, Wo, each ≈ C×C)
   FFN:        8 · C²        (W₁ ≈ C×4C, W₂ ≈ 4C×C)
   ───────────────────────
   total ≈    12 · C²  per block   (+ small LayerNorm & bias terms)
```

So for `C = 128`, one block ≈ `12 × 128² ≈ 197K` params; a handful of blocks lands DevLLM
in its 1–4M target. Notice the FFN carries **twice** the attention's parameters — a useful
thing to remember when budgeting 8 GB.

---

## Visual Diagrams

### One pre-norm transformer block

```
        x  (B, T, C)
        │
        ├───────────────────────────────┐  (residual highway — untouched)
        ▼                               │
   ┌─────────┐                          │
   │LayerNorm│                          │
   └────┬────┘                          │
        ▼                               │
   ┌──────────────────┐                 │
   │ multi-head        │                 │
   │ self-attention    │  ← doc 08       │
   └────┬─────────────┘                 │
        ▼                               │
       (+)◄─────────────────────────────┘
        │  a = x + MHSA(LN(x))
        ├───────────────────────────────┐  (residual highway — untouched)
        ▼                               │
   ┌─────────┐                          │
   │LayerNorm│                          │
   └────┬────┘                          │
        ▼                               │
   ┌──────────────────┐                 │
   │ feed-forward     │                 │
   │ (×4 GELU proj)   │                 │
   └────┬─────────────┘                 │
        ▼                               │
       (+)◄─────────────────────────────┘
        │  y = a + FFN(LN(a))
        ▼
        y  (B, T, C)   ── same shape in, same shape out
```

### Attention vs feed-forward: two axes of mixing

```
   ATTENTION mixes ACROSS positions        FEED-FORWARD mixes ACROSS features
   (tokens ← other tokens)                 (within one token, no cross-talk)

     t0 t1 t2 t3                             t0   t1   t2   t3
      ╲  │  │ ╱                              │    │    │    │
       ╲ │  │╱     each token pulls          ▼    ▼    ▼    ▼    each token
        ╳╳╳╳       from the others         [MLP][MLP][MLP][MLP]  transformed
       ╱ │  │╲                               │    │    │    │    on its own
      t0 t1 t2 t3                            t0'  t1'  t2'  t3'
```

A block alternates the two: gather context, then think. Stacking blocks repeats this —
gather, think, gather, think — building up increasingly abstract representations.

---

## Common Mistakes

- **Putting LayerNorm on the residual path.** The skip connection must stay clean:
  `x + Sublayer(LN(x))`, **not** `LN(x + Sublayer(x))` if you intend pre-norm. Normalizing
  the highway defeats the purpose and hurts deep-stack stability.
- **Forgetting the nonlinearity in the FFN.** `W₂ · (W₁ · x)` with no GELU is just one
  linear layer wearing two matrices — the model silently loses most of its capacity with
  no error.
- **Only one residual.** Both sublayers need their own residual and their own LayerNorm.
  Dropping one usually still runs, and trains worse, which makes it a nasty silent bug.
- **Wrong FFN expansion, then surprise at the parameter count.** The ×4 is where the
  parameters go; get it wrong and your 1M-param plan becomes a 4M-param model that won't
  fit your training batch on 8 GB.
- **Assuming more blocks is always better.** On 8 GB, depth costs memory (activations for
  every layer must be kept for backprop). Past a point, a wider/shallower model trains
  better within the budget — measure it (see [`14_BENCHMARKING.md`](14_BENCHMARKING.md)).
- **Mixing up post-norm and pre-norm** when reading reference code. The 2017 paper is
  post-norm; nanoGPT and GPT-2 are pre-norm like DevLLM. The diagrams look almost identical;
  the training stability is not.

---

## Future Improvements

- **RMSNorm** instead of LayerNorm — drops the mean-centering step; cheaper and works as
  well or better in practice (used by Llama). A clean experiment for this project.
- **SwiGLU / gated FFN** — a more expressive feed-forward variant that often improves
  quality per parameter; worth benchmarking against the plain GELU MLP.
- **Parallel attention + FFN** (as in some newer models) — run both sublayers off the same
  normalized input to save a little latency. A nice ablation for
  [`research/design_decisions/`](../research/design_decisions/).
- **Weight tying and sharing** across blocks — explored more in
  [`10_GPT_ARCHITECTURE.md`](10_GPT_ARCHITECTURE.md) and
  [`13_OPTIMIZATION_FOR_APPLE_SILICON.md`](13_OPTIMIZATION_FOR_APPLE_SILICON.md).

---

## Glossary

Canonical definitions in [`19_GLOSSARY.md`](19_GLOSSARY.md); the block-specific terms:

| Term | Meaning |
|------|---------|
| Transformer block | the stackable unit: attention + FFN, each with LayerNorm + residual |
| Feed-forward network (FFN) | per-position MLP: `W₂·GELU(W₁·x)`, usually ×4 expansion |
| Residual connection | `x + Sublayer(x)`; the skip path that lets gradients flow |
| LayerNorm | per-token normalization to zero mean, unit variance (then learned scale/shift) |
| Pre-norm | normalize *before* each sublayer (DevLLM's choice) vs post-norm (original) |
| GELU | smooth nonlinearity used in the FFN |
| Sublayer | one of the two components (attention, or FFN) inside a block |

---

## Learning Checklist

You understand the transformer block when you can:

- [ ] State the two things a block does (mix across positions; transform within a position)
      and which component does which.
- [ ] Write the two sublayer equations `a = x + MHSA(LN(x))`, `y = a + FFN(LN(a))` from
      memory.
- [ ] Explain why the residual connection helps both the forward signal and the backward
      gradient.
- [ ] Explain why the FFN needs a nonlinearity, and why it expands ×4.
- [ ] Explain why a block preserves the `(B, T, C)` shape, and why that enables stacking.
- [ ] Explain pre-norm vs post-norm and why DevLLM chose pre-norm.
- [ ] Estimate a block's parameter count as ≈ 12·C².

---

## References

- Vaswani et al., *Attention Is All You Need* (2017) — introduces the block (post-norm);
  see [`research/paper_notes/attention_is_all_you_need.md`](../research/paper_notes/attention_is_all_you_need.md).
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2, 2019) — the
  pre-norm decoder-only design DevLLM follows.
- Ba, Kiros, Hinton, *Layer Normalization* (2016) — the normalization used here.
- Hendrycks & Gimpel, *Gaussian Error Linear Units (GELUs)* (2016).

## Further Reading

- Andrej Karpathy, nanoGPT — `Block`, `CausalSelfAttention`, and `MLP` in ~30 lines each;
  read them right after this doc and every line will be familiar.
- [`architecture/model_architecture.md`](../architecture/model_architecture.md) — the block
  in DevLLM's concrete wiring, with shapes.
- [`10_GPT_ARCHITECTURE.md`](10_GPT_ARCHITECTURE.md) — **next:** stack blocks into a GPT and
  add the language-model head.

> **Next:** [`10_GPT_ARCHITECTURE.md`](10_GPT_ARCHITECTURE.md) — you have the brick; now
> build the wall.
