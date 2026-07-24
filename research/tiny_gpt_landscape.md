# The Tiny-GPT Landscape — what to borrow, what to avoid

> **Purpose:** understand the existing small-GPT projects so JimmyLabs can *learn from* them
> without *copying* them (principle 9, and the whole point of "from scratch"). For each: its
> one-line identity, strengths, weaknesses **for our goal** (educational, from-scratch, tiny,
> on an 8 GB M1), specific ideas worth borrowing, and specific things to avoid.
>
> "For our goal" is doing a lot of work in this document. Many of these are *excellent*
> projects that are simply optimized for a different goal than ours (production inference,
> framework generality, maximum speed). A weakness here is rarely a criticism of the project.

---

## At a glance

```
   project        language   trains?  infers?  Apple-Si?   best lesson for us
   ─────────────  ─────────  ───────  ───────  ──────────  ───────────────────────
   nanoGPT        PyTorch      ✓        ✓       via MPS*    clean train loop + config
   minGPT         PyTorch      ✓        ✓       via MPS*    readable model decomposition
   picoGPT        NumPy        ✗        ✓        n/a        the forward pass, laid bare
   litGPT         PyTorch      ✓        ✓       via MPS*    recipe/config discipline
   tinygrad       own fw       ✓        ✓        Metal      how a DL framework works
   MLX examples   MLX          ✓        ✓        native     unified-memory-native patterns
   llama.cpp      C/C++        ✗(infer) ✓        Metal      quantization + mmap + Metal infer
   TinyStories    (dataset)    —        —         —         tiny models CAN be coherent
   Zero to Hero   (videos)     —        —         —         the teaching order itself
```
`*` PyTorch runs on `mps`, but these repos are written/tuned with CUDA assumptions; expect
op-fallbacks and CUDA-only paths (see [`docs/13`](../docs/13_OPTIMIZATION_FOR_APPLE_SILICON.md)).

---

## The implementations

### nanoGPT (Karpathy)
**Identity:** ~300 lines of clean PyTorch that actually train and reproduce GPT-2-scale
models; the de-facto reference for "minimal but real."
- **Strengths:** genuinely readable; clean separation of `model.py` / `train.py` / config;
  correct modern choices (pre-norm, GELU, weight tying); a real training loop with
  checkpointing and LR schedule.
- **Weaknesses (for us):** written CUDA-first — uses `torch.compile`, flash attention, AMP,
  and DDP paths that don't map cleanly to MPS; dense in spots (packs a lot per line);
  assumes larger scale and multi-GPU as a possibility.
- **Borrow:** the overall skeleton (model/train/config split), the LR warmup+cosine
  schedule, weight tying as default, the config-as-source-of-truth habit.
- **Avoid:** copying its CUDA/DDP/compile machinery; matching its terseness at the cost of
  the teaching goal. Read it *after* [`docs/09`](../docs/09_TRANSFORMER.md) — you'll
  recognize every piece, which is the point.

### minGPT (Karpathy)
**Identity:** nanoGPT's more pedagogical ancestor — class-based, verbose, optimized for
*reading*.
- **Strengths:** the clearest object decomposition of a GPT anywhere (`CausalSelfAttention`,
  `Block`, `GPT`); excellent for a first structural read.
- **Weaknesses (for us):** older, less performant, not tuned for anything; superseded by
  nanoGPT for actual training.
- **Borrow:** the module boundaries and naming; the "explain by structure" style — it
  mirrors JimmyLabs's `src/` module map.
- **Avoid:** using it as a performance or training reference (that's nanoGPT's job).

### picoGPT (jaymody)
**Identity:** GPT-2 *forward pass* in ~40 lines of NumPy, no framework.
- **Strengths:** strips inference to pure arithmetic — no autograd, no framework magic;
  wonderful for seeing that "a transformer is just matmuls and a softmax."
- **Weaknesses (for us):** inference-only (loads pretrained weights), no training, NumPy is
  slow, no Apple-Silicon angle.
- **Borrow:** the *minimalism as pedagogy* idea — JimmyLabs's forward pass should be readable
  in one sitting like this. A great Week-8 companion ([`docs/CURRICULUM.md`](../docs/CURRICULUM.md)).
- **Avoid:** its "load someone else's weights" approach — we train our own.

### litGPT (Lightning AI)
**Identity:** clean, production-leaning implementations of many open LLMs, with
finetuning/pretraining recipes; nanoGPT lineage grown up.
- **Strengths:** disciplined config/recipe structure; broad model coverage; good engineering
  hygiene.
- **Weaknesses (for us):** more abstraction than a 1–4M educational model needs; aimed at
  real/large training and finetuning; the abstraction hides the very internals we want to
  build.
- **Borrow:** the *recipe* discipline (a run = a versioned config), which matches our
  [`configs/`](../configs/) + [experiment](../research/experiment_templates/) approach.
- **Avoid:** adopting its abstraction layers — they'd bury the learning.

### tinygrad (Tiny Corp / geohot)
**Identity:** a tiny deep-learning *framework* with its own autograd, lazy evaluation, and
multiple backends including Metal.
- **Strengths:** shows how a DL framework itself works (autograd, kernels, lazy graphs);
  real Metal support on Apple Silicon.
- **Weaknesses (for us):** deliberately terse to the point of unreadability in places; a
  fast-moving target; building *on* it conflicts with "from scratch in PyTorch."
- **Borrow:** conceptual inspiration for the **micrograd-style autograd toy** we build once
  in Phase 0 ([`ARCHITECTURE_REVIEW.md`](ARCHITECTURE_REVIEW.md) gap list); the idea of a
  backend abstraction; lazy-eval intuition (useful for understanding MLX too).
- **Avoid:** its terseness (directly violates principle 9); reimplementing a production
  framework — that's the overengineering the review warns against.

### MLX examples (Apple)
**Identity:** Apple's array framework with official transformer/LLM examples, designed
*around* unified memory with lazy evaluation.
- **Strengths:** native Apple-Silicon performance; clean NumPy-like API; unified-memory-first
  design; no CUDA baggage.
- **Weaknesses (for us):** Apple-only; newer/smaller ecosystem and docs; not what our
  from-scratch PyTorch pieces are built on.
- **Borrow:** unified-memory-native patterns; **use as a benchmarked comparison target** —
  "does the same tiny GPT train faster in MLX on this exact M1?" is a great experiment
  ([`docs/13`](../docs/13_OPTIMIZATION_FOR_APPLE_SILICON.md)).
- **Avoid:** making it a hard dependency; a port is an ADR-worthy decision, not a default.

### llama.cpp (ggerganov)
**Identity:** high-performance C/C++ **inference** of LLaMA-family models, with aggressive
quantization (GGUF) and excellent Metal support.
- **Strengths:** world-class optimization; int4/int8 quantization that runs big models on
  Macs; memory-mapped weights; first-class Metal backend.
- **Weaknesses (for us):** C/C++ and inference-only — not a learning resource for *training*;
  enormous and highly optimized (opaque to a beginner); a different universe of scale.
- **Borrow:** *ideas* — quantized inference, memory-mapping weights, what a serious Metal
  inference path looks like — as inspiration for our v1.0+ deployment experiments.
- **Avoid:** treating it as an architecture template; its complexity is the opposite of our
  goal.

---

## The non-implementations (equally important)

### TinyStories (Eldan & Li, dataset + paper)
**Why it matters most:** it demonstrates empirically that **models with only millions of
parameters can generate fluent, coherent English** when trained on a simple enough corpus.
This is the single strongest external validation of JimmyLabs's premise — that a 1–4M-param
model is worth building and can produce real (if simple) language.
- **Borrow:** the dataset itself (a primary training corpus, see
  [`docs/17_DATASET_GUIDE.md`](../docs/17_DATASET_GUIDE.md)) **and** the thesis: *match data
  complexity to model capacity.* Don't feed a 1M-param model Wikipedia and expect coherence.
- **Avoid:** over-reading it as an architecture recipe — it's about *data*, not a specific
  model design.

### Karpathy — *Neural Networks: Zero to Hero* (video course)
**Why it matters:** the pedagogical gold standard, and the spiritual model for JimmyLabs's
learning order (micrograd → makemore → build GPT).
- **Borrow:** the *teaching sequence* — build backprop before trusting it, build the tokenizer
  before the model, build the forward pass before training. [`docs/CURRICULUM.md`](../docs/CURRICULUM.md)
  is aligned to it.
- **Avoid:** passive watching. The videos work because you *type along*; our curriculum
  enforces the same.

---

## Synthesis — what JimmyLabs's identity is, by contrast

```
   picoGPT    minimal, but inference-only & no framework
   minGPT     readable, but not tuned or Apple-aware
   nanoGPT    real & clean, but CUDA-first & terse
   litGPT     disciplined, but abstracted beyond our scale
   tinygrad   framework-deep, but unreadably terse
   MLX        Apple-native, but a different stack
   llama.cpp  Apple-fast, but C++ inference at another scale
   ───────────────────────────────────────────────────────────
   JimmyLabs  =  readable like minGPT
            + real training loop like nanoGPT
            + Apple-Silicon-first like MLX/llama.cpp
            + documented & reproducible beyond all of them
            + explicitly educational: WHY before HOW, every step
```

No existing project sits at that intersection. That gap — *readable, trainable,
Apple-Silicon-first, and exhaustively documented at tiny scale* — is JimmyLabs's reason to
exist.

## See also

- [`ARCHITECTURE_REVIEW.md`](ARCHITECTURE_REVIEW.md) · [`../SPEC.md`](../SPEC.md) ·
  [`../docs/18_RESEARCH_PAPERS.md`](../docs/18_RESEARCH_PAPERS.md) ·
  [`../docs/23_RESOURCES.md`](../docs/23_RESOURCES.md)
