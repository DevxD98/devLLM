# 13 — Optimization for Apple Silicon

> **Prerequisites:** [`11_TRAINING_PIPELINE.md`](11_TRAINING_PIPELINE.md) and
> [`12_INFERENCE.md`](12_INFERENCE.md) (you should have something that *runs* before you
> optimize it — principle 1), and [`SPEC.md`](../SPEC.md) §6 (the memory arithmetic).
>
> **Next:** [`14_BENCHMARKING.md`](14_BENCHMARKING.md)

---

## Purpose

Most "train a GPT" material assumes an NVIDIA GPU with CUDA, dedicated VRAM, and a fan.
JimmyLabs assumes none of those. This document explains how Apple Silicon — the **M1's unified
memory**, the **MPS (Metal) backend** in PyTorch, and a **fanless thermal envelope** —
actually behaves, and how those facts should shape the model and the training loop. The
goal is not a bag of tricks; it's a *mental model* of the machine, from which the right
optimizations follow.

Golden rule, restated from [`ENGINEERING_PRINCIPLES.md`](../ENGINEERING_PRINCIPLES.md):
**profile first, optimize second, measure the difference always.** Everything here is a
hypothesis to benchmark ([`14`](14_BENCHMARKING.md)), not a guarantee — Apple's stack moves
fast and behavior varies by macOS and PyTorch version.

---

## Background

### What "Apple Silicon" means for us

The M1 is a **System-on-Chip (SoC)**: CPU cores, GPU cores, a Neural Engine, and the memory
controller all sit on one die, sharing one pool of RAM. The three facts that matter most
for training a transformer:

1. **Unified memory** — CPU and GPU address the *same* physical RAM. There is no separate
   "VRAM," and no `cudaMemcpy` across a PCIe bus.
2. **MPS backend** — PyTorch talks to the GPU through Apple's **Metal Performance Shaders**,
   via `torch.device("mps")`. It is younger and less complete than CUDA.
3. **Fanless (Air)** — the MacBook Air M1 has no fan. Sustained compute heats the chip,
   and the chip protects itself by **throttling** clock speed after minutes of load.

Each is a double-edged sword. The rest of this doc is about which edge you get.

---

## Concepts

- **Unified Memory Architecture (UMA):** one RAM pool for everything (OS, your Python
  process, CPU tensors, GPU tensors). No host↔device copies — but also no escape from the
  fact that your model, your data, *and macOS itself* all draw from the same 8 GB.
- **MPS (Metal Performance Shaders):** Apple's GPU compute layer; PyTorch's `mps` device
  backend is built on it.
- **Operator fallback:** if PyTorch hasn't implemented an op for MPS, it errors — or, with
  `PYTORCH_ENABLE_MPS_FALLBACK=1`, silently runs that op on the **CPU**, which means a
  hidden GPU→CPU→GPU round-trip.
- **Memory pressure / swap:** when RAM fills, macOS compresses memory and then swaps to
  SSD. Training doesn't crash — it silently crawls.
- **Thermal throttling:** the SoC reduces frequency under sustained heat; a fanless Air
  hits this sooner than a MacBook Pro.
- **MLX:** Apple's own array/ML framework, designed *around* unified memory with lazy
  evaluation. A comparison target, not a JimmyLabs dependency
  ([`research/tiny_gpt_landscape.md`](../research/tiny_gpt_landscape.md)).

---

## Detailed Explanation

### Unified memory: the gift and the tax

**The gift.** On a discrete GPU you constantly shuttle data host↔device, and that copy is
often the bottleneck. On the M1 there is *no copy*: a tensor "moved" to `mps` doesn't cross
a bus. Data loading can be simpler and lower-latency, and you never pay the classic
PCIe-transfer tax.

**The tax.** There is no dedicated VRAM to hide in. Your 8 GB is shared with macOS, the
window server, your browser, and VS Code. Realistically **~5–6 GB** is available to your
process before memory pressure begins. So the effective budget for *(weights + gradients +
optimizer state + activations + the dataset batch + framework overhead)* is smaller than
the sticker says.

Implication for design: at JimmyLabs's scale, weights are trivial (see [`SPEC.md`](../SPEC.md)
§6 — v0.1's weights + optimizer ≈ 13 MB). The thing that fills memory is **activations**,
and the biggest single activation is the **O(T²) attention matrix**. So on Apple Silicon,
the highest-leverage "memory optimizations" are the ones that shrink `B·T²`, not the ones
that shrink parameter count.

### The MPS backend: fast, young, occasionally surprising

Using it is simple:

```
   device = "mps" if torch.backends.mps.is_available() else "cpu"
   model.to(device);  x = x.to(device)
```

Three realities to internalize:

1. **Coverage is good but not total.** Most ops a GPT needs are supported and fast, but
   some (or some dtype/shape combinations) may be missing on a given PyTorch version.
   Setting `PYTORCH_ENABLE_MPS_FALLBACK=1` keeps you running by pushing unsupported ops to
   CPU — but a silent CPU fallback in your hot loop can quietly dominate runtime. **Profile
   for it** ([`14`](14_BENCHMARKING.md)); an unexplained slowdown is often a fallback.
2. **Kernels compile on first use.** The first few iterations are slower while Metal
   compiles shaders. Always **discard warmup iterations** before measuring — a benchmark
   that includes warmup is measuring the compiler, not the model.
3. **Async execution hides costs and traps.** MPS work is queued asynchronously. Any call
   that reads a value back to Python — `loss.item()`, `.cpu()`, `print(tensor)`,
   `tensor.tolist()` — forces a **synchronization** that stalls the GPU until the queue
   drains. Doing that every step (e.g. printing loss) can serialize training. Log
   sparingly, and accumulate metrics on-device.

### Precision: fp32, fp16, bf16 on MPS

- **fp32** is the correct, boring baseline: numerically safe, universally supported. Start
  here.
- **fp16 / bf16** roughly halve activation memory and can speed up matmuls — but MPS support
  and stability for lower precision have **varied significantly across PyTorch versions**,
  and `autocast` on `mps` has historically been less mature than on CUDA. bf16's wider
  exponent makes it the safer low-precision choice *when supported*.
- **The rule:** treat mixed precision as a **benchmarked experiment**, not a default. Run
  the [experiment template](../research/experiment_templates/), confirm the loss curve is
  unharmed *and* memory/throughput actually improved on *your* macOS+torch version, then
  adopt it. Do not cargo-cult `autocast` from a CUDA tutorial.

### Thermals: the fanless reality

On a MacBook Air M1, a short benchmark and a two-hour training run are different physical
regimes. The chip runs fast cold, then throttles as it heats — so **tokens/sec measured in
the first 30 seconds overstates sustained training throughput.** This has concrete
consequences:

- **Benchmark warm** for sustained-workload numbers, and **record thermal state** (the
  benchmark template has a field). Report medians, not bests.
- **Elevate the laptop** for airflow; heat soak into the chassis is real.
- Expect training to settle into a lower, steadier throughput after a few minutes. That
  steady number — not the cold burst — is your planning figure.
- This is a genuine argument for **smaller models and shorter runs**: they finish inside the
  thermal budget and iterate faster, which serves the educational goal better than a giant
  run that throttles.

### Gradient accumulation: bigger batches without bigger memory

If a batch size you want won't fit (activations too large), don't give up the effective
batch — **accumulate gradients** over several small micro-batches before stepping the
optimizer. This trades time for memory: same effective batch, lower peak activation memory.
On an 8 GB machine this is often the single most useful training knob, and it's a clean
config field, not a code hack (principle 8).

### Keep the GPU fed and un-stalled

Practical loop hygiene that matters more on MPS than people expect:

- **Move data to the device once**, not per-op. Avoid ping-ponging tensors CPU↔MPS.
- **Avoid `.item()` in the hot path.** Accumulate loss on-device; sync every `eval_interval`,
  not every step (see async trap above).
- **`torch.mps.empty_cache()`** can relieve pressure between phases, but it's not free and
  not a substitute for a model that fits — use it deliberately, and measure.
- **Memory-map the dataset** ([`17_DATASET_GUIDE.md`](17_DATASET_GUIDE.md)) so the corpus
  isn't sitting in your 8 GB alongside the model.

### When to reach for MLX

PyTorch/MPS is the right default for JimmyLabs (it's what the from-scratch pieces are built on,
and it's widely documented). **MLX** — Apple's framework — is built natively around unified
memory and lazy evaluation and can be faster or lighter for some transformer workloads. It's
worth a **benchmarked comparison experiment** (does the same tiny GPT train faster in MLX on
this exact machine?), documented in
[`research/tiny_gpt_landscape.md`](../research/tiny_gpt_landscape.md) — but adopting it is a
decision for an ADR, not a default.

---

## Visual Diagrams

### Unified memory vs discrete GPU

```
   DISCRETE GPU (CUDA)                 APPLE SILICON (M1, UMA)
   ┌────────┐   PCIe    ┌────────┐     ┌──────────────────────────┐
   │  CPU   │ ◀══════▶ │  GPU   │     │  CPU ┐              ┌ GPU │
   │  RAM   │  copy!    │  VRAM  │     │      ├─ ONE 8GB ───┤     │
   └────────┘          └────────┘     │      ┘   POOL      └     │
   two pools, explicit copies         │  no copy — same addresses │
   copy is often the bottleneck       └──────────────────────────┘
                                      no copy tax · but shared with macOS
```

### Where the 8 GB actually goes (v0.1, illustrative)

```
   ┌───────────────────────────── 8 GB total ─────────────────────────────┐
   │ macOS + window server + apps            ~2–3 GB   (not yours)         │
   │ ─────────────────────────────────────────────────────────────────────│
   │ your process budget                     ~5–6 GB   before pressure     │
   │   weights+grads+optimizer  ~13 MB   ▏ (negligible at this scale)      │
   │   dataset batch / mmap      small   ▏                                  │
   │   ACTIVATIONS (incl. O(T²) attn)  ← this is what you actually manage   │
   └───────────────────────────────────────────────────────────────────────┘
```

### The optimization decision tree

```
   is it too slow / OOM?
        │
        ├─ profiled first? ── no ──► go profile (docs/14). do not guess.
        │                    │
        yes                  └─ MPS→CPU fallback in hot loop? ─► fix the op / dtype
        │
        ├─ OOM?  ──► shrink B, then T (B·T² term) ; grad-accumulate ; try bf16 (benchmark)
        │
        └─ slow but fits? ──► kill per-step .item()/print syncs ; keep data on device ;
                              check thermals (warm vs cold) ; consider MLX comparison
```

---

## Common Mistakes

- **Optimizing before profiling.** The cardinal sin (principle 1). On MPS especially,
  intuition is unreliable — the culprit is often a silent CPU fallback, not the thing you'd
  guess.
- **Benchmarking cold and planning as if it's sustained.** Fanless throttling makes the
  first-30-seconds number a fantasy for a long run. Measure warm.
- **Calling `.item()` / printing loss every step.** Forces a GPU sync every iteration and
  can serialize training. Batch your logging.
- **Copy-pasting CUDA `autocast`/AMP recipes.** MPS mixed precision has different maturity
  and pitfalls; verify on *your* version before trusting it.
- **Chasing parameter-count reductions to save memory.** At JimmyLabs's scale, weights are
  ~megabytes; activations and the O(T²) attention matrix are the real consumers. Optimize
  the right thing.
- **Filling all 8 GB and blaming the model when it crawls.** You crossed into swap/pressure.
  Leave headroom for macOS.
- **Ignoring warmup iterations in benchmarks.** You'll "measure" the Metal shader compiler.

---

## Future Improvements

- **KV cache** for inference (O(T²)→O(T) per step) — the biggest single inference win; see
  [`12_INFERENCE.md`](12_INFERENCE.md) and [`OPTIMIZATION_BACKLOG.md`](../research/OPTIMIZATION_BACKLOG.md).
- **bf16 training** once validated stable on the target macOS/torch — roughly halves
  activation memory.
- **Tiled / FlashAttention-style attention** to avoid materializing the full `T×T` matrix,
  attacking the exact term that caps context length on 8 GB.
- **Grouped-Query Attention** to shrink the KV cache footprint.
- **Quantized inference** (int8/int4) for deployment — a v1.0+ experiment.
- **MLX port** as a benchmarked alternative runtime.

Every item above is a *hypothesis* until [`benchmarks/`](../benchmarks/) says otherwise.

---

## Glossary

| Term | Meaning |
|------|---------|
| Unified memory (UMA) | single RAM pool shared by CPU and GPU; no host↔device copy |
| MPS | Metal Performance Shaders; PyTorch's `mps` GPU backend on Apple Silicon |
| Operator fallback | running an MPS-unsupported op on CPU (hidden round-trip) |
| Memory pressure | macOS compressing/swapping RAM when it fills → silent slowdown |
| Thermal throttling | SoC lowering clocks under sustained heat (pronounced on fanless Air) |
| Gradient accumulation | summing grads over micro-batches to emulate a larger batch cheaply |
| Warmup iterations | early, slower iters (kernel compilation) discarded before measuring |
| MLX | Apple's unified-memory-native ML framework; a comparison target |

---

## Learning Checklist

- [ ] Explain why there's no host↔device copy on the M1, and why that's both good and a tax.
- [ ] State roughly how much of 8 GB is actually yours, and what fills it at JimmyLabs's scale.
- [ ] Explain why the O(T²) attention matrix — not parameter count — is the memory to watch.
- [ ] Describe the MPS async trap and name three calls that force a sync.
- [ ] Explain why fanless thermals make cold benchmarks misleading.
- [ ] Explain gradient accumulation and when to use it on 8 GB.
- [ ] Say why mixed precision on MPS is an experiment, not a default.

---

## References

- PyTorch MPS backend documentation and release notes (behavior is version-dependent — read
  the notes for *your* version).
- Apple, *Metal* and *MLX* documentation.
- Vaswani et al. 2017 (the O(T²) term); Dao et al., *FlashAttention* (attacking it) — see
  [`18_RESEARCH_PAPERS.md`](18_RESEARCH_PAPERS.md).

## Further Reading

- [`architecture/apple_silicon_strategy.md`](../architecture/apple_silicon_strategy.md) —
  JimmyLabs's concrete device/dtype placement decisions.
- [`architecture/memory_layout.md`](../architecture/memory_layout.md) — where each large
  tensor lives in 8 GB.
- [`14_BENCHMARKING.md`](14_BENCHMARKING.md) — **next:** how to measure all of the above
  honestly.

> **Next:** [`14_BENCHMARKING.md`](14_BENCHMARKING.md) — optimization without measurement is
> superstition.
