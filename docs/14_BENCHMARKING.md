# 14 — Benchmarking

> **Prerequisites:** [`13_OPTIMIZATION_FOR_APPLE_SILICON.md`](13_OPTIMIZATION_FOR_APPLE_SILICON.md)
> (the machine's quirks: warmup, async syncs, thermals). Governed by
> [`ENGINEERING_PRINCIPLES.md`](../ENGINEERING_PRINCIPLES.md) principles 1–2.
>
> **Next:** [`15_EXPERIMENT_GUIDE.md`](15_EXPERIMENT_GUIDE.md).

---

## Purpose

Optimization without measurement is superstition. This document defines DevLLM's
**benchmark suite** — *what* we measure, *how* we measure it honestly on a fanless M1, and
*where* results are recorded — so that every performance claim in the repo is backed by a
reproducible number, and every optimization has a before/after (principle 2).

---

## Background

### Why benchmarking is hard on this machine

Three properties of the M1 Air make naive timing lie to you (all from
[`docs/13`](13_OPTIMIZATION_FOR_APPLE_SILICON.md)):

- **Kernel compilation** — the first MPS iterations are slow while Metal compiles shaders.
- **Async execution** — GPU work is queued; timing without an explicit sync measures the
  *queue*, not the *work*.
- **Thermal throttling** — a fanless chip runs fast cold, then slows under sustained load.

A benchmark that ignores these produces confident, wrong numbers. The whole method below
exists to defeat them.

---

## Concepts

- **Metric** — a specific measurable quantity (tokens/sec, peak memory).
- **Warmup** — early iterations discarded before measuring (kernel compile + caches).
- **Synchronization** — forcing the GPU queue to drain so elapsed time reflects real work
  (`torch.mps.synchronize()` around the timed region).
- **Warm vs cold** — thermal state at measurement; must be recorded.
- **Median over mean** — robust to thermal drift and outliers; DevLLM reports medians.
- **Before/after** — the only valid form of an optimization result (one variable changed).

---

## Detailed Explanation — the suite

### What we measure

| Metric | Why it matters | How |
|--------|----------------|-----|
| **Tokens/sec (train)** | training throughput → how long a run takes | tokens processed ÷ synced wall-time, warm, median of N |
| **Tokens/sec (generate)** | interactive usability | tokens generated ÷ synced time, warmup discarded |
| **Peak memory** | the 8 GB ceiling | max resident / MPS allocated during a run |
| **Training step time** | spot regressions | median ms/step, warm |
| **Model size (params)** | capacity & memory | counted from the model, matches [`SPEC.md`](../SPEC.md) §5 |
| **Checkpoint size** | disk & resume cost | bytes of the saved `.pt` |
| **Startup latency** | dev-loop friction | import → model-on-device → first forward |
| **Prompt latency** | time-to-first-token | prompt encode → first generated token |

### How to measure honestly (the protocol)

```
   1. FIX the seed and the config (a benchmark is config + seed).
   2. RECORD machine state: macOS, torch, device, dtype, thermal (cold/warm), batch, block.
   3. WARM UP: run K iterations and discard them (K large enough that timings stabilize).
   4. SYNCHRONIZE around the timed region (drain the MPS queue) — else you time the queue.
   5. REPEAT N times; report the MEDIAN (and spread), not the best.
   6. CHANGE ONE THING for an optimization; re-run identically; report before/after + Δ.
   7. WRITE IT DOWN in benchmarks/ using the template. A number not written down is lost.
```

Two failure modes this specifically prevents: (a) reporting the cold burst as if it were
sustained throughput, and (b) a hidden MPS→CPU op-fallback silently dominating the timing
without you noticing — always sanity-check that the workload is actually on-device.

### Where results live

- Blank form: [`research/benchmark_templates/BENCHMARK_TEMPLATE.md`](../research/benchmark_templates/BENCHMARK_TEMPLATE.md)
- Filled records: [`benchmarks/`](../benchmarks/) (`NNN_short-name.md` + raw data in
  `benchmarks/results/`).
- Optimizations tie back to [`research/OPTIMIZATION_BACKLOG.md`](../research/OPTIMIZATION_BACKLOG.md):
  each item is only "done" when a benchmark proves its win.

### Establish a baseline first

Before any optimization, record **001_baseline** for the v0.1 config: train tokens/sec,
generate tokens/sec, peak memory, checkpoint size — all warm, all with the protocol above.
Every later optimization is measured *against this baseline*. No baseline → no valid
"faster."

---

## Visual Diagrams

### The measurement timeline

```
   │◄─ warmup (discard) ─►│◄──────── measured region (sync-bracketed) ────────►│
   ▁▁▂▃▅▇███              ████████████████████████████████████████████████████
   kernels compiling,     steady state — take the MEDIAN of N runs here
   caches filling
                          └─ but watch: is the chip still cold? run warm too.
```

### Cold vs warm (why we record thermal state)

```
   tokens/sec
     ▲
   ██│███▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
     │  ╲___ throttling as the fanless chip heats ___
     │              settles → THIS is your planning number
     └──────────────────────────────────────────────► time
       cold burst overstates sustained training throughput
```

---

## Common Mistakes

- **Including warmup in the measurement** → you benchmark the Metal compiler.
- **No synchronization** → you time the async queue, getting impossibly fast numbers.
- **Reporting the best (or cold) number** → unrepresentative of a real run; report warm
  medians.
- **Changing two things at once** → the Δ is uninterpretable (principle 7).
- **Not recording machine/thermal state** → the number is irreproducible (principle 2).
- **Ignoring a silent CPU fallback** → "the optimization did nothing" when really an op ran
  on CPU the whole time.
- **Trusting one run** → thermal drift and OS noise demand N runs.

---

## Future Improvements

- A reusable `scripts/benchmark.py` implementing the protocol (warmup, sync, median-of-N,
  machine-state capture) so benchmarks are one command.
- Automatic **regression detection**: fail if tokens/sec drops >X% vs the last baseline.
- Plots (tokens/sec vs time) to visualize throttling per run, saved to
  [`assets/`](../assets/).
- An MLX-vs-PyTorch comparison harness (same model, same machine).

---

## Glossary

| Term | Meaning |
|------|---------|
| Warmup | discarded early iterations (kernel compile, cache fill) |
| Synchronize | drain the MPS queue so timing reflects real work |
| Warm / cold | thermal state at measurement time |
| Median-of-N | robust central number across N repeated runs |
| Baseline | the reference measurement every optimization is compared against |
| Regression | an unintended performance drop vs a prior baseline |

---

## Learning Checklist

- [ ] List the eight metrics the suite tracks.
- [ ] State the 7-step honest-measurement protocol from memory.
- [ ] Explain why warmup must be discarded and why syncing is required on MPS.
- [ ] Explain why we report warm medians, not cold bests.
- [ ] Explain why every optimization needs a baseline and a one-variable before/after.

---

## References

- PyTorch MPS docs (synchronization, memory APIs) — version-specific.
- [`ENGINEERING_PRINCIPLES.md`](../ENGINEERING_PRINCIPLES.md) principles 1, 2, 7.
- [`13_OPTIMIZATION_FOR_APPLE_SILICON.md`](13_OPTIMIZATION_FOR_APPLE_SILICON.md) (the quirks
  this protocol defeats).

## Further Reading

- [`research/OPTIMIZATION_BACKLOG.md`](../research/OPTIMIZATION_BACKLOG.md) — what to
  benchmark next, and why.
- [`15_EXPERIMENT_GUIDE.md`](15_EXPERIMENT_GUIDE.md) — **next:** the same rigor for
  *accuracy* experiments, not just speed.
