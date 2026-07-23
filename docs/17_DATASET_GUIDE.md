# 17 — Dataset Guide

> **Prerequisites:** [`06_TOKENIZER.md`](06_TOKENIZER.md) (text → ids) and the TinyStories
> thesis from [`research/tiny_gpt_landscape.md`](../research/tiny_gpt_landscape.md).
>
> **Next:** [`18_RESEARCH_PAPERS.md`](18_RESEARCH_PAPERS.md). Provenance:
> [`datasets/SOURCES.md`](../datasets/SOURCES.md).

---

## Purpose

The data decides what the model can possibly learn. This document explains **which corpora
DevLLM uses and why**, the licensing and cleaning each requires, and — crucially — the one
principle that governs all of it: **match data complexity to model capacity.** A 1–4M-param
model fed the whole internet learns to produce fluent-sounding mush; the same model fed
simple, consistent English can produce genuinely coherent text.

---

## Background

### The central principle (from TinyStories)

Eldan & Li's *TinyStories* showed that models with only millions of parameters generate
coherent English **when the training text is simple enough** (short stories using a small
vocabulary a 3–4-year-old would know). The lesson is not "small models are bad" — it's
"small models need appropriately simple data." This directly shapes DevLLM's choices: we
prefer **small, clean, consistent** corpora over large, messy, diverse ones.

```
   capacity ── too big for data ──► memorizes / overfits
      match  ─────────────────────► learns real structure  ✅
   capacity ── too small for data ─► fluent-sounding mush
```

### The 8 GB angle

Data must be **streamed or memory-mapped**, never fully resident in RAM alongside the model
([`13_OPTIMIZATION_FOR_APPLE_SILICON.md`](13_OPTIMIZATION_FOR_APPLE_SILICON.md)). At
character level, a corpus is pre-tokenized once into a flat array of int IDs on disk and
`mmap`-ed, so the OS pages in only what a batch needs.

---

## Concepts

- **Corpus** — the raw text collection.
- **Token budget** — total tokens seen during training (steps × batch × block). Chinchilla
  ([`18`](18_RESEARCH_PAPERS.md)) argues params and tokens should scale together; at our
  scale, small models want *modest* token budgets, not endless data.
- **Provenance** — where data came from, its version, and its license. Recorded in
  [`datasets/SOURCES.md`](../datasets/SOURCES.md); a corpus without provenance is not used.
- **Cleaning** — normalization (encoding, whitespace, control chars) before tokenizing.
- **Train/val split** — a seeded hold-out for honest evaluation ([`SPEC.md`](../SPEC.md) §13).

---

## Detailed Explanation — the candidate corpora

| Corpus | Size | Complexity | Fit for a 1–4M char model | License (verify!) |
|--------|------|-----------|---------------------------|-------------------|
| **Shakespeare (tiny)** | ~1 MB | medium, archaic | **sanity/overfit set** — perfect first target | public domain |
| **TinyStories** | ~hundreds MB | *low* (by design) | **primary corpus** — ideal capacity match | check upstream terms |
| WikiText-2 / -103 | ~12 MB / ~500 MB | medium-high | -2 usable subset; -103 too broad for us | CC BY-SA |
| OpenWebText | ~40 GB | high, diverse | **too big/broad** — mush risk; only tiny subsets | scraped; check terms |
| SlimPajama | ~600 GB+ | very high | far beyond our scale | mixed; check terms |
| FineWeb | ~TBs | very high | far beyond our scale | check terms |
| Code (e.g. small Stack subset) | varies | structured | fun v0.3+ experiment (structure ≠ prose) | **permissive licenses only** |

### The verdict for DevLLM

1. **Shakespeare first** — ~1 MB, one file, public domain. It's the *sanity* dataset: small
   enough to overfit deliberately (proving the model can learn — see the overfit-a-batch
   test in [`tests/TESTING_STRATEGY.md`](../tests/TESTING_STRATEGY.md)), and it produces
   recognizable output fast. This is where v0.1 starts.
2. **TinyStories as the primary corpus** — its deliberately simple English is the best
   *capacity match* for a 1–4M-param model. This is where coherent generation is most
   likely.
3. **WikiText-2** — an optional step up in complexity once TinyStories is working, to see
   the model strain against harder text.
4. **A small, permissively-licensed code subset** — a v0.3+ experiment in modeling
   *structured* text; interesting precisely because code has different statistics than prose.
5. **OpenWebText / SlimPajama / FineWeb** — **not used for training** at our scale. Their
   size and diversity guarantee mush from a tiny model, and they can't fit or finish
   sensibly on an 8 GB Air. They're named here so the *reason* they're excluded is on record.

### Licensing — non-negotiable

Only train on data we're permitted to use. Public-domain (Shakespeare) and clearly-licensed
sets are safe; scraped web corpora carry murkier terms. **Every corpus's exact license,
URL, version, and download date goes in [`datasets/SOURCES.md`](../datasets/SOURCES.md)
before training** (principle: no result without a record). For code, restrict to permissive
licenses.

### Cleaning pipeline

```
   raw text
     │  1. normalize encoding → UTF-8; drop/replace invalid bytes
     │  2. normalize whitespace & line endings (consistent \n)
     │  3. strip control chars; optionally lowercase (char-model choice — log it!)
     │  4. build vocab from the cleaned corpus (char set)
     │  5. tokenize → flat int array on disk
     ▼  6. seeded train/val split (e.g. 90/10)
   *.idx (mmap-able token ids, git-ignored)
```

Each cleaning choice is a *decision* that affects results — record non-obvious ones (e.g.
"lowercased everything") as a note or ADR, because they change the vocab and therefore the
model.

### Expected training time (honest, order-of-magnitude)

Concrete tokens/sec must be **measured** on the target machine
([`14_BENCHMARKING.md`](14_BENCHMARKING.md)) — it depends on config, dtype, thermals, and
macOS/torch version, and this doc will not invent a number. Directionally:

- **Shakespeare + v0.1:** minutes to low hours to reach recognizable output — fast enough to
  iterate many times a day. This fast loop is *why* it's the starting corpus.
- **TinyStories + v0.1–v0.3:** hours, potentially across sessions (mind fanless thermals);
  use checkpointing so runs resume.
- Anything web-scale: effectively never, on this machine — which is the point of not using it.

Fill the real numbers into a benchmark record once measured.

---

## Visual Diagrams

### Data → batches pipeline

```
   corpus.txt ─clean─► ids.idx (mmap) ─split─► train.idx / val.idx
                                         │
                          sample random (B, T) windows
                                         ▼
                                   (B, T) int tensor ──► model
   memory-mapped: only the sampled windows are paged into RAM (8 GB safe)
```

### Complexity-vs-capacity map

```
   simple ├─ Shakespeare ─ TinyStories ─┼─ WikiText ─┼─ OpenWebText ─ FineWeb ─┤ complex
          │   ✅ sanity      ✅ primary    │  ⚠ stretch │      ❌ mush for 1–4M       │
          └──────────── DevLLM lives here ─┘            └── excluded (on record) ─────┘
```

---

## Common Mistakes

- **Feeding a tiny model a huge, diverse corpus** and concluding "small models don't work."
  Wrong corpus, not wrong model — the TinyStories result disproves it.
- **Loading the whole dataset into RAM.** On 8 GB this competes with the model. Memory-map.
- **Skipping provenance/licensing** "just to experiment." A result on unrecorded data is
  unreproducible and possibly unusable. Record first.
- **Changing cleaning silently.** Lowercasing or stripping punctuation changes the vocab and
  the task; log it or you can't compare runs.
- **No fixed val split / leaking val into train.** Makes every metric a lie. Seed the split.
- **Over-training on 1 MB Shakespeare and calling it good.** It'll memorize; that proves the
  loop works, not that the model generalizes. Move to TinyStories for real learning.

---

## Future Improvements

- A tiny **data-cleaning script** with recorded, reversible steps (v0.2).
- **Curriculum learning** experiment: Shakespeare → TinyStories → WikiText in stages.
- **Dataset ablations** (how much data does v0.1 actually need before diminishing returns?)
  — a natural experiment tied to the Chinchilla reading.
- A **streaming loader** for larger subsets without full download.

---

## Glossary

| Term | Meaning |
|------|---------|
| Corpus | the raw text used for training |
| Provenance | recorded source, version, license, and date of a dataset |
| Token budget | total tokens seen in training (steps·batch·block) |
| mmap | memory-mapping a file so the OS pages in only what's needed |
| Train/val split | seeded partition into training and held-out evaluation data |
| Capacity match | choosing data complexity appropriate to model size |

---

## Learning Checklist

- [ ] State the "match data complexity to capacity" principle and cite TinyStories.
- [ ] Explain why Shakespeare is the *sanity* corpus and TinyStories the *primary* one.
- [ ] Explain why OpenWebText/FineWeb are excluded at this scale.
- [ ] List what must be recorded in `datasets/SOURCES.md` before training.
- [ ] Explain why data is memory-mapped on 8 GB.
- [ ] Explain why cleaning choices must be logged.

---

## References

- Eldan & Li, *TinyStories: How Small Can Language Models Be and Still Speak Coherent
  English?* (2023) — the guiding result.
- Hoffmann et al., *Training Compute-Optimal LLMs* (Chinchilla, 2022) — params↔tokens.
- Merity et al., *WikiText* — the language-modeling benchmark.
- Corpus specifics and licenses: [`datasets/SOURCES.md`](../datasets/SOURCES.md).

## Further Reading

- [`research/tiny_gpt_landscape.md`](../research/tiny_gpt_landscape.md) (TinyStories section).
- [`18_RESEARCH_PAPERS.md`](18_RESEARCH_PAPERS.md) — **next.**
