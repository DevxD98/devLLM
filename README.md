<!--
        this readme is a small piece of abstract art.
        it is also a map. read it slowly. nothing here is decoration
        without meaning — every glyph points at an idea inside the repo.
-->

```
        ██████╗ ███████╗██╗   ██╗██╗     ██╗     ███╗   ███╗
        ██╔══██╗██╔════╝██║   ██║██║     ██║     ████╗ ████║
        ██║  ██║█████╗  ██║   ██║██║     ██║     ██╔████╔██║
        ██║  ██║██╔══╝  ╚██╗ ██╔╝██║     ██║     ██║╚██╔╝██║
        ██████╔╝███████╗ ╚████╔╝ ███████╗███████╗██║ ╚═╝ ██║
        ╚═════╝ ╚══════╝  ╚═══╝  ╚══════╝╚══════╝╚═╝     ╚═╝
```

```
   a language model, grown by hand, in eight gigabytes of silence.
```

<!-- ───────────────────────────────────────────────────────────── -->

```
                            ·  .  ˚    ✦
                     .      the void      ˚   ·
                  ˚    (untrained weights, pure noise)
                              │
                              │  gradient rain
                              ▼
        ·  ˖  ✦   ┌─────────────────────────────┐   ✦  ˖  ·
                  │   t o k e n s   →   m e a n  │
          ˚       └─────────────────────────────┘        ˚
                              │
                              ▼
                   ✶  a sentence that was
                       never written before  ✶
```

---

```
 ▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚
 ▞                                                            ▞
 ▞   D e v L L M  —  a GPT built from nothing on an M1 Air.   ▞
 ▞                                                            ▞
 ▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚
```

**DevLLM** is a from-scratch, GPT-style language model — **1–4 million parameters** —
engineered, trained, and profiled entirely on a **MacBook Air M1 with 8 GB of unified
memory**. No CUDA. No pretrained weights. No prebuilt transformer blocks.

It is **not** trying to beat GPT-4, Claude, Gemini, or Llama. It is trying to *understand
them* — every embedding, every attention head, every gradient — by rebuilding the machine
one comprehensible piece at a time, and writing down *why* each piece exists before *how*
it works.

> This repository is designed to read like a small research lab, not a tutorial.
> The code will be tiny. The **understanding** is meant to be enormous.

---

## The idea in one diagram

```
      raw text                                          new text
         │                                                 ▲
         ▼                                                 │
   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌──────────────┐
   │ tokenizer │ → │ embeddings│ → │transformer│ → │  lm head +   │
   │  (chars)  │   │  + pos.   │   │  blocks   │   │  sampling    │
   └───────────┘   └───────────┘   └───────────┘   └──────────────┘
         └──────────── everything below is built by hand ─────────┘

   PyTorch is allowed to do exactly four things:
      tensors  ·  autograd  ·  optimizers  ·  MPS (Metal) acceleration
   nothing else is imported that we could have written ourselves.
```

---

## How to read this repository

DevLLM is documentation-first. The knowledge is organized as a **path**, not a pile.

```
   START HERE
      │
      ▼
   docs/00 → 03      the "why": vision, roadmap, architecture, learning path
      │
      ▼
   docs/04 → 10      the "what": math → neurons → tokens → attention → GPT
      │
      ▼
   docs/11 → 14      the "how it runs": training, inference, Apple-Silicon, benchmarks
      │
      ▼
   docs/15 → 23      the "how we work": experiments, configs, papers, glossary, devlog
      │
      ▼
   architecture/     the wiring diagrams behind every doc above
      │
      ▼
   research/         paper notes · experiment logs · design decisions (ADRs)
```

New to machine learning? Open **[docs/03_LEARNING_PATH.md](docs/03_LEARNING_PATH.md)**.
It assumes basic Python and *almost nothing else*, and routes you through the rest.

---

## Repository map

```
DevLLM/
├── README.md            ← you are here
├── docs/                the textbook (numbered, read in order)
├── architecture/        deep-dive wiring diagrams
├── research/            paper notes · experiments · benchmarks · design decisions
├── experiments/         run logs (one folder per experiment)
├── benchmarks/          reproducible performance measurements
├── notes/               loose thinking, kept honest
├── assets/              images referenced by docs
├── diagrams/            source-of-truth ASCII diagrams reused across docs
├── configs/             annotated model/training configurations
├── scripts/             utility entry points (documented, not yet code)
├── datasets/            data provenance, licensing, preparation notes
├── src/                 the model itself (module map mirrors architecture/)
├── tests/               what every module must prove about itself
├── checkpoints/         saved weights (git-ignored; format documented)
└── outputs/             generated text samples
```

Every folder contains a `README.md` explaining **why it exists**, what belongs inside,
and — just as importantly — what does *not*.

---

## Design commitments

```
   ①  understanding  >  speed
   ②  correctness    >  optimization
   ③  measure every optimization — before & after, or it didn't happen
   ④  small, elegant, maintainable code
   ⑤  every experiment reproducible from a config + a seed
```

These are not slogans. They are enforced by the repository's structure: every
optimization lives next to its benchmark, every architecture change next to its ADR,
every claim next to the evidence for it.

---

## The machine we build on

```
   ┌──────────────────────────────────────────────────────────┐
   │  MacBook Air M1                                           │
   │  ┌────────────┐   8 GB UNIFIED MEMORY   ┌────────────┐    │
   │  │  CPU cores │ ◀────── one pool ──────▶ │  GPU (MPS) │    │
   │  └────────────┘   no copy across bus     └────────────┘    │
   │                                                            │
   │  the constraint is the point. small memory forces small,   │
   │  honest models — and every optimization must earn its RAM. │
   └──────────────────────────────────────────────────────────┘
```

Apple Silicon is treated as a first-class target, not an afterthought — see
**[docs/13_OPTIMIZATION_FOR_APPLE_SILICON.md](docs/13_OPTIMIZATION_FOR_APPLE_SILICON.md)**
and **[architecture/apple_silicon_strategy.md](architecture/apple_silicon_strategy.md)**.

---

## Status

```
   phase 0  foundations ......  ▓▓▓▓▓▓▓▓▓▓  documentation scaffolding
   phase 1  nlp foundations ..  ░░░░░░░░░░  planned
   phase 2  tinygpt .........   ░░░░░░░░░░  planned
   phase 3  training ........   ░░░░░░░░░░  planned
   phase 4  optimization ....   ░░░░░░░░░░  planned
   phase 5  research ........   ░░░░░░░░░░  ongoing
```

This is an active, in-progress **3–4 month educational build**. The documentation is
being written ahead of the code on purpose: you cannot engineer clearly what you cannot
explain clearly.

---

```
        every large model was once a small one
        that someone refused to stop understanding.
```

<sub>License: TBD · Built by hand on Apple Silicon · Documentation-first, forever.</sub>
