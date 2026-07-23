# `datasets/` — data provenance & preparation

## Why this folder exists

The data determines what the model can possibly learn. This folder documents **where each
corpus came from, its license, its size, and how we turn it into token IDs** — so training
runs are reproducible and legally clean. The raw data itself is generally **not committed**
(it's large and often licensed); this folder holds the *recipe*, not the groceries.

## Planned corpora

| Corpus | Why | Size | License |
|--------|-----|------|---------|
| **Shakespeare** (tiny) | classic overfit-and-verify sanity set | ~1 MB | public domain |
| **TinyStories** | simple, coherent English at small scale | ~hundreds MB | check upstream |
| small code corpus | test structured-text modeling | tbd | permissive only |

## What belongs here

- `SOURCES.md` — exact URLs, versions, licenses, download dates
- Preparation notes: cleaning, encoding, train/val split ratio and seed
- Small sample snippets for illustration (not the full corpus)

## What does NOT belong here

- The full raw datasets (git-ignored — see repo `.gitignore`)
- The tokenizer implementation → [`src/`](../src/)
- Token-id caches → generated into a git-ignored path

## The rule on 8 GB

Datasets are streamed or memory-mapped, never fully loaded into RAM. See
[`docs/17_DATASET_GUIDE.md`](../docs/17_DATASET_GUIDE.md) and
[`architecture/memory_layout.md`](../architecture/memory_layout.md).
