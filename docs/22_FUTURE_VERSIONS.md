# 22 — Future Versions & Roadmap Extensions

> **Prerequisites:** [`21_DEVLOG.md`](21_DEVLOG.md).
>
> **Next:** [`23_RESOURCES.md`](23_RESOURCES.md)

---

## Purpose

This document outlines the planned evolutionary path for JimmyLabs beyond the v0.1 baseline model (~0.82M parameters). It details architectural upgrades, tokenization scaling, and performance optimizations planned for future versions.

---

## Version Growth Roadmap

```
   v0.1 Baseline (Current)      v0.2 Scaling               v0.3 & v1.0 Target
   ├── Char Tokenizer (~65)     ├── Char Tokenizer (~65)   ├── BPE Tokenizer (512)
   ├── 4 Layers, 128 Embed      ├── 6 Layers, 192 Embed    ├── 6 Layers, 256 Embed
   ├── ~0.82M Parameters        ├── ~2.7M Parameters       ├── ~4.0M Parameters
   └── FP32 Baseline (MPS)      └── KV Cache Generation    └── FlashAttn / SwiGLU
```

---

## Architectural Version Specs

### Version 0.1 — Computed Baseline
- **Focus:** Complete educational transparency and baseline MPS execution on 8 GB Mac.
- **Config:** `vocab_size=65`, `n_layer=4`, `n_head=4`, `n_embd=128`, `block_size=128`, weight tying active.
- **Param Count:** ~0.82M parameters ([`SPEC.md` §5](../SPEC.md#5-parameter-count-worked-for-v01)).

### Version 0.2 — Layer & Width Scaling
- **Focus:** Expanding depth and width to reach ~2.7M parameters while remaining within tight activation memory budgets.
- **Config:** `vocab_size=65`, `n_layer=6`, `n_head=6`, `n_embd=192`, `block_size=256`.
- **Key Feature:** Introduction of KV Cache for fast autoregressive generation ([`12_INFERENCE.md`](12_INFERENCE.md)).

### Version 0.3 — BPE Tokenizer Integration
- **Focus:** Transitioning from character-level tokenization to Byte-Pair Encoding (BPE).
- **Config:** `vocab_size=512`, `n_layer=6`, `n_head=8`, `n_embd=256`, `block_size=256`.
- **Key Feature:** Reduced sequence length $T$ per sentence and improved text generation coherence.

### Version 1.0 — Apple-Silicon Tuned Flagship
- **Focus:** Optimized 4M parameter model trained on larger datasets (TinyStories / OpenWebText subset).
- **Features:** RMSNorm, SwiGLU activations, FlashAttention tiling exploration ([`research/OPTIMIZATION_BACKLOG.md`](../research/OPTIMIZATION_BACKLOG.md)).

---

> **Next:** [`23_RESOURCES.md`](23_RESOURCES.md)
