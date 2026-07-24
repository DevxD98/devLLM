# Memory Layout & Allocation on 8 GB Unified Memory

> **Concept Doc:** [`docs/13_OPTIMIZATION_FOR_APPLE_SILICON.md`](../docs/13_OPTIMIZATION_FOR_APPLE_SILICON.md)  
> **Technical Spec:** [`SPEC.md` §6](../SPEC.md#6-memory-usage-fp32-on-8-gb)

This document maps memory allocations across CPU, MPS (GPU), weights, optimizer states, and dynamic activation tensors for DevLLM running on an 8 GB Apple Silicon M1 Mac.

---

## Memory Footprint Breakdown (v0.1 Baseline, fp32)

### 1. Fixed Memory Footprint (~13 MB total)

$$N_{\text{params}} \approx 0.82\text{M parameters}$$

```
   Component                  Formula                      v0.1 Footprint
   ────────────────────────────────────────────────────────────────────────
   Model Weights (fp32)       N_params × 4 B               ≈  3.3 MB
   Gradients (fp32)           N_params × 4 B               ≈  3.3 MB
   Adam Moments (m_t, v_t)    2 × N_params × 4 B           ≈  6.6 MB
   ────────────────────────────────────────────────────────────────────────
   Fixed Footprint Subtotal   4 × N_params × 4 B           ≈ 13.2 MB (Trivial)
```

---

### 2. Dynamic Activation & Attention Footprint

Dynamic activation memory scales with batch size $B$, sequence length $T$, embedding width $C$, and layer count $L$.

#### Attention Matrix Ceiling
The primary memory ceiling is the $O(T^2)$ attention score matrix ($QK^T$), stored for backpropagation across all $L$ layers:

$$\text{Attn Memory} = B \cdot h \cdot T \cdot T \cdot 4\text{ bytes} \times L \quad \text{(fp32)}$$

```
   Context Length T    Batch Size B    Attn Memory / Layer    Total Attn (L=4)
   ──────────────────────────────────────────────────────────────────────────
   T = 128             B = 32          ~1.0 MB                ~4.0 MB   (Comfortable)
   T = 256             B = 32          ~4.0 MB                ~16.0 MB  (Comfortable)
   T = 512             B = 32          ~16.0 MB               ~64.0 MB  (Moderate)
   T = 1024            B = 64          ~256.0 MB              ~1.0 GB   (High risk on 8 GB)
```

---

## Memory Layout Diagram

```
   Apple Silicon 8 GB Unified Memory Pool
   ┌───────────────────────────────────────────────────────────────┐
   │ macOS OS & System Applications (~3.5 - 4.5 GB)                │
   ├───────────────────────────────────────────────────────────────┤
   │ PyTorch MPS Unified Allocation Pool                           │
   │ ├── Fixed Weights & Adam State (~13.2 MB)                     │
   │ ├── Dataset Batch Buffers      (~2.0 MB)                      │
   │ └── Dynamic Activations & Attention Tensors                   │
   │     ├── Block Residuals (B, T, C, L)                          │
   │     ├── FFN Activations  (B, T, 4C, L)                         │
   │     └── QK^T Attn Tensor (B, h, T^2, L) ◄── Primary Ceiling    │
   └───────────────────────────────────────────────────────────────┘
```
