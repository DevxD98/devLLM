# Apple Silicon Execution Strategy & Hardware Integration

> **Concept Doc:** [`docs/13_OPTIMIZATION_FOR_APPLE_SILICON.md`](../docs/13_OPTIMIZATION_FOR_APPLE_SILICON.md)  
> **Technical Spec:** [`SPEC.md` §10](../SPEC.md#10-dependencies)

This document defines execution placement, MPS device dispatch, zero-copy unified memory hygiene, and thermal throttling strategies for running DevLLM on Apple Silicon (M1/M2/M3).

---

## Device Dispatch & Execution Flow

```
   [ PyTorch Host Python ]
             │
             ├── Device Selection ──► device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
             │
             ▼  Move Model & Tensors
   [ MPS Device Memory ]  (Unified Memory Architecture)
             │
             ▼  PyTorch Metal Performance Shaders (MPS) Graph
   ┌───────────────────────────────────────────────────────────┐
   │ Apple Silicon Unified Memory Pool                         │
   │ ├── CPU Core Access   (Data Loading, Tokenization)         │
   │ └── GPU Core Access   (Matmuls, Attention, LayerNorm)     │
   └───────────────────────────────────────────────────────────┘
```

---

## Core Optimization Rules for Apple Silicon

1. **Avoid CPU $\leftrightarrow$ MPS Ping-Pong:** Keep data batches on MPS device once loaded. Avoid calling `.item()`, `.cpu()`, or `.numpy()` inside training hot loops, as CPU synchronization stalls Metal GPU dispatch pipelines.
2. **Use `set_to_none=True` in `zero_grad()`:** Deallocates gradient tensor memory immediately rather than writing zeros, reducing memory allocation churn on unified memory pools.
3. **Thermal Throttling Management:** On fanless MacBook Air M1 devices, sustained 100% GPU load triggers thermal throttling after extended execution. Set evaluation and logging intervals to allow short thermal rest windows.
4. **FP32 Baseline Precision:** FP32 is the baseline precision contract for v0.1 MPS execution. FP16/BF16 mixed precision operations are evaluated strictly through benchmark-gated experiments ([`research/OPTIMIZATION_BACKLOG.md` #6](../research/OPTIMIZATION_BACKLOG.md#6)).
