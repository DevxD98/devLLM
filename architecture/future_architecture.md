# Future Architecture & Optimization Designs

> **Concept Docs:** [`docs/12_INFERENCE.md`](../docs/12_INFERENCE.md), [`docs/22_FUTURE_VERSIONS.md`](../docs/22_FUTURE_VERSIONS.md)  
> **Backlog Reference:** [`research/OPTIMIZATION_BACKLOG.md`](../research/OPTIMIZATION_BACKLOG.md)

This document outlines structural tensor designs for future architectural upgrades planned for v0.2 through v1.0, including KV Cache, Rotary Positional Embeddings (RoPE), and Grouped-Query Attention (GQA).

---

## 1. KV Cache Tensor Specifications (Planned v0.2)

### Tensor Shapes & Storage
During autoregressive generation, Keys and Values for past tokens are cached across $L$ layers:

```
   Cache Store Shape per Layer:
   Key Cache   (B, h, t_past, d_k)
   Value Cache (B, h, t_past, d_k)
```

```
   New Token Input        (B, 1)
        │
        ▼  Attention Projection
   New Q_new, K_new, V_new (B, h, 1, d_k)
        │
        ├── K_cached = concat([K_cached, K_new], dim=2)  ──► (B, h, t_past + 1, d_k)
        └── V_cached = concat([V_cached, V_new], dim=2)  ──► (B, h, t_past + 1, d_k)
             │
             ▼  Attention Matmul
   Attention Score = Q_new @ K_cached^T                  ──► (B, h, 1, t_past + 1)
```

---

## 2. Rotary Positional Embeddings (RoPE) (Planned v0.3)

Instead of adding absolute positional embeddings to token vectors at the input layer ([`07_EMBEDDINGS.md`](../docs/07_EMBEDDINGS.md)), RoPE rotates Query and Key vectors in 2D plane pairs by an angle proportional to sequence position $m$:

$$R_{\Theta, m}^d R_x(m) = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$$

---

## 3. Grouped-Query Attention (GQA) (Planned v1.0)

Grouped-Query Attention groups multiple Query heads to share a single Key/Value head:

```
   Multi-Head Attention (MHA)       Grouped-Query Attention (GQA)
   Query Heads:  8                  Query Heads:  8 (Grouped 4:1)
   Key/Val Heads: 8                  Key/Val Heads: 2
   (High KV Cache Memory)           (4x Reduction in KV Cache Memory)
```

GQA reduces memory footprint for the KV cache during generation on Apple Silicon 8 GB unified memory.
