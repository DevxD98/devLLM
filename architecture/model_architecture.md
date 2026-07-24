# Model Architecture Wiring & Tensor Shapes

> **Concept Doc:** [`docs/10_GPT_ARCHITECTURE.md`](../docs/10_GPT_ARCHITECTURE.md)  
> **Technical Spec:** [`SPEC.md` §2–4](../SPEC.md#2-architecture)

This document defines the concrete module wiring, tensor data paths, and shape contracts for JimmyLabs's decoder-only GPT model (`src/model/`).

---

## End-to-End Tensor Flow & Shapes

Notation: `B` = Batch Size, `T` = Sequence Length ($\le \text{block\_size}$), `C` = Embedding Dimension ($\text{n\_embd}$), `h` = Head Count ($\text{n\_head}$), `d_k` = $C / h$, `V` = Vocabulary Size ($\text{vocab\_size}$).

```
   Input Tensor idx                 (B, T)                int64
      │
      ├── Token Embedding Lookup    (B, T, C)             wte(idx)
      └── Positional Embedding      (1, T, C)             wpe(0..T-1)
           │
           ▼  Sum (+)
   Hidden State H0                  (B, T, C)             fp32
      │
      ├── Block 0 (Pre-Norm)        (B, T, C)             block_0(H0)
      ├── Block 1 (Pre-Norm)        (B, T, C)             block_1(H1)
      ├── ...                       (B, T, C)
      └── Block L-1 (Pre-Norm)      (B, T, C)             block_(L-1)
           │
           ▼
   Hidden State HL                  (B, T, C)
      │
      ▼  Final LayerNorm
   Normalized State H_norm          (B, T, C)             ln_f(HL)
      │
      ▼  LM Head (Weight Tied)
   Output Logits                    (B, T, V)             H_norm @ wte.weight.T
```

---

## Module Boundary Contracts (`src/model/`)

### 1. `GPT` (`src/model/gpt.py`)
- **Inputs:** `idx: torch.Tensor` of shape `(B, T)`, `dtype=torch.int64`.
- **Outputs:** `logits: torch.Tensor` of shape `(B, T, V)`, `dtype=torch.float32`.
- **Submodules:** `wte` (`nn.Embedding`), `wpe` (`nn.Embedding`), `blocks` (`nn.ModuleList`), `ln_f` (`nn.LayerNorm`).

### 2. `Block` (`src/model/block.py`)
- **Inputs:** `x: torch.Tensor` of shape `(B, T, C)`.
- **Outputs:** `y: torch.Tensor` of shape `(B, T, C)`.
- **Internal Wiring:**
  ```python
  x = x + self.attn(self.ln_1(x))  # Attention sublayer (B, T, C)
  x = x + self.mlp(self.ln_2(x))   # Feed-Forward sublayer (B, T, C)
  ```

### 3. `MLP` (`src/model/mlp.py`)
- **Inputs:** `x: torch.Tensor` of shape `(B, T, C)`.
- **Outputs:** `y: torch.Tensor` of shape `(B, T, C)`.
- **Shape Transformations:**
  ```
  (B, T, C) ──► c_fc ──► (B, T, 4C) ──► GELU ──► (B, T, 4C) ──► c_proj ──► (B, T, C)
  ```
