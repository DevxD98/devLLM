# Multi-Head Attention Tensor Flow & Shapes

> **Concept Doc:** [`docs/08_ATTENTION.md`](../docs/08_ATTENTION.md)  
> **Technical Spec:** [`SPEC.md` §4](../SPEC.md#4-tensor-shapes-the-contract-every-module-honors)

This document defines the detailed tensor transformations, matrix multiplications, and head split/merge mechanics inside `CausalSelfAttention` (`src/model/attention.py`).

---

## Detailed Dataflow & Tensor Transformations

Notation: `B` = Batch Size, `T` = Sequence Length ($\le \text{block\_size}$), `C` = Hidden Dim ($\text{n\_embd}$), `h` = Head Count ($\text{n\_head}$), `d_k` = $C / h$.

```
   Input Tensor x                  (B, T, C)
      │
      ▼  Linear Projection (c_attn)
   Combined Q, K, V                (B, T, 3C)
      │
      ▼  Split into Q, K, V & Reshape
   Q, K, V Tensors                 (B, h, T, d_k)        each
      │
      ├───────────────────────┐
      ▼                       ▼
   Query Q (B, h, T, d_k)   Key K (B, h, T, d_k)
      │                       │  Transpose (K^T)
      │                       ▼  (B, h, d_k, T)
      └───────────┬───────────┘
                  ▼  Batched Matmul (Q @ K^T)
   Raw Scores                      (B, h, T, T)          ⚠ O(T²) Tensor
                  │
                  ▼  Scale by ÷ √(d_k)
   Scaled Scores                   (B, h, T, T)
                  │
                  ▼  Causal Mask (-∞ above diagonal)
   Masked Scores                   (B, h, T, T)
                  │
                  ▼  Softmax (dim=-1)
   Attention Weights               (B, h, T, T)
                  │
                  ├───────────────────────┐
                  ▼                       ▼
   Weights (B, h, T, T)             Value V (B, h, T, d_k)
                  │                       │
                  └───────────┬───────────┘
                              ▼  Matmul (Weights @ V)
   Context Output                  (B, h, T, d_k)
                              │
                              ▼  Transpose & Contiguous Reshape
   Merged Heads                    (B, T, C)
                              │
                              ▼  Linear Projection (c_proj)
   Attention Output                (B, T, C)
```

---

## PyTorch Module Implementation Contract (`src/model/attention.py`)

### Input / Output Contract
- **Input:** `x: torch.Tensor` of shape `(B, T, C)`.
- **Output:** `y: torch.Tensor` of shape `(B, T, C)`.

### Key Code Transformation Steps
```python
# 1. Combined QKV projection
qkv = self.c_attn(x) # (B, T, 3C)

# 2. Split and reshape for multi-head attention
q, k, v = qkv.split(self.n_embd, dim=2)
q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, h, T, d_k)
k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, h, T, d_k)
v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, h, T, d_k)

# 3. Scaled dot-product attention with causal mask
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, h, T, T)
att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
att = F.softmax(att, dim=-1)
y = att @ v # (B, h, T, d_k)

# 4. Re-assemble heads and project
y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)
return self.c_proj(y) # (B, T, C)
```
