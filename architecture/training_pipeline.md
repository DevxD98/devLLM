# Training Pipeline Execution & Data Flow

> **Concept Doc:** [`docs/11_TRAINING_PIPELINE.md`](../docs/11_TRAINING_PIPELINE.md)  
> **Technical Spec:** [`SPEC.md` §7](../SPEC.md#7-training-pipeline)

This document maps the concrete module boundaries, execution order, tensor shapes, and storage locations during a training iteration (`src/train.py`).

---

## Iteration Data Flow & Module Boundaries

```
   [ Dataset Sampler ]
           │
           ├── Inputs X   (B, T)    dtype: int64  ──► MPS Memory
           └── Targets Y  (B, T)    dtype: int64  ──► MPS Memory
                │
                ▼
   [ GPT Model Forward Pass ]
                │
                ▼
   Logits Tensor                    (B, T, V)     dtype: float32
                │
                ▼  Shifted View Reshape
   Logits (B*T, V) vs Targets (B*T)
                │
                ▼  PyTorch F.cross_entropy()
   Loss Scalar                      ()            dtype: float32
                │
                ▼  loss.backward()
   Autograd Computation Graph
                │
                ▼  Grad Clipping
   clip_grad_norm_(parameters, 1.0)
                │
                ▼  Scheduler Update
   lr = get_lr(step)
                │
                ▼  AdamW Step
   optimizer.step()
```

---

## Module Boundary Contracts (`src/train.py`)

### 1. Data Loader / Sampler (`src/dataset.py`)
- **Returns:** Tuple `(X, Y)` where `X` is input token IDs `(B, T)` and `Y` is target token IDs `(B, T)` shifted by +1 token position.

### 2. Loss Engine
- **Input Shapes:** `logits.view(-1, vocab_size)` of shape `(B*T, V)`, `targets.view(-1)` of shape `(B*T)`.
- **Output:** Scalar float loss tensor.

### 3. Checkpointing Contract (`checkpoints/`)
Saved dictionary schema:
```python
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'config': config,
    'step': step,
    'val_loss': val_loss,
    'seed': seed,
}
```
