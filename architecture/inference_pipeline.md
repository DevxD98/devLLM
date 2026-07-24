# Inference Pipeline Execution & Data Flow

> **Concept Doc:** [`docs/12_INFERENCE.md`](../docs/12_INFERENCE.md)  
> **Technical Spec:** [`SPEC.md` §8](../SPEC.md#8-inference-pipeline)

This document details module interactions, tensor transformations, and sampling boundaries during text generation (`src/generate.py`).

---

## Inference Execution Pipeline

```
   Prompt Text String
        │
        ▼  Tokenizer encode()
   Input Token IDs                 (1, t)            int64
        │
        ├──► Autoregressive Generation Loop (max_new_tokens)
        │       │
        │       ▼  Crop to block_size if sequence > T
        │    Context Tensor        (1, t_crop)       t_crop <= block_size
        │       │
        │       ▼  Model Forward Pass
        │    Logits Tensor         (1, t_crop, V)
        │       │
        │       ▼  Extract Last Position
        │    Last Logits           (1, V)            logits[:, -1, :]
        │       │
        │       ▼  Apply Temperature Scaling (÷ T_temp)
        │    Scaled Logits         (1, V)
        │       │
        │       ▼  Top-K / Top-P Masking (-∞)
        │    Filtered Logits       (1, V)
        │       │
        │       ▼  Softmax (dim=-1)
        │    Probabilities         (1, V)            sum = 1.0
        │       │
        │       ▼  torch.multinomial(probs, num_samples=1)
        │    Sampled Token ID      (1, 1)            int64
        │       │
        │       ▼  Concat to Sequence
        │    Sequence Tensor       (1, t + 1)
        └───────┘
        │
        ▼  Tokenizer decode()
   Generated Text String
```

---

## Module Boundary Contracts (`src/generate.py`)

### `generate()` Function Interface
```python
@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.Tensor,         # Shape: (1, t), dtype: int64
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None
) -> torch.Tensor:              # Shape: (1, t + max_new_tokens)
```
