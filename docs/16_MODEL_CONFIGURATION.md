# 16 — Model Configuration

> **Prerequisites:** [`02_ARCHITECTURE.md`](02_ARCHITECTURE.md), [`09_TRANSFORMER.md`](09_TRANSFORMER.md), and [`SPEC.md`](../SPEC.md).
>
> **Next:** [`17_DATASET_GUIDE.md`](17_DATASET_GUIDE.md)

---

## Purpose

A model in JimmyLabs is not defined by arbitrary inline values scattered across code—it is defined strictly as **a configuration file plus a random seed**. 

This document explains the specification, role, memory cost, and quality trade-offs of every configuration field in JimmyLabs. It enforces **Principle 8** (*Configuration is separate from implementation*) and **Principle 5** (*No unexplained magic numbers*) from [`ENGINEERING_PRINCIPLES.md`](../ENGINEERING_PRINCIPLES.md). By decoupling hyperparameter declaration from execution logic, we ensure that every training run and model architecture variant is diffable, reproducible, and explicitly budgeted for an 8 GB Apple Silicon unified memory environment.

---

## Background

### Why hardcoded constants break research

In many beginner implementations, architectural parameters (like hidden dimension or sequence length) and training parameters (like learning rate or batch size) are hardcoded directly into Python scripts. This anti-pattern introduces severe issues:
1. **Unreproducible experiments:** Subtle changes in script constants overwrite prior state, making past runs unrepeatable.
2. **Hidden memory cliffs:** On an 8 GB M1 Mac, increasing context length or batch size without calculating tensor memory leading to sudden out-of-memory (OOM) crashes on the Metal Performance Shaders (MPS) backend.
3. **Implicit assumptions:** Magic numbers obscure design decisions. A reader cannot tell if `n_embd = 128` was chosen for quality, hardware limits, or convenience.

### The 8 GB unified memory constraint

Apple Silicon utilizes a unified memory architecture where CPU and GPU (MPS) share a common pool of 8 GB LPDDR4X RAM. Unlike dedicated GPUs with VRAM isolation, OS overhead and background applications reduce the usable memory pool. 

While static parameter weights and Adam optimizer states for a 1–4M parameter model require only a few megabytes (as computed in [`SPEC.md` §6](../SPEC.md#6-memory-usage-fp32-on-8-gb)), dynamic activation memory—specifically the $O(T^2)$ self-attention score matrix—scales rapidly with batch size $B$ and block size $T$. Configuration parameters are therefore memory decisions first.

---

## Concepts

Hyperparameters fall into two canonical categories, stored together in version-controlled YAML files under [`configs/`](../configs/):

- **Model Hyperparameters:** Define the neural network graph architecture (`vocab_size`, `n_layer`, `n_head`, `n_embd`, `block_size`, `dropout`, `weight_tying`). Modifying any of these changes the parameter count or layer shapes, producing an incompatible checkpoint tensor structure.
- **Training Hyperparameters:** Control the optimization process (`batch_size`, `lr`, `warmup_steps`, `max_steps`, `weight_decay`, `grad_clip`, `eval_interval`, `seed`). Modifying these changes optimization dynamics without altering the model's architectural contract.

```
   ┌─────────────────────────────────────────────────────────────┐
   │                       configs/*.yaml                        │
   ├──────────────────────────────┬──────────────────────────────┤
   │     Model Hyperparams        │     Training Hyperparams     │
   │  n_layer, n_head, n_embd,    │  batch_size, lr, seed,       │
   │  block_size, weight_tying... │  warmup_steps, max_steps...  │
   └──────────────┬───────────────┴──────────────┬───────────────┘
                  │                              │
                  ▼                              ▼
          [ Model Instantiation ]        [ Optimizer & Loop ]
```

---

## Detailed Explanation

Below is the field-by-field specification for all 15 configuration keys. Parameter math and memory equations match [`SPEC.md` §5–6](../SPEC.md#5-parameter-count-worked-for-v01).

### 1. `vocab_size` (int)
- **What it is:** The total number of unique tokens in the vocabulary (characters in character-level tokenization, or sub-words in BPE). Canonical tokenization details live in [`06_TOKENIZER.md`](06_TOKENIZER.md).
- **Trade-off:** Larger vocabularies increase sequence efficiency (fewer tokens per text), but enlarge the embedding lookup matrix and logit output layer.
- **Memory & Parameter Cost:** 
  - Token embedding parameters: $V \cdot C$ (where $V = \text{vocab\_size}$, $C = \text{n\_embd}$).
  - With weight tying active ([`ADR-0003`](../research/design_decisions/ADR-0003-weight-tying-default.md)), the output LM head adds $0$ extra parameters. Without tying, it adds an additional $V \cdot C$.
  - At $V = 65, C = 128$, embedding parameters $= 65 \cdot 128 = 8,320$ (~33.3 KB in fp32).

### 2. `n_layer` (int)
- **What it is:** The number of stacked pre-norm transformer blocks. Block mechanics are defined in [`09_TRANSFORMER.md`](09_TRANSFORMER.md).
- **Trade-off:** Increasing depth enables higher compositional abstraction and reasoning depth, but linearly increases parameter count, compute FLOPs, and activation memory stored for backpropagation.
- **Memory & Parameter Cost:** 
  - Adds $\approx 12 \cdot C^2 \cdot L$ parameters across $L$ layers.
  - Linear scaling on fixed weights, gradients, and activation memory ($L \times$).

### 3. `n_head` (int)
- **What it is:** The number of parallel attention heads in multi-head self-attention ([`08_ATTENTION.md`](08_ATTENTION.md)).
- **Trade-off:** More heads allow the model to simultaneously attend to different representation sub-spaces (e.g. syntax, positional distance). Must evenly divide `n_embd` so per-head dimension $d_k = C / h$ is an integer.
- **Memory & Parameter Cost:** 
  - Multi-head attention splits $C$ into $h$ heads of width $d_k$; total projection parameters ($W_q, W_k, W_v, W_o$) remain $4 \cdot C^2$ regardless of $h$.
  - Attention score matrix shape per layer is $(B, h, T, T)$. Memory scales directly with $h$.

### 4. `n_embd` (int)
- **What it is:** The hidden embedding dimension ($C$) across transformer blocks.
- **Trade-off:** Dominates model capacity. Higher $C$ increases representation width, but scales parameter count quadratically ($\approx 12 \cdot C^2$ per block).
- **Memory & Parameter Cost:** 
  - Quadratic parameter scaling: One block holds $\approx 12 \cdot C^2$ parameters (Attention $\approx 4C^2$, Feed-Forward $\approx 8C^2$).
  - For $C = 128$, one block holds $\approx 197\text{K}$ parameters. For $C = 256$, one block holds $\approx 786\text{K}$ parameters.

### 5. `block_size` (int)
- **What it is:** The maximum sequence length ($T$) or context window the model can process in one forward pass. Position embeddings are defined in [`07_EMBEDDINGS.md`](07_EMBEDDINGS.md).
- **Trade-off:** Longer context allows capturing long-range dependencies, but attention memory explodes quadratically $O(T^2)$.
- **Memory & Parameter Cost:** 
  - Position embedding parameters: $T \cdot C$ (e.g., $128 \cdot 128 = 16,384$ params for $T=128, C=128$).
  - Attention score tensor memory per layer (fp32): $B \cdot h \cdot T^2 \cdot 4\text{ bytes}$.
  - At $T=128, B=32, h=4, L=4$, total attention memory $\approx 32\text{ MB}$. At $T=1024, B=64$, attention memory reaches gigabytes and exceeds 8 GB limits.

### 6. `dropout` (float)
- **What it is:** The probability of zeroing activations during training for regularization ([`05_NEURAL_NETWORKS.md`](05_NEURAL_NETWORKS.md)).
- **Trade-off:** Prevents overfitting on small datasets like Shakespeare or TinyStories. Set to `0.0` during evaluation/inference.
- **Memory & Parameter Cost:** 0 parameter cost. Minimal activation memory cost for storing binary dropout masks during forward pass.

### 7. `weight_tying` (bool)
- **What it is:** Toggles sharing weights between the input token embedding matrix and the final output LM head projection matrix ([`ADR-0003`](../research/design_decisions/ADR-0003-weight-tying-default.md)).
- **Trade-off:** `true` (v0.1 default) saves $V \cdot C$ parameters and memory without loss in model quality. `false` instantiates a standalone linear layer.
- **Memory & Parameter Cost:** Saves $V \cdot C \cdot 4\text{ bytes}$ for weights, plus $V \cdot C \cdot 4\text{ bytes}$ for gradients, and $2 \cdot V \cdot C \cdot 4\text{ bytes}$ for Adam moments.

### 8. `batch_size` (int)
- **What it is:** The number of independent sequence samples ($B$) processed in parallel per optimization step.
- **Trade-off:** Larger batches stabilize gradient estimates and maximize MPS GPU parallelism, but scale activation memory linearly ($O(B)$).
- **Memory & Parameter Cost:** 0 parameter cost. Linear scaling on all intermediate activation tensors ($B \cdot T \cdot C$) and attention matrices ($B \cdot h \cdot T^2 \cdot L$).

### 9. `lr` (float)
- **What it is:** The peak learning rate for the AdamW optimizer ([`11_TRAINING_PIPELINE.md`](11_TRAINING_PIPELINE.md)).
- **Trade-off:** High LR speeds up convergence but risks divergence/loss NaNs; low LR causes sluggish training.
- **Memory & Parameter Cost:** 0 parameter/memory cost.

### 10. `warmup_steps` (int)
- **What it is:** The number of initial steps over which learning rate increases linearly from 0 to `lr`.
- **Trade-off:** Prevents early gradient explosion while LayerNorm and attention projections stabilize.
- **Memory & Parameter Cost:** 0 parameter/memory cost.

### 11. `max_steps` (int)
- **What it is:** The total number of training iterations before termination.
- **Trade-off:** Longer training improves loss until overfitting occurs.
- **Memory & Parameter Cost:** 0 parameter/memory cost.

### 12. `weight_decay` (float)
- **What it is:** $L_2$ penalty applied to matrix weights (matmuls) to prevent parameter inflation. Excludes 1D biases and LayerNorm scales.
- **Trade-off:** Regularizes model; setting too high suppresses capacity.
- **Memory & Parameter Cost:** 0 parameter/memory cost.

### 13. `grad_clip` (float)
- **What it is:** Maximum threshold for clipping global gradient norm ($|g\|_2 \le \text{grad\_clip}$).
- **Trade-off:** Prevents catastrophic gradient spikes on MPS.
- **Memory & Parameter Cost:** 0 parameter cost. Requires one temporary float scalar for global norm.

### 14. `eval_interval` (int)
- **What it is:** Step frequency for running evaluation on held-out validation data and generating prompt samples.
- **Trade-off:** Frequent evaluation provides fine-grained loss tracking but adds validation runtime overhead.
- **Memory & Parameter Cost:** 0 parameter/memory cost.

### 15. `seed` (int)
- **What it is:** Deterministic random seed for PyTorch, NumPy, Python random, and MPS generators ([`ENGINEERING_PRINCIPLES.md`](../ENGINEERING_PRINCIPLES.md) Principle 4).
- **Trade-off:** Guarantees exact run reproducibility.
- **Memory & Parameter Cost:** 0 parameter/memory cost.

---

## Visual Diagrams

### Configuration Loading & Model Lifecycle

```
    configs/model_v0_1_char_100k.yaml
                  │
                  ▼
       ┌─────────────────────┐
       │ Config Parser &     │ ── validate schema & seed
       │ Memory Budget Check │
       └──────────┬──────────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
  ┌───────────┐       ┌───────────┐
  │ GPT Model │       │ Training  │
  │ Structure │       │  Harness  │
  └─────┬─────┘       └─────┬─────┘
        │                   │
        └─────────┬─────────┘
                  ▼
        [ MPS Device Memory ]
```

### Memory Footprint Breakdown on 8 GB Unified Memory

```
  Memory Allocation (fp32)
  ├── Fixed Footprint (~13 MB for v0.1)
  │   ├── Model Weights  (params × 4 B)
  │   ├── Gradients      (params × 4 B)
  │   └── Adam State     (2 × params × 4 B)
  └── Dynamic Footprint (Scales during forward/backward)
      ├── Activations    (B × T × C × L × 4 B)
      └── Attn Matrix    (B × h × T² × L × 4 B) ◄── Primary Ceiling
```

---

## Common Mistakes

- **Confusing `n_embd` and `n_head` constraints:** Failing to ensure `n_embd % n_head == 0`, which causes runtime tensor division crashes when splitting attention projections into heads.
- **Unbudgeted Context Explosion:** Increasing `block_size` $T$ from 128 to 1024 without accounting for the $O(T^2)$ memory growth, leading to sudden MPS allocations failure.
- **Applying Weight Decay to Biases:** Applying weight decay indiscriminately across all parameters instead of restricting it to 2D matmul weights, hurting LayerNorm performance.
- **Ignoring Seed Initialization:** Changing hyperparameter values across runs without locking `seed`, making it impossible to distinguish algorithmic gains from random initialization noise.

---

## Future Improvements

- **Pydantic Schema Validation:** Implement strict type and value range checking (`n_embd > 0`, `dropout <= 1.0`) during config parsing in `src/config.py`.
- **Automatic MPS Memory Estimator:** Compute expected peak memory in MB prior to allocating PyTorch tensors and warn if estimated memory exceeds available RAM.
- **Dynamic Config Diffing:** Automatically log hyperparameter diffs between baseline and experiment configs in experiment artifacts.

---

## Glossary

Canonical definitions live in [`19_GLOSSARY.md`](19_GLOSSARY.md). Key configuration terms:

| Term | Meaning |
|------|---------|
| `n_embd` | Hidden dimension width $C$ across transformer blocks |
| `block_size` | Maximum sequence context length $T$ |
| `weight_tying` | Sharing parameters between token embedding and output LM head |
| `grad_clip` | Global gradient norm threshold for preventing exploding gradients |
| `seed` | Integer initializer for pseudo-random number generators |

---

## Learning Checklist

You master model configuration when you can:

- [ ] Explain why configuration is kept separate from script implementation (Principle 8).
- [ ] State the parameter formula for a pre-norm block ($\approx 12 \cdot C^2$).
- [ ] Compute the memory cost of weights, gradients, and Adam state for a given parameter count.
- [ ] Derive why attention score memory grows as $O(B \cdot h \cdot T^2 \cdot L)$.
- [ ] Verify that `n_embd % n_head == 0` for any chosen model architecture.

---

## References

- [`SPEC.md` §5–6](../SPEC.md#5-parameter-count-worked-for-v01) — Parameter count arithmetic and 8 GB memory budget.
- [`ENGINEERING_PRINCIPLES.md`](../ENGINEERING_PRINCIPLES.md) — Principles 4, 5, and 8.
- Press & Wolf (2017), *Using the Output Embedding to Improve Language Models* — Weight tying foundation.
- Radford et al. (2019), *Language Models are Unsupervised Multitask Learners* — GPT-2 configuration parameters.

## Further Reading

- [`15_EXPERIMENT_GUIDE.md`](15_EXPERIMENT_GUIDE.md) — How configs are tied to hypothesis testing.
- [`13_OPTIMIZATION_FOR_APPLE_SILICON.md`](13_OPTIMIZATION_FOR_APPLE_SILICON.md) — Apple Silicon MPS memory optimization.
- [`configs/README.md`](../configs/README.md) — Config directory conventions and usage.

> **Next:** [`17_DATASET_GUIDE.md`](17_DATASET_GUIDE.md)
