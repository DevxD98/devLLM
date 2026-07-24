# 11 — Training Pipeline

> **Prerequisites:** [`05_NEURAL_NETWORKS.md`](05_NEURAL_NETWORKS.md) (backpropagation and training loop) and [`10_GPT_ARCHITECTURE.md`](10_GPT_ARCHITECTURE.md) (GPT forward pass).
>
> **Next:** [`12_INFERENCE.md`](12_INFERENCE.md)

---

## Purpose

Having constructed the complete GPT architecture ([`10_GPT_ARCHITECTURE.md`](10_GPT_ARCHITECTURE.md)), we must train its parameters. A training pipeline transforms a raw sequence of token IDs into a optimized language model through an iterative optimization loop.

This document details the mechanics of JimmyLabs's training pipeline:
1. **Cross-Entropy Loss Computation** on autoregressive target tokens.
2. **AdamW Optimizer Mechanics** — momentum, weight decay exclusion, and state tracking.
3. **Learning Rate Scheduling** — linear warmup followed by cosine decay.
4. **Gradient Clipping** — protecting stability on Apple Silicon MPS hardware.
5. **Checkpointing & Evaluation Contracts** — saving model state and measuring validation loss.

---

## Background

### Autoregressive Next-Token Supervision

Training a decoder-only GPT uses self-supervised learning: the dataset itself provides labels. For an input sequence of tokens $[x_0, x_1, x_2, \dots, x_T]$, the model predicts the shifted sequence $[x_1, x_2, x_3, \dots, x_{T+1}]$ as targets.

```
   INPUTS  (X):   "The",  "cat",  "sat",  "on"
   TARGETS (Y):   "cat",  "sat",  "on",   "the"
```

Every position in a batch of size $(B \times T)$ provides a training signal in a single parallel forward pass.

---

## Concepts

Core training pipeline components:

- **Shifted Cross-Entropy Loss:** Cross-entropy loss ([`05_NEURAL_NETWORKS.md`](05_NEURAL_NETWORKS.md)) computed by aligning input logits at position $t$ against target token ID at position $t+1$.
- **AdamW Optimizer:** Variant of Adam optimizer incorporating decoupled weight decay ($L_2$ penalty applied directly to parameter weights rather than gradients).
- **Linear Warmup + Cosine Decay:** Learning rate schedule that ramps LR linearly during early steps, then decays it following a cosine curve to a minimum floor.
- **Global Gradient Clipping:** Rescaling gradient vectors if their global $L_2$ norm exceeds threshold $g_{\text{max}}$ (typically $1.0$).
- **Checkpointing:** Periodically saving model weights, optimizer state, step count, and val loss to disk under `checkpoints/`.

---

## Detailed Explanation

### 1. Shifted Cross-Entropy Loss

Given logits of shape $(B, T, V)$ and target token IDs $Y$ of shape $(B, T)$:

1. Flatten logits tensor to shape $(B \cdot T, V)$.
2. Flatten target tensor $Y$ to shape $(B \cdot T)$.
3. Compute cross-entropy loss across all $B \cdot T$ predictions:

$$\text{Loss} = -\frac{1}{B \cdot T} \sum_{i=1}^{B \cdot T} \log \left(\frac{e^{\text{logits}[i, Y[i]]}}{\sum_{j=1}^{V} e^{\text{logits}[i, j]}}\right)$$

In PyTorch: `F.cross_entropy(logits.view(-1, V), targets.view(-1))`.

---

### 2. AdamW Optimizer Mechanics

AdamW maintains two running moments for every parameter $w$:
1. First moment $m_t$ (exponential moving average of gradients):
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
2. Second moment $v_t$ (exponential moving average of squared gradients):
   $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

Standard hyperparameter defaults: $\beta_1 = 0.9$, $\beta_2 = 0.95$, $\epsilon = 10^{-8}$.

#### Weight Decay Exclusion Rule
Weight decay ($L_2$ regularization) must be applied **only** to 2D matrix multiplication weights ($W_q, W_k, W_v, W_o, W_1, W_2, W_{\text{token}}$). 1D biases and LayerNorm scale/shift parameters are explicitly excluded from weight decay to prevent degradation of normalization dynamics.

#### Memory Impact
As derived in [`SPEC.md` §6](../SPEC.md#6-memory-usage-fp32-on-8-gb), AdamW stores $m_t$ and $v_t$ for every parameter, requiring $2 \times N_{\text{params}} \times 4\text{ bytes}$ of memory.

---

### 3. Learning Rate Schedule (Warmup + Cosine Decay)

The learning rate $\eta(t)$ varies over step $t \in [0, T_{\text{max}}]$:

```
   LR
   η_max ┼         ┌───'''''───┐
         │       ╱              ╲
         │     ╱                  ╲  Cosine Decay
         │   ╱                      ╲
   η_min ┼─┴──────────────────────────┴─────► Step t
         0   Warmup                   T_max
```

#### Schedule Formula
1. **Warmup Phase ($t < T_{\text{warmup}}$):**
   $$\eta(t) = \eta_{\text{max}} \cdot \frac{t}{T_{\text{warmup}}}$$

2. **Cosine Decay Phase ($T_{\text{warmup}} \le t \le T_{\text{max}}$):**
   $$\eta(t) = \eta_{\text{min}} + 0.5 (\eta_{\text{max}} - \eta_{\text{min}}) \left(1 + \cos\left(\pi \frac{t - T_{\text{warmup}}}{T_{\text{max}} - T_{\text{warmup}}}\right)\right)$$

where $\eta_{\text{min}} = 0.1 \cdot \eta_{\text{max}}$.

---

### 4. Global Gradient Clipping

To prevent exploding gradients from destabilizing training on Apple Silicon MPS hardware:

1. Compute global $L_2$ norm across all model parameter gradients:
   $$\|g\|_2 = \sqrt{\sum_{p} \|\nabla_p L\|_2^2}$$

2. If $\|g\|_2 > g_{\text{max}}$ (where $g_{\text{max}} = 1.0$), rescale all gradients:
   $$\nabla_p L \leftarrow \nabla_p L \cdot \frac{g_{\text{max}}}{\max(\|g\|_2, g_{\text{max}})}$$

In PyTorch: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`.

---

### 5. Training Loop Step Sequence

Every iteration follows this execution sequence:

```
   1. Sample Batch (X, Y)  ──► Load (B, T) token IDs on MPS device
   2. Zero Gradients      ──► optimizer.zero_grad(set_to_none=True)
   3. Forward Pass        ──► logits = model(X)
   4. Loss Calculation    ──► loss = F.cross_entropy(logits, Y)
   5. Backward Pass       ──► loss.backward()
   6. Gradient Clip       ──► clip_grad_norm_(model, 1.0)
   7. Schedule LR Step    ──► update lr per warmup/cosine schedule
   8. Optimizer Step      ──► optimizer.step()
```

---

## Visual Diagrams

### Complete Training Pipeline Loop

```
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. Data Sampler (B, T) ──► Input X (B,T), Targets Y (B,T)  │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 2. Forward Pass        ──► Logits (B, T, V)                 │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 3. Loss Calculation    ──► Shifted Cross-Entropy Loss       │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 4. Backward Pass       ──► PyTorch Autograd (dL/dw)         │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 5. Grad Clip & AdamW   ──► Clip norm <= 1.0; update weights │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 6. Eval & Checkpoint   ──► Log val loss & save checkpoint   │
  └─────────────────────────────────────────────────────────────┘
```

---

## Common Mistakes

- **Applying Weight Decay to Biases or LayerNorm:** Applying $L_2$ decay to 1D biases or scale parameters, degrading layer normalization performance.
- **Forgetting `set_to_none=True`:** Calling `zero_grad()` without setting gradients to `None`, wasting memory allocations on Apple Silicon MPS.
- **Unclipped Gradient Spikes:** Omitting `clip_grad_norm_`, leading to sudden `NaN` loss during training spikes.
- **Evaluating with Training Dropout:** Forgetting `model.eval()` during validation runs, resulting in noisy, inaccurate validation metrics.

---

## Future Improvements

- Implement Gradient Accumulation to simulate large effective batch sizes without increasing peak activation memory ([`research/OPTIMIZATION_BACKLOG.md` #3](../research/OPTIMIZATION_BACKLOG.md#3)).
- Add mixed precision (fp16/bf16) training support gated by benchmarks on MPS.

---

## Glossary

Canonical definitions live in [`19_GLOSSARY.md`](19_GLOSSARY.md). Training terms:

| Term | Definition |
|------|------------|
| Shifted Cross-Entropy | Loss computed by matching position $t$ logits against target $t+1$ |
| AdamW | Optimizer with decoupled weight decay applied directly to matmul parameters |
| Cosine Decay | Learning rate decay following a smooth cosine curve down to $\eta_{\text{min}}$ |
| Gradient Clipping | Rescaling global gradient norm to prevent numeric explosion |
| Checkpoint | Saved model file containing weights, optimizer state, step, and val loss |

---

## Learning Checklist

You master the training pipeline when you can:

- [ ] Explain how target tokens are shifted relative to input tokens.
- [ ] Compute memory required for AdamW optimizer states given parameter count.
- [ ] State why weight decay is excluded from biases and LayerNorm parameters.
- [ ] Write the mathematical formula for cosine learning rate decay with warmup.
- [ ] Explain how gradient clipping protects MPS execution stability.

---

## References

- [`SPEC.md` §7](../SPEC.md#7-training-pipeline) — Technical training pipeline specifications.
- Loshchilov & Hutter (2019), *Decoupled Weight Decay Regularization* (AdamW).
- [`16_MODEL_CONFIGURATION.md`](16_MODEL_CONFIGURATION.md) — Configuration hyperparameters for training.

## Further Reading

- [`12_INFERENCE.md`](12_INFERENCE.md) — **Next:** Generating text from trained model checkpoints.
- [`architecture/training_pipeline.md`](../architecture/training_pipeline.md) — Concrete data flow and module boundaries for training in JimmyLabs.

> **Next:** [`12_INFERENCE.md`](12_INFERENCE.md)
