# 05 — Neural Networks

> **Prerequisites:** [`04_MATHEMATICS.md`](04_MATHEMATICS.md) (vectors, dot products, matrix multiplication, derivatives, chain rule).
>
> **Next:** [`06_TOKENIZER.md`](06_TOKENIZER.md)

---

## Purpose

A transformer is a specialized neural network. Before constructing attention layers or multi-layer transformer blocks, you must understand how a basic neural network processes inputs, calculates errors, and updates its weights.

This document builds neural networks from the ground up:
1. **The Artificial Neuron** — the fundamental building block.
2. **The Multi-Layer Perceptron (MLP)** — stacking neurons into layers with non-linearities.
3. **The Loss Function** — quantifying how wrong predictions are.
4. **Backpropagation** — applying the chain rule to derive gradients for every weight.
5. **The Training Loop** — the iterative cycle of forward, loss, backward, and step.

This document is the **canonical home** in DevLLM for the **overfit-a-batch** debugging technique—the single most effective rule for catching structural bugs in deep learning implementations.

---

## Background

### From single linear functions to universal approximators

A single matrix multiplication $y = x W + b$ can only represent linear relationships (straight lines, flat planes). Language is highly non-linear. By combining linear transformations with non-linear activation functions (like ReLU or GELU), multi-layer neural networks become **universal function approximators** capable of learning complex linguistic relationships.

### The core learning cycle

Every neural network learns via the exact same four-step loop:

```
   1. FORWARD PASS    ──► Compute predictions for input data
   2. LOSS COMPUTATION ──► Calculate single scalar error number
   3. BACKWARD PASS   ──► Compute gradients (dL/dw) via chain rule
   4. OPTIMIZER STEP  ──► Update weights: w = w - lr * grad
```

---

## Concepts

Core neural network building blocks:

- **Neuron:** Takes input vector $x$, multiplies by weight vector $w$, adds bias scalar $b$, and passes result through non-linearity $\sigma$.
- **Activation Function:** Non-linear transformation (e.g. ReLU, GELU, Softmax) enabling networks to learn non-linear functions.
- **Multi-Layer Perceptron (MLP):** A feed-forward network consisting of an input layer, one or more hidden layers, and an output layer.
- **Loss Function:** A function measuring error between predictions and target labels (e.g. Cross-Entropy loss).
- **Backpropagation:** Efficient algorithm for computing gradients of loss with respect to all parameters using the chain rule ([`04_MATHEMATICS.md`](04_MATHEMATICS.md)).
- **Overfit-a-Batch:** Debugging procedure where a model is trained on a tiny fixed batch of data (e.g., 2–10 samples) until loss reaches near-zero, proving that gradient flow and optimization work.

---

## Detailed Explanation

### 1. The Artificial Neuron

A single neuron computes a weighted sum of inputs plus a bias, then applies an activation function:

$$y = \sigma(w \cdot x + b) = \sigma\left(\sum_{i=1}^{C} w_i x_i + b\right)$$

- $x$: Input vector $[x_1, x_2, \dots, x_C]$.
- $w$: Learned weight vector $[w_1, w_2, \dots, w_C]$.
- $b$: Learned scalar bias.
- $\sigma$: Non-linear activation function.

```
   x1 ──► [ × w1 ] ──┐
   x2 ──► [ × w2 ] ──┼──► ( Sum + b ) ──► [ Activation σ ] ──► Output y
   x3 ──► [ × w3 ] ──┘
```

### 2. Multi-Layer Perceptron (MLP)

When multiple neurons are grouped in parallel, they form a **layer**. Stacking layers creates an **MLP**.

#### Layer Computation
For an input tensor $X$ (shape $B \times C$), a linear layer with weight matrix $W$ (shape $C \times C_{\text{out}}$) and bias $b$ computes:

$$Z = X W + b \quad \text{shape: } (B \times C_{\text{out}})$$

Applying an activation function yields hidden representations:

$$H = \sigma(X W + b)$$

#### Why Non-Linearities are Mandatory
If we stack two linear layers without non-linearities:

$$H = X W_1, \quad Y = H W_2 = (X W_1) W_2 = X (W_1 W_2) = X W_{\text{combined}}$$

Two matrix multiplications collapse into a single matrix multiplication $W_{\text{combined}}$. Without non-linearities, a 100-layer network has no more capacity than a 1-layer network.

#### Activations: ReLU vs GELU
- **ReLU (Rectified Linear Unit):** $\text{ReLU}(x) = \max(0, x)$. Simple and fast, but gradient is zero for $x < 0$ ("dying ReLU").
- **GELU (Gaussian Error Linear Unit):** Used in modern GPTs ([`09_TRANSFORMER.md`](09_TRANSFORMER.md)). A smooth approximation: $\text{GELU}(x) \approx 0.5 x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} (x + 0.044715 x^3)\right)\right)$. Smooth non-zero gradients for small negative values.

---

### 3. The Loss Function (Cross-Entropy)

For next-token prediction, the model outputs logit scores $z$ over vocabulary $V$. Passing logits through softmax ([`04_MATHEMATICS.md`](04_MATHEMATICS.md)) yields probability distribution $p$.

If the true target token index is $y^*$, **Cross-Entropy Loss** measures the negative log probability assigned to the correct target token:

$$\text{Loss} = -\log p_{y^*}$$

- If the model assigns $p_{y^*} = 1.0$ (100% confidence to target), $\text{Loss} = -\log(1.0) = 0.0$.
- If the model assigns $p_{y^*} = 0.01$ (low confidence), $\text{Loss} = -\log(0.01) \approx 4.60$.
- Average loss across a batch of $B \times T$ tokens:

$$\text{Loss}_{\text{batch}} = -\frac{1}{B \cdot T} \sum_{b=1}^{B} \sum_{t=1}^{T} \log p_{b, t, y^*_{b,t}}$$

---

### 4. Backpropagation

Backpropagation calculates $\frac{\partial \text{Loss}}{\partial W}$ for every weight matrix $W$ by traversing the network graph in reverse.

#### Step-by-Step Chain Rule Example
Consider a 1-layer network with scalar operations:

$$z = w x + b, \quad a = \text{ReLU}(z), \quad L = (a - y^*)^2$$

1. **Forward pass:** Compute $z$, $a$, and $L$.
2. **Backward pass:**
   - Derivative of loss w.r.t activation $a$: $\frac{\partial L}{\partial a} = 2(a - y^*)$.
   - Derivative of activation w.r.t $z$: $\frac{\partial a}{\partial z} = 1$ if $z > 0$ else $0$.
   - By chain rule: $\frac{\partial L}{\partial z} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z}$.
   - Derivative w.r.t weight $w$: $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial w} = \frac{\partial L}{\partial z} \cdot x$.

PyTorch's `autograd` engine automates this backward chain rule for arbitrary tensor operations.

---

### 5. Overfit-a-Batch — Canonical Debugging Technique

> **The Golden Rule of ML Debugging:** Before running a multi-hour training run on a full dataset, take a tiny batch of data (e.g., 2–4 sentences) and train the model on that exact same batch repeatedly.

#### Why Overfit-a-Batch works
A correctly implemented model with sufficient capacity has enough parameters to memorize a tiny dataset completely. 

```
   Training Step 0     ──► Loss ≈ 4.18  (random guess over vocab)
   Training Step 100   ──► Loss ≈ 0.50
   Training Step 500   ──► Loss ≈ 0.001 (perfect memorization)
```

#### Diagnostic Value
- If loss drops to $\approx 0.0$: Your model architecture, forward pass, loss calculation, backprop, and optimizer step are **structurally sound**.
- If loss remains flat or explodes to `NaN`: You have a fundamental bug (e.g., disconnected autograd graph, inverted mask, zeroed gradients, wrong learning rate).

---

## Visual Diagrams

### Forward and Backward Pass Architecture

```
   INPUT X ──► [ Layer 1: X·W1+b1 ] ──► [ GELU ] ──► [ Layer 2: H·W2+b2 ]
                                                              │
                                                              ▼
                                                        [ Softmax ]
                                                              │
                                                              ▼
   GRADIENTS ◄── [ Backprop ] ◄── [ Cross-Entropy Loss L ] ◄── Logits
   (dL/dW1, dL/dW2)
```

### Overfit-a-Batch Loss Trajectory

```
   Loss
   4.0 ┼───┐
       │   └───┐
   2.0 │       └───┐
       │           └───┐
   0.0 ┼───────────────┴────────► Steps
       0      100     500
       [ Healthy model memorizes tiny batch to ~0 loss ]
```

---

## Common Mistakes

- **Forgetting `optimizer.zero_grad()`:** PyTorch accumulates gradients by default. Omitting `zero_grad()` adds new gradients to previous steps, causing immediate gradient explosion.
- **Skipping Non-Linearities:** Stacking linear layers without GELU/ReLU, reducing deep network capacity to a single linear matrix.
- **Evaluating in Training Mode:** Leaving `dropout` active during evaluation, causing artificial noise in validation metrics.
- **Not Testing Overfit-a-Batch First:** Launching long runs on large datasets before proving the pipeline can memorize a single batch.

---

## Future Improvements

- Add a scalar autograd engine toy (micrograd-style) under `scripts/autograd_toy.py` to demonstrate manual backprop code.
- Implement explicit gradient-checking unit tests in `tests/`.

---

## Glossary

Canonical definitions live in [`19_GLOSSARY.md`](19_GLOSSARY.md). Core neural network terms:

| Term | Definition |
|------|------------|
| MLP | Multi-Layer Perceptron; feed-forward stacked linear layers + non-linearities |
| Activation | Non-linear function applied to layer outputs (e.g. GELU, ReLU) |
| Cross-Entropy | Loss function measuring distance between predicted probabilities and target labels |
| Backpropagation | Reverse-mode automatic differentiation algorithm calculating gradients |
| Overfit-a-Batch | Verification method testing if model can memorize 2–10 samples to ~0 loss |

---

## Learning Checklist

You master neural network foundations when you can:

- [ ] Explain why non-linear activation functions are necessary between linear layers.
- [ ] Differentiate between ReLU and GELU activation functions.
- [ ] Trace a forward pass through a 2-layer MLP with shapes.
- [ ] Explain how the chain rule enables backpropagation.
- [ ] Execute an overfit-a-batch test and state what near-zero loss proves.

---

## References

- Karpathy, *Neural Networks: Zero to Hero* — Building micrograd & MLP from scratch.
- [`04_MATHEMATICS.md`](04_MATHEMATICS.md) — Matrix multiplication and softmax definitions.
- [`ENGINEERING_PRINCIPLES.md`](../ENGINEERING_PRINCIPLES.md) — Principle 6 (Testing & Shape Verification).

## Further Reading

- [`06_TOKENIZER.md`](06_TOKENIZER.md) — **Next:** Converting text into discrete token IDs.
- [`09_TRANSFORMER.md`](09_TRANSFORMER.md) — How the FFN sublayer uses an MLP inside every transformer block.

> **Next:** [`06_TOKENIZER.md`](06_TOKENIZER.md)
