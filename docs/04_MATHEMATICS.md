# 04 — Mathematics for Transformers

> **Prerequisites:** [`03_LEARNING_PATH.md`](03_LEARNING_PATH.md) (orientation and mental models). Basic high-school algebra (variables, functions, graphs). *No* calculus or linear algebra pre-requisites assumed.
>
> **Next:** [`05_NEURAL_NETWORKS.md`](05_NEURAL_NETWORKS.md)

---

## Purpose

Machine learning literature is filled with mathematical notation that often conceals simple geometric and arithmetic ideas. You do not need to master all of university linear algebra or multivariable calculus to build a GPT. You only need to master **five mathematical operations**:

1. **Vectors and Dot Products** — how similarity is measured.
2. **Matrices and Matrix Multiplication** — how an entire neural network layer transforms data at once.
3. **Derivatives and the Chain Rule** — how we calculate direction to fix errors.
4. **Softmax** — how raw scores are turned into probability distributions.

This document is the **canonical home** in DevLLM for matrix multiplication and the softmax function. Other documents ([`05_NEURAL_NETWORKS.md`](05_NEURAL_NETWORKS.md), [`08_ATTENTION.md`](08_ATTENTION.md)) use these operations, but this document explains *why* they work, *how* to compute them, and *what* their tensor shapes mean.

---

## Background

### Why math first?

If you try to read about self-attention ([`08_ATTENTION.md`](08_ATTENTION.md)) without a concrete grasp of dot products and matrix multiplication, attention will seem like magic. When you realize that query-key matching is just dot products, and multi-head attention is just matrix multiplication, attention stops being magic and becomes mechanical.

### The geometric intuition

In deep learning, every piece of text is mapped to a point in a high-dimensional space (a vector). Learning is the process of moving those points around until geometric closeness corresponds to semantic similarity.

```
   "king"  • ──────► vector space
             ╲
              ╲  dot product measures angle/similarity
               ▼
   "queen" •
```

---

## Concepts

Here are the primary mathematical building blocks:

- **Scalar:** A single number (e.g., $x = 3.14$). In PyTorch, a 0-D tensor.
- **Vector:** An ordered 1-D list of numbers representing a direction/point in space (e.g., $v = [1.0, -2.5, 0.5]$ of length $C$).
- **Matrix:** A 2-D grid of numbers with rows and columns (e.g., shape $R \times C$).
- **Tensor:** An $N$-dimensional array of numbers. Scalars (0-D), vectors (1-D), and matrices (2-D) are all tensors.
- **Dot Product:** An operation taking two equal-length vectors and producing a single scalar measuring their directional alignment.
- **Matrix Multiplication (Matmul):** An operation combining two matrices by taking dot products of rows and columns.
- **Derivative:** The rate of change of a function with respect to an input; tells us which direction to tweak a weight to decrease loss.
- **Softmax:** A function converting an array of unnormalized numbers (logits) into positive probabilities that sum to 1.0.

---

## Detailed Explanation

### 1. Vectors and Dot Products

A vector $u$ of dimension $C$ is a list of $C$ numbers: $u = [u_1, u_2, \dots, u_C]$.

The **dot product** of two vectors $u$ and $v$ of the same length $C$ is the sum of their element-wise products:

$$u \cdot v = \sum_{i=1}^{C} u_i v_i = u_1 v_1 + u_2 v_2 + \dots + u_C v_C$$

#### Geometric Meaning
The dot product can also be written as:

$$u \cdot v = \|u\| \|v\| \cos(\theta)$$

where $\|u\|$ is the length (magnitude) of vector $u$, and $\theta$ is the angle between them:
- If $u$ and $v$ point in the **same direction**, $\cos(\theta) = 1$, so $u \cdot v$ is large and positive.
- If $u$ and $v$ are **perpendicular (orthogonal)**, $\cos(\theta) = 0$, so $u \cdot v = 0$.
- If $u$ and $v$ point in **opposite directions**, $\cos(\theta) = -1$, so $u \cdot v$ is negative.

> **Key takeaway:** The dot product is a universal engine for measuring **relevance or similarity**. Higher dot product = greater alignment.

---

### 2. Matrix Multiplication (Matmul) — Canonical Definition

Matrix multiplication is the workhorse of deep learning. In DevLLM, over 99% of floating-point operations (FLOPs) happen inside matrix multiplications.

#### The Rule of Shapes
To multiply matrix $A$ (shape $M \times K$) by matrix $B$ (shape $K \times N$), the **inner dimensions must match** ($K$). The resulting matrix $C = A B$ has shape $M \times N$:

```
   (M × K)  ×  (K × N)  ──►  (M × N)
        ▲         ▲
        └────┬────┘
      must match!
```

#### How it is Computed
Entry $C_{i,j}$ (row $i$, column $j$ of the output) is the **dot product** of row $i$ of $A$ and column $j$ of $B$:

$$C_{i,j} = \sum_{k=1}^{K} A_{i,k} B_{k,j}$$

#### Worked Example
Let $A$ be a $2 \times 3$ matrix and $B$ be a $3 \times 2$ matrix:

$$A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}, \quad B = \begin{bmatrix} 7 & 8 \\ 9 & 1 \\ 2 & 3 \end{bmatrix}$$

Calculate $C = A B$ (shape $2 \times 2$):

- $C_{1,1} = (1 \cdot 7) + (2 \cdot 9) + (3 \cdot 2) = 7 + 18 + 6 = 31$
- $C_{1,2} = (1 \cdot 8) + (2 \cdot 1) + (3 \cdot 3) = 8 + 2 + 9 = 19$
- $C_{2,1} = (4 \cdot 7) + (5 \cdot 9) + (6 \cdot 2) = 28 + 45 + 12 = 85$
- $C_{2,2} = (4 \cdot 8) + (5 \cdot 1) + (6 \cdot 3) = 32 + 5 + 18 = 55$

$$C = \begin{bmatrix} 31 & 19 \\ 85 & 55 \end{bmatrix}$$

#### Why Matmul powers Transformers
When processing a sequence of $T$ tokens, each represented by a vector of size $C$, we stack them into a matrix $X$ of shape $(T \times C)$. Multiplying $X$ by a weight matrix $W$ of shape $(C \times 4C)$ transforms all $T$ tokens simultaneously in one parallel operation.

---

### 3. Derivatives and the Chain Rule

To train a model, we need to know how to adjust a weight $w$ to reduce the loss $L$. The **derivative** $\frac{dL}{dw}$ tells us how much $L$ changes when $w$ changes by a tiny amount.

- If $\frac{dL}{dw} > 0$: Increasing $w$ increases loss. To decrease loss, **decrease** $w$.
- If $\frac{dL}{dw} < 0$: Increasing $w$ decreases loss. To decrease loss, **increase** $w$.

#### Gradient Descent Rule
We update weights by moving opposite to the gradient:

$$w_{\text{new}} = w_{\text{old}} - \eta \cdot \frac{dL}{dw}$$

where $\eta$ (eta) is the **learning rate**.

#### The Chain Rule
Neural networks stack functions: $y = f(x)$, $z = g(y)$. To find how $z$ changes with $x$, we multiply the intermediate derivatives:

$$\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$$

Backpropagation ([`05_NEURAL_NETWORKS.md`](05_NEURAL_NETWORKS.md)) is simply applying the chain rule backward through every layer in the network.

---

### 4. The Softmax Function — Canonical Definition

The **softmax function** takes a vector of arbitrary real numbers (called **logits**) and transforms them into a valid probability distribution.

#### Mathematical Definition
For a vector of logits $z = [z_1, z_2, \dots, z_V]$, the softmax output $p_i$ for index $i$ is:

$$p_i = \text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}$$

#### Properties of Softmax
1. **Positivity:** Because $e^x > 0$ for all real $x$, every $p_i > 0$.
2. **Normalized:** Sum of probabilities across all indices equals exactly $1.0$: $\sum_{i=1}^{V} p_i = 1.0$.
3. **Exponentiation amplifies differences:** Larger logits receive exponentially more probability mass than smaller logits.

#### Numerical Stability (The Softmax Trick)
Naively computing $e^{z_i}$ when $z_i = 100$ leads to floating-point overflow (`inf`). In practice, we subtract the maximum logit $m = \max(z)$ from all entries before exponentiating:

$$\text{softmax}(z)_i = \frac{e^{z_i - m}}{\sum_{j=1}^{V} e^{z_j - m}}$$

Because $\frac{e^{z_i - m}}{e^{z_j - m}} = \frac{e^{z_i} e^{-m}}{e^{z_j} e^{-m}} = \frac{e^{z_i}}{e^{z_j}}$, subtracting $m$ does not alter the output, but keeps the largest exponent at $e^0 = 1.0$, preventing overflow.

---

## Visual Diagrams

### Matrix Multiplication Shape Contract

```
      A (M × K)             B (K × N)             C (M × N)
   ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
   │             │       │             │       │             │
   │  Row i      │   ×   │   Col j     │   =   │   C[i,j]    │
   │  (K elements)       │  (K elements)       │  (scalar)   │
   └─────────────┘       └─────────────┘       └─────────────┘
```

### Softmax Transformation Flow

```
   Raw Logits z           Exponentiate e^z         Normalize (÷ sum)
   ┌──────────┐            ┌──────────┐             ┌──────────┐
   │   2.0    │  ──────►   │   7.39   │   ──────►   │   0.66   │ (66%)
   │   1.0    │  e^(z-m)   │   2.72   │  ÷ 11.16    │   0.24   │ (24%)
   │   0.0    │            │   1.00   │             │   0.10   │ (10%)
   └──────────┘            └──────────┘             └──────────┘
   Sum = 3.0               Sum = 11.16              Sum = 1.00 (100%)
```

---

## Common Mistakes

- **Mismatching inner matmul dimensions:** Attempting to multiply $(M \times K)$ by $(N \times K)$ without transposing the second matrix to $(K \times N)$.
- **Confusing element-wise multiplication with matrix multiplication:** $A * B$ (Hadamard product) requires identical shapes and multiplies corresponding entries. $A @ B$ is matrix multiplication and requires inner dimension agreement.
- **Forgetting the softmax numerical stability trick:** Computing $\exp(z)$ directly in floating-point without subtracting $\max(z)$, causing `NaN` or `inf` during forward passes.
- **Expecting dot products to be normalized:** A dot product $u \cdot v$ is not bounded between $-1$ and $+1$ unless $u$ and $v$ are unit-normalized vectors.
- **Applying softmax over the wrong dimension:** Softmax must run over the target axis (e.g. key dimension in attention, or vocabulary dimension in language model head).

---

## Future Improvements

- Add interactive visual tensor shape playground scripts under `scripts/`.
- Document hardware-accelerated matrix multiplication operations (e.g., Apple Silicon AMX/Metal matrix instructions).

---

## Glossary

Canonical definitions live in [`19_GLOSSARY.md`](19_GLOSSARY.md). Core mathematical terms:

| Term | Definition |
|------|------------|
| Dot Product | Sum of element-wise products of two equal-length vectors |
| Matmul ($@$) | Matrix multiplication combining rows and columns via dot products |
| Softmax | Function converting logits into a normalized probability distribution |
| Gradient | Vector of partial derivatives indicating direction of steepest loss increase |
| Chain Rule | Calculus formula for computing derivatives of composite functions |

---

## Learning Checklist

You master transformer mathematics when you can:

- [ ] Compute the dot product of two 3-element vectors by hand.
- [ ] Determine if matrix $A (4 \times 128)$ and $B (128 \times 512)$ can be multiplied, and state the resulting shape.
- [ ] Explain why dot products measure geometric similarity.
- [ ] Compute softmax for logits $[2.0, 1.0, 0.0]$ and verify the sum equals $1.0$.
- [ ] Explain the numerical stability trick for softmax.

---

## References

- 3Blue1Brown, *Essence of Linear Algebra* — Intuitive visual linear algebra.
- Goodfellow, Bengio, Courville, *Deep Learning* (MIT Press) — Chapter 2: Linear Algebra & Chapter 3: Probability.
- [`SPEC.md`](../SPEC.md) — Technical parameter arithmetic using these matrix shape contracts.

## Further Reading

- [`05_NEURAL_NETWORKS.md`](05_NEURAL_NETWORKS.md) — **Next:** Building neurons, MLPs, and backprop using these operations.
- [`08_ATTENTION.md`](08_ATTENTION.md) — How scaled dot-product attention builds on dot products and softmax.

> **Next:** [`05_NEURAL_NETWORKS.md`](05_NEURAL_NETWORKS.md)
