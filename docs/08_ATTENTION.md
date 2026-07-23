# 08 — Attention

> **Prerequisites:** [`04_MATHEMATICS.md`](04_MATHEMATICS.md) (dot products, matrix
> multiplication, softmax) and [`07_EMBEDDINGS.md`](07_EMBEDDINGS.md) (tokens are now
> vectors). If the softmax and the dot product aren't comfortable, go back — this document
> is *made* of them.
>
> **Next:** [`09_TRANSFORMER.md`](09_TRANSFORMER.md)

---

## Purpose

Attention is the mechanism that lets a language model **use context**: it lets the vector
for one token be rewritten as a blend of the tokens around it, weighted by relevance. It is
the single idea that separated transformers from everything before them, and it is the
conceptual center of this entire project. When attention clicks, the rest of a GPT is
assembly work.

This document builds attention from the ground up — the problem it solves, the query/key/
value idea, the exact math with a worked numerical example, causal masking, and multi-head
attention — and does it **why-first**, so you could rederive the formula rather than
memorize it.

---

## Background

### The problem attention solves

After [`07_EMBEDDINGS.md`](07_EMBEDDINGS.md), each token is a vector. But those vectors are
**context-free**: the vector for "bank" is identical in "river bank" and "savings bank."
Language is not context-free — meaning comes from how words relate. We need a layer that
lets each position **look at other positions and pull in information from them**.

Older approaches did this sequentially. A recurrent network (RNN) read left to right,
carrying a hidden state:

```
   the  ►  cat  ►  sat  ►  on  ►  the  ►  mat
    │       │       │      │      │       │
    h1  ►   h2  ►   h3  ►  h4  ►  h5  ►   h6      (state passed step by step)
```

Two problems: it's inherently **sequential** (can't parallelize across time — painful on a
GPU/MPS), and information from far-back tokens gets **diluted** as it's squeezed through
many steps ("the long-range dependency problem"). Attention throws out the recurrence
entirely. Every position can look **directly** at every other position, in one parallel
operation:

```
   the  cat  sat  on  the  mat
    ▲    ▲    ▲    ▲    ▲    ▲
    └────┴────┴──┬─┴────┴────┘
              "sat" attends directly to every position at once
```

### The one-sentence idea

> **Attention = for each position, compute how relevant every other position is, then
> replace this position's vector with a relevance-weighted average of information from all
> positions.**

Everything below is making that sentence precise.

---

## Concepts

Attention borrows a metaphor from **information retrieval** (think of a database lookup).
Three vectors are derived from each token, each by multiplying the token's embedding by a
separate learned weight matrix:

- **Query (Q)** — "what am I looking for?" The question this position is asking.
- **Key (K)** — "what do I offer?" A label advertising what this position contains.
- **Value (V)** — "what do I actually hand over if you attend to me?" The content.

The analogy: you (a query) walk into a library. Each book has a spine label (a key). You
compare your need against every label, and from the best-matching books you take the
*contents* (the values), weighted by how well each matched.

```
   token embedding x
        │
        ├──► × Wq ──► q   (what I'm looking for)
        ├──► × Wk ──► k   (what I advertise)
        └──► × Wv ──► v   (what I give)
```

`Wq`, `Wk`, `Wv` are **learned** — the model discovers, through training, what makes a
useful question, label, and payload. This is the crucial point: attention has no hand-coded
notion of relevance. It *learns* one.

---

## Detailed Explanation

### Step 1 — Similarity via dot products

Relevance between a query and a key is measured by their **dot product**. Two vectors
pointing the same way give a large positive number; orthogonal vectors give ~0. So
`q · k` is a natural "how well does this key answer my query" score.

For a sequence of `T` tokens, we compute the score of every query against every key. Stack
the queries into a matrix `Q` (shape `T × d_k`) and keys into `K` (shape `T × d_k`); then

```
   scores = Q Kᵀ            shape (T × T)
```

Entry `scores[i][j]` = "how much should token *i* attend to token *j*." One matrix multiply
produces the entire `T × T` grid of pairwise relevances — this is why attention is so
GPU/MPS-friendly.

### Step 2 — Scaling

Raw dot products grow with the dimension `d_k`: add up `d_k` product terms and the variance
grows like `d_k`. Large values push the softmax (next step) into a region where it becomes
extremely peaked and its gradients vanish. The fix, from the original paper, is to divide by
`√d_k`:

```
   scaled = (Q Kᵀ) / √d_k
```

This keeps the scores' variance roughly constant regardless of head size, so the softmax
stays in a healthy, trainable range. It is a small detail with an outsized effect on whether
training works at all.

### Step 3 — Softmax into weights

Each **row** of `scaled` is turned into a probability distribution with a softmax, so the
weights for token *i* across all tokens *j* are non-negative and sum to 1:

```
   weights[i] = softmax(scaled[i])       each row sums to 1.0
```

Now `weights` is an `T × T` matrix where row *i* says "here is how I, token *i*, divide my
attention across all tokens."

### Step 4 — Weighted sum of values

Finally, mix the values by those weights:

```
   output = weights · V        shape (T × d_v)
```

Row *i* of the output is the relevance-weighted average of every token's value vector — the
new, **context-aware** representation of token *i*.

### The whole thing in one line

Putting the four steps together gives the famous formula — **scaled dot-product attention**:

```
                    ⎛  Q Kᵀ  ⎞
   Attention(Q,K,V) = softmax⎜ ─────── ⎟ V
                    ⎝  √d_k  ⎠
```

You did not need to memorize this. You *derived* it: similarity (QKᵀ) → stabilize (÷√d_k) →
normalize (softmax) → gather (·V).

### A tiny worked example

Let `T = 3` tokens, `d_k = d_v = 2`. Suppose after the projections:

```
   Q = [[1, 0],      K = [[1, 0],      V = [[10, 0],
        [0, 1],           [1, 1],           [ 0,10],
        [1, 1]]           [0, 1]]           [ 5, 5]]
```

Take token 0's query `q0 = [1,0]`. Scores against each key:

```
   q0·k0 = 1·1 + 0·0 = 1
   q0·k1 = 1·1 + 0·1 = 1
   q0·k2 = 1·0 + 0·1 = 0
```

Scale by √d_k = √2 ≈ 1.414 → `[0.71, 0.71, 0]`. Softmax → `[0.40, 0.40, 0.20]`
(the two equal, larger scores share most of the weight). Output for token 0:

```
   0.40·[10,0] + 0.40·[0,10] + 0.20·[5,5]
   = [4.0, 0.0] + [0.0, 4.0] + [1.0, 1.0] = [5.0, 5.0]
```

Token 0's new vector `[5,5]` is a blend dominated by tokens 0 and 1. That blend — computed
from learned queries and keys — *is* attention. Change `Wq/Wk` and the blend changes; that's
what training tunes.

### Causal masking — the rule that makes it a *language* model

A GPT predicts the **next** token. When producing the token at position *i*, it must not be
allowed to look at positions *i+1, i+2, …* — those are the future, the answer. If it could,
training would be trivial and generation impossible (at generation time the future doesn't
exist yet).

We enforce this with a **causal mask**: before the softmax, set every score where `j > i`
to −∞, so softmax gives those positions weight 0.

```
   allowed?  (row = query i, col = key j)

          j=0   j=1   j=2   j=3
   i=0 [  ✓    -∞    -∞    -∞  ]
   i=1 [  ✓    ✓     -∞    -∞  ]     lower triangle kept,
   i=2 [  ✓    ✓     ✓     -∞  ]     upper triangle masked to -∞
   i=3 [  ✓    ✓     ✓     ✓   ]
```

After softmax, each position attends only to itself and the past. This single masking step
is the entire difference between "attention" and "*causal* self-attention," and it is what
makes autoregressive generation ([`12_INFERENCE.md`](12_INFERENCE.md)) possible.

> **Self-attention** means Q, K, and V all come from the *same* sequence (the tokens attend
> to each other). That's what a GPT uses. (Cross-attention, where queries come from one
> sequence and keys/values from another, is used in encoder–decoder models — not here.)

### Multi-head attention — many questions at once

One attention operation forces every position to ask a *single* question. But "what should
I attend to?" has many simultaneously-useful answers: the previous word, the subject of the
sentence, the matching bracket, the topic. **Multi-head attention** runs several attention
operations in parallel, each with its own `Wq/Wk/Wv`, so each **head** can specialize.

```
   x ──┬─► head 1 (Wq1,Wk1,Wv1) ─► out1 ┐
       ├─► head 2 (Wq2,Wk2,Wv2) ─► out2 ├─ concat ─► × Wo ─► output
       ├─► head 3 ...            ─► out3 │
       └─► head h ...            ─► outh ┘
```

Mechanically: split the embedding dimension `C` across `h` heads (each head works in
`d_k = C/h` dimensions — so multi-head is *not* more expensive than single-head of the same
total width), run scaled dot-product attention in each, **concatenate** the outputs back to
width `C`, and mix them with a final learned projection `Wo`. Different heads reliably learn
different relationships — inspecting them is one of the most illuminating things you can do
in this project (see [`15_EXPERIMENT_GUIDE.md`](15_EXPERIMENT_GUIDE.md)).

### Complexity — and why it matters on 8 GB

The `Q Kᵀ` matrix is `T × T`. So attention's compute and memory scale as **O(T²)** in the
sequence length `T`. Double the context, quadruple the attention cost. On a MacBook Air M1
with 8 GB of unified memory this is a *hard* constraint: it's why DevLLM keeps `block_size`
(context length) modest, and why the `T × T` attention matrix — not the parameters — is
often the thing that decides your maximum batch size. See
[`architecture/memory_layout.md`](../architecture/memory_layout.md) and
[`13_OPTIMIZATION_FOR_APPLE_SILICON.md`](13_OPTIMIZATION_FOR_APPLE_SILICON.md); the O(T²)
memory term is exactly what FlashAttention ([`18_RESEARCH_PAPERS.md`](18_RESEARCH_PAPERS.md))
attacks.

---

## Visual Diagrams

### The full data flow of one attention head

```
   input x            (T tokens, each a C-dim vector)      shape (T, C)
     │
     ├──────────────► × Wq ──► Q   (T, d_k)
     ├──────────────► × Wk ──► K   (T, d_k)
     └──────────────► × Wv ──► V   (T, d_v)
                          │
              Q Kᵀ  ┌─────┴─────┐
        (T,d_k)(d_k,T)          ▼
                        scores  (T, T)   ── pairwise relevance
                          │
                     ÷ √d_k              ── keep variance sane
                          │
                   + causal mask (-∞ above diagonal)   ── no peeking ahead
                          │
                     softmax (per row)  ── weights sum to 1
                          │
                    weights (T, T)
                          │
                     weights · V
                          ▼
                     output  (T, d_v)   ── context-aware vectors
```

### Where attention sits in a transformer block (preview of doc 09)

```
   x ─►(+)─────────────────────────────────► ... to feed-forward
        ▲
        │   residual
   x ─► LayerNorm ─► MULTI-HEAD ATTENTION ───┘
                     (this document)
```

Attention never acts alone: it's wrapped in a LayerNorm and a residual connection. That
wrapping is the subject of [`09_TRANSFORMER.md`](09_TRANSFORMER.md).

### Batched shapes (what the code really sees)

```
   real tensors carry a batch dimension B and a head dimension h:

   x        (B, T, C)
   Q,K,V    (B, h, T, C/h)          ← split into heads
   scores   (B, h, T, T)            ← the O(T²) tensor to watch on 8 GB
   output   (B, h, T, C/h) ─concat─► (B, T, C)
```

---

## Common Mistakes

- **Forgetting the causal mask** (or masking the wrong triangle). The model appears to
  train beautifully — loss drops fast — because it's cheating by seeing the future, then
  generates garbage. A classic, silent bug. Always test that the mask zeroes the *upper*
  triangle.
- **Dropping the √d_k scaling.** Training becomes unstable or stalls as the softmax
  saturates. Easy to omit, hard to diagnose.
- **Softmax over the wrong axis.** The softmax is over **keys** (the last dimension, each
  query's row). Softmax over queries silently computes nonsense that still has the right
  shape.
- **Confusing d_k and C.** With `h` heads, each head works in `d_k = C/h` dimensions, but Q
  and K feed the same head so they share `d_k`; V can differ but usually matches. Mixing
  these up gives shape errors — or worse, no error and wrong results.
- **Thinking multi-head is more compute than single-head.** It isn't: you split the same C
  across heads. Same FLOPs, more expressive.
- **Assuming attention "understands" language.** It computes weighted averages of learned
  vectors. The understanding is emergent, distributed across weights and layers — there is
  no rule in there that says "verbs attend to subjects." Training discovers it, or doesn't.

---

## Future Improvements

- **KV cache** for inference: at generation time, keys and values for past tokens don't
  change, so recomputing them every step is waste. Caching them turns per-step work from
  O(T²) back toward O(T). Covered in [`12_INFERENCE.md`](12_INFERENCE.md) and
  [`architecture/future_architecture.md`](../architecture/future_architecture.md).
- **Rotary Positional Embeddings (RoPE):** fold position into attention itself rather than
  adding it at the embedding stage — often better length generalization.
- **FlashAttention-style tiling:** never materialize the full `T × T` matrix; compute
  attention in blocks to cut memory from O(T²) to O(T). Directly relevant to the 8 GB
  budget.
- **Grouped-Query Attention (GQA):** share keys/values across query heads to shrink the KV
  cache — a memory win worth exploring on Apple Silicon.

---

## Glossary

Canonical definitions live in [`19_GLOSSARY.md`](19_GLOSSARY.md); the essentials here:

| Term | Meaning |
|------|---------|
| Query / Key / Value | the three learned projections of a token: its question, its label, its content |
| Scaled dot-product attention | `softmax(QKᵀ/√d_k)·V` — the core operation |
| Self-attention | attention where Q, K, V all come from the same sequence |
| Causal mask | forces each position to attend only to itself and earlier positions |
| Head | one independent attention operation; multi-head runs several in parallel |
| `d_k` | per-head key/query dimension, `= C / n_head` |
| O(T²) | attention's cost scales with the square of the sequence length |

---

## Learning Checklist

You understand attention when you can:

- [ ] Explain query, key, and value in the library-lookup metaphor **and** as matrix
      projections.
- [ ] Rederive `softmax(QKᵀ/√d_k)·V` from the four steps, without looking.
- [ ] Say why we divide by `√d_k`, and what breaks if we don't.
- [ ] State which axis the softmax runs over, and why.
- [ ] Draw the causal mask and explain why a GPT needs it.
- [ ] Explain what a head specializes in and why multi-head isn't extra compute.
- [ ] Explain why attention is O(T²) and what that means for batch size on 8 GB.

---

## References

- Vaswani et al., *Attention Is All You Need* (2017) — the origin; see the worked note in
  [`research/paper_notes/attention_is_all_you_need.md`](../research/paper_notes/attention_is_all_you_need.md).
- Radford et al., *Improving Language Understanding by Generative Pre-Training* (GPT-1) —
  the decoder-only, causal-masked configuration DevLLM uses.
- Bahdanau et al. (2014) — attention's earlier appearance in translation, for historical
  context.

## Further Reading

- Jay Alammar, *The Illustrated Transformer* — the best-known visual walkthrough; a great
  second pass after this doc.
- Andrej Karpathy, *Let's build GPT: from scratch* — watch him implement exactly this.
- [`architecture/attention_flow.md`](../architecture/attention_flow.md) — DevLLM's concrete
  tensor-shape wiring of the above.
- [`09_TRANSFORMER.md`](09_TRANSFORMER.md) — **next:** wrap attention in a full block.

> **Next:** [`09_TRANSFORMER.md`](09_TRANSFORMER.md) — attention is the engine; now we build
> the car around it.
