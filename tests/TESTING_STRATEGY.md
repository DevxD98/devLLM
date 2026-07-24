# Testing Strategy

> Most hobby ML repos have zero tests. JimmyLabs treats tests as **executable claims about what
> each component is** — the clearest possible statement of what attention, a block, or the
> tokenizer is *supposed to do* — not just regression insurance (principle 6). This document
> is the full strategy; [`tests/README.md`](README.md) is the short version.
>
> Governs: every `src/` module. See the module map in [`src/README.md`](../src/README.md) and
> failure cases in [`SPEC.md`](../SPEC.md) §11.

---

## The testing pyramid (bottom = most, fastest, cheapest)

```
                    ▲   fewer, slower, higher-confidence
          ┌─────────────────────┐
          │  regression tests   │   guard numbers we've earned (loss, tokens/sec)
          ├─────────────────────┤
          │ integration:        │   overfit-a-batch · generate-runs · train-step-decreases
          │ overfit-a-batch     │
          ├─────────────────────┤
          │  gradient checks    │   hand-understanding == autograd
          ├─────────────────────┤
          │ known-answer tests  │   softmax sums to 1 · mask zeroes future · LN unit var
          ├─────────────────────┤
          │    shape tests      │   (B,T,C) in → (B,T,C) out — THE most common bug class
          └─────────────────────┘
              many, fast, run on every change     ▼
```

Build from the bottom up. A module isn't "done" until at least its shape and known-answer
tests pass.

---

## The five layers

### 1. Shape tests — *always first*
The #1 bug in a from-scratch transformer is a wrong tensor shape (a bad reshape, a transpose,
a head-split off by one). Every module asserts input→output shapes before anything else,
against the contract in [`SPEC.md`](../SPEC.md) §4.
```
   attention:  (B,T,C) → (B,T,C);   scores (B,h,T,T)
   block:      (B,T,C) → (B,T,C)     (shape-invariant — enables stacking)
   gpt:        (B,T)   → (B,T,V)
```

### 2. Known-answer tests — *math with a right answer*
Where closed-form truth exists, assert it exactly (within float tolerance):
- **softmax** rows sum to 1 and are all in (0,1).
- **causal mask** zeroes the strict upper triangle (no attention to the future) — the single
  most important correctness test in the repo; a missing mask trains beautifully and
  generates garbage ([`08`](../docs/08_ATTENTION.md)).
- **LayerNorm** output has ~zero mean and ~unit variance per token (pre affine).
- **tokenizer** round-trips: `decode(encode(x)) == x` for all x.
- **weight tying**: the output-projection weight *is* the token-embedding weight (identity,
  not just equal).

### 3. Gradient checks — *understanding == autograd*
We trust PyTorch autograd, but not our *wiring*. Two checks:
- **Connectivity:** after a backward pass, every parameter that should have a gradient has a
  non-`None`, non-zero grad (catches disconnected sub-graphs).
- **Numerical gradient check** on the small hand-built pieces (e.g. the Phase-0 micrograd
  toy): finite-difference vs analytic gradient agree. This is where your *understanding* of
  backprop gets verified, not just the framework's.

### 4. Integration tests — *does the whole thing learn?*
- **Overfit-a-batch** (the flagship integration test): train on one tiny fixed batch; loss
  must fall to ~0. If it can't memorize 10 examples, something is disconnected — and you find
  out in seconds. Run this before *every* real training run.
- **Generate-runs:** given a checkpoint, generation produces the right-shaped token stream
  and terminates; sampling knobs (temperature/top-k/top-p) don't crash at their extremes.
- **Train-step-decreases:** N steps on a fixed batch strictly reduce loss.

### 5. Regression tests — *protect what we've earned*
Once we have a trusted baseline (a val loss, a tokens/sec from [`14`](../docs/14_BENCHMARKING.md)),
lock it in: fail if a change regresses loss beyond noise or drops throughput past a threshold.
This is what makes optimization safe — you can refactor knowing the numbers are watched.

---

## Determinism & the MPS caveat

Reproducibility (principle 4) requires a **seed-everything** utility (Python, NumPy, torch,
torch.mps) run at the top of every test and run. Caveat: exact bitwise reproducibility on
`mps` can differ from `cpu` and across macOS/torch versions — so cross-device tests use
**tolerances**, not equality, and the determinism test asserts *same-device, same-seed*
repeatability. This subtlety is itself documented so it's not mistaken for a bug (see
[`SPEC.md`](../SPEC.md) §11).

---

## Cross-reference lint (docs, not code)

The repo's cross-reference rule (a concept is defined once, linked elsewhere) is a kind of
test too. Treat a term *defined* in two docs as a defect — the same way you'd treat duplicated
logic. Until automated, this is a review checklist item; noted here so the discipline isn't
forgotten.

---

## What each module must prove (the contract)

```
   tokenizer     round-trip · vocab stable · shapes
   embedding     lookup shape · positions added · tying identity
   attention     shapes · softmax sums to 1 · causal mask zeroes future · scale present
   feedforward   shape preserved · nonlinearity present (not two linears)
   block         shape-invariant · both residuals present · both norms present
   gpt           forward shape (B,T,V) · param count matches SPEC · overfit-a-batch
   training      loss finite · loss decreases · grad clip active · checkpoint round-trips
   inference     generate shape · sampling extremes safe · KV cache == no-cache output
```

That last one — **KV cache produces identical output to the naive path** — is the correctness
gate for the top inference optimization ([`OPTIMIZATION_BACKLOG.md`](../research/OPTIMIZATION_BACKLOG.md) #2).

---

## Tooling & convention

- **`pytest`**, one `test_<module>.py` per `src/` module.
- Fast tests (shape/known-answer) run on every change; slow ones (overfit, generate) on
  demand and before commits that touch the model.
- Tiny fixed configs and seeds for speed and determinism.
- A failing test is a *specification of expected behavior* — read it before "fixing" it
  (maybe the code is right and the test encodes a misunderstanding; resolve which).

## See also

- [`tests/README.md`](README.md) (short version) · [`SPEC.md`](../SPEC.md) §11–12 (failure
  cases & summary) · [`ENGINEERING_PRINCIPLES.md`](../ENGINEERING_PRINCIPLES.md) (principle 6)
  · [`docs/05_NEURAL_NETWORKS.md`](../docs/05_NEURAL_NETWORKS.md) (overfit-a-batch, from first
  principles).
