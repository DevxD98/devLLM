# `tests/` — what every module must prove

## Why this folder exists

Repository standard (from `02_ARCHITECTURE.md`): *every module has unit tests.* But in an
educational project, tests do more than catch regressions — they are **executable claims
about what a component is supposed to do**. A good test for the attention module is also
the clearest possible statement of what attention *is*.

## Testing philosophy

- **Shape tests first.** The most common bug in a transformer from scratch is a wrong
  tensor shape. Every module test asserts input/output shapes `(B, T, C)` before anything
  else.
- **Known-answer tests.** Where math has a closed form (softmax sums to 1, a causal mask
  zeroes the upper triangle, LayerNorm gives unit variance), assert it exactly.
- **Gradient sanity.** Autograd is trusted, but connectivity is not: assert that loss
  actually produces non-zero gradients on the parameters it should.
- **Overfit-a-batch.** The fastest integration test of a model: can it memorize one tiny
  batch to near-zero loss? If not, something is disconnected.

## Planned layout

```
tests/
├── test_tokenizer.py       round-trip: decode(encode(x)) == x
├── test_attention.py       shapes · causal mask · softmax rows sum to 1
├── test_block.py           residual path preserves shape
├── test_gpt.py             forward shape · overfit-a-batch
└── test_training.py        loss decreases on a fixed batch
```

## Status

Documentation-first. This README defines the test contract each `src/` module must satisfy
as it is built. See [`docs/05_NEURAL_NETWORKS.md`](../docs/05_NEURAL_NETWORKS.md) for the
overfit-a-batch technique explained from first principles.
