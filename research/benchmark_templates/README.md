# `research/benchmark_templates/`

The blank forms every performance measurement copies. A number without its conditions is
noise; this template records the conditions.

- `BENCHMARK_TEMPLATE.md` — copy into [`benchmarks/`](../../benchmarks/) as
  `NNN_short-name.md`.

## What a benchmark must record

```
   machine state  ·  macOS + torch version  ·  device (mps/cpu)
   dtype          ·  batch size  ·  sequence length  ·  model config
   metric         ·  seed        ·  before / after (if it's an optimization)
```

On a fanless MacBook Air, **thermal state matters**: the same code benchmarked warm and
cold gives different tokens/second. The template has a field for it. See
[`docs/14_BENCHMARKING.md`](../../docs/14_BENCHMARKING.md).
