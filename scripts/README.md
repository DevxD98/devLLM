# `scripts/` — entry points

## Why this folder exists

`src/` holds the model's *library* code — importable pieces with no side effects. But a
project also needs **verbs**: train, generate, prepare-data, benchmark. Those command-line
entry points live here, thin and documented, so the library stays clean and testable while
the "things you actually run" are discoverable in one place.

## What will belong here

```
scripts/
├── prepare_data.py     text → token ids → train/val split
├── train.py            config → trained checkpoint
├── generate.py         checkpoint + prompt → text
└── benchmark.py        checkpoint → tokens/second, memory
```

Each script is a thin shell: parse a config, call into `src/`, write to `checkpoints/` or
`outputs/`. Business logic lives in `src/`, not here — a script you can't import is a
script you can't test.

## Status

Documentation-first: this README defines the intended entry points and their contracts.
The scripts themselves arrive with the code phases (roadmap phases 1–3).

See [`docs/11_TRAINING_PIPELINE.md`](../docs/11_TRAINING_PIPELINE.md) and
[`docs/12_INFERENCE.md`](../docs/12_INFERENCE.md).
