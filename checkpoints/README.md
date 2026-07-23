# `checkpoints/` — saved weights

## Why this folder exists

Training is expensive and interruptible — especially on a fanless laptop that may thermal-
throttle or need to sleep. Checkpoints let training resume, let inference load a specific
model, and let experiments be reproduced. This folder documents the **format** and holds
checkpoints locally; the weight files themselves are **git-ignored** (they are large,
binary, and regenerable).

## What a checkpoint contains

```
   ┌──────────────────────────────────────────────┐
   │ checkpoint = {                                 │
   │   model_state_dict     the learned weights     │
   │   optimizer_state_dict momentum / Adam moments │
   │   config               exact model config      │
   │   step / epoch         where training was       │
   │   val_loss             best metric so far       │
   │   rng_state            for exact reproducibility│
   │ }                                              │
   └──────────────────────────────────────────────┘
```

Saving *config + rng_state* alongside weights is what makes a checkpoint reproducible
rather than merely resumable.

## Convention

```
checkpoints/
├── v0_1/ckpt_step_1000.pt      (git-ignored)
└── v0_1/best.pt                (git-ignored)
```

## Why git-ignored

Binary weight blobs bloat git history permanently and are fully regenerable from a config +
seed + dataset. We version the *recipe* (config, code, data provenance), not the *output*.
See [`architecture/training_pipeline.md`](../architecture/training_pipeline.md) and
[`docs/11_TRAINING_PIPELINE.md`](../docs/11_TRAINING_PIPELINE.md).
