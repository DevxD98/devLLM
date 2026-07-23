# `assets/` — images referenced by docs

## Why this folder exists

DevLLM's diagrams are ASCII first (see [`diagrams/`](../diagrams/)) because ASCII lives in
plain text, diffs cleanly in git, and never rots. But some things — a loss curve, an
attention heatmap, a photo of a whiteboard — are genuinely raster. Those live here.

## What belongs here

- PNG/SVG figures embedded in `docs/` or `research/`
- Exported plots from `experiments/` worth pinning into prose
- Screenshots of profiler output

## Conventions

- Reference from docs with a relative path: `![caption](../assets/loss_curve_001.png)`
- Name by what it shows and its source: `attention_heatmap_exp007.png`
- Keep files small. Large binaries bloat the repo forever — prefer SVG or compressed PNG.

## What does NOT belong here

- ASCII diagrams → [`diagrams/`](../diagrams/)
- Raw experiment data → `experiments/*/results/`
