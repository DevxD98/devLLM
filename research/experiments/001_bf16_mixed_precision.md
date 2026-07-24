# Experiment 001: bfloat16 Mixed Precision

**Date**: 2026-07-25
**Status**: ❌ REJECTED

## Hypothesis
Using `torch.autocast(device_type='mps', dtype=torch.bfloat16)` will reduce memory footprint by storing activations in half-precision, and potentially increase training throughput if the hardware can execute `bfloat16` matrix multiplications natively.

## The Test
We wrapped the forward passes and loss calculations in `benchmark.py` with `autocast` using `dtype=torch.bfloat16` and re-ran the benchmark against the baseline (using `--use_cache`). 

## Results (Median, Warm)

| Metric | FP32 Baseline | BF16 Autocast | Delta |
|--------|---------------|---------------|-------|
| **Train Throughput** | 65,027 tok/s | 55,161 tok/s | 🔻 **-15% slower** |
| **Gen Throughput** | 129 tok/s | 88 tok/s | 🔻 **-31% slower** |
| **MPS Driver Memory** | 257.6 MB | 188.5 MB | 🟢 **-26% smaller** |

## Conclusion
**Why we are rejecting it**: While `bfloat16` *does* save a measurable amount of memory (down to 188.5 MB from 257.6 MB), it catastrophically harms throughput. 

**The architectural reason**: The Apple M1 chip lacks native hardware instructions for `bfloat16` matrix multiplication (unlike NVIDIA Ampere+ GPUs which have dedicated BF16 Tensor Cores). Consequently, the PyTorch MPS backend must insert casting operations or emulate the math, which introduces massive overhead. 

Since our goal on a MacBook Air is local speed, slowing down the model by 15-30% for a few megabytes of memory is a bad trade. We will stick to `fp32` natively for the baseline.
