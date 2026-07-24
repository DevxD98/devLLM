# Paper Note — FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

| Field | Value |
|-------|-------|
| Authors | Dao, Fu, Ermon, Rudra, Ré |
| Year | 2022 |
| Venue | NeurIPS |
| Read on | 2026-07-24 |
| Reused in JimmyLabs | memory management philosophy |

> A note is not a summary. It records what **we** understood and what **we** borrowed —
> in our own words. The teaching version lives in
> [`docs/13_OPTIMIZATION_FOR_APPLE_SILICON.md`](../../docs/13_OPTIMIZATION_FOR_APPLE_SILICON.md); this is the lab's reading record.

## Why we read it

Standard attention requires manifesting the entire `T x T` attention matrix in memory. This quadratic memory requirement makes long context windows impossible on constrained hardware. FlashAttention solves this by computing exact attention without ever materializing the full `T x T` matrix. Understanding this is crucial for managing our strict 8GB memory budget on the M1 MacBook.

## Core idea in plain words

Memory bandwidth (moving data between slow HBM memory and fast SRAM cache on the GPU) is the real bottleneck for attention, not math operations. FlashAttention restructures the attention calculation into tiles (blocks). It loads a block of Queries, Keys, and Values into fast SRAM, computes the attention for that block, and writes the result back, using clever math to keep track of the softmax denominator (the scaling factor) as it goes. This eliminates the need to store the huge intermediate `T x T` matrix.

## Mechanisms we reuse

- **The concept of tiling.** While we do not implement custom CUDA or Metal kernels for FlashAttention from scratch, understanding tiling informs how we view the memory constraints of the `QK^T` operation.
  → [`docs/08_ATTENTION.md`](../../docs/08_ATTENTION.md).
- **Justifying framework features.** We rely on PyTorch's `F.scaled_dot_product_attention`, which under the hood uses FlashAttention algorithms if the hardware and context support it. Reading this paper explains *why* we must use that function instead of writing the naive matrix multiplication out manually.
  → [`docs/13_OPTIMIZATION_FOR_APPLE_SILICON.md`](../../docs/13_OPTIMIZATION_FOR_APPLE_SILICON.md) and [`research/OPTIMIZATION_BACKLOG.md` #11](../../research/OPTIMIZATION_BACKLOG.md#11).

## What we deliberately ignore (and why)

- **The low-level SRAM management.** The paper details block sizes and SRAM layout for specific Nvidia GPUs. We are on Apple Silicon (MPS/Metal), so the specific hardware implementation details are irrelevant to us, but the algorithmic trick for streaming softmax is universal.

## Open questions this raised

- How effective is PyTorch's MPS backend at utilizing FlashAttention algorithms compared to CUDA? Can we measure the memory high-water mark with and without `F.scaled_dot_product_attention` on our M1?

## One-line takeaway

> By tiling the attention computation and streaming the softmax calculation, you can compute exact attention without ever storing the massive N² attention matrix, turning a memory bottleneck into a math problem.
