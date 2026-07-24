# Model Card: JimmyLabs

> **Status:** Living Document. This model card is updated as the model scales from v0.1 to v1.0.

## Model Details
- **Name:** JimmyLabs
- **Versions:** v0.1 (~0.82M), v0.2 (~2.7M), v0.3 (~4.8M), v1.0 (~4M tuned). See [`SPEC.md`](SPEC.md) §3 for specific hyperparameter configurations per version.
- **Model Type:** Decoder-only, causal self-attention, autoregressive next-token language model.
- **License:** MIT License.
- **Hardware:** Built, trained, and tested entirely on a MacBook Air M1 (8GB unified memory) using PyTorch MPS/Metal.

## Intended Use
- **Primary Use Case:** **Educational only.** JimmyLabs is designed to be understood down to the last gradient, serving as a pedagogical tool for those learning how transformer-based language models work from scratch.
- **Out-of-Scope Uses:** **NOT for production.** This model is tiny (1–4 million parameters). It cannot perform complex reasoning, code generation, translation, or general-purpose instruction following. It will produce simple, often-wrong text. Do not deploy this in user-facing applications.

## Training Data
The data dictates what a model of this capacity can learn. We explicitly match data complexity to the model's 1–4M parameter capacity.
- **Corpora:**
  - *Shakespeare (tiny):* ~1 MB public domain dataset used for v0.1 as a sanity/overfit test.
  - *TinyStories:* A dataset of syntactically simple short stories used as the primary corpus for v0.2+.
- **Data Details & Licensing:** Refer to [`docs/17_DATASET_GUIDE.md`](docs/17_DATASET_GUIDE.md) and [`datasets/SOURCES.md`](datasets/SOURCES.md) for full provenance, versioning, cleaning pipelines, and licenses of the corpora used.

## Architecture
- **Description:** A from-scratch implementation of the standard GPT-style transformer.
- **Key Features:** Pre-norm LayerNorm, GELU activations, learned positional embeddings, and output weight tying ($W_{\text{out}} = W_{\text{token}}^T$).
- **Technical Specification:** See [`SPEC.md`](SPEC.md) for precise tensor shapes, parameter counts, and pipeline descriptions.

## Evaluation
We hold strictly to the evaluation contract defined in [`SPEC.md`](SPEC.md) §13.
- **Quantitative Metrics:**
  - Held-out cross-entropy loss: *(to be measured)*
  - Validation perplexity: *(to be measured)*
- **Qualitative Metrics:**
  - Fixed-prompt generation samples, recorded at every checkpoint and stored in `outputs/`.
- **Performance Benchmarks:**
  - Tokens/sec and peak memory usage on the M1 Air: *(to be measured)*
- *Note: No metrics are reported without their definition and fixed seed. Currently, numerical metrics are pending actual training runs.*

## Ethical Considerations
- **Hallucinations & Reliability:** As a very small model trained on narrow corpora, JimmyLabs will hallucinate confidently. It does not possess real-world knowledge or factual accuracy.
- **Safety:** The model has not been subjected to RLHF or safety fine-tuning. It may reflect biases present in its training data (e.g., historical biases in Shakespeare). It must not be relied upon for safety-critical, medical, legal, or advisory purposes.

## How to Cite
If you found this educational repository helpful, you can cite it as:

```bibtex
@software{jimmylabs_2026,
  author = {JimmyLabs Contributors},
  title = {JimmyLabs: A From-Scratch Educational GPT},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/DevxD98/JimmyLabs}}
}
```
