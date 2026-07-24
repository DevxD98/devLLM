# Tokenizer Architecture & Module Boundaries

> **Concept Doc:** [`docs/06_TOKENIZER.md`](../docs/06_TOKENIZER.md)  
> **Technical Decision:** [`ADR-0001`](../research/design_decisions/ADR-0001-character-tokenizer-first.md)

This document defines module interfaces, dictionary storage, and serialization contracts for character and subword tokenization in `src/tokenizer/`.

---

## Tokenizer Data Flow & Interfaces

```
   Raw Input Text String
           │
           ▼  CharTokenizer.encode(text)
   Lookup Character Indices in `stoi` Map
           │
           ▼  Convert to PyTorch Tensor
   Integer Token Tensor (B, T)          dtype: int64
           │
           ▼  (Model Execution & Next-Token Sampling)
   Output Token Tensor  (B, T + N)      dtype: int64
           │
           ▼  CharTokenizer.decode(tensor)
   Lookup Indices in `itos` Map
           │
           ▼
   Generated Text String
```

---

## Python Module Contract (`src/tokenizer/char_tokenizer.py`)

```python
class CharTokenizer:
    def __init__(self, text_corpus: str):
        self.chars = sorted(list(set(text_corpus)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> List[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids: List[int]) -> str:
        return ''.join([self.itos[i] for i in ids])
```

---

## Round-Trip Contract Verification
- For any valid string $S$ within the dataset alphabet:
  $$\text{decode}(\text{encode}(S)) \equiv S$$
