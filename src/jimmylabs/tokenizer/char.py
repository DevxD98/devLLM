import json

class CharTokenizer:
    """
    A simple character-level tokenizer.
    Aligns with ADR-0001: tiny vocabulary, trivial encoding/decoding, no external libs.
    """
    def __init__(self, corpus: str = None, vocab: list[str] = None):
        if vocab is not None:
            self.vocab = vocab
        elif corpus is not None:
            # Sort to ensure determinism and stability across runs
            self.vocab = sorted(list(set(corpus)))
        else:
            raise ValueError("Must provide either corpus or vocab.")
            
        self.vocab_size = len(self.vocab)
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}

    def encode(self, text: str) -> list[int]:
        """Convert a string into a list of integer token IDs."""
        return [self.stoi[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        """Convert a list of integer token IDs back into a string."""
        return ''.join([self.itos[i] for i in ids])

    def save(self, filepath: str):
        """Save the vocabulary to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({'vocab': self.vocab}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'CharTokenizer':
        """Load the vocabulary from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(vocab=data['vocab'])
