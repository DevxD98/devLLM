import os
from jimmylabs.tokenizer.char import CharTokenizer

def test_char_tokenizer_roundtrip():
    """
    Test that decode(encode(x)) == x for the full corpus and edge strings.
    """
    corpus = "Hello, world! \nThis is a test of the 1-4M param tokenizer."
    tokenizer = CharTokenizer(corpus=corpus)
    
    # Check simple round trip
    assert tokenizer.decode(tokenizer.encode(corpus)) == corpus
    
    # Check sub-string round trip
    sub_str = "Hello \n1-4M"
    assert tokenizer.decode(tokenizer.encode(sub_str)) == sub_str

def test_char_tokenizer_vocab():
    """
    Test that vocab is stable/sorted and every encoded id is in [0, vocab_size).
    """
    corpus = "zxy abc"
    tokenizer = CharTokenizer(corpus=corpus)
    
    # Vocab should be space, a, b, c, x, y, z sorted
    assert tokenizer.vocab == [' ', 'a', 'b', 'c', 'x', 'y', 'z']
    assert tokenizer.vocab_size == 7
    
    ids = tokenizer.encode(corpus)
    for i in ids:
        assert 0 <= i < tokenizer.vocab_size

def test_char_tokenizer_save_load(tmp_path):
    """
    Test saving and loading vocabulary from JSON.
    """
    corpus = "save and load me"
    tokenizer1 = CharTokenizer(corpus=corpus)
    
    save_path = tmp_path / "vocab.json"
    tokenizer1.save(str(save_path))
    
    tokenizer2 = CharTokenizer.load(str(save_path))
    
    assert tokenizer1.vocab == tokenizer2.vocab
    assert tokenizer2.decode(tokenizer2.encode(corpus)) == corpus
