import importlib.util

import numpy as np
import pytest

from newsclf.text import SequenceTokenizer, pad_sequences, text_to_word_sequence


def test_word_sequence_matches_keras_rules():
    assert text_to_word_sequence("Hello, World!\tIt's  NEW-DELHI") == ["hello", "world", "it's", "new", "delhi"]


def test_fit_ranks_by_frequency_then_first_seen_and_respects_num_words():
    tok = SequenceTokenizer(num_words=3).fit(["b a", "a c", "c d"])
    assert tok.word_index == {"a": 1, "c": 2, "b": 3, "d": 4}
    assert tok.texts_to_sequences(["a b c d unknown"]) == [[1, 2]]  # ids >= num_words dropped


def test_pad_pre_and_truncate_pre():
    out = pad_sequences([[1, 2, 3], [], [4]], maxlen=2)
    assert out.tolist() == [[2, 3], [0, 0], [0, 4]]


def test_save_load_roundtrip(tmp_path):
    tok = SequenceTokenizer(num_words=10).fit(["x y z", "x"])
    tok.save(tmp_path / "t.json")
    again = SequenceTokenizer.load(tmp_path / "t.json")
    assert again.texts_to_sequences(["z y x q"]) == tok.texts_to_sequences(["z y x q"])


@pytest.mark.skipif(importlib.util.find_spec("tf_keras") is None, reason="tf-keras not installed")
def test_identical_to_keras_tokenizer():
    from tf_keras.preprocessing.sequence import pad_sequences as kpad
    from tf_keras.preprocessing.text import Tokenizer

    texts = ["The quick, brown fox!", "the lazy dog; the END", "Quick quick QUICK 42", "új ünïcode text"] * 3
    k = Tokenizer(num_words=6)
    k.fit_on_texts(texts)
    m = SequenceTokenizer(6).fit(texts)
    assert k.word_index == m.word_index
    assert np.array_equal(kpad(k.texts_to_sequences(texts), maxlen=4), pad_sequences(m.texts_to_sequences(texts), 4))
