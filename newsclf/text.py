"""A dependency-free re-implementation of the Keras 2 ``Tokenizer`` + ``pad_sequences``.

The original project re-fitted a Keras tokenizer on the training texts inside
every web request. Here the fitted vocabulary is saved as JSON next to the
model, so inference uses exactly the mapping the model was trained with and
the web app does not need TensorFlow just to preprocess text.

Behaviour matches ``keras.preprocessing.text.Tokenizer(num_words=N)`` with its
default filters and ``pad_sequences(maxlen=L)`` defaults (pre-padding,
pre-truncation); a test checks this against tf-keras when it is installed.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

KERAS_FILTERS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
_TRANSLATE = str.maketrans({c: " " for c in KERAS_FILTERS})


def text_to_word_sequence(text: str) -> list[str]:
    return [w for w in str(text).lower().translate(_TRANSLATE).split(" ") if w]


class SequenceTokenizer:
    def __init__(self, num_words: int, word_index: dict[str, int] | None = None):
        self.num_words = num_words
        self.word_index = word_index or {}

    def fit(self, texts) -> "SequenceTokenizer":
        counts: Counter = Counter()
        order: dict[str, int] = {}
        for text in texts:
            for w in text_to_word_sequence(text):
                counts[w] += 1
                order.setdefault(w, len(order))
        # Keras sorts by count (descending); ties keep first-seen order.
        ranked = sorted(counts, key=lambda w: (-counts[w], order[w]))
        self.word_index = {w: i + 1 for i, w in enumerate(ranked)}
        return self

    def texts_to_sequences(self, texts) -> list[list[int]]:
        out = []
        for text in texts:
            seq = []
            for w in text_to_word_sequence(text):
                i = self.word_index.get(w)
                if i is not None and i < self.num_words:
                    seq.append(i)
            out.append(seq)
        return out

    def save(self, path: Path) -> None:
        # Only the ids that can ever be emitted are needed at inference time.
        kept = {w: i for w, i in self.word_index.items() if i < self.num_words}
        Path(path).write_text(json.dumps({"num_words": self.num_words, "word_index": kept}))

    @classmethod
    def load(cls, path: Path) -> "SequenceTokenizer":
        data = json.loads(Path(path).read_text())
        return cls(data["num_words"], data["word_index"])


def pad_sequences(sequences, maxlen: int) -> np.ndarray:
    """Pre-pad with 0 and keep the *last* ``maxlen`` tokens (Keras defaults)."""
    out = np.zeros((len(sequences), maxlen), dtype="int32")
    for row, seq in enumerate(sequences):
        seq = seq[-maxlen:]
        if seq:
            out[row, -len(seq):] = seq
    return out
