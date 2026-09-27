"""Tiny synthetic datasets for tests and CI (no real news data needed)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

VOCAB = {
    "sports": "match team goal league coach player score season win cup",
    "politics": "election minister party vote parliament policy leader campaign bill court",
    "technology": "software chip startup phone app data cloud device launch ai",
}
FAKE_REAL = {
    "fake": "viral video claim shared social media image false morphed fact check",
    "real": "said official reported on monday statement police department announced ministry",
}


def _docs(vocab: dict, n_per_class: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    filler = "the a of in and to for with on at by from".split()
    rows = []
    for label, words in vocab.items():
        words = words.split()
        for i in range(n_per_class):
            tokens = list(rng.choice(words, 12)) + list(rng.choice(filler, 8)) + [f"doc{label}{i}"]
            rng.shuffle(tokens)
            rows.append((label, " ".join(tokens)))
    return pd.DataFrame(rows, columns=["label", "text"])


def write_synthetic_data(data_dir: Path, n_per_class: int = 60, seed: int = 0) -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    fr = _docs(FAKE_REAL, n_per_class, seed)
    fr["label"] = fr["label"].str.upper()                      # original file uses FAKE/REAL
    fr = pd.concat([fr, fr.head(10)])                          # some exact duplicates, like the real data
    fr.to_csv(data_dir / "NEWS.csv", index=False)
    cat = _docs(VOCAB, n_per_class, seed + 1)
    cat = pd.concat([cat, cat.head(5).assign(label="sports")])  # duplicates, some with conflicting labels
    cat.rename(columns={"text": "news_article", "label": "news_category"}).assign(
        news_headline="headline").to_csv(data_dir / "CATEGORY.csv", index=True)
