"""Tasks, paths and shared constants. Paths are overridable with environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("NEWS_DATA_DIR", REPO_ROOT / "data"))
ARTIFACTS_DIR = Path(os.environ.get("NEWS_ARTIFACTS_DIR", REPO_ROOT / "artifacts"))
RESULTS_DIR = Path(os.environ.get("NEWS_RESULTS_DIR", REPO_ROOT / "results"))

SEED = 42
MAX_WORDS = 10_000        # vocabulary size used by the original neural models
MAX_SEQUENCE_LENGTH = 100  # tokens per document used by the original neural models


@dataclass(frozen=True)
class Task:
    key: str
    title: str
    csv_name: str
    text_column: str
    label_column: str
    legacy_model: str       # file name of the original Keras model
    legacy_architecture: str


TASKS: dict[str, Task] = {
    t.key: t
    for t in (
        Task("fake_news", "Fake vs. real news", "NEWS.csv", "text", "label",
             "NEWS.h5", "BiSimpleRNN"),
        Task("category", "News category (7 classes)", "CATEGORY.csv", "news_article", "news_category",
             "category.h5", "BiLSTM"),
    )
}
