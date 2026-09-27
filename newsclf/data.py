"""Loading, auditing and leakage-safe splitting of the two news datasets."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from .config import DATA_DIR, SEED, TASKS


def normalise(text) -> str:
    """Key used to detect duplicates: lower-case, collapsed whitespace."""
    return re.sub(r"\s+", " ", str(text) if pd.notna(text) else "").strip().lower()


def load_task(task_key: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Return a frame with columns text, label, key (normalised text)."""
    task = TASKS[task_key]
    path = Path(data_dir or DATA_DIR) / task.csv_name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See data/README.md for how to obtain it.")
    raw = pd.read_csv(path)
    missing = {task.text_column, task.label_column} - set(raw.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns {sorted(missing)}")
    df = pd.DataFrame({
        "text": raw[task.text_column].fillna("").astype(str),
        "label": raw[task.label_column].astype(str).str.strip().str.lower(),
    })
    df["key"] = df["text"].map(normalise)
    return df


def audit(df: pd.DataFrame) -> dict:
    groups = df.groupby("key")["label"].nunique()
    lengths = df["text"].str.split().str.len()
    return {
        "rows": int(len(df)),
        "empty_texts": int((df["key"] == "").sum()),
        "unique_texts": int(df["key"].nunique()),
        "duplicate_rows": int(df["key"].duplicated().sum()),
        "duplicate_share": float(df["key"].duplicated().mean()),
        "largest_duplicate_group": int(df["key"].value_counts().max()),
        "texts_with_conflicting_labels": int((groups > 1).sum()),
        "class_counts_raw": {k: int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "median_words": float(lengths.median()),
    }


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep one row per distinct text; drop texts that carry conflicting labels."""
    n_labels = df.groupby("key")["label"].transform("nunique")
    conflicting = df[n_labels > 1]["key"].nunique()
    clean = df[(n_labels == 1) & (df["key"] != "")].drop_duplicates("key").reset_index(drop=True)
    return clean, {
        "rows_before": int(len(df)),
        "rows_after": int(len(clean)),
        "dropped_conflicting_texts": int(conflicting),
        "class_counts_clean": {k: int(v) for k, v in clean["label"].value_counts().sort_index().items()},
    }


def clean_split(df: pd.DataFrame, seed: int = SEED, val_size=0.15, test_size=0.15):
    """Stratified train/validation/test split of de-duplicated data."""
    clean, info = deduplicate(df)
    train_val, test = train_test_split(clean, test_size=test_size, stratify=clean["label"], random_state=seed)
    train, val = train_test_split(train_val, test_size=val_size / (1 - test_size),
                                  stratify=train_val["label"], random_state=seed)
    splits = {"train": train.reset_index(drop=True), "val": val.reset_index(drop=True),
              "test": test.reset_index(drop=True)}
    info["split_sizes"] = {k: int(len(v)) for k, v in splits.items()}
    info["exact_overlap"] = {
        "train_test": int(splits["test"]["key"].isin(set(splits["train"]["key"])).sum()),
        "train_val": int(splits["val"]["key"].isin(set(splits["train"]["key"])).sum()),
    }
    info["near_duplicates_test_vs_train"] = near_duplicate_count(splits["train"]["key"], splits["test"]["key"])
    return splits, info


def original_split(df: pd.DataFrame):
    """The original notebooks' protocol: random 80/20 split of the raw rows (seed 42)."""
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def near_duplicate_count(train_texts, test_texts, threshold: float = 0.9) -> dict:
    """Test documents whose TF-IDF cosine similarity to some training document >= threshold."""
    vec = TfidfVectorizer(min_df=1, sublinear_tf=True).fit(list(train_texts) + list(test_texts))
    a, b = vec.transform(train_texts), vec.transform(test_texts)
    best = np.zeros(b.shape[0])
    for start in range(0, b.shape[0], 500):
        sims = (b[start:start + 500] @ a.T).toarray()
        best[start:start + 500] = sims.max(axis=1)
    return {"threshold": threshold, "count": int((best >= threshold).sum()), "of": int(b.shape[0])}
