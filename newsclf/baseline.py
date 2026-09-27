"""TF-IDF + logistic regression baseline, with C tuned on the validation split."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

C_GRID = (0.3, 1.0, 3.0, 10.0, 30.0)


def make_pipeline(C: float = 10.0) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)),
        ("clf", LogisticRegression(C=C, max_iter=2000, class_weight="balanced")),
    ])


def tune_and_fit(train, val) -> tuple[Pipeline, dict]:
    scores = {}
    for C in C_GRID:
        pipe = make_pipeline(C).fit(train["text"], train["label"])
        scores[C] = f1_score(val["label"], pipe.predict(val["text"]), average="macro")
    best_c = max(scores, key=scores.get)
    return make_pipeline(best_c).fit(train["text"], train["label"]), {
        "val_macro_f1_by_C": {str(k): float(v) for k, v in scores.items()}, "chosen_C": best_c}
