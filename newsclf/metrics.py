"""Classification metrics with bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .config import SEED


def classification_metrics(y_true, y_pred, labels: list[str], n_boot: int = 1000) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    rng = np.random.default_rng(SEED)
    boots = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(f1_score(y_true[idx], y_pred[idx], labels=labels, average="macro", zero_division=0))
    return {
        "n": int(n),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "per_class": {
            lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, lab in enumerate(labels)
        },
        "confusion_matrix": {"labels": labels, "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()},
    }
