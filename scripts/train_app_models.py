"""Train the models the web app serves and write artifacts/<task>/.

For each task (de-duplicated, stratified 70/15/15 split, seed 42):
  * TF-IDF + logistic regression: C chosen on validation, then refit on train+val
  * optionally the bidirectional RNN (--neural), trained on train with early
    stopping on validation; its tokenizer is saved as tokenizer.json
Hold-out test metrics for every model are stored in manifest.json and shown by the app.

Usage: python scripts/train_app_models.py [--neural] [--data-dir DIR] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

from newsclf.baseline import make_pipeline, tune_and_fit  # noqa: E402
from newsclf.config import ARTIFACTS_DIR, DATA_DIR, TASKS  # noqa: E402
from newsclf.data import clean_split, load_task  # noqa: E402
from newsclf.metrics import classification_metrics  # noqa: E402


def summarise(m: dict) -> dict:
    return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "macro_f1_ci95": m["macro_f1_ci95"],
            "per_class_f1": {k: v["f1"] for k, v in m["per_class"].items()}, "n_test": m["n"]}


def train_task(key: str, data_dir: Path, out_dir: Path, neural: bool) -> dict:
    task = TASKS[key]
    df = load_task(key, data_dir)
    labels = sorted(df["label"].unique())
    splits, info = clean_split(df)
    tr, va, te = splits["train"], splits["val"], splits["test"]
    task_dir = out_dir / key
    task_dir.mkdir(parents=True, exist_ok=True)

    _, tuning = tune_and_fit(tr, va)
    # Score the tuned configuration on test, then refit on train+val for serving.
    scored = make_pipeline(tuning["chosen_C"]).fit(tr["text"], tr["label"])
    lr_metrics = classification_metrics(te["label"], scored.predict(te["text"]), labels)
    final = make_pipeline(tuning["chosen_C"]).fit(pd.concat([tr["text"], va["text"]]), pd.concat([tr["label"], va["label"]]))
    joblib.dump(final, task_dir / "tfidf_logreg.joblib", compress=3)
    models = {"tfidf_logreg": {"type": "sklearn", "file": "tfidf_logreg.joblib",
                               "title": "TF-IDF + logistic regression",
                               "test_metrics": summarise(lr_metrics), "C": tuning["chosen_C"]}}

    if neural:
        from newsclf.neural import predict_labels, train

        model, tok, fit_info = train(task.legacy_architecture, tr, va, labels)
        nn_metrics = classification_metrics(te["label"], predict_labels(model, tok, te["text"], labels), labels)
        model.save(task_dir / "rnn.h5")
        tok.save(task_dir / "tokenizer.json")
        models["rnn"] = {"type": "keras", "file": "rnn.h5", "tokenizer": "tokenizer.json",
                         "title": f"{task.legacy_architecture} (retrained)", "test_metrics": summarise(nn_metrics),
                         **fit_info}

    manifest = {"task": key, "title": task.title, "labels": labels, "default_model": "tfidf_logreg",
                "models": models, "split": info["split_sizes"], "sklearn_version": sklearn.__version__}
    (task_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=ARTIFACTS_DIR)
    ap.add_argument("--neural", action="store_true")
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    args = ap.parse_args(argv)
    for key in args.tasks:
        m = train_task(key, args.data_dir, args.out_dir, args.neural)
        for name, spec in m["models"].items():
            t = spec["test_metrics"]
            print(f"{key:<10} {name:<13} acc={t['accuracy']:.3f} macroF1={t['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
