"""Audit the data, reproduce the original evaluation, and run the de-duplicated comparison.

Writes results/metrics.json and results/RESULTS.md.

  python scripts/run_experiments.py                  # baseline + neural (needs requirements-neural.txt)
  python scripts/run_experiments.py --skip-neural    # scikit-learn only
  python scripts/run_experiments.py --legacy-models-dir /path/to/original/h5/files
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import sklearn  # noqa: E402

from newsclf.baseline import make_pipeline, tune_and_fit  # noqa: E402
from newsclf.config import DATA_DIR, MAX_SEQUENCE_LENGTH, MAX_WORDS, RESULTS_DIR, TASKS  # noqa: E402
from newsclf.data import audit, clean_split, load_task, original_split  # noqa: E402
from newsclf.metrics import classification_metrics  # noqa: E402
from newsclf.text import SequenceTokenizer, pad_sequences  # noqa: E402

NEURAL_SEEDS = (42, 43, 44)


def majority(train, test, labels):
    top = train["label"].value_counts().idxmax()
    return classification_metrics(test["label"], [top] * len(test), labels)


def legacy_section(df, task, labels, legacy_dir, run_neural):
    """Original protocol: random 80/20 split of raw rows, duplicates included."""
    train, test = original_split(df)
    unseen = ~test["key"].isin(set(train["key"]))
    out = {"n_test": int(len(test)), "n_test_unseen": int(unseen.sum()),
           "test_rows_with_text_in_train": int((~unseen).sum()), "models": {}}
    pipe = make_pipeline().fit(train["text"], train["label"])
    pred = pipe.predict(test["text"])
    out["models"]["TF-IDF + LogReg (C=10)"] = {
        "all_test": classification_metrics(test["label"], pred, labels, n_boot=200),
        "unseen_test": classification_metrics(test["label"][unseen], pred[unseen.to_numpy()], labels, n_boot=200),
    }
    h5 = legacy_dir / task.legacy_model if legacy_dir else None
    if run_neural and h5 and h5.exists():
        from newsclf.neural import load_legacy

        model = load_legacy(h5)
        # Reproduce the original preprocessing: tokenizer fitted on the lower-cased training split.
        tok = SequenceTokenizer(MAX_WORDS).fit(train["text"].str.lower())
        probs = model.predict(pad_sequences(tok.texts_to_sequences(test["text"].str.lower()), MAX_SEQUENCE_LENGTH), verbose=0)
        pred = np.array([labels[i] for i in probs.argmax(axis=1)])
        out["models"][f"Original {task.legacy_architecture} ({task.legacy_model})"] = {
            "all_test": classification_metrics(test["label"], pred, labels, n_boot=200),
            "unseen_test": classification_metrics(test["label"][unseen], pred[unseen.to_numpy()], labels, n_boot=200),
            "note": "saved model trained by the original project on this same split",
        }
    return out


def clean_section(df, task, labels, run_neural, artifacts_out):
    splits, info = clean_split(df)
    tr, va, te = splits["train"], splits["val"], splits["test"]
    res = {"protocol": info, "models": {}}
    res["models"]["Majority class"] = majority(tr, te, labels)
    pipe, tuning = tune_and_fit(tr, va)
    res["models"]["TF-IDF + LogReg"] = {**classification_metrics(te["label"], pipe.predict(te["text"]), labels),
                                        "tuning": tuning}
    if run_neural:
        from newsclf.neural import predict_labels, train

        runs = []
        for seed in NEURAL_SEEDS:
            model, tok, fit_info = train(task.legacy_architecture, tr, va, labels, seed=seed)
            m = classification_metrics(te["label"], predict_labels(model, tok, te["text"], labels), labels)
            runs.append({"seed": seed, **fit_info, **m})
            if artifacts_out is not None and seed == NEURAL_SEEDS[0]:
                artifacts_out[task.key] = (model, tok)
        f1s = [r["macro_f1"] for r in runs]
        accs = [r["accuracy"] for r in runs]
        res["models"][f"{task.legacy_architecture} (retrained)"] = {
            **runs[0], "seeds": list(NEURAL_SEEDS),
            "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
            "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
            "runs": [{k: r[k] for k in ("seed", "epochs_run", "accuracy", "macro_f1")} for r in runs],
        }
    return res


def pct(x):
    return f"{100 * x:.1f}"


def report(results) -> str:
    L = [f"# Results (re-run {results['generated_utc']})", "",
         f"Python {results['python']}, scikit-learn {results['sklearn']}"
         + (f", TensorFlow {results['tensorflow']}" if results.get("tensorflow") else "") + ".", ""]
    for key, r in results["tasks"].items():
        a = r["audit"]
        L += [f"## {TASKS[key].title}", "",
              f"Raw rows {a['rows']}, unique texts {a['unique_texts']} "
              f"({pct(a['duplicate_share'])}% duplicate rows), texts with conflicting labels {a['texts_with_conflicting_labels']}.", "",
              "### 1. Original protocol (random 80/20 split of raw rows)", "",
              f"{r['legacy']['test_rows_with_text_in_train']} of {r['legacy']['n_test']} test rows have an identical text in the training split.", "",
              "| Model | Accuracy (all test) | Macro F1 (all test) | Accuracy (unseen texts only) | Macro F1 (unseen) |",
              "|---|---:|---:|---:|---:|"]
        for name, m in r["legacy"]["models"].items():
            L.append(f"| {name} | {pct(m['all_test']['accuracy'])} | {pct(m['all_test']['macro_f1'])} | "
                     f"{pct(m['unseen_test']['accuracy'])} | {pct(m['unseen_test']['macro_f1'])} |")
        p = r["clean"]["protocol"]
        L += ["", "### 2. De-duplicated protocol (stratified 70/15/15, no shared texts)", "",
              f"Sizes {p['split_sizes']}; exact train/test overlap {p['exact_overlap']['train_test']}; "
              f"near-duplicates (TF-IDF cosine >= {p['near_duplicates_test_vs_train']['threshold']}) "
              f"{p['near_duplicates_test_vs_train']['count']} of {p['near_duplicates_test_vs_train']['of']}.", "",
              "| Model | Accuracy | Macro F1 [95% bootstrap CI] | Notes |", "|---|---:|---:|---|"]
        for name, m in r["clean"]["models"].items():
            note = ""
            if "macro_f1_mean" in m:
                note = f"seed 42 shown; mean±sd over seeds {m['seeds']}: F1 {pct(m['macro_f1_mean'])}±{pct(m['macro_f1_std'])}"
            elif "tuning" in m:
                note = f"C={m['tuning']['chosen_C']} chosen on validation"
            ci = m["macro_f1_ci95"]
            L.append(f"| {name} | {pct(m['accuracy'])} | {pct(m['macro_f1'])} [{pct(ci[0])}, {pct(ci[1])}] | {note} |")
        L += ["", "Per-class F1 (de-duplicated test):", "",
              "| Class | Support | " + " | ".join(n for n in r["clean"]["models"] if n != "Majority class") + " |",
              "|---|---:|" + "---:|" * (len(r["clean"]["models"]) - 1)]
        first = next(iter(r["clean"]["models"].values()))
        for lab in first["per_class"]:
            cells = [pct(m["per_class"][lab]["f1"]) for n, m in r["clean"]["models"].items() if n != "Majority class"]
            L.append(f"| {lab} | {first['per_class'][lab]['support']} | " + " | ".join(cells) + " |")
        L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--legacy-models-dir", type=Path, default=None)
    ap.add_argument("--skip-neural", action="store_true")
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    args = ap.parse_args(argv)

    results = {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
               "python": platform.python_version(), "sklearn": sklearn.__version__, "tasks": {}}
    if not args.skip_neural:
        import tensorflow as tf
        results["tensorflow"] = tf.__version__
    for key in args.tasks:
        task = TASKS[key]
        df = load_task(key, args.data_dir)
        labels = sorted(df["label"].unique())
        print(f"[{key}] auditing / legacy protocol", flush=True)
        entry = {"labels": labels, "audit": audit(df),
                 "legacy": legacy_section(df, task, labels, args.legacy_models_dir, not args.skip_neural)}
        print(f"[{key}] de-duplicated protocol", flush=True)
        entry["clean"] = clean_section(df, task, labels, not args.skip_neural, None)
        results["tasks"][key] = entry

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    (args.out_dir / "RESULTS.md").write_text(report(results))
    print(report(results))


if __name__ == "__main__":
    main()
