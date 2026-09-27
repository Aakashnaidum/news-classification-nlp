import json

import pytest

from newsclf.metrics import classification_metrics
from newsclf.predictor import ArtifactError, TaskPredictor


@pytest.fixture(scope="module")
def artifacts(data_dir, tmp_path_factory):
    import scripts.train_app_models as tam  # noqa: PLC0415

    out = tmp_path_factory.mktemp("artifacts")
    tam.main(["--data-dir", str(data_dir), "--out-dir", str(out)])
    return out


def test_metrics_include_macro_f1_and_per_class():
    m = classification_metrics(["a", "a", "b", "c"], ["a", "b", "b", "c"], ["a", "b", "c"], n_boot=50)
    assert m["accuracy"] == 0.75
    assert 0 < m["macro_f1"] <= 1 and m["macro_f1_ci95"][0] <= m["macro_f1"] <= m["macro_f1_ci95"][1] + 1e-9
    assert m["per_class"]["a"]["recall"] == 0.5
    assert m["confusion_matrix"]["matrix"][0] == [1, 1, 0]


def test_manifest_and_prediction(artifacts):
    p = TaskPredictor(artifacts / "category")
    assert p.labels == ["politics", "sports", "technology"]
    pred = p.predict("the coach said the team will win the league cup this season")
    assert pred.label == "sports"
    assert abs(sum(pred.probabilities.values()) - 1) < 1e-6
    assert p.manifest["models"]["tfidf_logreg"]["test_metrics"]["macro_f1"] > 0.9


def test_input_validation(artifacts):
    p = TaskPredictor(artifacts / "fake_news")
    with pytest.raises(ValueError):
        p.predict("   ")
    with pytest.raises(ValueError):
        p.predict("x" * 20_001)
    with pytest.raises(ArtifactError):
        p.predict("text", "no-such-model")


def test_label_order_mismatch_is_detected(artifacts, tmp_path):
    import shutil

    d = tmp_path / "bad"
    shutil.copytree(artifacts / "category", d)
    manifest = json.loads((d / "manifest.json").read_text())
    manifest["labels"] = list(reversed(manifest["labels"]))
    (d / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError, match="Label order mismatch"):
        TaskPredictor(d).predict("some text")


def test_missing_artifacts_raise_clear_error(tmp_path):
    with pytest.raises(ArtifactError, match="train_app_models"):
        TaskPredictor(tmp_path / "nothing")


@pytest.mark.skipif(__import__("importlib").util.find_spec("tf_keras") is None, reason="tf-keras not installed")
def test_neural_artifact_roundtrip(data_dir, tmp_path):
    """The web app path (saved .h5 + tokenizer.json + manifest labels) must match the in-memory model."""
    import numpy as np

    from newsclf.config import MAX_SEQUENCE_LENGTH
    from newsclf.data import clean_split, load_task
    from newsclf.neural import train
    from newsclf.text import pad_sequences

    df = load_task("category", data_dir)
    labels = sorted(df["label"].unique())
    splits, _ = clean_split(df)
    model, tok, _ = train("BiLSTM", splits["train"], splits["val"], labels, max_epochs=2)
    d = tmp_path / "category"
    d.mkdir()
    model.save(d / "rnn.h5")
    tok.save(d / "tokenizer.json")
    (d / "manifest.json").write_text(json.dumps({"labels": labels, "default_model": "rnn", "models": {
        "rnn": {"type": "keras", "file": "rnn.h5", "tokenizer": "tokenizer.json", "title": "rnn"}}}))
    text = splits["test"]["text"].iloc[0]
    expected = model.predict(pad_sequences(tok.texts_to_sequences([text]), MAX_SEQUENCE_LENGTH), verbose=0)[0]
    got = TaskPredictor(d).predict(text)
    assert np.allclose([got.probabilities[lab] for lab in labels], expected, atol=1e-5)
