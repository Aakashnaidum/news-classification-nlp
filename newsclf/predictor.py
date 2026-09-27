"""Load trained artifacts and classify text consistently with training.

Layout written by scripts/train_app_models.py::

    artifacts/<task>/manifest.json      labels, default model, evaluation evidence
    artifacts/<task>/tfidf_logreg.joblib
    artifacts/<task>/rnn.h5 + tokenizer.json   (optional; needs TensorFlow + tf-keras)

The label order and the tokenizer come from the files saved at training time;
nothing is re-fitted at inference time.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from .config import MAX_SEQUENCE_LENGTH
from .text import SequenceTokenizer, pad_sequences

MAX_INPUT_CHARS = 20_000


class ArtifactError(RuntimeError):
    pass


@dataclass
class Prediction:
    model: str
    label: str
    probabilities: dict[str, float]


class TaskPredictor:
    def __init__(self, task_dir: Path):
        self.dir = Path(task_dir)
        manifest_path = self.dir / "manifest.json"
        if not manifest_path.exists():
            raise ArtifactError(f"{manifest_path} not found; run scripts/train_app_models.py")
        self.manifest = json.loads(manifest_path.read_text())
        self.labels: list[str] = self.manifest["labels"]
        self._models: dict = {}

    @property
    def available_models(self) -> list[str]:
        names = []
        for name, spec in self.manifest["models"].items():
            if spec["type"] == "keras" and not _keras_available():
                continue
            names.append(name)
        return names

    def _load(self, name: str):
        if name in self._models:
            return self._models[name]
        spec = self.manifest["models"].get(name)
        if spec is None:
            raise ArtifactError(f"Unknown model '{name}'")
        if spec["type"] == "sklearn":
            model = joblib.load(self.dir / spec["file"])
            classes = [str(c) for c in model.classes_]
            if classes != self.labels:
                raise ArtifactError(f"Label order mismatch: model {classes} vs manifest {self.labels}")
            loaded = ("sklearn", model, None)
        elif spec["type"] == "keras":
            if not _keras_available():
                raise ArtifactError("TensorFlow/tf-keras is not installed")
            import tf_keras

            model = tf_keras.models.load_model(self.dir / spec["file"], compile=False)
            if model.output_shape[-1] != len(self.labels):
                raise ArtifactError("Model output size does not match the saved labels")
            loaded = ("keras", model, SequenceTokenizer.load(self.dir / spec["tokenizer"]))
        else:
            raise ArtifactError(f"Unsupported model type {spec['type']}")
        self._models[name] = loaded
        return loaded

    def predict(self, text: str, model_name: str | None = None) -> Prediction:
        text = (text or "").strip()
        if not text:
            raise ValueError("Enter some text to classify.")
        if len(text) > MAX_INPUT_CHARS:
            raise ValueError(f"Text is too long (max {MAX_INPUT_CHARS} characters).")
        name = model_name or self.manifest["default_model"]
        kind, model, tok = self._load(name)
        if kind == "sklearn":
            probs = model.predict_proba([text])[0]
        else:
            x = pad_sequences(tok.texts_to_sequences([text]), MAX_SEQUENCE_LENGTH)
            probs = model.predict(x, verbose=0)[0]
        probs = np.asarray(probs, dtype=float)
        return Prediction(name, self.labels[int(probs.argmax())],
                          {lab: float(p) for lab, p in zip(self.labels, probs)})


def _keras_available() -> bool:
    try:
        import tf_keras  # noqa: F401
        return True
    except Exception:
        return False
