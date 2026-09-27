"""Bidirectional RNN models matching the original architecture (needs TensorFlow + tf-keras).

Differences from the original training code, all aimed at a fair evaluation:
  * validation split is a separate, de-duplicated set (not the tail of the training data)
  * early stopping on validation loss restores the best weights; the original
    checkpointed on *training* accuracy for 50 epochs
  * the binary model uses categorical cross-entropy (the original used binary
    cross-entropy on a 2-unit softmax)
"""
from __future__ import annotations

import os

import numpy as np

from .config import MAX_SEQUENCE_LENGTH, MAX_WORDS, SEED
from .text import SequenceTokenizer, pad_sequences

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")


def _keras():
    import tf_keras as keras  # Keras 2 API; also loads the original .h5 files
    return keras


def build_model(architecture: str, num_classes: int, units: int = 128, embedding_dim: int = 100):
    keras = _keras()
    rnn = {"BiLSTM": keras.layers.LSTM, "BiSimpleRNN": keras.layers.SimpleRNN}[architecture]
    model = keras.Sequential([
        keras.layers.Embedding(MAX_WORDS, embedding_dim, input_length=MAX_SEQUENCE_LENGTH),
        keras.layers.Bidirectional(rnn(units, dropout=0.2, recurrent_dropout=0.2)),
        keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def train(architecture: str, train_df, val_df, labels: list[str], seed: int = SEED, max_epochs: int = 30):
    keras = _keras()
    keras.utils.set_random_seed(seed)
    tok = SequenceTokenizer(MAX_WORDS).fit(train_df["text"])
    to_x = lambda s: pad_sequences(tok.texts_to_sequences(s), MAX_SEQUENCE_LENGTH)  # noqa: E731
    to_y = lambda s: keras.utils.to_categorical([labels.index(v) for v in s], len(labels))  # noqa: E731
    model = build_model(architecture, len(labels))
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    hist = model.fit(to_x(train_df["text"]), to_y(train_df["label"]), validation_data=(to_x(val_df["text"]), to_y(val_df["label"])),
                     epochs=max_epochs, batch_size=32, callbacks=[stop], verbose=0)
    return model, tok, {"epochs_run": len(hist.history["loss"]),
                        "best_val_loss": float(np.min(hist.history["val_loss"]))}


def predict_labels(model, tok: SequenceTokenizer, texts, labels: list[str]) -> list[str]:
    probs = model.predict(pad_sequences(tok.texts_to_sequences(texts), MAX_SEQUENCE_LENGTH), verbose=0)
    return [labels[i] for i in probs.argmax(axis=1)]


def load_legacy(path):
    return _keras().models.load_model(path, compile=False)
