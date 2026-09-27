# Results (re-run 2026-09-27 04:06 UTC)

Python 3.11.15, scikit-learn 1.9.1, TensorFlow 2.19.1.

## Fake vs. real news

Raw rows 3729, unique texts 2230 (40.2% duplicate rows), texts with conflicting labels 1.

### 1. Original protocol (random 80/20 split of raw rows)

307 of 746 test rows have an identical text in the training split.

| Model | Accuracy (all test) | Macro F1 (all test) | Accuracy (unseen texts only) | Macro F1 (unseen) |
|---|---:|---:|---:|---:|
| TF-IDF + LogReg (C=10) | 99.7 | 99.7 | 99.8 | 99.6 |
| Original BiSimpleRNN (NEWS.h5) | 96.4 | 96.4 | 93.8 | 88.2 |

### 2. De-duplicated protocol (stratified 70/15/15, no shared texts)

Sizes {'train': 1559, 'val': 335, 'test': 335}; exact train/test overlap 0; near-duplicates (TF-IDF cosine >= 0.9) 0 of 335.

| Model | Accuracy | Macro F1 [95% bootstrap CI] | Notes |
|---|---:|---:|---|
| Majority class | 83.0 | 45.4 [44.0, 46.5] |  |
| TF-IDF + LogReg | 99.4 | 99.0 [97.4, 100.0] | C=0.3 chosen on validation |
| BiSimpleRNN (retrained) | 95.8 | 92.4 [88.1, 95.9] | seed 42 shown; mean±sd over seeds [42, 43, 44]: F1 78.3±16.7 |

Per-class F1 (de-duplicated test):

| Class | Support | TF-IDF + LogReg | BiSimpleRNN (retrained) |
|---|---:|---:|---:|
| fake | 278 | 99.6 | 97.5 |
| real | 57 | 98.3 | 87.3 |

## News category (7 classes)

Raw rows 4817, unique texts 1998 (58.5% duplicate rows), texts with conflicting labels 65.

### 1. Original protocol (random 80/20 split of raw rows)

836 of 964 test rows have an identical text in the training split.

| Model | Accuracy (all test) | Macro F1 (all test) | Accuracy (unseen texts only) | Macro F1 (unseen) |
|---|---:|---:|---:|---:|
| TF-IDF + LogReg (C=10) | 94.7 | 94.7 | 93.8 | 81.5 |
| Original BiLSTM (category.h5) | 91.7 | 91.3 | 79.7 | 57.7 |

### 2. De-duplicated protocol (stratified 70/15/15, no shared texts)

Sizes {'train': 1353, 'val': 290, 'test': 290}; exact train/test overlap 0; near-duplicates (TF-IDF cosine >= 0.9) 0 of 290.

| Model | Accuracy | Macro F1 [95% bootstrap CI] | Notes |
|---|---:|---:|---|
| Majority class | 22.1 | 5.2 [4.3, 6.1] |  |
| TF-IDF + LogReg | 90.7 | 88.1 [82.7, 92.8] | C=30.0 chosen on validation |
| BiLSTM (retrained) | 74.1 | 58.5 [53.4, 63.1] | seed 42 shown; mean±sd over seeds [42, 43, 44]: F1 56.1±2.0 |

Per-class F1 (de-duplicated test):

| Class | Support | TF-IDF + LogReg | BiLSTM (retrained) |
|---|---:|---:|---:|
| automobile | 11 | 73.7 | 0.0 |
| entertainment | 64 | 94.6 | 93.4 |
| politics | 35 | 95.5 | 64.9 |
| science | 20 | 85.0 | 25.0 |
| sports | 55 | 95.5 | 91.3 |
| technology | 43 | 82.4 | 57.4 |
| world | 62 | 89.9 | 77.9 |
