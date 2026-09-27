# Data

The two CSV files came with the original project package **without documented sources or licences**, so they
are not redistributed here. Place them in this folder (or set `NEWS_DATA_DIR`):

| File | Columns used | Raw rows | Unique texts | Likely origin (unverified) |
|---|---|---:|---:|---|
| `NEWS.csv` | `text`, `label` (`FAKE`/`REAL`) | 3,729 | 2,230 | Indian news. The "fake" items are fact-check write-ups in the style of Indian fact-checking sites, and the "real" items look like Times of India reports. Similar to Kaggle "Indian Fake News" datasets. |
| `CATEGORY.csv` | `news_article`, `news_category` (7 classes) | 4,817 | 1,998 | Inshorts-style 60-word summaries with `news_headline` / `news_article` / `news_category` columns, as in Kaggle "Inshorts news" datasets. |

If you use a public dataset with the same columns instead, check its licence first. The loader only needs the
columns listed above. Labels are lower-cased and texts are de-duplicated before splitting (see `newsclf/data.py`).

Tests and CI do not need these files; they generate synthetic data (`newsclf/synthetic.py`).
