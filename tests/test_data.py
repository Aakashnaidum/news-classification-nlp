import pandas as pd

from newsclf.data import audit, clean_split, deduplicate, load_task, normalise, original_split


def test_normalise_collapses_case_and_whitespace():
    assert normalise("  Hello\n\nWORLD ") == "hello world"
    assert normalise(None) == ""


def test_load_task_uses_expected_columns(data_dir):
    df = load_task("category", data_dir)
    assert list(df.columns) == ["text", "label", "key"]
    assert set(df["label"]) == {"politics", "sports", "technology"}
    assert set(load_task("fake_news", data_dir)["label"]) == {"fake", "real"}  # lower-cased


def test_dedup_drops_duplicates_and_conflicting_texts():
    df = pd.DataFrame({"text": ["a", "A ", "b", "b", "c"], "label": ["x", "x", "x", "y", "y"]})
    df["key"] = df["text"].map(normalise)
    clean, info = deduplicate(df)
    assert sorted(clean["key"]) == ["a", "c"]
    assert info["dropped_conflicting_texts"] == 1


def test_clean_split_has_no_shared_texts(data_dir):
    df = load_task("category", data_dir)
    splits, info = clean_split(df)
    keys = [set(s["key"]) for s in splits.values()]
    assert not (keys[0] & keys[1]) and not (keys[0] & keys[2]) and not (keys[1] & keys[2])
    assert info["exact_overlap"] == {"train_test": 0, "train_val": 0}


def test_original_split_leaks_duplicates(data_dir):
    df = load_task("fake_news", data_dir)
    assert audit(df)["duplicate_rows"] == 10
    train, test = original_split(df)
    assert len(train) + len(test) == len(df)
