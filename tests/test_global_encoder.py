import json
from pathlib import Path

import pandas as pd

from src.partition.relational_partition import TABLE_NAMES
from src.preprocessing.global_encoder import fit_global_encoder


def _fixture(root: Path):
    tables = root / "tables"
    tables.mkdir(parents=True)
    split = pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3, 4, 5, 6],
        "split": ["train", "train", "train", "train", "validation", "test"],
    })
    split_path = root / "customer_split.csv"
    split.to_csv(split_path, index=False)
    for table in TABLE_NAMES:
        frame = pd.DataFrame({
            "SK_ID_CURR": [1, 2, 3, 4, 5, 6],
            "TARGET": [0, 1, 0, 1, 0, 1],
            "numeric": [1.0, 2.0, 3.0, 4.0, 9999.0, -9999.0],
            "category": ["a", "b", "a", "b", "leak", "leak"],
            "mostly_missing": [None, None, None, None, 2.0, 3.0],
        })
        frame.to_csv(tables / f"{table}.csv", index=False)
    return tables, split_path


def _portable(result):
    copy = json.loads(json.dumps(result))
    copy.pop("fingerprint")
    copy.pop("split_file")
    copy.pop("split_hash")
    for table in copy["tables"].values():
        table.pop("source")
    return copy


def test_encoder_uses_train_only_and_drops_by_train_missingness(tmp_path):
    tables, split_path = _fixture(tmp_path / "first")
    first = fit_global_encoder(tables, split_path, tmp_path / "first.json", chunksize=2)
    application = first["tables"]["application_train"]
    assert application["fit_rows"] == 4
    assert application["dropped_columns"] == ["mostly_missing"]
    numeric = next(x for x in application["columns"] if x["column"] == "numeric")
    category = next(x for x in application["columns"] if x["column"] == "category")
    assert numeric["mean"] == 2.5
    assert category["categories"] == ["a", "b"]

    changed_tables, changed_split = _fixture(tmp_path / "changed")
    for table in TABLE_NAMES:
        path = changed_tables / f"{table}.csv"
        frame = pd.read_csv(path)
        frame.loc[frame.SK_ID_CURR.isin([5, 6]), ["numeric", "category"]] = [123456, "changed"]
        frame.to_csv(path, index=False)
    second = fit_global_encoder(
        changed_tables, changed_split, tmp_path / "changed.json", chunksize=3
    )
    assert _portable(first) == _portable(second)


def test_invalid_missing_threshold_is_rejected(tmp_path):
    tables, split_path = _fixture(tmp_path / "invalid")
    try:
        fit_global_encoder(tables, split_path, tmp_path / "x.json", 1.1)
    except ValueError as error:
        assert "missing_threshold" in str(error)
    else:
        raise AssertionError("Expected ValueError")
