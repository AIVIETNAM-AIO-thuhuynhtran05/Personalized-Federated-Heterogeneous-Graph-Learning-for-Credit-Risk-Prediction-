import importlib.util
from pathlib import Path

import pandas as pd
import tempfile
import unittest

from src.preprocessing.aggregate_bureau import merge_bureau_balance


def test_merge_preserves_bureau_rows_and_uses_latest_month():
    bureau = pd.DataFrame({"SK_ID_BUREAU": [10, 20], "SK_ID_CURR": [1, 2]})
    balance = pd.DataFrame({"SK_ID_BUREAU": [10, 10, 10],
                            "MONTHS_BALANCE": [-1, -3, -2], "STATUS": ["C", "2", "0"]})
    result = merge_bureau_balance(bureau, balance)
    assert result["SK_ID_BUREAU"].tolist() == [10, 20]
    assert result.loc[0, "BB_MONTH_COUNT"] == 3
    assert result.loc[0, "BB_OVERDUE_MONTH_COUNT"] == 1
    assert result.loc[0, "BB_STATUS_C_RATIO"] == (1 / 3)
    assert result.loc[0, "BB_LATEST_STATUS"] == "C"
    assert pd.isna(result.loc[1, "BB_MONTH_COUNT"])


def test_prepare_tables_protects_keys_and_matches_test_columns(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/01_prepare_tables.py"
    spec = importlib.util.spec_from_file_location("prepare_tables", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = tmp_path / "raw"
    raw.mkdir()
    frame = pd.DataFrame({"SK_ID_CURR": range(5), "SK_ID_PREV": [None] * 5,
                          "keep_80": [1, None, None, None, None], "drop": [None] * 5})
    for name in module.TABLES:
        data = frame.copy()
        if name == "bureau":
            data["SK_ID_BUREAU"] = range(5)
        if name == "application_test":
            data["drop"] = 1
            data["keep_80"] = None
        data.to_csv(raw / f"{name}.csv", index=False)
    pd.DataFrame({"SK_ID_BUREAU": range(5), "MONTHS_BALANCE": [-1] * 5,
                  "STATUS": ["0"] * 5}).to_csv(raw / "bureau_balance.csv", index=False)
    report = module.prepare_tables(raw, tmp_path / "output")
    for name in module.TABLES:
        result = pd.read_csv(tmp_path / "output" / f"{name}.csv")
        assert len(result) == 5
        assert "SK_ID_PREV" in result
        assert "keep_80" in result
        assert "drop" not in result
    assert report["tables"]["bureau"]["columns_before_drop"] > len(frame.columns)
    assert (tmp_path / "output/missing_report.json").is_file()


class PrepareTablesTests(unittest.TestCase):
    def test_merge(self):
        test_merge_preserves_bureau_rows_and_uses_latest_month()

    def test_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            test_prepare_tables_protects_keys_and_matches_test_columns(Path(directory))


if __name__ == "__main__":
    unittest.main()
