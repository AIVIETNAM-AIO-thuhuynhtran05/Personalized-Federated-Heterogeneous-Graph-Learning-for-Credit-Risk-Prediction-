# Bước 1: raw → interim/tables. Lọc cột thiếu trước partition; đây chưa phải lọc chỉ trên local-train.
"""Merge bureau history, then drop columns with >80% missing in each table."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.aggregate_bureau import merge_bureau_balance
from src.preprocessing.clean_application import drop_high_missing_columns

PROTECTED = (
    "SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "TARGET",
    "REGION_RATING_CLIENT_W_CITY", "OCCUPATION_TYPE",
)
TABLES = (
    "application_train", "application_test", "bureau", "previous_application",
    "installments_payments", "POS_CASH_balance", "credit_card_balance",
)


def prepare_tables(raw_dir: Path, output_dir: Path, threshold: float = 0.80) -> dict:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if raw_dir.resolve() == output_dir.resolve():
        raise ValueError("output_dir must differ from raw_dir")
    for name in (*TABLES, "bureau_balance"):
        if not (raw_dir / f"{name}.csv").is_file():
            raise FileNotFoundError(raw_dir / f"{name}.csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"missing_threshold": threshold, "protected_columns": list(PROTECTED), "tables": {}}
    train_dropped = []
    # Read one main table at a time to limit peak memory usage.
    for name in TABLES:
        print(f"Processing {name}...", flush=True)
        df = pd.read_csv(raw_dir / f"{name}.csv")
        if name == "bureau":
            balance = pd.read_csv(raw_dir / "bureau_balance.csv", dtype={"STATUS": "string"})
            df = merge_bureau_balance(df, balance)
            del balance
        before = df.shape[1]
        # Giữ lựa chọn cột nhất quán với application_train thay vì học ngưỡng riêng trên application_test.
        if name == "application_test":
            dropped = [col for col in train_dropped if col in df.columns]
            cleaned = df.drop(columns=dropped)
        else:
            cleaned, dropped = drop_high_missing_columns(df, threshold, PROTECTED)
        if name == "application_train":
            train_dropped = dropped
        output = output_dir / f"{name}.csv"
        cleaned.to_csv(output, index=False)
        report["tables"][name] = {
            "rows": len(cleaned), "columns_before_drop": before,
            "columns_after_drop": cleaned.shape[1], "dropped_columns": dropped,
            "retained_columns": cleaned.columns.tolist(),
            "drop_policy": "application_train_columns" if name == "application_test" else "own_missing_ratio",
            "output": str(output),
        }
        print(f"  Saved {len(cleaned)} rows; dropped {len(dropped)} columns.", flush=True)
        del df, cleaned
    (output_dir / "missing_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/interim/tables")
    parser.add_argument("--missing-threshold", type=float, default=0.80)
    args = parser.parse_args()
    prepare_tables(args.raw_dir, args.output_dir, args.missing_threshold)


if __name__ == "__main__":
    main()
