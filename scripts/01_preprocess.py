# Tiện ích application riêng: làm sạch và tạo đặc trưng; không phải bước chuẩn bị sáu bảng graph.
"""Clean application_train and create domain features."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocessing.clean_application import clean_application
from src.preprocessing.feature_transformer import engineer_application_features

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/application_train.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/interim/application_features.csv")
    parser.add_argument("--missing-threshold", type=float, default=0.80)
    args = parser.parse_args()
    cleaned, report = clean_application(pd.read_csv(args.input), args.missing_threshold)
    featured = engineer_application_features(cleaned)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    featured.to_csv(args.output, index=False)
    report["output"] = str(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
