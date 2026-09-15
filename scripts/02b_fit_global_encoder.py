"""Fit the strict centralized encoder from global Train customers only."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.global_encoder import fit_global_encoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table-dir", type=Path,
        default=ROOT / "data/interim/global_cleaned_tables"
    )
    parser.add_argument(
        "--split", type=Path,
        default=ROOT / "data/processed/global/customer_split.csv"
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/processed/global/global_encoder.json"
    )
    parser.add_argument("--missing-threshold", type=float, default=0.80)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--max-categories", type=int, default=1_000)
    args = parser.parse_args()
    fit_global_encoder(
        args.table_dir, args.split, args.output, args.missing_threshold,
        args.chunksize, args.max_categories
    )


if __name__ == "__main__":
    main()
