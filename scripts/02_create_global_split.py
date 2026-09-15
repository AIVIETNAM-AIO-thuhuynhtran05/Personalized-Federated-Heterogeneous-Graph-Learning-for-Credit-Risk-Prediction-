"""Create strict centralized global train/validation/test customer split."""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.split.global_split import save_global_split


def main():
    parser = argparse.ArgumentParser(description="Create a stratified global customer split.")
    parser.add_argument(
        "--application",
        type=Path,
        default=Path("data/interim/tables/application_train.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/global")
    )
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = save_global_split(
        args.application,
        args.output_dir,
        args.train_size,
        args.val_size,
        args.test_size,
        args.seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
