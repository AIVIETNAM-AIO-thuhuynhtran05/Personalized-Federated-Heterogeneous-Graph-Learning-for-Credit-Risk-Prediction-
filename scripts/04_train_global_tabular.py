"""Train strict global-first centralized tabular benchmarks."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.global_tabular import train_global_tabular


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--application",
        type=Path,
        default=ROOT / "data/interim/global_cleaned_tables/application_train.csv",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=ROOT / "data/processed/global/customer_split.csv",
    )
    parser.add_argument(
        "--encoder",
        type=Path,
        default=ROOT / "data/processed/global/global_encoder.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/global_tabular",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("logistic_regression", "hist_gradient_boosting"),
        default=("logistic_regression",),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    missing = [path for path in (args.application, args.split, args.encoder)
               if not path.is_file()]
    if missing:
        parser.error("Missing required input files:\n  " + "\n  ".join(map(str, missing)))

    rows = train_global_tabular(
        table_path=args.application,
        split_path=args.split,
        encoder_path=args.encoder,
        output_dir=args.output_dir,
        model_names=args.models,
        seed=args.seed,
    )
    print("\nStrict centralized tabular Test results:")
    for row in rows:
        if row["subset"] == "test":
            print(
                f"{row['model']} [{row['threshold_rule']}]: "
                f"threshold={row['threshold']:.6f}, "
                f"ROC-AUC={row['roc_auc']:.6f}, PR-AUC={row['pr_auc']:.6f}, "
                f"F1={row['f1']:.6f}, Brier={row['brier_score']:.6f}"
            )


if __name__ == "__main__":
    main()
