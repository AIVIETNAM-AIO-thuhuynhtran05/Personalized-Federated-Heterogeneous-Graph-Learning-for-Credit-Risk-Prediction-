"""Train one strict centralized HeteroGNN over global graph shards."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.global_gnn import train_global_gnn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-dir", type=Path,
        default=ROOT / "data/processed/global/graphs"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results/global_gnn"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    if not (args.graph_dir / "manifest.json").is_file():
        parser.error(
            f"Missing {args.graph_dir / 'manifest.json'}. "
            "Run scripts/03_build_global_graph.py first."
        )

    rows = train_global_gnn(
        graph_dir=args.graph_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden=args.hidden,
        layers=args.layers,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        threads=args.threads,
    )
    print("\nStrict centralized HeteroGNN results:")
    for row in rows:
        if row["subset"] == "test":
            print(
                f"{row['threshold_rule']}: threshold={row['threshold']:.6f}, "
                f"ROC-AUC={row['roc_auc']:.6f}, PR-AUC={row['pr_auc']:.6f}, "
                f"F1={row['f1']:.6f}, Brier={row['brier_score']:.6f}"
            )


if __name__ == "__main__":
    main()
