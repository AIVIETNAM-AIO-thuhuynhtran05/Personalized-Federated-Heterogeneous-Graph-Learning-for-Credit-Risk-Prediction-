"""Build the strict centralized heterogeneous graph as technical shards."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.graph.global_heterograph_builder import build_global_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=ROOT / "data/interim/global_cleaned_tables",
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
        default=ROOT / "data/processed/global/graphs",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=ROOT / "data/processed/global/graph_staging",
    )
    parser.add_argument("--num-shards", type=int, default=20)
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()

    required = [args.split, args.encoder]
    required.extend(
        args.table_dir / f"{name}.csv"
        for name in (
            "application_train",
            "bureau",
            "previous_application",
            "installments_payments",
            "POS_CASH_balance",
            "credit_card_balance",
        )
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("Missing required input files:\n  " + "\n  ".join(missing))
    if args.output_dir.resolve() == args.staging_dir.resolve():
        parser.error("--output-dir and --staging-dir must be different")

    manifest = build_global_graph(
        table_dir=args.table_dir,
        split_path=args.split,
        encoder_path=args.encoder,
        output_dir=args.output_dir,
        staging_dir=args.staging_dir,
        num_shards=args.num_shards,
        chunksize=args.chunksize,
    )
    print(
        f"Completed {manifest['num_shards']} global graph shards in "
        f"{args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
