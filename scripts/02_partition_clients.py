"""Partition prepared relational tables, then split customers locally by TARGET."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.partition.relational_partition import partition_tables

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/interim/tables")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/clients")
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--strategy", choices=["dirichlet", "semantic"], default="dirichlet")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--min-per-class", type=int, default=2)
    parser.add_argument("--partition-columns", nargs="+",
                        default=["REGION_RATING_CLIENT_W_CITY", "OCCUPATION_TYPE"])
    args = parser.parse_args()
    report = partition_tables(args.input_dir, args.output_dir, args.num_clients,
                              tuple(args.partition_columns), strategy=args.strategy,
                              alpha=args.alpha, seed=args.seed, test_size=args.test_size,
                              min_per_class=args.min_per_class)
    print(json.dumps(report["tables"], indent=2))

if __name__ == "__main__":
    main()
