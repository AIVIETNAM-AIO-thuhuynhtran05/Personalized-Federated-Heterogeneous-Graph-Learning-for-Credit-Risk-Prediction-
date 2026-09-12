"""Fit one shared transformer from pooled local-train rows only."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocessing.shared_encoder import fit_shared_encoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/processed/clients")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/shared_encoder.json")
    args = parser.parse_args()
    import json
    report_path = args.input_dir / "partition_report.json"
    if not report_path.is_file():
        parser.error("Run scripts/02_partition_clients.py first: partition_report.json is missing.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if any(key not in report for key in ("strategy", "alpha", "seed", "num_clients")):
        parser.error("Old partition output detected. Rerun scripts/02_partition_clients.py to create local train/test splits.")
    missing = [client for client in report["clients"] if not (args.input_dir / client / "customer_split.csv").is_file()]
    if missing:
        parser.error(f"Missing customer_split.csv for {missing}. Run scripts/02_partition_clients.py first.")
    fit_shared_encoder(args.input_dir, args.output)
