"""Train independent GNNs on all clients or selected clients; no server aggregation."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.local_training import train_clients


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, default=ROOT / "data/processed/graphs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/local_gnn")
    parser.add_argument("--clients", nargs="+", help="For example: client_000 client_001; default: all clients")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    train_clients(**vars(parser.parse_args()))
