# Bước 4: gọi thí nghiệm FedAvg kèm local-only; các tham số CLI quyết định ngân sách train.
"""Train FedAvg and independent local-only GNNs; log both AUC readouts each round."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.federated.server import run_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, default=ROOT / "data/processed/graphs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/default")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run_experiment(**vars(args))
