# Mỗi cặp số client/seed có partition, encoder, graph và kết quả riêng; không tái dùng encoder khác split.
"""Sweep client count, fitting one fresh pooled-train transformer per experiment."""
import argparse
import json
from pathlib import Path
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.partition.relational_partition import partition_tables
from src.preprocessing.shared_encoder import fit_shared_encoder
from src.graph.heterograph_builder import build_client_graph
from src.graph.schema import GRAPH_SCHEMA_VERSION
from src.federated.server import run_experiment


def sweep(input_dir, output_dir, client_counts=(10, 20, 30, 40, 50), alpha=0.5,
          seeds=(42,), test_size=0.2, min_per_class=2, **training):
    if len(set(client_counts)) != len(client_counts) or len(set(seeds)) != len(seeds):
        raise ValueError("Client counts and seeds must be unique")
    records = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        for count in client_counts:
            # Đường dẫn hiện phân biệt N và seed; chạy lại cùng cặp sẽ dùng lại thư mục đầu ra này.
            root = output_dir / f"n_{count:02d}_seed_{seed}"
            client_dir, graph_dir = root / "clients", root / "graphs"
            report = partition_tables(input_dir, client_dir, count, alpha=alpha, seed=seed,
                                      test_size=test_size, min_per_class=min_per_class)
            # N/seed khác tạo split khác nên phải fit encoder riêng, không dùng encoder từ vòng trước.
            encoder = fit_shared_encoder(client_dir, root / "shared_encoder.json")
            graph_dir.mkdir(parents=True, exist_ok=True)
            for client in report["clients"]:
                build_client_graph(client_dir / client, graph_dir / client, report["schema"], encoder)
            (graph_dir / "manifest.json").write_text(json.dumps({
                "clients": report["clients"], "encoder_fingerprint": encoder["fingerprint"],
                "graph_schema_version": GRAPH_SCHEMA_VERSION,
                "partition": {k: report[k] for k in ("strategy", "alpha", "seed", "num_clients")}}))
            rows = run_experiment(graph_dir, root / "training", seed=seed, **training)
            records.extend({"num_clients": count, "alpha": alpha, "seed": seed, **row} for row in rows)
            pd.DataFrame(records).to_csv(output_dir / "sweep_metrics.csv", index=False)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/interim/tables")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/sweep")
    parser.add_argument("--client-counts", nargs="+", type=int, default=[10, 20, 30, 40, 50])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--min-per-class", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    sweep(**vars(parser.parse_args()))
