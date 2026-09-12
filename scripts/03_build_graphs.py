# Bước 3: bảng từng client + encoder → graph NPZ và manifest; không fit encoder trong bước này.
"""Build one encoded six-type graph with Customer train/test masks per client."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.graph.heterograph_builder import build_client_graph
from src.graph.schema import GRAPH_SCHEMA_VERSION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/processed/clients")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/graphs")
    parser.add_argument("--encoder", type=Path, default=ROOT / "data/processed/shared_encoder.json")
    args = parser.parse_args()
    if not args.encoder.is_file():
        parser.error(f"Missing encoder: {args.encoder}. Run scripts/02_partition_clients.py, then scripts/02b_fit_encoder.py before building graphs.")
    if not (args.input_dir / "partition_report.json").is_file():
        parser.error("Missing partition_report.json. Run scripts/02_partition_clients.py first.")
    report = json.loads((args.input_dir / "partition_report.json").read_text(encoding="utf-8"))
    encoder = json.loads(args.encoder.read_text(encoding="utf-8"))
    # Dùng danh sách manifest partition thay vì quét thư mục, tránh lấy nhầm client cũ còn trên đĩa.
    for client in report["clients"]:
        print(f"Building {client}...", flush=True)
        build_client_graph(args.input_dir / client, args.output_dir / client, report["schema"], encoder)
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "clients": report["clients"], "encoder_fingerprint": encoder["fingerprint"],
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "partition": {key: report[key] for key in ("strategy", "alpha", "seed", "num_clients")}
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
