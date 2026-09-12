# Đọc checkpoint cuối và graph của lần chạy để tính lại metric; không chọn checkpoint theo test.
"""Re-evaluate final saved federated and local-only checkpoints on local test masks."""
import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.hetero_gnn import HeteroGNN
from src.graph.graph_dataset import load_graph
from src.federated.client import predict_test
from src.evaluation.metrics import auc_metrics


def evaluate(run_dir, device="cpu"):
    settings = json.loads((run_dir / "run_config.json").read_text())
    checkpoint = torch.load(run_dir / "federated.pt", map_location=device, weights_only=True)
    model = HeteroGNN(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    rows = []
    pooled = {method: ([], []) for method in ("federated", "local_only")}
    for client in settings["manifest"]["clients"]:
        graph = load_graph(Path(settings["graph_dir"]) / f"{client}.npz")
        # Ngăn đánh giá checkpoint với encoder/schema khác lần huấn luyện.
        if graph["metadata"]["encoder_fingerprint"] != settings["manifest"]["encoder_fingerprint"]:
            raise ValueError("Graph encoder differs from the training run")
        if graph["metadata"].get("graph_schema_version", "legacy") != settings.get("graph_schema_version", "legacy"):
            raise ValueError("Graph schema differs from checkpoint training schema")
        local = HeteroGNN(**checkpoint["config"]).to(device)
        local.load_state_dict(torch.load(run_dir / "local_models" / f"{client}.pt",
                                         map_location=device, weights_only=True)["model"])
        for method, current in (("federated", model), ("local_only", local)):
            y, scores = predict_test(current, graph, settings["batch_size"], device)
            rows.append({"client": client, "method": method, **auc_metrics(y, scores)})
            pooled[method][0].extend(y.tolist())
            pooled[method][1].extend(scores.tolist())
    # Ghép nhãn và xác suất trước khi tính pooled metric; không lấy mean metric từng client.
    for method, (y, scores) in pooled.items():
        rows.append({"client": "pooled", "method": method, **auc_metrics(y, scores)})
    pd.DataFrame(rows).to_csv(run_dir / "final_evaluation.csv", index=False)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/default")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.run_dir, args.device), indent=2))
