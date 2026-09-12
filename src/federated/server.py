# Điều phối cả FedAvg và đối chứng local-only; gộp trọng số theo số Customer train, không theo số node lịch sử.
"""All-client FedAvg versus equally initialized independent local GNNs."""
import copy
import csv
import json
from pathlib import Path
import numpy as np
import torch
from src.graph.graph_dataset import load_graph
from src.models.hetero_gnn import HeteroGNN
from src.federated.client import train_local, predict_test
from src.evaluation.metrics import auc_metrics


def run_experiment(graph_dir: Path, output_dir: Path, rounds=20, local_epochs=1,
                   batch_size=256, learning_rate=1e-3, hidden=32, layers=2, seed=42, device="cpu"):
    if min(rounds, local_epochs, batch_size, hidden, layers) < 1 or learning_rate <= 0:
        raise ValueError("Training parameters must be positive")
    manifest = json.loads((graph_dir / "manifest.json").read_text())
    clients = manifest["clients"]
    if not clients:
        raise ValueError("No clients")
    torch.manual_seed(seed)
    np.random.seed(seed)
    first = load_graph(graph_dir / f"{clients[0]}.npz")
    dimensions = {key[3:]: value.shape[1] for key, value in first.items() if key.startswith("x__")}
    relations = sorted(key[6:] for key in first if key.startswith("edge__"))
    graph_schema_version = first["metadata"].get("graph_schema_version", "legacy")
    if manifest.get("graph_schema_version", graph_schema_version) != graph_schema_version:
        raise ValueError("Graph schema differs from manifest; rebuild all client graphs")
    del first
    config = {"dimensions": dimensions, "relations": relations, "hidden": hidden, "layers": layers}
    model = HeteroGNN(**config).to(device)
    # Giữ bản trọng số khởi tạo chung để các mô hình local-only bắt đầu giống mô hình global.
    initial = copy.deepcopy(model.cpu().state_dict())
    model.to(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    local_dir = output_dir / "local_models"
    local_dir.mkdir(exist_ok=True)
    metadata = {"model": config, "rounds": rounds, "local_epochs": local_epochs,
                "graph_schema_version": graph_schema_version,
                "batch_size": batch_size, "learning_rate": learning_rate, "seed": seed,
                "device": device, "graph_dir": str(graph_dir.resolve()), "manifest": manifest,
                "aggregation": "FedAvg weighted by train customer count",
                "comparison": "same initial weights and same local epochs per round; fixed final round"}
    (output_dir / "run_config.json").write_text(json.dumps(metadata, indent=2))
    rows = []
    for round_id in range(1, rounds + 1):
        # Chụp trạng thái đầu round; tất cả client FedAvg đều xuất phát từ cùng trạng thái này.
        global_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        accumulator = {k: torch.zeros_like(v) for k, v in global_state.items()}
        total = 0
        for i, client in enumerate(clients):
            graph = load_graph(graph_dir / f"{client}.npz")
            if graph["metadata"]["encoder_fingerprint"] != manifest["encoder_fingerprint"]:
                raise ValueError("Graphs use different encoders")
            if graph["metadata"].get("graph_schema_version", "legacy") != graph_schema_version:
                raise ValueError("Mixed graph schemas; rebuild all client graphs")
            # Trọng số gộp là số Customer train; lịch sử nhiều node không làm tăng trọng số client.
            count = int(graph["train_mask"].sum())
            local = HeteroGNN(**config).to(device)
            local.load_state_dict(global_state)
            # Không truyền optimizer nên Adam của FedAvg được reset mỗi round/client.
            train_local(local, graph, local_epochs, batch_size, learning_rate,
                        seed + round_id * len(clients) + i, device)
            for key, value in local.state_dict().items():
                accumulator[key] += value.detach().cpu() * count
            total += count
            path = local_dir / f"{client}.pt"
            standalone = HeteroGNN(**config).to(device)
            optimizer = torch.optim.Adam(standalone.parameters(), lr=learning_rate)
            if round_id == 1:
                standalone.load_state_dict(initial)
            else:
                saved = torch.load(path, map_location=device, weights_only=True)
                standalone.load_state_dict(saved["model"])
                # Local-only giữ cả moment của Adam qua round, khác cơ chế reset của FedAvg ở trên.
                optimizer.load_state_dict(saved["optimizer"])
            train_local(standalone, graph, local_epochs, batch_size, learning_rate,
                        seed + round_id * len(clients) + i, device, optimizer)
            torch.save({"model": standalone.state_dict(), "optimizer": optimizer.state_dict()}, path)
            del graph, local, standalone, optimizer
        if total == 0:
            raise ValueError("No train customers")
        # Chia tổng có trọng số cho tổng Customer train để nhận trọng số global round mới.
        model.load_state_dict({k: v / total for k, v in accumulator.items()})
        pooled = {method: ([], []) for method in ("federated", "local_only")}
        for client in clients:
            graph = load_graph(graph_dir / f"{client}.npz")
            standalone = HeteroGNN(**config).to(device)
            standalone.load_state_dict(torch.load(local_dir / f"{client}.pt", map_location=device,
                                                   weights_only=True)["model"])
            for method, current in (("federated", model), ("local_only", standalone)):
                y, scores = predict_test(current, graph, batch_size, device)
                rows.append({"round": round_id, "client": client, "method": method, **auc_metrics(y, scores)})
                pooled[method][0].extend(y.tolist())
                pooled[method][1].extend(scores.tolist())
            del graph, standalone
        # Pooled AUC tính từ dự đoán ghép lại, không phải trung bình AUC các client.
        for method, (y, scores) in pooled.items():
            rows.append({"round": round_id, "client": "pooled", "method": method, **auc_metrics(y, scores)})
        with (output_dir / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        # Ghi checkpoint round hiện tại; không dùng test để chọn checkpoint tốt nhất.
        torch.save({"model": model.state_dict(), "config": config, "round": round_id}, output_dir / "federated.pt")
        print(f"Round {round_id}/{rounds}: pooled FL ROC-AUC={rows[-2]['roc_auc']}, PR-AUC={rows[-2]['pr_auc']}; local-only ROC-AUC={rows[-1]['roc_auc']}, PR-AUC={rows[-1]['pr_auc']}", flush=True)
    return rows
