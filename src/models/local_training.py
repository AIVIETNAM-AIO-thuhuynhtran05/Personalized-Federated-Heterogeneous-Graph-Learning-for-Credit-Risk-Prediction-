"""Independent client GNN training using shared preprocessing and fixed epochs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.models.hetero_gnn import HeteroGNN
from src.graph.graph_dataset import load_graph
from src.graph.validate_training_graph import validate_training_graph
from src.federated.client import train_local, predict_test
from src.evaluation.metrics import auc_metrics


def train_clients(graph_dir: Path, output_dir: Path, clients=None, epochs=20, batch_size=128,
                  learning_rate=1e-3, hidden=32, layers=2, seed=42, device="auto", threads=4):
    if min(epochs, batch_size, hidden, layers, threads) < 1 or learning_rate <= 0:
        raise ValueError("Training settings must be positive")
    manifest_path = graph_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Encoded graph manifest missing. Run partition, 02b_fit_encoder.py, then 03_build_graphs.py first.")
    manifest = json.loads(manifest_path.read_text())
    selected = manifest["clients"] if clients is None else list(clients)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(manifest["clients"]):
        raise ValueError("Select unique client names from the graph manifest")
    for name in selected:
        if not (graph_dir / f"{name}.npz").is_file():
            raise FileNotFoundError(f"Missing encoded graph {name}.npz; rebuild graphs first")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use --device cpu")
    torch.set_num_threads(threads)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = {"graph_dir": str(graph_dir.resolve()), "clients": selected, "epochs": epochs,
                "batch_size": batch_size, "learning_rate": learning_rate, "hidden": hidden,
                "layers": layers, "seed": seed, "device": device, "threads": threads,
                "checkpoint_selection": "fixed_final_epoch", "algorithm": "independent_local_gnn",
                "encoder_fingerprint": manifest["encoder_fingerprint"]}
    (output_dir / "run_config.json").write_text(json.dumps(settings, indent=2))
    summary, expected_config = [], None
    for name in selected:
        print(f"Loading {name}...", flush=True)
        graph = load_graph(graph_dir / f"{name}.npz")
        if graph["metadata"]["client"] != name:
            raise ValueError("Graph client identity differs from filename")
        config = {**validate_training_graph(graph, manifest), "hidden": hidden, "layers": layers}
        if expected_config is not None and config != expected_config:
            raise ValueError("Clients have incompatible feature dimensions or relations")
        expected_config = config
        n_train, n_test = int(graph["train_mask"].sum()), int(graph["test_mask"].sum())
        if n_train == 0:
            summary.append({"client": name, "status": "skipped_no_train", "train_count": 0, "test_count": n_test,
                            "auc": None, "roc_auc": None, "pr_auc": None, "average_precision": None})
        else:
            # Identical initialization across clients; independent optimizers and weights.
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = HeteroGNN(**config).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            directory = output_dir / name
            directory.mkdir(parents=True, exist_ok=True)
            history = []
            for epoch in range(1, epochs + 1):
                stats = {}
                train_local(model, graph, 1, batch_size, learning_rate, seed + epoch,
                            device, optimizer, stats)
                y, probabilities = predict_test(model, graph, batch_size, device)
                metrics = auc_metrics(y, probabilities)
                history.append({"epoch": epoch, **stats, **metrics})
                pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
                print(f"{name} epoch {epoch}/{epochs}: loss={stats['train_loss']:.5f}, ROC-AUC={metrics['roc_auc']}, PR-AUC={metrics['pr_auc']}", flush=True)
            torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "config": config, "epoch": epochs, "client": name,
                        "encoder_fingerprint": manifest["encoder_fingerprint"],
                        "graph_schema_version": graph["metadata"]["graph_schema_version"],
                        "checkpoint_selection": "fixed_final_epoch"}, directory / "model.pt")
            pd.DataFrame({"SK_ID_CURR": graph["customer_ids"][graph["test_mask"]],
                          "TARGET": y.astype(int), "probability_default": probabilities}).to_csv(directory / "test_predictions.csv", index=False)
            summary.append({"client": name, "status": "trained", "train_count": n_train,
                            "final_epoch": epochs, **metrics})
            del model, optimizer
        pd.DataFrame(summary).to_csv(output_dir / "summary.csv", index=False)
        del graph
    return summary
