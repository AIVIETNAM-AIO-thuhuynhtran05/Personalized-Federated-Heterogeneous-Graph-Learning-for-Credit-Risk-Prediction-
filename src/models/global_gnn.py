"""Train one centralized HeteroGNN over technical global-graph shards."""
from __future__ import annotations

import copy
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.federated.client import train_local
from src.graph.graph_dataset import component_batch, load_graph
from src.models.global_tabular import metrics, select_f1_threshold
from src.models.hetero_gnn import HeteroGNN


def _device(name):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return name


def _config(graph, hidden, layers):
    dimensions = {key[3:]: value.shape[1] for key, value in graph.items()
                  if key.startswith("x__")}
    relations = sorted(key[6:] for key in graph if key.startswith("edge__"))
    return {"dimensions": dimensions, "relations": relations,
            "hidden": hidden, "layers": layers}


def _validate(graph, manifest, config):
    for key in ("train_mask", "val_mask", "test_mask"):
        if key not in graph or graph[key].dtype != bool:
            raise ValueError(f"Missing boolean {key}")
    masks = sum(graph[key].astype(int) for key in
                ("train_mask", "val_mask", "test_mask"))
    if not np.all(masks == 1):
        raise ValueError("Global masks must be disjoint and exhaustive")
    if graph["metadata"]["encoder_fingerprint"] != manifest["encoder_fingerprint"]:
        raise ValueError("Graph encoder fingerprint differs from manifest")
    if _config(graph, config["hidden"], config["layers"]) != config:
        raise ValueError("Incompatible shard dimensions or relations")


def _predict(model, graph, mask_name, batch_size, device):
    customers = np.flatnonzero(graph[mask_name])
    labels, scores, ids = [], [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(customers), batch_size):
            chosen = customers[start:start + batch_size]
            x, edges, y, indices = component_batch(graph, chosen, device)
            labels.extend(y.cpu().numpy())
            scores.extend(torch.sigmoid(model(x, edges)).cpu().numpy())
            ids.extend(graph["customer_ids"][indices])
    return np.asarray(ids), np.asarray(labels), np.asarray(scores)


def _evaluate_shards(model, graph_dir, shards, mask_name, batch_size, device,
                     manifest, config):
    all_ids, all_labels, all_scores = [], [], []
    for filename in shards:
        graph = load_graph(graph_dir / filename)
        _validate(graph, manifest, config)
        ids, labels, scores = _predict(model, graph, mask_name, batch_size, device)
        all_ids.extend(ids); all_labels.extend(labels); all_scores.extend(scores)
        del graph
        gc.collect()
    return np.asarray(all_ids), np.asarray(all_labels), np.asarray(all_scores)


def train_global_gnn(graph_dir: Path, output_dir: Path, epochs=20, batch_size=256,
                     learning_rate=1e-3, hidden=16, layers=2, patience=5,
                     seed=42, device="auto", threads=8):
    if min(epochs, batch_size, hidden, layers, patience, threads) < 1:
        raise ValueError("Training settings must be positive")
    manifest = json.loads((graph_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("artifact") != "sharded_logical_global_graph":
        raise ValueError("Expected a sharded logical global graph")
    shards = list(manifest["shards"])
    device = _device(device)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    first = load_graph(graph_dir / shards[0])
    config = _config(first, hidden, layers)
    _validate(first, manifest, config)
    del first
    model = HeteroGNN(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    output_dir.mkdir(parents=True, exist_ok=True)
    history, best_state, best_pr, best_epoch, stale = [], None, -np.inf, 0, 0
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        order = shards.copy(); rng.shuffle(order)
        weighted_loss = examples = steps = 0
        for index, filename in enumerate(order):
            print(f"Epoch {epoch}/{epochs}: training {filename}...", flush=True)
            graph = load_graph(graph_dir / filename)
            _validate(graph, manifest, config)
            stats = {}
            train_local(model, graph, epochs=1, batch_size=batch_size,
                        learning_rate=learning_rate,
                        seed=seed + epoch * len(shards) + index,
                        device=device, optimizer=optimizer, stats=stats)
            weighted_loss += stats["train_loss"] * stats["examples"]
            examples += stats["examples"]; steps += stats["optimizer_steps"]
            del graph
            gc.collect()
        _, val_y, val_scores = _evaluate_shards(
            model, graph_dir, shards, "val_mask", batch_size, device, manifest, config)
        val_result = metrics(val_y, val_scores, 0.5)
        row = {"epoch": epoch, "train_loss": weighted_loss / examples,
               "train_examples": examples, "optimizer_steps": steps,
               "val_roc_auc": val_result["roc_auc"], "val_pr_auc": val_result["pr_auc"]}
        history.append(row); pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        print(f"Epoch {epoch}: loss={row['train_loss']:.6f}, val_PR-AUC={row['val_pr_auc']:.6f}", flush=True)
        if row["val_pr_auc"] > best_pr:
            best_pr, best_epoch, stale = row["val_pr_auc"], epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break
    model.load_state_dict(best_state)
    val_ids, val_y, val_scores = _evaluate_shards(
        model, graph_dir, shards, "val_mask", batch_size, device, manifest, config)
    threshold = select_f1_threshold(val_y, val_scores)
    test_ids, test_y, test_scores = _evaluate_shards(
        model, graph_dir, shards, "test_mask", batch_size, device, manifest, config)
    rows = []
    for subset, labels, scores in (("validation", val_y, val_scores), ("test", test_y, test_scores)):
        rows.append({"model": "global_heterognn", "subset": subset,
                     "threshold_rule": "validation_f1", **metrics(labels, scores, threshold)})
        rows.append({"model": "global_heterognn", "subset": subset,
                     "threshold_rule": "fixed_0.5", **metrics(labels, scores, 0.5)})
    pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame({"SK_ID_CURR": val_ids, "TARGET": val_y,
                  "probability_default": val_scores}).to_csv(output_dir / "validation_predictions.csv", index=False)
    pd.DataFrame({"SK_ID_CURR": test_ids, "TARGET": test_y,
                  "probability_default": test_scores}).to_csv(output_dir / "test_predictions.csv", index=False)
    torch.save({"model": best_state, "config": config, "best_epoch": best_epoch,
                "threshold": threshold, "encoder_fingerprint": manifest["encoder_fingerprint"]},
               output_dir / "global_gnn.pt")
    run = {"protocol": "strict_global_first", "method": "centralized_heterognn",
           "optimizer_scope": "one_shared_optimizer_across_all_shards",
           "checkpoint_selection": "validation_pr_auc", "threshold_scope": "validation_only",
           "test_scope": "global_test_only", "epochs_requested": epochs,
           "epochs_completed": len(history), "best_epoch": best_epoch,
           "batch_size": batch_size, "learning_rate": learning_rate, "hidden": hidden,
           "layers": layers, "patience": patience, "seed": seed, "device": device,
           "threads": threads, "num_shards": len(shards)}
    (output_dir / "run_config.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    return rows
