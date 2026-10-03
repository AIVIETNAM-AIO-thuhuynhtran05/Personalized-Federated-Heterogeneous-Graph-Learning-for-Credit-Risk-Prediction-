"""Centralized HeteroGNN trên global heterogeneous graph.

Train trên khách hàng split=train, early stopping + threshold trên split=val,
đánh giá một lần trên split=test (cùng split với LR/LightGBM).

    python scripts/04b_train_centralized_gnn.py
"""
import copy
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.evaluation.metrics import evaluate, select_threshold
from src.graph.graph_split import CustomerSubgraphLoader
from src.graph.schema import CUSTOMER
from src.models.hetero_gnn import HeteroGNN
from src.utils.io import load_config, resolve, save_json
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("centralized_gnn")


@torch.no_grad()
def predict(model, loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ids, probs = [], []
    for batch in loader:
        probs.append(torch.sigmoid(model(batch.x_dict, batch.edge_index_dict)))
        ids.append(batch[CUSTOMER].n_id)
    return torch.cat(ids).numpy(), torch.cat(probs).numpy()


def main():
    cfg, gcfg = load_config(), load_config("graph.yaml")
    mcfg = load_config("model.yaml")["hetero_gnn"]
    set_seed(cfg["seed"])
    torch.set_num_threads(mcfg["num_threads"])
    processed, results = resolve(cfg["paths"]["processed_dir"]), resolve(cfg["paths"]["results_dir"])

    data = torch.load(processed / gcfg["files"]["centralized_graph"], weights_only=False)
    splits = pd.read_parquet(processed / cfg["files"]["splits"])
    pos = pd.Series(np.arange(data[CUSTOMER].num_nodes), index=data[CUSTOMER].sk_id_curr.numpy())
    idx = {s: pos.loc[splits.loc[splits["split"] == s, "SK_ID_CURR"]].values for s in ["train", "val", "test"]}
    y_all = data[CUSTOMER].y.numpy()

    bs = mcfg["batch_size"]
    train_loader = CustomerSubgraphLoader(data, idx["train"], bs, shuffle=True, seed=cfg["seed"])
    eval_loaders = {s: CustomerSubgraphLoader(data, idx[s], bs * 4) for s in ["val", "test"]}

    model = HeteroGNN({t: data[t].x.size(1) for t in data.node_types}, data.edge_types,
                      mcfg["hidden"], mcfg["num_layers"], mcfg["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=mcfg["lr"], weight_decay=mcfg["weight_decay"])
    loss_fn = torch.nn.BCEWithLogitsLoss()
    log.info("Params: %d | train/val/test customers: %s", sum(p.numel() for p in model.parameters()),
             {s: len(v) for s, v in idx.items()})

    best_auc, best_state, best_epoch, bad, history = -1.0, None, 0, 0, []
    t0 = time.time()
    for epoch in range(1, mcfg["max_epochs"] + 1):
        model.train()
        total, n = 0.0, 0
        for batch in train_loader:
            opt.zero_grad()
            logits = model(batch.x_dict, batch.edge_index_dict)
            loss = loss_fn(logits, batch[CUSTOMER].y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(logits)
            n += len(logits)
        ids, p = predict(model, eval_loaders["val"])
        val_auc = roc_auc_score(y_all[ids], p)
        history.append({"epoch": epoch, "train_loss": total / n, "val_auc": val_auc})
        log.info("epoch %2d | loss %.4f | val AUC %.5f | %.0fs", epoch, total / n, val_auc, time.time() - t0)
        if val_auc > best_auc:
            best_auc, best_state, best_epoch, bad = val_auc, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= mcfg["patience"]:
                log.info("Early stopping (best epoch %d)", best_epoch)
                break
    train_time = time.time() - t0
    model.load_state_dict(best_state)

    out = {s: predict(model, eval_loaders[s]) for s in ["val", "test"]}
    thr = select_threshold(y_all[out["val"][0]], out["val"][1], cfg["threshold"]["criterion"])

    metrics_path = results / "metrics" / "centralized_baselines.json"
    report = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    report["hetero_gnn"] = {
        "val": evaluate(y_all[out["val"][0]], out["val"][1], thr),
        "test": evaluate(y_all[out["test"][0]], out["test"][1], thr),
        "threshold_criterion": cfg["threshold"]["criterion"],
        "train_time_sec": round(train_time, 1),
        "best_epoch": best_epoch,
        "history": history,
        "config": mcfg,
        "orphan_policy": gcfg["orphan_policy"],
    }
    save_json(report, metrics_path)

    preds_path = results / "metrics" / "centralized_predictions.parquet"
    preds = pd.read_parquet(preds_path)
    sk = data[CUSTOMER].sk_id_curr.numpy()
    p_gnn = pd.Series(np.concatenate([out["val"][1], out["test"][1]]),
                      index=sk[np.concatenate([out["val"][0], out["test"][0]])])
    preds["prob_hgnn"] = preds["SK_ID_CURR"].map(p_gnn).values
    preds.to_parquet(preds_path, index=False)

    ckpt = results / "checkpoints" / "centralized_hetero_gnn.pt"
    torch.save({"state_dict": best_state, "config": mcfg,
                "in_dims": {t: data[t].x.size(1) for t in data.node_types}}, ckpt)

    cols = ["roc_auc", "pr_auc", "ks", "f1", "precision", "recall", "threshold"]
    table = pd.DataFrame({m: {c: r["test"][c] for c in cols} for m, r in report.items()}).T
    log.info("TEST results (threshold chọn trên Validation):\n%s", table.round(4).to_string())


if __name__ == "__main__":
    main()
