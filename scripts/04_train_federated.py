"""Nhánh Federated với HeteroGNN trên local graph của từng client.

Baseline: Local-only, FedAvg. Personalized FL: FedAvg + fine-tune, FedProx, FedPer, Ditto.
Checkpoint + threshold chọn trên Validation của từng client, đánh giá trên Test của từng client.

    python scripts/04_train_federated.py                                   # tất cả
    python scripts/04_train_federated.py --methods fedper ditto
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from src.evaluation.metrics import evaluate_by_client, select_threshold
from src.federated.client import FLClient
from src.federated.fed_baselines import run_local_only
from src.federated.personalization import run_finetune
from src.federated.server import run_federated
from src.models.hetero_gnn import HeteroGNN
from src.utils.io import load_config, load_scenario, save_json
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("federated")

# fedavg_ft cần FedAvg chạy trước (hoặc đã có checkpoint)
METHODS = ["local", "fedavg", "fedavg_ft", "fedprox", "fedper", "ditto"]


def collect(models: dict, clients, criterion: str):
    """Dự đoán Val/Test của từng client với mô hình tương ứng; threshold chọn trên Val của client."""
    preds, thresholds = [], {}
    for c in clients:
        val, test = c.predict(models[c.cid], "val"), c.predict(models[c.cid], "test")
        thresholds[c.cid] = select_threshold(val["TARGET"], val["prob"], criterion)
        preds += [val, test]
    return pd.concat(preds, ignore_index=True), thresholds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--scenario")
    args = parser.parse_args()

    cfg, fcfg = load_config(), load_config("federated.yaml")
    name, pcfg, paths = load_scenario(args.scenario)
    mcfg = load_config("model.yaml")["hetero_gnn"]
    set_seed(cfg["seed"])
    torch.set_num_threads(mcfg["num_threads"])
    log.info("Scenario: %s", name)

    assign = pd.read_parquet(paths["assignments"])
    clients = []
    for k in sorted(assign["client"].unique()):
        data = torch.load(paths["graphs"] / f"client_{k}.pt", weights_only=False)
        clients.append(FLClient(int(k), data, assign[assign["client"] == k], mcfg["batch_size"], cfg["seed"]))
        log.info("client %d: train/val/test = %d/%d/%d", k, clients[-1].n("train"),
                 clients[-1].n("val"), clients[-1].n("test"))

    d0 = clients[0].data
    in_dims = {t: d0[t].x.size(1) for t in d0.node_types}
    assert all({t: c.data[t].x.size(1) for t in d0.node_types} == in_dims for c in clients), \
        "Client có số chiều đặc trưng khác nhau -> encoder chưa thống nhất"

    def model_fn():
        torch.manual_seed(cfg["seed"])  # cùng khởi tạo cho mọi phương pháp / client
        return HeteroGNN(in_dims, d0.edge_types, mcfg["hidden"], mcfg["num_layers"], mcfg["dropout"])

    out_path = paths["metrics"] / "federated_results.json"
    report = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    preds_path = paths["metrics"] / "federated_predictions.parquet"
    preds = pd.read_parquet(preds_path) if preds_path.exists() else None
    criterion = cfg["threshold"]["criterion"]
    ckpt_dir = paths["checkpoints"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    fedavg_global = None
    for method in args.methods:
        log.info("=== %s ===", method)
        if method == "local":
            models, info = {}, {}
            for c in clients:
                models[c.cid], info[c.cid] = run_local_only(c, model_fn, mcfg, fcfg["local_only"])
        elif method == "fedavg_ft":
            # Fine-tune từ checkpoint FedAvg (của lần chạy này hoặc đã lưu)
            if fedavg_global is None:
                fedavg_global = model_fn()
                fedavg_global.load_state_dict(torch.load(ckpt_dir / "federated_fedavg.pt"))
            models, info = run_finetune(clients, fedavg_global, mcfg, fcfg["fedavg_ft"])
        else:
            gmodel, models, info = run_federated(clients, model_fn, mcfg, fcfg[method], method)
            if method == "fedavg":
                fedavg_global = gmodel
        torch.save({k: m.state_dict() for k, m in models.items()} if method != "fedavg"
                   else fedavg_global.state_dict(), ckpt_dir / f"federated_{method}.pt")

        p, thresholds = collect(models, clients, criterion)
        test = p[p["split"] == "test"]
        report[method] = {"test": evaluate_by_client(test, "prob", thresholds),
                          "thresholds": thresholds, "training": info,
                          "config": {"model": mcfg, "federated": fcfg, "partition": pcfg}}
        save_json(report, out_path)

        p = p.rename(columns={"prob": f"prob_{method}"})
        keys = ["SK_ID_CURR", "client", "split", "TARGET"]
        preds = p if preds is None else preds.drop(columns=[f"prob_{method}"], errors="ignore").merge(
            p, on=keys, how="outer")
        preds.to_parquet(preds_path, index=False)

        pooled = report[method]["test"]["pooled"]
        log.info("%s TEST pooled: AUC %.4f | PR-AUC %.4f | F1 %.4f | client-avg AUC %.4f | worst %.4f",
                 method, pooled["roc_auc"], pooled["pr_auc"], pooled["f1"],
                 pooled["client_avg_roc_auc"], pooled["client_worst_roc_auc"])


if __name__ == "__main__":
    main()
