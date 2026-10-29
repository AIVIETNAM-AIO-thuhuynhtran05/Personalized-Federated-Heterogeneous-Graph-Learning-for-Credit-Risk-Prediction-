"""Đo concept shift giữa các client của một kịch bản chia (dùng lại mô hình đã train).

  1. Ma trận chéo: Local-only của client i dự đoán Test của client j.
  2. Hiệu chỉnh theo client trên Test: hệ số chặn / độ dốc riêng theo client có cần không (LRT).
  3. Kiểm định tương tác bằng LightGBM: thêm mã client vào đặc trưng bảng có giúp không,
     khi có và khi bỏ các cột vùng (vì client được suy ra từ cột vùng).

    python scripts/06_concept_shift.py [--scenario NAME]
"""
import argparse
import gc
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.evaluation.concept_shift import (bootstrap_slopes, calibration_shift_test, cross_client_summary,
                                          paired_bootstrap_by_client, per_client_auc)
from src.federated.client import FLClient
from src.models.baselines import train_lightgbm
from src.models.hetero_gnn import HeteroGNN
from src.preprocessing.feature_transformer import TARGET, split_feature_types
from src.utils.io import load_config, load_scenario, resolve, save_json
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("concept_shift")

REGION_COLS = ["REGION_POPULATION_RELATIVE", "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY"]


def cross_client_matrix(paths, assign, cfg, mcfg) -> tuple[pd.DataFrame, dict]:
    clients = []
    for k in sorted(assign["client"].unique()):
        data = torch.load(paths["graphs"] / f"client_{k}.pt", weights_only=False)
        clients.append(FLClient(int(k), data, assign[assign["client"] == k], mcfg["batch_size"], cfg["seed"]))
    d0 = clients[0].data
    in_dims = {t: d0[t].x.size(1) for t in d0.node_types}

    def load(state):
        m = HeteroGNN(in_dims, d0.edge_types, mcfg["hidden"], mcfg["num_layers"], mcfg["dropout"])
        m.load_state_dict(state)
        return m.eval()

    local = torch.load(paths["checkpoints"] / "federated_local.pt")
    models = {f"Local-only C{k}": load(local[k]) for k in sorted(local)}
    models["FedAvg"] = load(torch.load(paths["checkpoints"] / "federated_fedavg.pt"))
    rows = {}
    for name, m in models.items():
        rows[name] = {f"C{c.cid}": c.auc(m, "test") for c in clients}
        log.info("%-15s %s", name, {k: round(v, 4) for k, v in rows[name].items()})
    del clients
    gc.collect()
    mat = pd.DataFrame(rows).T
    local_mat = mat.loc[[f"Local-only C{k}" for k in sorted(local)]]
    local_mat.index = mat.columns
    return mat, cross_client_summary(local_mat).to_dict(orient="index")


def _tabular_data(cfg, assign) -> pd.DataFrame:
    processed = resolve(cfg["paths"]["processed_dir"])
    df = pd.read_parquet(processed / cfg["files"]["features"])
    df = df.merge(pd.read_parquet(processed / cfg["files"]["splits"])[["SK_ID_CURR", "split"]], on="SK_ID_CURR")
    df = df.merge(assign[["SK_ID_CURR", "client"]], on="SK_ID_CURR")
    df["CLIENT_ID"] = "C" + df["client"].astype(str)
    return df


def cross_client_matrix_tabular(cfg, assign, rounds, rng) -> tuple[pd.DataFrame, dict, dict]:
    """Như cross_client_matrix nhưng mô hình local là LightGBM train trên Train của từng client
    (early stopping trên Val của client). Dòng tham chiếu: Centralized LightGBM (toàn bộ dữ liệu)."""
    mcfg = load_config("model.yaml")["lightgbm"]
    df = _tabular_data(cfg, assign)
    feats = [c for c in df.columns if c not in ("SK_ID_CURR", TARGET, "split", "client", "CLIENT_ID")]
    _, cat_cols = split_feature_types(df[feats])
    test = df[df["split"] == "test"]
    cpred = pd.read_parquet(resolve(cfg["paths"]["results_dir"]) / "metrics" / "centralized_predictions.parquet")
    test = test.merge(cpred[["SK_ID_CURR", "prob_lgbm"]], on="SK_ID_CURR")
    clients = sorted(df["client"].unique())
    probs = {}
    for k in clients:
        tr = df[(df["split"] == "train") & (df["client"] == k)]
        va = df[(df["split"] == "val") & (df["client"] == k)]
        model, inf = train_lightgbm(tr[feats], tr[TARGET].values, va[feats], va[TARGET].values,
                                    cat_cols, mcfg, cfg["seed"])
        probs[k] = model.predict_proba(test[feats])[:, 1]
        log.info("LightGBM local C%d: n_train %d, best_iter %d", k, len(tr), inf["best_iteration"])
    y, cl = test[TARGET].values, test["client"].values
    rows = {f"Local LightGBM C{i}": {f"C{j}": float(roc_auc_score(y[cl == j], probs[i][cl == j])) for j in clients}
            for i in clients}
    rows["Centralized LightGBM"] = {f"C{j}": float(roc_auc_score(y[cl == j], test["prob_lgbm"].values[cl == j]))
                                    for j in clients}
    mat = pd.DataFrame(rows).T
    local_mat = mat.iloc[:len(clients)].copy()
    local_mat.index = mat.columns
    own = np.zeros(len(test))
    for k in clients:
        own[cl == k] = probs[k][cl == k]
    vs_global = paired_bootstrap_by_client(y, own, test["prob_lgbm"].values, cl, rounds, rng)
    return mat, cross_client_summary(local_mat).to_dict(orient="index"), vs_global


def interaction_test(cfg, assign, rounds, rng) -> dict:
    mcfg = load_config("model.yaml")["lightgbm"]
    df = _tabular_data(cfg, assign)
    parts = {s: df[df["split"] == s] for s in ["train", "val", "test"]}
    y = {s: p[TARGET].values for s, p in parts.items()}
    test_client = parts["test"]["client"].values

    variants = {
        "full": [c for c in df.columns if c not in ("SK_ID_CURR", TARGET, "split", "client", "CLIENT_ID")],
    }
    variants["full+client"] = variants["full"] + ["CLIENT_ID"]
    variants["no_region"] = [c for c in variants["full"] if c not in REGION_COLS]
    variants["no_region+client"] = variants["no_region"] + ["CLIENT_ID"]

    probs, info = {}, {}
    for name, cols in variants.items():
        _, cat_cols = split_feature_types(df[cols])
        model, inf = train_lightgbm(parts["train"][cols], y["train"], parts["val"][cols], y["val"],
                                    cat_cols, mcfg, cfg["seed"])
        probs[name] = model.predict_proba(parts["test"][cols])[:, 1]
        imp = inf["top_features_gain"]
        info[name] = {"best_iteration": inf["best_iteration"], "n_features": len(cols),
                      "client_id_gain_rank": (list(imp).index("CLIENT_ID") + 1) if "CLIENT_ID" in imp else None,
                      "per_client_auc": per_client_auc(y["test"], probs[name], test_client)}
        log.info("LightGBM %-17s best_iter %4d | per-client AUC %s", name, inf["best_iteration"],
                 {k: round(v, 4) for k, v in info[name]["per_client_auc"].items()})
    deltas = {
        "full+client − full": paired_bootstrap_by_client(y["test"], probs["full+client"], probs["full"],
                                                         test_client, rounds, rng),
        "no_region+client − no_region": paired_bootstrap_by_client(y["test"], probs["no_region+client"],
                                                                   probs["no_region"], test_client, rounds, rng),
        "full − no_region": paired_bootstrap_by_client(y["test"], probs["full"], probs["no_region"],
                                                       test_client, rounds, rng),
    }
    return {"variants": info, "delta_auc": deltas}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario")
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--tabular", action="store_true",
                        help="Ma trận chéo bằng LightGBM theo client (không cần graph / mô hình FL đã train)")
    args = parser.parse_args()

    cfg = load_config()
    mcfg = load_config("model.yaml")["hetero_gnn"]
    name, _, paths = load_scenario(args.scenario)
    set_seed(cfg["seed"])
    torch.set_num_threads(mcfg["num_threads"])
    rng = np.random.default_rng(cfg["seed"])
    assign = pd.read_parquet(paths["assignments"])
    log.info("Scenario %s", name)
    report = {"scenario": name}

    # 1. Ma trận chéo
    out_name = "concept_shift_tabular.json" if args.tabular else "concept_shift.json"
    if args.tabular:
        mat, summary, vs_global = cross_client_matrix_tabular(cfg, assign, args.bootstrap, rng)
        report["local_minus_centralized_lgbm"] = vs_global
        log.info("ΔAUC LightGBM local của client − Centralized LightGBM: %s",
                 {c: f"{d['delta_auc']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"
                  for c, d in vs_global.items()})
    else:
        mat, summary = cross_client_matrix(paths, assign, cfg, mcfg)
    report["cross_client_auc"] = mat.round(5).to_dict(orient="index")
    report["cross_client_summary"] = summary
    log.info("Ma trận chéo (dòng = mô hình, cột = Test của client):\n%s", mat.round(4).to_string())
    log.info("Tóm tắt theo client đích:\n%s", pd.DataFrame(summary).T.round(4).to_string())

    # 2. Hiệu chỉnh theo client (dự đoán Centralized luôn có; FedAvg nếu kịch bản đã được train)
    metrics_dir = resolve(cfg["paths"]["results_dir"]) / "metrics"
    cpred = pd.read_parquet(metrics_dir / "centralized_predictions.parquet")
    test = cpred[cpred["split"] == "test"].merge(assign[["SK_ID_CURR", "client"]], on="SK_ID_CURR")
    candidates = [("prob_hgnn", "Centralized HeteroGNN"), ("prob_lgbm", "Centralized LightGBM")]
    fpred_path = paths["metrics"] / "federated_predictions.parquet"
    if fpred_path.exists():
        fpred = pd.read_parquet(fpred_path)
        test = test.merge(fpred.loc[fpred["split"] == "test", ["SK_ID_CURR", "prob_fedavg"]], on="SK_ID_CURR")
        candidates.insert(0, ("prob_fedavg", "FedAvg"))
    report["calibration"] = {}
    for col, label in candidates:
        res = calibration_shift_test(test[TARGET].values, test[col].values, test["client"].values)
        res["slope_ci95"] = bootstrap_slopes(test[TARGET].values, test[col].values, test["client"].values,
                                             args.bootstrap // 2, rng)
        report["calibration"][label] = res
        per = pd.DataFrame(res["per_client"]).T
        per["slope_ci95"] = [f"[{a:.3f}, {b:.3f}]" for a, b in res["slope_ci95"].values()]
        log.info("%s | LRT hệ số chặn: stat %.1f p=%.2g | LRT độ dốc: stat %.1f p=%.2g\n%s", label,
                 res["lrt_intercepts"]["stat"], res["lrt_intercepts"]["p_value"],
                 res["lrt_slopes"]["stat"], res["lrt_slopes"]["p_value"], per.round(4).to_string())
    save_json(report, paths["metrics"] / out_name)

    # 3. Kiểm định tương tác
    report["interaction"] = interaction_test(cfg, assign, args.bootstrap, rng)
    for k, v in report["interaction"]["delta_auc"].items():
        log.info("ΔAUC %s: %s", k, {c: f"{d['delta_auc']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"
                                     for c, d in v.items()})
    save_json(report, paths["metrics"] / out_name)


if __name__ == "__main__":
    main()
