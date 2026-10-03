"""So sánh mọi phương pháp trên CÙNG Test của từng client trong một kịch bản chia.

Centralized (LR, LightGBM, HeteroGNN) dùng threshold toàn cục chọn trên Val chung;
Local-only/FedAvg dùng threshold chọn trên Val của từng client.
Khoảng tin cậy 95% của chênh lệch AUC: paired bootstrap trên Test của từng client.

    python scripts/05_evaluate.py [--scenario NAME]
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.evaluation.metrics import evaluate_by_client
from src.utils.io import load_config, load_scenario, resolve, save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("evaluate")

CENTRAL = {"lr": "logistic_regression", "lgbm": "lightgbm", "hgnn": "hetero_gnn"}
LABELS = {"lr": "Centralized LR", "lgbm": "Centralized LightGBM", "hgnn": "Centralized HeteroGNN",
          "local": "Local-only", "fedavg": "FedAvg", "fedavg_ft": "FedAvg + Fine-tune",
          "fedprox": "FedProx", "fedper": "FedPer", "ditto": "Ditto"}
FED = ["local", "fedavg", "fedavg_ft", "fedprox", "fedper", "ditto"]
PFL = ["fedavg_ft", "fedprox", "fedper", "ditto"]


def comparison_pairs(available) -> list[tuple[str, str]]:
    """(A, B) -> ΔAUC = AUC(A) - AUC(B). Baseline FL so với nhau và với Centralized;
    mỗi phương pháp Personalized FL so với FedAvg và với Centralized HeteroGNN."""
    pairs = [("fedavg", "local"), ("fedavg", "hgnn"), ("local", "hgnn")]
    pairs += [(m, ref) for m in PFL if m in available for ref in ["fedavg", "hgnn"]]
    return [(a, b) for a, b in pairs if a in available and b in available]


def paired_bootstrap(y, a, b, rounds: int, rng) -> dict:
    """ΔAUC = AUC(a) - AUC(b) trên cùng mẫu; CI 95% bằng bootstrap có ghép cặp."""
    diffs = []
    n = len(y)
    for _ in range(rounds):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        diffs.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    diffs = np.array(diffs)
    return {"delta_auc": float(roc_auc_score(y, a) - roc_auc_score(y, b)),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "p_delta_le_0": float((diffs <= 0).mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario")
    args = parser.parse_args()

    cfg, fcfg = load_config(), load_config("federated.yaml")
    name, _, paths = load_scenario(args.scenario)
    central_dir = resolve(cfg["paths"]["results_dir"]) / "metrics"
    assign = pd.read_parquet(paths["assignments"])[["SK_ID_CURR", "client"]]
    clients = sorted(assign["client"].unique())

    central = json.loads((central_dir / "centralized_baselines.json").read_text(encoding="utf-8"))
    cpred = pd.read_parquet(central_dir / "centralized_predictions.parquet").merge(assign, on="SK_ID_CURR")
    fed = json.loads((paths["metrics"] / "federated_results.json").read_text(encoding="utf-8"))
    fpred = pd.read_parquet(paths["metrics"] / "federated_predictions.parquet")

    fed_methods = [m for m in FED if m in fed and f"prob_{m}" in fpred]
    test = cpred[cpred["split"] == "test"].merge(
        fpred.loc[fpred["split"] == "test", ["SK_ID_CURR"] + [f"prob_{m}" for m in fed_methods]],
        on="SK_ID_CURR", how="inner", validate="one_to_one")
    assert len(test) == (cpred["split"] == "test").sum(), "Test của các client phải phủ đúng Test chung"

    results = {}
    for key, cname in CENTRAL.items():
        thr = central[cname]["test"]["threshold"]
        results[key] = evaluate_by_client(test, f"prob_{key}", {k: thr for k in clients})
    for key in fed_methods:
        thr = {int(k): v for k, v in fed[key]["thresholds"].items()}
        results[key] = evaluate_by_client(test, f"prob_{key}", thr)
    PAIRS = comparison_pairs(results)

    rng = np.random.default_rng(cfg["seed"])
    boot = {}
    for a, b in PAIRS:
        boot[f"{a}_vs_{b}"] = {
            int(k): paired_bootstrap(g["TARGET"].values, g[f"prob_{a}"].values, g[f"prob_{b}"].values,
                                     fcfg["evaluation"]["bootstrap_rounds"], rng)
            for k, g in test.groupby("client")}
    save_json({"scenario": name, "results": results, "bootstrap": boot}, paths["metrics"] / "comparison.json")

    sizes = test.groupby("client").size()
    per_auc = pd.DataFrame({LABELS[k]: {f"client {c} (n_test={sizes[c]})": r["per_client"][c]["roc_auc"]
                                        for c in clients} for k, r in results.items()}).T
    per_auc["client-avg"] = [r["pooled"]["client_avg_roc_auc"] for r in results.values()]
    per_auc["worst"] = [r["pooled"]["client_worst_roc_auc"] for r in results.values()]
    per_pr = pd.DataFrame({LABELS[k]: {f"client {c}": r["per_client"][c]["pr_auc"] for c in clients}
                           for k, r in results.items()}).T
    per_f1 = pd.DataFrame({LABELS[k]: {f"client {c}": r["per_client"][c]["f1"] for c in clients}
                           for k, r in results.items()}).T
    delta = pd.DataFrame({f"{LABELS[a]} − {LABELS[b]}": {
        f"client {c}": f"{v['delta_auc']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]"
        for c, v in boot[f"{a}_vs_{b}"].items()} for a, b in PAIRS}).T

    per_auc.round(4).to_csv(paths["metrics"] / "comparison_per_client_auc.csv")
    delta.to_csv(paths["metrics"] / "comparison_delta_auc_ci.csv")
    log.info("Scenario %s", name)
    log.info("ROC-AUC theo client:\n%s", per_auc.round(4).to_string())
    log.info("PR-AUC theo client:\n%s", per_pr.round(4).to_string())
    log.info("F1 theo client (threshold chọn trên Val):\n%s", per_f1.round(4).to_string())
    log.info("ΔAUC [CI 95%% paired bootstrap]:\n%s", delta.to_string())


if __name__ == "__main__":
    main()
