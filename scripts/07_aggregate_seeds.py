"""Tổng hợp kết quả qua nhiều seed huấn luyện (cùng một cách chia client).

Seed mặc định (config.yaml) đọc từ thư mục gốc, seed khác đọc từ .../seeds/seed_<n>/.
Với mỗi seed tính ROC-AUC trên Test của từng client, rồi báo cáo mean ± std qua các seed
và ΔAUC giữa các cặp phương pháp kèm số seed có chênh lệch dương.

    python scripts/07_aggregate_seeds.py --seeds 42 1 2 [--scenario NAME]
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.utils.io import load_config, load_scenario, resolve, save_json, seed_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("aggregate_seeds")

LABELS = {"hgnn": "Centralized HeteroGNN", "local": "Local-only", "fedavg": "FedAvg",
          "fedavg_ft": "FedAvg + Fine-tune", "fedprox": "FedProx", "fedper": "FedPer", "ditto": "Ditto"}
PAIRS = [("fedavg", "local"), ("fedavg", "hgnn"), ("local", "hgnn")] + \
        [(m, ref) for m in ["fedavg_ft", "fedprox", "fedper", "ditto"] for ref in ["fedavg", "hgnn"]]


def load_test(seed: int, cfg: dict, paths: dict, assign: pd.DataFrame) -> pd.DataFrame:
    """Dự đoán Test của mọi phương pháp ở một seed: SK_ID_CURR, client, TARGET, prob_<method>."""
    central = seed_dir(resolve(cfg["paths"]["results_dir"]) / "metrics", seed, cfg["seed"])
    fed = seed_dir(paths["metrics"], seed, cfg["seed"])
    c = pd.read_parquet(central / "centralized_predictions.parquet")
    c = c.loc[c["split"] == "test", ["SK_ID_CURR", "TARGET", "prob_hgnn"]]
    f = pd.read_parquet(fed / "federated_predictions.parquet")
    f = f.loc[f["split"] == "test"].drop(columns=["split", "TARGET", "client"])
    test = c.merge(assign, on="SK_ID_CURR").merge(f, on="SK_ID_CURR", validate="one_to_one")
    assert len(test) == len(c), f"seed {seed}: Test của các client phải phủ đúng Test chung"
    return test


def per_client_auc(test: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """Bảng AUC: hàng = phương pháp, cột = client + client-avg (trọng số n_test) + worst."""
    groups = dict(tuple(test.groupby("client")))
    n = np.array([len(g) for g in groups.values()])
    rows = {}
    for m in methods:
        auc = {k: roc_auc_score(g["TARGET"], g[f"prob_{m}"]) for k, g in groups.items()}
        rows[m] = {**{f"C{k}": v for k, v in auc.items()},
                   "client-avg": float(np.average(list(auc.values()), weights=n)), "worst": min(auc.values())}
    return pd.DataFrame(rows).T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()

    cfg = load_config()
    name, _, paths = load_scenario(args.scenario)
    assign = pd.read_parquet(paths["assignments"])[["SK_ID_CURR", "client"]]

    tests = {s: load_test(s, cfg, paths, assign) for s in args.seeds}
    # Chỉ giữ phương pháp có ở mọi seed
    methods = [m for m in LABELS if all(f"prob_{m}" in t for t in tests.values())]
    log.info("Scenario %s | seeds %s | methods %s", name, args.seeds, methods)

    auc = {s: per_client_auc(t, methods) for s, t in tests.items()}
    stack = np.stack([auc[s].loc[methods].values for s in args.seeds])  # seed × method × cột
    cols = auc[args.seeds[0]].columns
    mean = pd.DataFrame(stack.mean(0), index=methods, columns=cols)
    std = pd.DataFrame(stack.std(0, ddof=1), index=methods, columns=cols)

    pairs = [(a, b) for a, b in PAIRS if a in methods and b in methods]
    delta = {}
    for a, b in pairs:
        d = np.stack([(auc[s].loc[a] - auc[s].loc[b]).values for s in args.seeds])  # seed × cột
        delta[f"{a}_vs_{b}"] = {col: {"per_seed": dict(zip(args.seeds, d[:, j].tolist())),
                                      "mean": float(d[:, j].mean()), "std": float(d[:, j].std(ddof=1)),
                                      "n_positive": int((d[:, j] > 0).sum())}
                                for j, col in enumerate(cols)}

    out_dir = paths["metrics"] / "seeds"
    save_json({"scenario": name, "seeds": args.seeds, "methods": methods,
               "auc_per_seed": {s: auc[s].to_dict(orient="index") for s in args.seeds},
               "auc_mean": mean.to_dict(orient="index"), "auc_std": std.to_dict(orient="index"),
               "delta_auc": delta}, out_dir / "aggregate.json")

    fmt = mean.map(lambda v: f"{v:.4f}") + " ± " + std.map(lambda v: f"{v:.4f}")
    fmt.index = [LABELS[m] for m in methods]
    k = len(args.seeds)
    dfmt = pd.DataFrame({f"{LABELS[a]} − {LABELS[b]}": {
        col: f"{v['mean']:+.4f} ± {v['std']:.4f} ({v['n_positive']}/{k} dương)"
        for col, v in delta[f"{a}_vs_{b}"].items()} for a, b in pairs}).T
    fmt.to_csv(out_dir / "aggregate_auc.csv")
    dfmt.to_csv(out_dir / "aggregate_delta_auc.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        log.info("ROC-AUC Test theo client, mean ± std qua %d seed:\n%s", k, fmt.to_string())
        log.info("ΔAUC mean ± std (số seed có Δ > 0):\n%s", dfmt.to_string())


if __name__ == "__main__":
    main()
