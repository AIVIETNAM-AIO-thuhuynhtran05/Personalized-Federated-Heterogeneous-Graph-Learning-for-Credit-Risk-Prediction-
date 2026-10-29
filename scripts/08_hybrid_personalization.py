"""Hybrid Personalized FL: mỗi client trộn dự đoán của mô hình federated (tri thức chung) với
một mô hình riêng của client (tri thức cục bộ), trọng số trộn chọn trên Val của client.

    logit(p_hybrid) = w * logit(p_A) + (1 - w) * logit(p_B),   w ∈ {0, 0.05, ..., 1}

w = 1 hoặc 0 là ứng viên, nên trên Val hybrid không kém mô hình thành phần tốt hơn.
Thành phần cục bộ:
  ditto       : mô hình riêng Ditto (HeteroGNN), đã train bởi 04_train_federated.py
  local_lgbm  : LightGBM train CHỈ trên Train của client (601 đặc trưng bảng), early stopping
                trên Val của client -> không dữ liệu nào rời khỏi client
Các cặp được thử: fedavg+ditto, fedavg+local_lgbm, ditto+local_lgbm.

    python scripts/08_hybrid_personalization.py --seeds 42 1 2 [--scenario NAME]
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.evaluation.metrics import evaluate_by_client, select_threshold
from src.models.baselines import train_lightgbm
from src.preprocessing.feature_transformer import TARGET, split_feature_types
from src.utils.io import load_config, load_scenario, resolve, save_json, seed_dir
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("hybrid")

HYBRIDS = [("fedavg", "ditto"), ("fedavg", "local_lgbm"), ("ditto", "local_lgbm")]
GRID = np.round(np.arange(0, 1.0001, 0.05), 2)
LABELS = {"lgbm": "Centralized LightGBM", "hgnn": "Centralized HeteroGNN", "local": "Local-only",
          "fedavg": "FedAvg", "ditto": "Ditto", "local_lgbm": "Local LightGBM",
          "hyb_fedavg_ditto": "Hybrid FedAvg+Ditto", "hyb_fedavg_local_lgbm": "Hybrid FedAvg+LocalLGBM",
          "hyb_ditto_local_lgbm": "Hybrid Ditto+LocalLGBM"}


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def local_lightgbm(seed: int, cfg: dict, assign: pd.DataFrame, out: Path) -> pd.DataFrame:
    """LightGBM riêng từng client; cache dự đoán Val/Test theo seed."""
    path = out / "local_lgbm_predictions.parquet"
    if path.exists():
        return pd.read_parquet(path)
    processed = resolve(cfg["paths"]["processed_dir"])
    df = pd.read_parquet(processed / cfg["files"]["features"]).merge(assign, on="SK_ID_CURR",
                                                                     validate="one_to_one")
    num_cols, cat_cols = split_feature_types(df.drop(columns=["split", "client"]))
    feats = num_cols + cat_cols
    lcfg = load_config("model.yaml")["lightgbm"]
    preds = []
    for k, g in df.groupby("client"):
        set_seed(seed)
        parts = {s: g[g["split"] == s] for s in ["train", "val", "test"]}
        model, info = train_lightgbm(parts["train"][feats], parts["train"][TARGET].values,
                                     parts["val"][feats], parts["val"][TARGET].values, cat_cols, lcfg, seed)
        for s in ["val", "test"]:
            preds.append(pd.DataFrame({"SK_ID_CURR": parts[s]["SK_ID_CURR"].values, "split": s,
                                       "prob_local_lgbm": model.predict_proba(parts[s][feats])[:, 1]}))
        log.info("seed %d client %d local LightGBM | best iter %d | val AUC %.4f", seed, k,
                 info["best_iteration"],
                 roc_auc_score(parts["val"][TARGET], preds[-2]["prob_local_lgbm"]))
    p = pd.concat(preds, ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    p.to_parquet(path, index=False)
    return p


def add_hybrids(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Thêm cột prob_hyb_<a>_<b>; w của từng client chọn theo AUC Val của client."""
    weights = {}
    for a, b in HYBRIDS:
        col, weights[f"hyb_{a}_{b}"] = f"prob_hyb_{a}_{b}", {}
        df[col] = np.nan
        for k, g in df.groupby("client"):
            la, lb = logit(g[f"prob_{a}"].values), logit(g[f"prob_{b}"].values)
            val = (g["split"] == "val").values
            aucs = [roc_auc_score(g["TARGET"].values[val], w * la[val] + (1 - w) * lb[val]) for w in GRID]
            w = float(GRID[int(np.argmax(aucs))])
            weights[f"hyb_{a}_{b}"][int(k)] = w
            df.loc[g.index, col] = 1 / (1 + np.exp(-(w * la + (1 - w) * lb)))
    return df, weights


def per_client_auc(test: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    groups = dict(tuple(test.groupby("client")))
    n = np.array([len(g) for g in groups.values()])
    rows = {}
    for m in methods:
        auc = {k: roc_auc_score(g["TARGET"], g[f"prob_{m}"]) for k, g in groups.items()}
        rows[m] = {**{f"C{k}": v for k, v in auc.items()},
                   "client-avg": float(np.average(list(auc.values()), weights=n)), "worst": min(auc.values())}
    return pd.DataFrame(rows).T


def paired_bootstrap(y, a, b, rounds: int, rng) -> list[float]:
    diffs = []
    for _ in range(rounds):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            diffs.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = parser.parse_args()

    cfg, fcfg = load_config(), load_config("federated.yaml")
    name, _, paths = load_scenario(args.scenario)
    assign = pd.read_parquet(paths["assignments"])[["SK_ID_CURR", "client", "split"]]
    central_root = resolve(cfg["paths"]["results_dir"]) / "metrics"
    # Centralized LightGBM chỉ có ở seed mặc định -> dùng chung làm mốc cho mọi seed
    lgbm = pd.read_parquet(central_root / "centralized_predictions.parquet")[["SK_ID_CURR", "prob_lgbm"]]

    auc, weights, results = {}, {}, {}
    for seed in args.seeds:
        fed_dir = seed_dir(paths["metrics"], seed, cfg["seed"])
        out = fed_dir / "hybrid"
        f = pd.read_parquet(fed_dir / "federated_predictions.parquet")
        f = f[["SK_ID_CURR", "client", "split", "TARGET", "prob_local", "prob_fedavg", "prob_ditto"]]
        c = pd.read_parquet(seed_dir(central_root, seed, cfg["seed"]) / "centralized_predictions.parquet")
        df = (f.merge(local_lightgbm(seed, cfg, assign, out), on=["SK_ID_CURR", "split"], validate="one_to_one")
              .merge(c[["SK_ID_CURR", "prob_hgnn"]], on="SK_ID_CURR").merge(lgbm, on="SK_ID_CURR"))
        df, weights[seed] = add_hybrids(df.reset_index(drop=True))
        df.to_parquet(out / "hybrid_predictions.parquet", index=False)

        methods = list(LABELS)
        test = df[df["split"] == "test"]
        assert len(test) == len(c[c["split"] == "test"]), "Test của client phải phủ đúng Test chung"
        auc[seed] = per_client_auc(test, methods)
        # Threshold riêng từng client chọn trên Val (max F1), như nhánh Federated
        results[seed] = {}
        for m in methods:
            val = df[df["split"] == "val"]
            thr = {int(k): select_threshold(g["TARGET"], g[f"prob_{m}"], cfg["threshold"]["criterion"])
                   for k, g in val.groupby("client")}
            results[seed][m] = evaluate_by_client(test, f"prob_{m}", thr)
        log.info("seed %d | trọng số w (phần federated/Ditto) theo client: %s", seed, weights[seed])

        if seed == cfg["seed"]:
            rng = np.random.default_rng(seed)
            boot = {}
            for h in [m for m in methods if m.startswith("hyb_")]:
                for ref in ["fedavg", "ditto", "hgnn", "lgbm"]:
                    boot[f"{h}_vs_{ref}"] = {int(k): {
                        "delta_auc": float(roc_auc_score(g["TARGET"], g[f"prob_{h}"])
                                           - roc_auc_score(g["TARGET"], g[f"prob_{ref}"])),
                        "ci95": paired_bootstrap(g["TARGET"].values, g[f"prob_{h}"].values,
                                                 g[f"prob_{ref}"].values, fcfg["evaluation"]["bootstrap_rounds"], rng)}
                        for k, g in test.groupby("client")}
            save_json(boot, out / "hybrid_bootstrap.json")
            rows = {key: {f"C{k}": f"{v['delta_auc']:+.4f}{' *' if v['ci95'][0] > 0 or v['ci95'][1] < 0 else ''}"
                          f" [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]" for k, v in d.items()}
                    for key, d in boot.items()}
            with pd.option_context("display.width", 250, "display.max_columns", 20):
                log.info("seed %d ΔAUC kèm CI 95%% (paired bootstrap):\n%s", seed, pd.DataFrame(rows).T.to_string())

    methods = list(LABELS)
    stack = np.stack([auc[s].loc[methods].values for s in args.seeds])
    cols = auc[args.seeds[0]].columns
    mean = pd.DataFrame(stack.mean(0), index=methods, columns=cols)
    std = pd.DataFrame(stack.std(0, ddof=1) if len(args.seeds) > 1 else np.zeros_like(stack[0]),
                       index=methods, columns=cols)
    delta = {}
    for h in [m for m in methods if m.startswith("hyb_")]:
        for ref in ["fedavg", "ditto", "hgnn", "lgbm"]:
            d = np.stack([(auc[s].loc[h] - auc[s].loc[ref]).values for s in args.seeds])
            delta[f"{LABELS[h]} − {LABELS[ref]}"] = {
                col: f"{d[:, j].mean():+.4f} ({int((d[:, j] > 0).sum())}/{len(args.seeds)} dương)"
                for j, col in enumerate(cols)}

    out_dir = paths["metrics"] / "hybrid"
    save_json({"scenario": name, "seeds": args.seeds, "grid": GRID.tolist(),
               "weights": {str(s): w for s, w in weights.items()},
               "auc_per_seed": {str(s): auc[s].to_dict(orient="index") for s in args.seeds},
               "auc_mean": mean.to_dict(orient="index"), "auc_std": std.to_dict(orient="index"),
               "results_per_seed": {str(s): r for s, r in results.items()}}, out_dir / "hybrid_summary.json")
    fmt = mean.map(lambda v: f"{v:.4f}") + " ± " + std.map(lambda v: f"{v:.4f}")
    fmt.index = [LABELS[m] for m in methods]
    fmt.to_csv(out_dir / "hybrid_auc.csv")
    pd.DataFrame(delta).T.to_csv(out_dir / "hybrid_delta_auc.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        log.info("ROC-AUC Test theo client, mean ± std qua %d seed:\n%s", len(args.seeds), fmt.to_string())
        log.info("ΔAUC trung bình qua seed (số seed có Δ > 0):\n%s", pd.DataFrame(delta).T.to_string())


if __name__ == "__main__":
    main()
