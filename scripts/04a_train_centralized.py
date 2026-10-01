"""Centralized tabular baselines (Logistic Regression, LightGBM).

Train trên split=train, chọn siêu tham số + threshold trên split=val,
đánh giá MỘT LẦN trên split=test (cùng tập Test sẽ dùng cho nhánh Federated).

    python scripts/04a_train_centralized.py                # cả 2 mô hình
    python scripts/04a_train_centralized.py --models lgbm
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from src.evaluation.metrics import evaluate, select_threshold
from src.models.baselines import train_lightgbm, train_logistic_regression
from src.preprocessing.feature_transformer import TARGET, split_feature_types
from src.utils.io import load_config, resolve, save_json
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("centralized")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["lr", "lgbm"], choices=["lr", "lgbm"])
    args = parser.parse_args()

    cfg, model_cfg = load_config(), load_config("model.yaml")
    set_seed(cfg["seed"])
    processed = resolve(cfg["paths"]["processed_dir"])
    results = resolve(cfg["paths"]["results_dir"])

    df = pd.read_parquet(processed / cfg["files"]["features"])
    splits = pd.read_parquet(processed / cfg["files"]["splits"])[["SK_ID_CURR", "split"]]
    df = df.merge(splits, on="SK_ID_CURR", how="inner", validate="one_to_one")

    num_cols, cat_cols = split_feature_types(df.drop(columns=["split"]))
    feats = num_cols + cat_cols
    parts = {s: df[df["split"] == s] for s in ["train", "val", "test"]}
    X = {s: p[feats] for s, p in parts.items()}
    y = {s: p[TARGET].values for s, p in parts.items()}
    log.info("Features: %d numeric + %d categorical | %s",
             len(num_cols), len(cat_cols), {s: len(v) for s, v in y.items()})

    trainers = {
        "lr": ("logistic_regression",
               lambda: train_logistic_regression(X["train"], y["train"], X["val"], y["val"],
                                                 num_cols, cat_cols,
                                                 model_cfg["logistic_regression"], cfg["seed"])),
        "lgbm": ("lightgbm",
                 lambda: train_lightgbm(X["train"], y["train"], X["val"], y["val"], cat_cols,
                                        model_cfg["lightgbm"], cfg["seed"])),
    }

    metrics_path = results / "metrics" / "centralized_baselines.json"
    # Giữ kết quả mô hình khác nếu chỉ chạy lại một mô hình
    report = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    preds_path = results / "metrics" / "centralized_predictions.parquet"
    preds = pd.concat([parts[s][["SK_ID_CURR", "split", TARGET]] for s in ["val", "test"]])
    if preds_path.exists() and preds_path.stat().st_size > 0:
        old = pd.read_parquet(preds_path)
        keep = [c for c in old.columns if c.startswith("prob_")]
        preds = preds.merge(old[["SK_ID_CURR", *keep]], on="SK_ID_CURR", how="left")

    for key in args.models:
        name, train_fn = trainers[key]
        log.info("=== Training %s ===", name)
        t = time.time()
        model, info = train_fn()
        train_time = time.time() - t

        p_val = model.predict_proba(X["val"])[:, 1]
        p_test = model.predict_proba(X["test"])[:, 1]
        thr = select_threshold(y["val"], p_val, cfg["threshold"]["criterion"])

        report[name] = {
            "val": evaluate(y["val"], p_val, thr),
            "test": evaluate(y["test"], p_test, thr),
            "threshold_criterion": cfg["threshold"]["criterion"],
            "train_time_sec": round(train_time, 1),
            **info,
        }
        preds[f"prob_{key}"] = list(p_val) + list(p_test)

        ckpt = results / "checkpoints" / f"centralized_{key}.joblib"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, ckpt)
        log.info("%s done in %.0fs | val AUC=%.4f | test AUC=%.4f", name, train_time,
                 report[name]["val"]["roc_auc"], report[name]["test"]["roc_auc"])
        # Lưu ngay sau mỗi mô hình để không mất kết quả nếu mô hình sau lỗi
        save_json(report, metrics_path)
        preds.to_parquet(preds_path, index=False)

    cols = ["roc_auc", "pr_auc", "ks", "f1", "precision", "recall", "threshold"]
    table = pd.DataFrame({m: {c: r["test"][c] for c in cols} for m, r in report.items()}).T
    log.info("TEST results (threshold chọn trên Validation):\n%s", table.round(4).to_string())


if __name__ == "__main__":
    main()
