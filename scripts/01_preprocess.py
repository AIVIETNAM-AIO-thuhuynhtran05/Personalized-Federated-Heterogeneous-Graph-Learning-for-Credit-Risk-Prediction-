"""Làm sạch application_train, aggregate các bảng phụ, tạo split Train/Val/Test dùng chung.

    python scripts/01_preprocess.py               # tạo features + split (giữ split cũ nếu đã có)
    python scripts/01_preprocess.py --force-split # tạo lại split (sẽ làm lệch với các kết quả cũ)
"""
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing.aggregate_bureau import aggregate_bureau
from src.preprocessing.aggregate_payments import (aggregate_credit_card, aggregate_installments,
                                                  aggregate_pos_cash)
from src.preprocessing.clean_application import clean_application
from src.preprocessing.clean_previous import aggregate_previous
from src.utils.io import downcast, load_config, read_raw, resolve
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("preprocess")


def build_features(cfg: dict) -> pd.DataFrame:
    # Chỉ application_train có TARGET; application_test không được dùng ở đây.
    app = clean_application(read_raw(cfg, "application_train"))
    ids = set(app["SK_ID_CURR"])
    log.info("application_train: %s", app.shape)

    def load(table, **kw):
        df = read_raw(cfg, table, **kw)
        return df[df["SK_ID_CURR"].isin(ids)] if "SK_ID_CURR" in df.columns else df

    t = time.time()
    bureau = load("bureau")
    bb = read_raw(cfg, "bureau_balance", dtype={"STATUS": "category"})
    bb = bb[bb["SK_ID_BUREAU"].isin(set(bureau["SK_ID_BUREAU"]))]
    tables = {"bureau": aggregate_bureau(bureau, bb)}
    del bureau, bb
    log.info("bureau + bureau_balance: %s (%.0fs)", tables["bureau"].shape, time.time() - t)

    for name, fn in [("previous_application", aggregate_previous),
                     ("installments_payments", aggregate_installments),
                     ("POS_CASH_balance", aggregate_pos_cash),
                     ("credit_card_balance", aggregate_credit_card)]:
        t = time.time()
        tables[name] = fn(load(name))
        log.info("%s: %s (%.0fs)", name, tables[name].shape, time.time() - t)

    df = app.set_index("SK_ID_CURR")
    for agg in tables.values():
        df = df.join(agg)
    df = df.reset_index()

    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    # Bỏ cột hằng (kể cả cột toàn NaN)
    constant = [c for c in df.columns if df[c].nunique(dropna=True) <= 1 and c != "TARGET"]
    df = df.drop(columns=constant)
    log.info("Dropped %d constant columns", len(constant))
    return downcast(df)


def make_splits(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    s = cfg["split"]
    ids, y = df["SK_ID_CURR"].values, df[s["stratify"]].values
    rest_ids, test_ids, rest_y, _ = train_test_split(
        ids, y, test_size=s["test_size"], stratify=y, random_state=cfg["seed"])
    val_frac = s["val_size"] / (1 - s["test_size"])
    train_ids, val_ids = train_test_split(
        rest_ids, test_size=val_frac, stratify=rest_y, random_state=cfg["seed"])

    split = pd.Series("train", index=df["SK_ID_CURR"].values)
    split.loc[val_ids] = "val"
    split.loc[test_ids] = "test"
    return pd.DataFrame({"SK_ID_CURR": df["SK_ID_CURR"].values,
                         "TARGET": df["TARGET"].values,
                         "split": split.loc[df["SK_ID_CURR"].values].values})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-split", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    out_dir = resolve(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = build_features(cfg)
    df.to_parquet(out_dir / cfg["files"]["features"], index=False)
    log.info("Saved features: %s -> %s", df.shape, cfg["files"]["features"])

    split_path = out_dir / cfg["files"]["splits"]
    if split_path.exists() and split_path.stat().st_size > 0 and not args.force_split:
        splits = pd.read_parquet(split_path)
        assert set(splits["SK_ID_CURR"]) == set(df["SK_ID_CURR"]), "Split cũ không khớp ID"
        log.info("Giữ nguyên split đã có: %s", split_path.name)
    else:
        splits = make_splits(df, cfg)
        splits.to_parquet(split_path, index=False)
        log.info("Saved new split -> %s", split_path.name)

    summary = splits.groupby("split")["TARGET"].agg(n="size", default_rate="mean")
    log.info("Split summary:\n%s", summary)


if __name__ == "__main__":
    main()
