"""Đo mức non-IID của một cách chia client theo 3 chiều:
quantity skew (kích thước), label skew (tỷ lệ default) và feature skew (phân phối đặc trưng)."""
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

PROFILE_COLS = ["REGION_POPULATION_RELATIVE", "REGION_RATING_CLIENT", "AMT_INCOME_TOTAL",
                "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_BIRTH"]


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return (kl(p, m) + kl(q, m)) / 2


def _feature_shift(g: pd.Series, ref: pd.Series) -> float:
    """Khoảng cách Wasserstein giữa phân phối của client và toàn cục, chuẩn hóa theo độ lệch chuẩn toàn cục."""
    g, ref = g.dropna(), ref.dropna()
    return float(wasserstein_distance(g, ref) / ref.std()) if len(g) else float("nan")


def partition_report(assign: pd.DataFrame, profile: pd.DataFrame | None = None) -> dict:
    global_dist = np.bincount(assign["TARGET"], minlength=2).astype(float)
    df = assign if profile is None else assign.merge(profile, on="SK_ID_CURR", how="left")
    if profile is not None:
        df["LOG_INCOME"] = np.log1p(df["AMT_INCOME_TOTAL"])
        df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    clients = {}
    for k, g in df.groupby("client"):
        dist = np.bincount(g["TARGET"], minlength=2).astype(float)
        c = {
            "n": int(len(g)),
            "share": round(len(g) / len(df), 4),
            "default_rate": round(float(g["TARGET"].mean()), 4),
            "positives": int(g["TARGET"].sum()),
            "label_js_divergence": round(_js_divergence(dist, global_dist), 5),
            "split_sizes": g["split"].value_counts().to_dict(),
            "split_positives": g.groupby("split")["TARGET"].sum().astype(int).to_dict(),
        }
        if profile is not None:
            c.update({
                "n_regions": int(g["REGION_POPULATION_RELATIVE"].nunique()),
                "region_rating_share": g["REGION_RATING_CLIENT"].value_counts(normalize=True)
                                        .sort_index().round(4).to_dict(),
                "median_income": float(g["AMT_INCOME_TOTAL"].median()),
                "mean_ext_source_2": round(float(g["EXT_SOURCE_2"].mean()), 4),
                "mean_ext_source_3": round(float(g["EXT_SOURCE_3"].mean()), 4),
                "feature_shift_wasserstein": {
                    col: round(_feature_shift(g[col], df[col]), 4)
                    for col in ["EXT_SOURCE_2", "EXT_SOURCE_3", "LOG_INCOME", "AGE_YEARS"]},
            })
        clients[int(k)] = c
    sizes = np.array([c["n"] for c in clients.values()])
    rates = np.array([c["default_rate"] for c in clients.values()])
    return {
        "clients": clients,
        "global_default_rate": round(float(df["TARGET"].mean()), 4),
        "size_max_min_ratio": round(float(sizes.max() / sizes.min()), 2),
        "default_rate_range": [float(rates.min()), float(rates.max())],
    }
