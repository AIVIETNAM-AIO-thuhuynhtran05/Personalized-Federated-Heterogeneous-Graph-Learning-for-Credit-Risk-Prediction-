import re

import numpy as np
import pandas as pd


def one_hot(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """One-hot các cột phân loại (NaN thành cột riêng) để lấy tỷ lệ khi aggregate."""
    before = set(df.columns)
    df = pd.get_dummies(df, columns=cols, dummy_na=True, dtype=np.uint8)
    return df, [c for c in df.columns if c not in before]


def aggregate(df: pd.DataFrame, key: str, spec: dict[str, list[str]], prefix: str) -> pd.DataFrame:
    """groupby(key).agg(spec) rồi làm phẳng tên cột thành PREFIX_COL_STAT."""
    agg = df.groupby(key).agg(spec)
    # Tên cột chỉ gồm [A-Z0-9_] (LightGBM không nhận ký tự đặc biệt như , " : /)
    names = [re.sub(r"[^A-Z0-9_]+", "_", f"{prefix}_{col}_{stat}".upper()) for col, stat in agg.columns]
    agg.columns = names
    agg = agg.loc[:, ~agg.columns.duplicated()]
    return agg.astype(np.float32)


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b.replace(0, np.nan)).astype(np.float32)
