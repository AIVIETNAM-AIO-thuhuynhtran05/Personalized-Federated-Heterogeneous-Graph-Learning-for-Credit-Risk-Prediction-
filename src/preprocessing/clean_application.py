import numpy as np
import pandas as pd

from src.preprocessing.agg_utils import safe_div

# Giá trị placeholder của Home Credit cho "không có" (≈1000 năm)
DAYS_ANOMALY = 365243


def clean_application(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # DAYS_EMPLOYED = 365243 xuất hiện ở người hưu trí / thất nghiệp
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == DAYS_ANOMALY).astype(np.uint8)
    df.loc[df["DAYS_EMPLOYED"] == DAYS_ANOMALY, "DAYS_EMPLOYED"] = np.nan

    df.loc[df["CODE_GENDER"] == "XNA", "CODE_GENDER"] = np.nan
    df.loc[df["NAME_FAMILY_STATUS"] == "Unknown", "NAME_FAMILY_STATUS"] = np.nan
    df.loc[df["ORGANIZATION_TYPE"] == "XNA", "ORGANIZATION_TYPE"] = np.nan
    df.loc[df["DAYS_LAST_PHONE_CHANGE"] == 0, "DAYS_LAST_PHONE_CHANGE"] = np.nan

    # Cờ Y/N -> 1/0
    for col in ["FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
        df[col] = df[col].map({"Y": 1, "N": 0}).astype(np.float32)

    # Đặc trưng tỷ lệ: chỉ dùng thông tin của chính hồ sơ, không dùng nhãn
    df["CREDIT_INCOME_RATIO"] = safe_div(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    df["ANNUITY_INCOME_RATIO"] = safe_div(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    df["PAYMENT_RATE"] = safe_div(df["AMT_ANNUITY"], df["AMT_CREDIT"])
    df["GOODS_CREDIT_RATIO"] = safe_div(df["AMT_GOODS_PRICE"], df["AMT_CREDIT"])
    df["EMPLOYED_BIRTH_RATIO"] = safe_div(df["DAYS_EMPLOYED"], df["DAYS_BIRTH"])
    df["INCOME_PER_PERSON"] = safe_div(df["AMT_INCOME_TOTAL"], df["CNT_FAM_MEMBERS"])
    df["CAR_AGE_BIRTH_RATIO"] = safe_div(df["OWN_CAR_AGE"], df["DAYS_BIRTH"])

    ext = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    df["EXT_SOURCE_MEAN"] = ext.mean(axis=1).astype(np.float32)
    df["EXT_SOURCE_STD"] = ext.std(axis=1).astype(np.float32)
    df["EXT_SOURCE_MIN"] = ext.min(axis=1).astype(np.float32)
    df["EXT_SOURCE_PROD"] = ext.prod(axis=1, min_count=3).astype(np.float32)

    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    df["DOCUMENT_COUNT"] = df[doc_cols].sum(axis=1).astype(np.float32)

    return df
