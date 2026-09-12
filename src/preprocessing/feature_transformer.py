"""Domain feature engineering that keeps partition metadata intact."""
from __future__ import annotations
import numpy as np
import pandas as pd

PARTITION_COLUMNS = ("REGION_RATING_CLIENT_W_CITY", "OCCUPATION_TYPE")

def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)

def engineer_application_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create credit-risk features without prematurely encoding categories."""
    result = data.copy()
    if "DAYS_BIRTH" in result:
        result["AGE_YEARS"] = -result["DAYS_BIRTH"] / 365.25
    if "DAYS_EMPLOYED" in result:
        result["EMPLOYED_YEARS"] = -result["DAYS_EMPLOYED"] / 365.25
    ratios = {
        "CREDIT_INCOME_RATIO": ("AMT_CREDIT", "AMT_INCOME_TOTAL"),
        "ANNUITY_INCOME_RATIO": ("AMT_ANNUITY", "AMT_INCOME_TOTAL"),
        "CREDIT_ANNUITY_RATIO": ("AMT_CREDIT", "AMT_ANNUITY"),
        "GOODS_CREDIT_RATIO": ("AMT_GOODS_PRICE", "AMT_CREDIT"),
        "INCOME_PER_PERSON": ("AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"),
        "EMPLOYED_AGE_RATIO": ("DAYS_EMPLOYED", "DAYS_BIRTH"),
    }
    for new, (numerator, denominator) in ratios.items():
        if numerator in result and denominator in result:
            result[new] = _safe_ratio(result[numerator], result[denominator])
    ext = [c for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3") if c in result]
    if ext:
        result["EXT_SOURCE_MEAN"] = result[ext].mean(axis=1)
        result["EXT_SOURCE_STD"] = result[ext].std(axis=1)
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result
