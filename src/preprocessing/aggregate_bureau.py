import numpy as np
import pandas as pd

from src.preprocessing.agg_utils import aggregate, one_hot, safe_div


def aggregate_bureau_balance(bb: pd.DataFrame) -> pd.DataFrame:
    """bureau_balance -> 1 dòng / SK_ID_BUREAU."""
    bb, status_cols = one_hot(bb, ["STATUS"])
    spec = {"MONTHS_BALANCE": ["min", "max", "size"]}
    spec.update({c: ["mean"] for c in status_cols})
    agg = aggregate(bb, "SK_ID_BUREAU", spec, "BB")
    # Tỷ lệ tháng có quá hạn (STATUS 1..5)
    dpd_cols = [f"BB_STATUS_{k}_MEAN" for k in "12345" if f"BB_STATUS_{k}_MEAN" in agg.columns]
    agg["BB_DPD_RATE"] = agg[dpd_cols].sum(axis=1)
    return agg


def aggregate_bureau(bureau: pd.DataFrame, bb: pd.DataFrame) -> pd.DataFrame:
    """bureau (+ bureau_balance) -> 1 dòng / SK_ID_CURR."""
    bureau = bureau.join(aggregate_bureau_balance(bb), on="SK_ID_BUREAU")
    bureau = bureau.drop(columns=["SK_ID_BUREAU"])

    bureau["DEBT_CREDIT_RATIO"] = safe_div(bureau["AMT_CREDIT_SUM_DEBT"], bureau["AMT_CREDIT_SUM"])
    bureau["OVERDUE_DEBT_RATIO"] = safe_div(bureau["AMT_CREDIT_SUM_OVERDUE"], bureau["AMT_CREDIT_SUM_DEBT"])

    bureau, oh_cols = one_hot(bureau, ["CREDIT_ACTIVE", "CREDIT_CURRENCY", "CREDIT_TYPE"])

    num_spec = {
        "DAYS_CREDIT": ["min", "max", "mean"],
        "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
        "DAYS_ENDDATE_FACT": ["min", "max"],
        "DAYS_CREDIT_UPDATE": ["mean"],
        "CREDIT_DAY_OVERDUE": ["max", "mean"],
        "AMT_CREDIT_MAX_OVERDUE": ["max", "mean"],
        "CNT_CREDIT_PROLONG": ["sum"],
        "AMT_CREDIT_SUM": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_DEBT": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["max", "sum"],
        "AMT_CREDIT_SUM_LIMIT": ["mean", "sum"],
        "AMT_ANNUITY": ["max", "mean"],
        "DEBT_CREDIT_RATIO": ["max", "mean"],
        "OVERDUE_DEBT_RATIO": ["max"],
        "BB_MONTHS_BALANCE_MIN": ["min"],
        "BB_MONTHS_BALANCE_SIZE": ["mean", "sum"],
        "BB_DPD_RATE": ["max", "mean"],
    }
    spec = dict(num_spec)
    spec.update({c: ["mean"] for c in oh_cols})
    out = aggregate(bureau, "SK_ID_CURR", spec, "BURO")
    out["BURO_COUNT"] = bureau.groupby("SK_ID_CURR").size().astype(np.float32)

    # Thống kê riêng cho các khoản đang Active / đã Closed
    for name, flag in [("ACTIVE", "CREDIT_ACTIVE_Active"), ("CLOSED", "CREDIT_ACTIVE_Closed")]:
        sub = bureau[bureau[flag] == 1]
        out = out.join(aggregate(sub, "SK_ID_CURR", num_spec, f"BURO_{name}"))

    return out
