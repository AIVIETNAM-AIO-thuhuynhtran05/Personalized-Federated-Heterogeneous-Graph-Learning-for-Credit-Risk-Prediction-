import numpy as np
import pandas as pd

from src.preprocessing.agg_utils import aggregate, one_hot, safe_div
from src.preprocessing.clean_application import DAYS_ANOMALY

DAYS_COLS = ["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION",
             "DAYS_LAST_DUE", "DAYS_TERMINATION"]

CAT_COLS = ["NAME_CONTRACT_TYPE", "NAME_CONTRACT_STATUS", "NAME_PAYMENT_TYPE",
            "CODE_REJECT_REASON", "NAME_CLIENT_TYPE", "NAME_PORTFOLIO", "NAME_PRODUCT_TYPE",
            "CHANNEL_TYPE", "NAME_YIELD_GROUP", "PRODUCT_COMBINATION", "NAME_GOODS_CATEGORY",
            "NAME_CASH_LOAN_PURPOSE", "NAME_SELLER_INDUSTRY"]


def aggregate_previous(prev: pd.DataFrame) -> pd.DataFrame:
    """previous_application -> 1 dòng / SK_ID_CURR."""
    prev = prev.copy()
    for col in DAYS_COLS:
        prev.loc[prev[col] == DAYS_ANOMALY, col] = np.nan

    prev["APP_CREDIT_RATIO"] = safe_div(prev["AMT_APPLICATION"], prev["AMT_CREDIT"])
    prev["CREDIT_GOODS_RATIO"] = safe_div(prev["AMT_CREDIT"], prev["AMT_GOODS_PRICE"])
    prev["ANNUITY_CREDIT_RATIO"] = safe_div(prev["AMT_ANNUITY"], prev["AMT_CREDIT"])
    prev["DAYS_LAST_DUE_DIFF"] = (prev["DAYS_LAST_DUE_1ST_VERSION"] - prev["DAYS_LAST_DUE"]).astype(np.float32)

    prev, oh_cols = one_hot(prev, CAT_COLS)

    num_spec = {
        "AMT_ANNUITY": ["min", "max", "mean"],
        "AMT_APPLICATION": ["min", "max", "mean"],
        "AMT_CREDIT": ["min", "max", "mean"],
        "AMT_DOWN_PAYMENT": ["max", "mean"],
        "AMT_GOODS_PRICE": ["min", "max", "mean"],
        "RATE_DOWN_PAYMENT": ["max", "mean"],
        "HOUR_APPR_PROCESS_START": ["mean"],
        "DAYS_DECISION": ["min", "max", "mean"],
        "CNT_PAYMENT": ["mean", "sum"],
        "DAYS_FIRST_DUE": ["min", "max"],
        "DAYS_LAST_DUE": ["max"],
        "DAYS_TERMINATION": ["max"],
        "DAYS_LAST_DUE_DIFF": ["mean"],
        "NFLAG_INSURED_ON_APPROVAL": ["mean"],
        "APP_CREDIT_RATIO": ["min", "max", "mean", "var"],
        "CREDIT_GOODS_RATIO": ["mean"],
        "ANNUITY_CREDIT_RATIO": ["mean"],
    }
    spec = dict(num_spec)
    spec.update({c: ["mean"] for c in oh_cols})
    out = aggregate(prev, "SK_ID_CURR", spec, "PREV")
    out["PREV_COUNT"] = prev.groupby("SK_ID_CURR").size().astype(np.float32)

    for name, flag in [("APPROVED", "NAME_CONTRACT_STATUS_Approved"),
                       ("REFUSED", "NAME_CONTRACT_STATUS_Refused")]:
        sub = prev[prev[flag] == 1]
        out = out.join(aggregate(sub, "SK_ID_CURR", num_spec, f"PREV_{name}"))

    return out
