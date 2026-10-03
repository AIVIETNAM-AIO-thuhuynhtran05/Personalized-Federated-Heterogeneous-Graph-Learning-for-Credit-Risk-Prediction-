import numpy as np
import pandas as pd

from src.preprocessing.agg_utils import aggregate, one_hot, safe_div


def aggregate_installments(ins: pd.DataFrame) -> pd.DataFrame:
    """installments_payments -> 1 dòng / SK_ID_CURR."""
    ins = ins.copy()
    ins["PAYMENT_RATIO"] = safe_div(ins["AMT_PAYMENT"], ins["AMT_INSTALMENT"])
    ins["PAYMENT_DIFF"] = (ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]).astype(np.float32)
    # Số ngày trả trễ (DPD) / trả sớm (DBD)
    ins["DPD"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]).clip(lower=0).astype(np.float32)
    ins["DBD"] = (ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]).clip(lower=0).astype(np.float32)
    ins["IS_LATE"] = (ins["DPD"] > 0).astype(np.uint8)
    ins["IS_UNDERPAID"] = (ins["PAYMENT_DIFF"] > 0).astype(np.uint8)

    spec = {
        "NUM_INSTALMENT_VERSION": ["nunique"],
        "DPD": ["max", "mean", "sum"],
        "DBD": ["max", "mean", "sum"],
        "IS_LATE": ["mean", "sum"],
        "IS_UNDERPAID": ["mean"],
        "PAYMENT_RATIO": ["min", "max", "mean", "var"],
        "PAYMENT_DIFF": ["max", "mean", "sum"],
        "AMT_INSTALMENT": ["max", "mean", "sum"],
        "AMT_PAYMENT": ["min", "max", "mean", "sum"],
        "DAYS_ENTRY_PAYMENT": ["max", "mean"],
    }
    out = aggregate(ins, "SK_ID_CURR", spec, "INSTAL")
    out["INSTAL_COUNT"] = ins.groupby("SK_ID_CURR").size().astype(np.float32)
    return out


def aggregate_pos_cash(pos: pd.DataFrame) -> pd.DataFrame:
    """POS_CASH_balance -> 1 dòng / SK_ID_CURR."""
    pos, oh_cols = one_hot(pos, ["NAME_CONTRACT_STATUS"])
    spec = {
        "MONTHS_BALANCE": ["max", "mean", "size"],
        "CNT_INSTALMENT": ["mean"],
        "CNT_INSTALMENT_FUTURE": ["mean"],
        "SK_DPD": ["max", "mean"],
        "SK_DPD_DEF": ["max", "mean"],
    }
    spec.update({c: ["mean"] for c in oh_cols})
    out = aggregate(pos, "SK_ID_CURR", spec, "POS")
    out["POS_COUNT"] = pos.groupby("SK_ID_CURR").size().astype(np.float32)
    return out


def aggregate_credit_card(cc: pd.DataFrame) -> pd.DataFrame:
    """credit_card_balance -> 1 dòng / SK_ID_CURR."""
    cc = cc.copy()
    cc["UTILIZATION"] = safe_div(cc["AMT_BALANCE"], cc["AMT_CREDIT_LIMIT_ACTUAL"])
    cc["PAYMENT_MIN_RATIO"] = safe_div(cc["AMT_PAYMENT_CURRENT"], cc["AMT_INST_MIN_REGULARITY"])
    cc, oh_cols = one_hot(cc, ["NAME_CONTRACT_STATUS"])
    num_cols = [c for c in cc.columns if c not in ("SK_ID_PREV", "SK_ID_CURR", *oh_cols)]
    spec = {c: ["min", "max", "mean"] for c in num_cols}
    spec.update({c: ["mean"] for c in oh_cols})
    out = aggregate(cc, "SK_ID_CURR", spec, "CC")
    out["CC_COUNT"] = cc.groupby("SK_ID_CURR").size().astype(np.float32)
    return out
