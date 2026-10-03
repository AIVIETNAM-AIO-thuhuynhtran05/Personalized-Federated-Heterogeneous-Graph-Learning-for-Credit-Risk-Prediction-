"""Đọc bảng quan hệ và tạo bảng node (1 dòng = 1 node) cho từng node type.
Hàm nhận tập SK_ID_CURR bất kỳ nên dùng lại được cho graph cục bộ của từng client."""
import logging

import numpy as np
import pandas as pd

from src.graph.schema import BUREAU, CC, CUSTOMER, INSTALLMENT, POS, PREV
from src.preprocessing.agg_utils import safe_div
from src.preprocessing.aggregate_bureau import aggregate_bureau_balance
from src.preprocessing.clean_application import DAYS_ANOMALY, clean_application
from src.preprocessing.clean_previous import DAYS_COLS
from src.utils.io import read_raw

log = logging.getLogger(__name__)


def _customer(cfg, customer_ids) -> pd.DataFrame:
    app = clean_application(read_raw(cfg, "application_train"))
    app = app.set_index("SK_ID_CURR").loc[customer_ids].reset_index()
    return app


def _bureau(cfg, ids: set) -> tuple[pd.DataFrame, dict]:
    bureau = read_raw(cfg, "bureau")
    bureau = bureau[bureau["SK_ID_CURR"].isin(ids)]
    bb = read_raw(cfg, "bureau_balance", dtype={"STATUS": "category"})
    bb_ids = bb["SK_ID_BUREAU"].unique()
    all_bureau_ids = set(read_raw(cfg, "bureau", usecols=["SK_ID_BUREAU"])["SK_ID_BUREAU"])
    stats = {
        # bureau_balance không trỏ tới bureau nào -> không gắn được vào khách hàng nào
        "bureau_balance_rows_without_bureau": int((~bb["SK_ID_BUREAU"].isin(all_bureau_ids)).sum()),
        "bureau_balance_ids_without_bureau": int(len(set(bb_ids) - all_bureau_ids)),
        # bureau không có lịch sử tháng nào -> giữ node, đánh cờ HAS_BUREAU_BALANCE=0
        "bureau_without_bureau_balance": int((~bureau["SK_ID_BUREAU"].isin(set(bb_ids))).sum()),
    }
    bb = bb[bb["SK_ID_BUREAU"].isin(set(bureau["SK_ID_BUREAU"]))]
    bureau = bureau.join(aggregate_bureau_balance(bb), on="SK_ID_BUREAU")
    bureau["HAS_BUREAU_BALANCE"] = bureau["BB_MONTHS_BALANCE_SIZE"].notna().astype(np.float32)
    bureau["DEBT_CREDIT_RATIO"] = safe_div(bureau["AMT_CREDIT_SUM_DEBT"], bureau["AMT_CREDIT_SUM"])
    bureau["OVERDUE_DEBT_RATIO"] = safe_div(bureau["AMT_CREDIT_SUM_OVERDUE"], bureau["AMT_CREDIT_SUM_DEBT"])
    return bureau.reset_index(drop=True), stats


def _prev(cfg, ids: set) -> pd.DataFrame:
    prev = read_raw(cfg, "previous_application")
    prev = prev[prev["SK_ID_CURR"].isin(ids)].copy()
    for col in DAYS_COLS:
        prev.loc[prev[col] == DAYS_ANOMALY, col] = np.nan
    prev["APP_CREDIT_RATIO"] = safe_div(prev["AMT_APPLICATION"], prev["AMT_CREDIT"])
    prev["CREDIT_GOODS_RATIO"] = safe_div(prev["AMT_CREDIT"], prev["AMT_GOODS_PRICE"])
    prev["ANNUITY_CREDIT_RATIO"] = safe_div(prev["AMT_ANNUITY"], prev["AMT_CREDIT"])
    prev["IS_PLACEHOLDER"] = np.float32(0)
    return prev.reset_index(drop=True)


def _installment(cfg, ids: set) -> pd.DataFrame:
    ins = read_raw(cfg, "installments_payments")
    ins = ins[ins["SK_ID_CURR"].isin(ids)].copy()
    ins["PAYMENT_RATIO"] = safe_div(ins["AMT_PAYMENT"], ins["AMT_INSTALMENT"])
    ins["PAYMENT_DIFF"] = (ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]).astype(np.float32)
    delay = ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]
    ins["DPD"] = delay.clip(lower=0).astype(np.float32)
    ins["DBD"] = (-delay).clip(lower=0).astype(np.float32)
    # Kỳ có lịch trả nhưng không ghi nhận thanh toán
    ins["IS_PAYMENT_MISSING"] = ins["AMT_PAYMENT"].isna().astype(np.float32)
    return ins.reset_index(drop=True)


def _pos(cfg, ids: set) -> pd.DataFrame:
    pos = read_raw(cfg, "POS_CASH_balance")
    return pos[pos["SK_ID_CURR"].isin(ids)].reset_index(drop=True)


def _cc(cfg, ids: set) -> pd.DataFrame:
    cc = read_raw(cfg, "credit_card_balance")
    cc = cc[cc["SK_ID_CURR"].isin(ids)].copy()
    cc["UTILIZATION"] = safe_div(cc["AMT_BALANCE"], cc["AMT_CREDIT_LIMIT_ACTUAL"])
    cc["PAYMENT_MIN_RATIO"] = safe_div(cc["AMT_PAYMENT_CURRENT"], cc["AMT_INST_MIN_REGULARITY"])
    return cc.reset_index(drop=True)


def load_node_tables(cfg: dict, customer_ids) -> tuple[dict[str, pd.DataFrame], dict]:
    """Trả về bảng node thô cho từng node type (chưa xử lý orphan) + thống kê bureau_balance."""
    ids = set(customer_ids)
    tables = {CUSTOMER: _customer(cfg, customer_ids)}
    tables[BUREAU], bb_stats = _bureau(cfg, ids)
    for t, fn in [(PREV, _prev), (INSTALLMENT, _installment), (POS, _pos), (CC, _cc)]:
        tables[t] = fn(cfg, ids)
        log.info("%s: %d nodes", t, len(tables[t]))
    log.info("%s: %d nodes | %s: %d nodes", CUSTOMER, len(tables[CUSTOMER]), BUREAU, len(tables[BUREAU]))
    return tables, bb_stats
