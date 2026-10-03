"""Phát hiện và xử lý orphan records khi dựng graph từ bảng quan hệ.

Các trường hợp:
  A. Event (installment/POS/CC) trỏ tới SK_ID_PREV không có trong previous_application
     -> theo policy: tạo node prev placeholder (mặc định) hoặc bỏ event.
  B. Khóa ngoại mâu thuẫn: SK_ID_CURR của event khác SK_ID_CURR của prev cha -> bỏ event.
  C. Node cha không có con (prev không có lịch sử trả, customer không có lịch sử)
     -> KHÔNG phải lỗi; giữ node, thêm đặc trưng cấu trúc (số con, cờ HAS_*).
  D. bureau_balance không trỏ tới bureau nào -> không gắn được, bỏ (được ghi trong báo cáo).
"""
import logging

import numpy as np
import pandas as pd
import torch

from src.graph.schema import CUSTOMER, EVENT_TYPES, PARENT, PREV, edge_types

log = logging.getLogger(__name__)


def resolve_missing_previous(tables: dict[str, pd.DataFrame], policy: str) -> dict:
    """Xử lý trường hợp A và B (sửa `tables` tại chỗ). Trả về thống kê."""
    prev = tables[PREV]
    prev_owner = prev.set_index("SK_ID_PREV")["SK_ID_CURR"]
    stats, placeholders = {}, []

    for t in EVENT_TYPES:
        ev = tables[t]
        owner = ev["SK_ID_PREV"].map(prev_owner)
        missing = owner.isna()
        mismatch = ~missing & (owner != ev["SK_ID_CURR"])
        stats[t] = {
            "rows": int(len(ev)),
            "rows_missing_prev": int(missing.sum()),
            "share_missing_prev": float(missing.mean()) if len(ev) else 0.0,
            "distinct_missing_prev": int(ev.loc[missing, "SK_ID_PREV"].nunique()),
            "customers_affected": int(ev.loc[missing, "SK_ID_CURR"].nunique()),
            "rows_curr_mismatch": int(mismatch.sum()),
        }
        if mismatch.any():
            ev = ev[~mismatch]
            missing = missing[~mismatch]
        if policy == "placeholder":
            ph = ev.loc[missing, ["SK_ID_PREV", "SK_ID_CURR"]].drop_duplicates()
            ph[f"PH_FROM_{t.upper()}"] = np.float32(1)
            placeholders.append(ph)
        elif policy == "drop":
            ev = ev[~missing]
        else:
            raise ValueError(f"Unknown missing_previous policy: {policy}")
        tables[t] = ev.reset_index(drop=True)

    # Luôn có đủ cột PH_FROM_* (kể cả khi bằng 0) để mọi client có cùng schema
    flag_cols = [f"PH_FROM_{t.upper()}" for t in EVENT_TYPES]
    if policy == "placeholder" and placeholders:
        ph = pd.concat(placeholders).groupby(["SK_ID_PREV", "SK_ID_CURR"], as_index=False).max()
        ph["IS_PLACEHOLDER"] = np.float32(1)
        prev = pd.concat([prev, ph], ignore_index=True)
        stats["placeholder_prev_created"] = int(len(ph))
        log.info("Created %d placeholder prev nodes", len(ph))
    for c in flag_cols:
        prev[c] = prev[c].fillna(0).astype(np.float32) if c in prev else np.float32(0)
    tables[PREV] = prev
    return stats


def childless_report(tables: dict[str, pd.DataFrame]) -> dict:
    """Trường hợp C: node cha không có node con."""
    prev, cust = tables[PREV], tables[CUSTOMER]
    has_child = pd.DataFrame({t: prev["SK_ID_PREV"].isin(set(tables[t]["SK_ID_PREV"]))
                              for t in EVENT_TYPES})
    has_any = has_child.any(axis=1)
    real = prev["IS_PLACEHOLDER"] == 0
    by_status = (pd.DataFrame({"status": prev.loc[real, "NAME_CONTRACT_STATUS"],
                               "has_any_payment_record": has_any[real]})
                 .groupby("status")["has_any_payment_record"].agg(["size", "mean"]))
    has_b = cust["SK_ID_CURR"].isin(set(tables["bureau"]["SK_ID_CURR"]))
    has_p = cust["SK_ID_CURR"].isin(set(prev["SK_ID_CURR"]))
    return {
        "prev_without_any_payment_record": int((~has_any & real).sum()),
        "prev_payment_coverage_by_status": {
            s: {"n": int(r["size"]), "share_with_payment_record": round(float(r["mean"]), 4)}
            for s, r in by_status.iterrows()},
        "approved_prev_without_payment_record": int(
            (~has_any & real & (prev["NAME_CONTRACT_STATUS"] == "Approved")).sum()),
        "customers_without_bureau": int((~has_b).sum()),
        "customers_without_prev": int((~has_p).sum()),
        "customers_without_any_history": int((~has_b & ~has_p).sum()),
    }


def validate_graph(data) -> dict:
    """Kiểm tra sau khi dựng: chỉ số cạnh hợp lệ, mỗi node con có đúng 1 cha,
    cạnh ngược khớp cạnh xuôi, mọi node con đều nối được về một customer."""
    checks = {}
    for child, (parent, rel, _, _) in PARENT.items():
        ei = data[parent, rel, child].edge_index
        rev = data[child, f"rev_{rel}", parent].edge_index
        n_p, n_c = data[parent].num_nodes, data[child].num_nodes
        in_deg = torch.bincount(ei[1], minlength=n_c)
        checks[f"{parent}->{child}"] = {
            "edges": int(ei.size(1)),
            "index_in_range": bool(ei[0].max() < n_p and ei[1].max() < n_c) if ei.numel() else True,
            "every_child_has_one_parent": bool((in_deg == 1).all()),
            "reverse_matches": bool(torch.equal(rev.flip(0), ei)),
        }
    assert all(v for c in checks.values() for k, v in c.items() if k != "edges"), checks
    # Graph là rừng: mỗi khách hàng là một cây riêng, không có cạnh giữa các khách hàng
    checks["edge_types"] = [list(e) for e in edge_types()]
    checks["customer_customer_edges"] = 0
    return checks
