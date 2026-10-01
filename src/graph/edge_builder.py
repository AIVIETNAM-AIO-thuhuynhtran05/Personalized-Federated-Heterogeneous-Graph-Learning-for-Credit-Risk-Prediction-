"""Tạo cạnh theo khóa ngoại. Node con được sắp theo chỉ số node cha để
các node con của cùng một cha nằm liền nhau (cho phép cắt subgraph nhanh theo CSR)."""
import numpy as np
import pandas as pd
import torch

from src.graph.schema import BUREAU, CUSTOMER, EVENT_TYPES, PARENT, PREV


def _link(child: pd.DataFrame, fk: str, parent_keys: pd.Series) -> tuple[pd.DataFrame, np.ndarray]:
    pos = pd.Series(np.arange(len(parent_keys)), index=parent_keys.values)
    parent = child[fk].map(pos)
    assert parent.notna().all(), f"Dangling foreign key {fk}: chạy resolve_missing_previous trước"
    order = np.argsort(parent.values, kind="stable")
    return child.iloc[order].reset_index(drop=True), parent.values[order].astype(np.int64)


def link_tables(tables: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    """Sắp xếp lại bảng node con (tại chỗ) và trả về chỉ số node cha của từng node con."""
    parents = {}
    cust_keys = tables[CUSTOMER]["SK_ID_CURR"]
    for t in [BUREAU, PREV]:
        tables[t], parents[t] = _link(tables[t], "SK_ID_CURR", cust_keys)
    prev_keys = tables[PREV]["SK_ID_PREV"]
    for t in EVENT_TYPES:
        tables[t], parents[t] = _link(tables[t], "SK_ID_PREV", prev_keys)
    return parents


def add_structural_features(tables: dict[str, pd.DataFrame], parents: dict[str, np.ndarray]) -> None:
    """Số node con của mỗi node cha -> đặc trưng; giúp mô hình phân biệt 'không có lịch sử'
    với 'có lịch sử nhưng giá trị nhỏ' (trường hợp node không có con)."""
    for child, (parent, _, _, _) in PARENT.items():
        cnt = np.bincount(parents[child], minlength=len(tables[parent])).astype(np.float32)
        tables[parent][f"N_{child.upper()}"] = np.log1p(cnt)
        tables[parent][f"HAS_{child.upper()}"] = (cnt > 0).astype(np.float32)
    prev_ph = tables[PREV]["IS_PLACEHOLDER"].values
    tables[CUSTOMER]["N_PREV_PLACEHOLDER"] = np.log1p(
        np.bincount(parents[PREV], weights=prev_ph, minlength=len(tables[CUSTOMER]))).astype(np.float32)


def root_customer(parents: dict[str, np.ndarray], n_customer: int) -> dict[str, np.ndarray]:
    """Chỉ số customer gốc của mọi node (dùng để biết node thuộc khách hàng Train/Val/Test)."""
    roots = {CUSTOMER: np.arange(n_customer), BUREAU: parents[BUREAU], PREV: parents[PREV]}
    for t in EVENT_TYPES:
        roots[t] = parents[PREV][parents[t]]
    return roots


def edge_index_dict(parents: dict[str, np.ndarray]) -> dict[tuple, torch.Tensor]:
    out = {}
    for child, (parent, rel, _, _) in PARENT.items():
        src = torch.from_numpy(parents[child])
        dst = torch.arange(len(src))
        out[(parent, rel, child)] = torch.stack([src, dst])
        out[(child, f"rev_{rel}", parent)] = torch.stack([dst, src])
    return out
