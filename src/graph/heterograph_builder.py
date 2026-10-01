"""Dựng HeteroData từ bảng quan hệ: node -> xử lý orphan -> cạnh -> encode đặc trưng -> kiểm tra."""
import logging

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from src.graph.edge_builder import add_structural_features, edge_index_dict, link_tables, root_customer
from src.graph.node_builder import load_node_tables
from src.graph.orphan_check import childless_report, resolve_missing_previous, validate_graph
from src.graph.schema import CATEGORICAL, CUSTOMER, NODE_TYPES
from src.preprocessing.feature_transformer import make_linear_preprocessor

log = logging.getLogger(__name__)

NON_FEATURE = {"SK_ID_CURR", "SK_ID_PREV", "SK_ID_BUREAU", "TARGET"}


def _columns(t: str, df: pd.DataFrame) -> tuple[list[str], list[str]]:
    if t == CUSTOMER:
        cat = [c for c in df.columns if df[c].dtype == object and c not in NON_FEATURE]
    else:
        cat = CATEGORICAL[t]
    num = [c for c in df.columns if c not in NON_FEATURE and c not in cat]
    return num, cat


def encode_nodes(tables, roots, train_customer_mask, gcfg, seed, chunk=1_000_000):
    """Fit encoder của từng node type CHỈ trên node thuộc khách hàng Train, áp cho mọi node."""
    rng = np.random.default_rng(seed)
    xs, names, encoders = {}, {}, {}
    for t in NODE_TYPES:
        df = tables[t]
        num, cat = _columns(t, df)
        enc = make_linear_preprocessor(num, cat, gcfg["clip_quantiles"], gcfg["onehot_min_frequency"])
        fit_idx = np.flatnonzero(train_customer_mask[roots[t]])
        if len(fit_idx) > gcfg["encoder_fit_max_rows"]:
            fit_idx = rng.choice(fit_idx, gcfg["encoder_fit_max_rows"], replace=False)
        enc.fit(df.iloc[np.sort(fit_idx)])
        parts = [enc.transform(df.iloc[i:i + chunk]).astype(np.float32) for i in range(0, len(df), chunk)]
        xs[t] = torch.from_numpy(np.concatenate(parts))
        names[t] = list(enc.get_feature_names_out())
        encoders[t] = enc
        log.info("encoded %-11s %9d nodes x %3d features", t, *xs[t].shape)
    return xs, names, encoders


def build_hetero_graph(cfg, gcfg, customer_ids, train_customer_ids):
    tables, bb_stats = load_node_tables(cfg, customer_ids)

    report = {"bureau_balance": bb_stats}
    report["missing_previous"] = resolve_missing_previous(tables, gcfg["orphan_policy"]["missing_previous"])
    report["childless"] = childless_report(tables)

    parents = link_tables(tables)
    add_structural_features(tables, parents)
    roots = root_customer(parents, len(tables[CUSTOMER]))
    train_mask = tables[CUSTOMER]["SK_ID_CURR"].isin(set(train_customer_ids)).values

    xs, names, encoders = encode_nodes(tables, roots, train_mask, gcfg, cfg["seed"])

    data = HeteroData()
    for t in NODE_TYPES:
        data[t].x = xs[t]
        data[t].num_nodes = xs[t].size(0)
    data[CUSTOMER].y = torch.from_numpy(tables[CUSTOMER]["TARGET"].values.astype(np.float32))
    data[CUSTOMER].sk_id_curr = torch.from_numpy(tables[CUSTOMER]["SK_ID_CURR"].values.astype(np.int64))
    for et, ei in edge_index_dict(parents).items():
        data[et].edge_index = ei

    report["validation"] = validate_graph(data)
    report["nodes"] = {t: int(data[t].num_nodes) for t in NODE_TYPES}
    report["feature_dims"] = {t: len(names[t]) for t in NODE_TYPES}
    return data, report, {"feature_names": names, "encoders": encoders}
