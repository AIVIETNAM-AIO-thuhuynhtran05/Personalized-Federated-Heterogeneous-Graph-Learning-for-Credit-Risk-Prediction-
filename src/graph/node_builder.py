# Tạo node_id liên tục riêng cho từng loại node; giữ khóa gốc và source_row_id để truy vết.
"""Create client-local contiguous node IDs, retaining original relationship keys."""
import numpy as np
import pandas as pd
from src.graph.schema import NODE_TABLES


def build_nodes(tables):
    nodes = {}
    for node_type, (table, key) in NODE_TABLES.items():
        frame = tables[table]
        if key and (frame[key].isna().any() or not frame[key].is_unique):
            raise ValueError(f"{table}: {key} must be unique and non-null")
        columns = [c for c in ("SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "is_orphan_prev") if c in frame]
        mapping = frame[columns].reset_index(drop=True).copy()
        # node_id liên tục từ 0 theo từng loại node, không dùng trực tiếp ID nghiệp vụ làm chỉ số tensor.
        mapping.insert(0, "node_id", np.arange(len(frame), dtype=np.int64))
        mapping["source_row_id"] = frame.index.to_numpy(dtype=np.int64)
        # Row order in each persisted client table identifies granular event nodes.
        nodes[node_type] = mapping
    return nodes
