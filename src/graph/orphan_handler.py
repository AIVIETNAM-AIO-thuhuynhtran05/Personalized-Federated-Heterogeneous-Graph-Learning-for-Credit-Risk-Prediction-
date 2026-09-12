# Phân biệt thiếu Previous có thể giữ lại với ownership sai phải cách ly; không sửa DataFrame nguồn tại chỗ.
"""Keep resolvable transactions; quarantine invalid customer ownership for review."""
import pandas as pd
from src.graph.schema import NODE_TABLES, TRANSACTION_TYPES


def prepare_transaction_nodes(tables):
    tables = dict(tables)
    customers = tables["application_train"].SK_ID_CURR
    previous = tables["previous_application"]
    if previous.SK_ID_PREV.isna().any() or not previous.SK_ID_PREV.is_unique:
        raise ValueError("Previous application IDs must be unique and non-null")
    previous_owner = previous.set_index("SK_ID_PREV").SK_ID_CURR
    quarantine = {}
    for node_type in TRANSACTION_TYPES:
        table = NODE_TABLES[node_type][0]
        frame = tables[table]
        exists = frame.SK_ID_PREV.isin(previous_owner.index)
        valid_customer = frame.SK_ID_CURR.notna() & frame.SK_ID_CURR.isin(customers)
        mismatch = exists & frame.SK_ID_CURR.ne(frame.SK_ID_PREV.map(previous_owner)).fillna(True)
        reason = pd.Series("", index=frame.index, dtype="string")
        reason.loc[mismatch] = "previous_customer_mismatch"
        # Lỗi Customer được ưu tiên nếu một hàng đồng thời có nhiều lỗi ownership.
        reason.loc[~valid_customer] = "customer_not_in_client"
        rejected = reason.ne("")
        review = frame.loc[rejected].copy()
        review.insert(0, "source_row_id", frame.index[rejected])
        review["quarantine_reason"] = reason.loc[rejected]
        quarantine[table] = review
        kept = frame.loc[~rejected].copy()
        # Thiếu Previous vẫn được giữ nếu Customer hợp lệ; cờ này điều khiển cạnh fallback.
        kept["is_orphan_prev"] = (~exists.loc[~rejected]).astype("int8")
        tables[table] = kept
    return tables, quarantine
