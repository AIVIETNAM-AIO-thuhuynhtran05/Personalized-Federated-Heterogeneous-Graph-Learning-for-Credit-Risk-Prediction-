"""Foreign-key edges and their reverses, never across customer subgraphs."""
import numpy as np
from src.graph.schema import TRANSACTION_TYPES


def build_edges(nodes):
    edges, missing = {}, {}
    customers = nodes["customer"].set_index("SK_ID_CURR")["node_id"]
    previous = nodes["previous_application"].set_index("SK_ID_PREV")

    def add(source, target, source_ids, valid, relation="has"):
        name = f"{source}__{relation}__{target}"
        array = np.vstack([source_ids.loc[valid].to_numpy(dtype=np.int64),
                           nodes[target].loc[valid, "node_id"].to_numpy(dtype=np.int64)])
        edges[name] = array
        edges[f"{target}__rev_{relation}__{source}"] = array[::-1].copy()

    for target, frame in nodes.items():
        if target == "customer":
            continue
        ids = frame.SK_ID_CURR.map(customers)
        if ids.isna().any():
            raise ValueError(f"{target}: customer not present in this client")
        customer_ids = ids
        if target in TRANSACTION_TYPES:
            ids = frame.SK_ID_PREV.map(previous.node_id)
            owners = frame.SK_ID_PREV.map(previous.SK_ID_CURR)
            valid = ids.notna()
            if (owners.loc[valid] != frame.loc[valid, "SK_ID_CURR"]).any():
                raise ValueError(f"{target}: previous application belongs to a different customer")
            missing[target] = int((~valid).sum())
            add("previous_application", target, ids, valid)
            add("customer", target, customer_ids, ~valid, relation="has_orphan")
        else:
            add("customer", target, customer_ids, customer_ids.notna())
    return edges, missing
