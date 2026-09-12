"""Check graph isolation and tensor contracts before training."""
import numpy as np
from src.graph.schema import NODE_TABLES, TRANSACTION_TYPES, GRAPH_SCHEMA_VERSION


def validate_training_graph(graph, manifest):
    meta = graph["metadata"]
    if meta.get("graph_schema_version") != GRAPH_SCHEMA_VERSION:
        raise ValueError("Rebuild graphs with scripts/03_build_graphs.py (orphan_fallback_v2 required)")
    if meta["encoder_fingerprint"] != manifest["encoder_fingerprint"]:
        raise ValueError("Graph and manifest use different encoders")
    y = graph["y"]
    if y.ndim != 1 or not np.isin(y, [0, 1]).all():
        raise ValueError("Customer labels must be binary and finite")
    for key in ("train_mask", "test_mask"):
        if graph[key].dtype != np.bool_ or graph[key].shape != y.shape:
            raise ValueError(f"Invalid {key}")
    if (graph["train_mask"] & graph["test_mask"]).any() or not (graph["train_mask"] | graph["test_mask"]).all():
        raise ValueError("Train/test masks must be disjoint and exhaustive")
    if len(graph["customer_ids"]) != len(y) or len(np.unique(graph["customer_ids"])) != len(y):
        raise ValueError("Customer IDs must match labels and be unique")
    dimensions = {}
    for node in NODE_TABLES:
        x, owner = graph[f"x__{node}"], graph[f"owner__{node}"]
        if x.ndim != 2 or x.shape[1] == 0 or not np.isfinite(x).all():
            raise ValueError(f"Invalid features: {node}")
        if owner.shape != (len(x),) or not np.issubdtype(owner.dtype, np.integer):
            raise ValueError(f"Invalid ownership: {node}")
        if len(owner) and (owner.min() < 0 or owner.max() >= len(y)):
            raise ValueError(f"Customer owner outside graph: {node}")
        dimensions[node] = x.shape[1]
        if node in TRANSACTION_TYPES:
            flag = graph[f"is_orphan_prev__{node}"]
            if flag.shape != (len(x),) or not np.isin(flag, [0, 1]).all() or not np.array_equal(x[:, -1], flag):
                raise ValueError(f"Invalid orphan flag feature: {node}")
    if not np.array_equal(graph["owner__customer"], np.arange(len(y))):
        raise ValueError("Customer node order differs from labels")
    relations = sorted(key[6:] for key in graph if key.startswith("edge__"))
    expected = [f"customer__has__{node}" for node in ("bureau", "previous_application")]
    expected += [f"previous_application__has__{node}" for node in TRANSACTION_TYPES]
    expected += [f"customer__has_orphan__{node}" for node in TRANSACTION_TYPES]
    expected += [f"{dst}__rev_{relation}__{src}" for src, relation, dst in [name.split("__") for name in expected]]
    if set(relations) != set(expected):
        raise ValueError("Graph relation schema is incomplete or outdated")
    for relation in relations:
        src, kind, dst = relation.split("__")
        edge = graph[f"edge__{relation}"]
        if edge.ndim != 2 or edge.shape[0] != 2 or not np.issubdtype(edge.dtype, np.integer):
            raise ValueError(f"Invalid edge array: {relation}")
        if edge.size:
            if edge.min() < 0 or edge[0].max() >= len(graph[f"x__{src}"]) or edge[1].max() >= len(graph[f"x__{dst}"]):
                raise ValueError(f"Out-of-range edge: {relation}")
            if not np.array_equal(graph[f"owner__{src}"][edge[0]], graph[f"owner__{dst}"][edge[1]]):
                raise ValueError(f"Cross-customer edge: {relation}")
        if not kind.startswith("rev_"):
            if not np.array_equal(edge[::-1], graph[f"edge__{dst}__rev_{kind}__{src}"]):
                raise ValueError(f"Incorrect reverse edge: {relation}")
    return {"dimensions": dimensions, "relations": relations}
