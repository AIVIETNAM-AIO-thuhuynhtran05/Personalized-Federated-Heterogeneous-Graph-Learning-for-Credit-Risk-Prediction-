# Ghép node, cạnh và đặc trưng; encoder có mặt tạo NPZ huấn luyện, encoder=None chỉ xuất topology.
"""Build client-local graphs using a previously fitted shared train-only encoder."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from src.graph.schema import NODE_TABLES, TRANSACTION_TYPES, GRAPH_SCHEMA_VERSION
from src.graph.orphan_handler import prepare_transaction_nodes
from src.graph.node_builder import build_nodes
from src.graph.edge_builder import build_edges
from src.preprocessing.shared_encoder import transform_table


def build_client_graph(client_dir: Path, output_dir: Path, schema: dict, encoder=None):
    tables = {}
    for table, _ in NODE_TABLES.values():
        dtypes = ({spec["column"]: "string" for spec in encoder["tables"][table]["columns"]}
                  if encoder is not None else None)
        frame = pd.read_csv(client_dir / f"{table}.csv", dtype=dtypes)
        if frame.columns.tolist() != schema[table]:
            raise ValueError(f"{client_dir.name}/{table}: inconsistent global schema")
        tables[table] = frame
    tables, quarantine = prepare_transaction_nodes(tables)
    quarantine_dir = output_dir.parent / "quarantine" / client_dir.name
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    quarantine_report = {}
    for table, rows in quarantine.items():
        rows.to_csv(quarantine_dir / f"{table}.csv", index=False)
        quarantine_report[table] = {"rows": len(rows), "reasons": rows.quarantine_reason.value_counts().to_dict()}
    nodes = build_nodes(tables)
    edges, missing = build_edges(nodes)
    if encoder is not None:
        split_path = client_dir / "customer_split.csv"
        # Không cho dùng encoder cũ khi nội dung split đã đổi.
        if hashlib.sha256(split_path.read_bytes()).hexdigest() != encoder["split_hashes"][client_dir.name]:
            raise ValueError("Split changed since encoder fit; refit on current local-train rows")
        split = pd.read_csv(split_path).set_index("SK_ID_CURR")
        customers = tables["application_train"]
        # Căn thứ tự mask theo thứ tự Customer trong graph, không dựa thứ tự CSV split.
        split = split.loc[customers.SK_ID_CURR]
        train = split.train_mask.to_numpy(dtype=bool)
        test = split.test_mask.to_numpy(dtype=bool)
        if np.any(train & test) or not np.all(train | test):
            raise ValueError("Train/test masks must be disjoint and exhaustive")
        payload = {"y": customers.TARGET.to_numpy(dtype=np.float32),
                   "train_mask": train, "test_mask": test,
                   "customer_ids": customers.SK_ID_CURR.to_numpy(dtype=np.int64)}
        owner = nodes["customer"].set_index("SK_ID_CURR").node_id
        for node_type, (table, _) in NODE_TABLES.items():
            payload[f"x__{node_type}"] = transform_table(tables[table], encoder["tables"][table])
            if node_type in TRANSACTION_TYPES:
                flag = tables[table].is_orphan_prev.to_numpy(dtype=np.float32)
                # Thêm cờ cấu trúc ở chiều cuối, giữ nguyên 0/1 thay vì chuẩn hóa bằng encoder.
                payload[f"x__{node_type}"] = np.column_stack((payload[f"x__{node_type}"], flag))
                payload[f"is_orphan_prev__{node_type}"] = flag.astype(np.int8)
            payload[f"owner__{node_type}"] = nodes[node_type].SK_ID_CURR.map(owner).to_numpy(dtype=np.int64)
            for key in nodes[node_type].columns:
                payload[f"mapping__{node_type}__{key}"] = pd.to_numeric(nodes[node_type][key]).to_numpy(dtype=np.float64)
        # Lưu cả quan hệ thuận/ngược, kể cả mảng rỗng để schema đồng nhất giữa client.
        payload.update({f"edge__{key}": value for key, value in edges.items()})
        metadata = {"client": client_dir.name, "encoder_fingerprint": encoder["fingerprint"],
                    "graph_schema_version": GRAPH_SCHEMA_VERSION,
                    "extra_features": {node: ["is_orphan_prev"] for node in TRANSACTION_TYPES},
                    "quarantine": quarantine_report, "quarantine_path": str(quarantine_dir.resolve()),
                    "node_counts": {k: len(v) for k, v in nodes.items()},
                    "missing_previous_links": missing, "artifact": "encoded_heterogeneous_graph"}
        payload["metadata"] = np.array(json.dumps(metadata))
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_dir.with_suffix(".npz"), **payload)
        return metadata
    # Nhánh encoder=None chỉ xuất topology; entry point huấn luyện yêu cầu graph đã mã hóa.
    output_dir.mkdir(parents=True, exist_ok=True)
    for node_type, mapping in nodes.items():
        mapping.to_csv(output_dir / f"{node_type}_nodes.csv", index=False)
    np.savez_compressed(output_dir / "edges.npz", **edges)
    report = {"client": client_dir.name, "node_counts": {k: len(v) for k, v in nodes.items()},
              "graph_schema_version": GRAPH_SCHEMA_VERSION, "quarantine": quarantine_report,
              "quarantine_path": str(quarantine_dir.resolve()),
              "edge_counts": {k: v.shape[1] for k, v in edges.items()},
              "missing_previous_links": missing, "schema": schema,
              "artifact": "topology_only", "source_tables": str(client_dir.resolve())}
    (output_dir / "graph_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
