"""Build a sharded logical global graph without federated client partitioning."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.graph.edge_builder import build_edges
from src.graph.node_builder import build_nodes
from src.graph.orphan_handler import prepare_transaction_nodes
from src.graph.schema import NODE_TABLES, TRANSACTION_TYPES, GRAPH_SCHEMA_VERSION
from src.partition.relational_partition import TABLE_NAMES
from src.preprocessing.global_encoder import transform_table


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_global_shards(table_dir: Path, split_path: Path, staging_dir: Path,
                        num_shards: int = 20, chunksize: int = 200_000) -> dict:
    """Place each customer and all owned history rows in one technical shard."""
    if num_shards <= 0 or chunksize <= 0:
        raise ValueError("num_shards and chunksize must be positive")
    customers = pd.read_csv(table_dir / "application_train.csv")
    split = pd.read_csv(split_path)
    if customers.SK_ID_CURR.isna().any() or not customers.SK_ID_CURR.is_unique:
        raise ValueError("Application customer IDs must be unique and non-null")
    if set(customers.SK_ID_CURR) != set(split.SK_ID_CURR):
        raise ValueError("Application customers and global split do not match")
    split = split.set_index("SK_ID_CURR").loc[customers.SK_ID_CURR].reset_index()
    shard_ids = np.arange(len(customers), dtype=np.int64) % num_shards
    owner = pd.Series(shard_ids, index=customers.SK_ID_CURR)
    schema = {name: pd.read_csv(table_dir / f"{name}.csv", nrows=0).columns.tolist()
              for name in TABLE_NAMES}
    staging_dir.mkdir(parents=True, exist_ok=True)
    names = [f"shard_{i:03d}" for i in range(num_shards)]
    counts = {}
    for shard_id, name in enumerate(names):
        directory = staging_dir / name
        directory.mkdir(parents=True, exist_ok=True)
        mask = shard_ids == shard_id
        customers.loc[mask].to_csv(directory / "application_train.csv", index=False)
        split.loc[mask].to_csv(directory / "customer_split.csv", index=False)
        for table in TABLE_NAMES[1:]:
            pd.DataFrame(columns=schema[table]).to_csv(directory / f"{table}.csv", index=False)
        counts[name] = {"customers": int(mask.sum())}
    table_counts = {}
    for table in TABLE_NAMES[1:]:
        assigned = unassigned = 0
        print(f"Staging {table}...", flush=True)
        for chunk in pd.read_csv(table_dir / f"{table}.csv", chunksize=chunksize):
            ids = chunk.SK_ID_CURR.map(owner)
            assigned += int(ids.notna().sum())
            unassigned += int(ids.isna().sum())
            for shard_id, rows in chunk.loc[ids.notna()].groupby(ids.loc[ids.notna()]):
                rows.to_csv(staging_dir / names[int(shard_id)] / f"{table}.csv",
                            mode="a", header=False, index=False)
        table_counts[table] = {"assigned_rows": assigned, "unassigned_rows": unassigned}
    report = {"artifact": "global_graph_staging", "num_shards": num_shards,
              "shards": names, "schema": schema, "split_hash": _hash(split_path),
              "tables": table_counts, "counts": counts}
    (staging_dir / "staging_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_global_shard(shard_dir: Path, output_path: Path, schema: dict,
                       encoder: dict, quarantine_root: Path) -> dict:
    tables = {}
    for table, _ in NODE_TABLES.values():
        dtype = {spec["column"]: "string" for spec in encoder["tables"][table]["columns"]}
        frame = pd.read_csv(shard_dir / f"{table}.csv", dtype=dtype)
        if frame.columns.tolist() != schema[table]:
            raise ValueError(f"{shard_dir.name}/{table}: inconsistent schema")
        tables[table] = frame
    tables, quarantine = prepare_transaction_nodes(tables)
    quarantine_dir = quarantine_root / shard_dir.name
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    quarantine_report = {}
    for table, rows in quarantine.items():
        rows.to_csv(quarantine_dir / f"{table}.csv", index=False)
        quarantine_report[table] = {"rows": len(rows)}
    nodes = build_nodes(tables)
    edges, missing = build_edges(nodes)
    split = pd.read_csv(shard_dir / "customer_split.csv").set_index("SK_ID_CURR")
    customers = tables["application_train"]
    split = split.loc[customers.SK_ID_CURR]
    train = split.train_mask.to_numpy(bool)
    val = split.val_mask.to_numpy(bool)
    test = split.test_mask.to_numpy(bool)
    if not np.all(train.astype(int) + val.astype(int) + test.astype(int) == 1):
        raise ValueError("Train/validation/test masks must be disjoint and exhaustive")
    payload = {"y": customers.TARGET.to_numpy(np.float32), "train_mask": train,
               "val_mask": val, "test_mask": test,
               "customer_ids": customers.SK_ID_CURR.to_numpy(np.int64)}
    customer_owner = nodes["customer"].set_index("SK_ID_CURR").node_id
    for node_type, (table, _) in NODE_TABLES.items():
        features = transform_table(tables[table], encoder["tables"][table])
        if node_type in TRANSACTION_TYPES:
            flag = tables[table].is_orphan_prev.to_numpy(np.float32)
            features = np.column_stack((features, flag))
        payload[f"x__{node_type}"] = features
        payload[f"owner__{node_type}"] = nodes[node_type].SK_ID_CURR.map(customer_owner).to_numpy(np.int64)
    payload.update({f"edge__{key}": value for key, value in edges.items()})
    metadata = {"shard": shard_dir.name, "artifact": "global_graph_shard",
                "graph_schema_version": GRAPH_SCHEMA_VERSION,
                "encoder_fingerprint": encoder["fingerprint"],
                "node_counts": {key: len(value) for key, value in nodes.items()},
                "edge_counts": {key: value.shape[1] for key, value in edges.items()},
                "missing_previous_links": missing, "quarantine": quarantine_report}
    payload["metadata"] = np.array(json.dumps(metadata))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return metadata


def build_global_graph(table_dir: Path, split_path: Path, encoder_path: Path,
                       output_dir: Path, staging_dir: Path,
                       num_shards: int = 20, chunksize: int = 200_000) -> dict:
    encoder = json.loads(encoder_path.read_text(encoding="utf-8"))
    if encoder["split_hash"] != _hash(split_path):
        raise ValueError("Global split changed since encoder fit")
    
    staging_report = staging_dir / "staging_report.json"
    staging = None

    if staging_report.is_file():
        candidate = json.loads(
            staging_report.read_text(encoding="utf-8")
        )
        expected = [
            staging_dir / name / f"{table}.csv"
            for name in candidate.get("shards", [])
            for table in TABLE_NAMES
        ]
        expected += [
            staging_dir / name / "customer_split.csv"
            for name in candidate.get("shards", [])
        ]

        if (
            candidate.get("num_shards") == num_shards
            and candidate.get("split_hash") == _hash(split_path)
            and len(candidate.get("shards", [])) == num_shards
            and all(path.is_file() for path in expected)
        ):
            staging = candidate
            print(
                "Reusing completed global graph staging...",
                flush=True,
            )

    if staging is None:
        staging = stage_global_shards(
            table_dir,
            split_path,
            staging_dir,
            num_shards,
            chunksize,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for name in staging["shards"]:
        print(f"Building {name}...", flush=True)
        reports.append(build_global_shard(staging_dir / name, output_dir / f"{name}.npz",
                                          staging["schema"], encoder, output_dir / "quarantine"))
    manifest = {"artifact": "sharded_logical_global_graph", "num_shards": num_shards,
                "shards": [f"{name}.npz" for name in staging["shards"]],
                "split_hash": staging["split_hash"],
                "encoder_fingerprint": encoder["fingerprint"],
                "graph_schema_version": GRAPH_SCHEMA_VERSION,
                "reports": reports}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
