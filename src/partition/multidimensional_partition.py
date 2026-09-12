"""Semantic non-IID client partitioning."""
from __future__ import annotations
import hashlib
from pathlib import Path
import pandas as pd

DEFAULT_PARTITION_COLUMNS = ("REGION_RATING_CLIENT_W_CITY", "OCCUPATION_TYPE")

def _stable_bucket(value: str, number_of_clients: int) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16) % number_of_clients

def partition_non_iid(data: pd.DataFrame,
                      partition_columns: tuple[str, ...] = DEFAULT_PARTITION_COLUMNS,
                      number_of_clients: int = 10, missing_label: str = "UNKNOWN"
                      ) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    """Keep every region/occupation group on exactly one federated client."""
    if number_of_clients < 2:
        raise ValueError("number_of_clients must be at least 2")
    missing = [c for c in partition_columns if c not in data.columns]
    if missing:
        raise KeyError(f"Missing partition columns: {missing}")
    keys = data.loc[:, partition_columns].fillna(missing_label).astype(str)
    group_key = keys.agg("|".join, axis=1)
    client_id = group_key.map(lambda value: _stable_bucket(value, number_of_clients))
    clients = {i: data.loc[client_id == i].copy() for i in range(number_of_clients)}
    assignments = pd.DataFrame({"partition_group": group_key, "client_id": client_id})
    if "SK_ID_CURR" in data:
        assignments.insert(0, "SK_ID_CURR", data["SK_ID_CURR"])
    return clients, assignments.reset_index(drop=True)

def save_client_partitions(clients: dict[int, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for client_id, frame in clients.items():
        frame.to_csv(output_dir / f"client_{client_id:03d}.csv", index=False)
