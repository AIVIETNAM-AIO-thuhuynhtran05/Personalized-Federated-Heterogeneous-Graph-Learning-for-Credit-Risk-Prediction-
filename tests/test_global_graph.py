import hashlib
import json

import numpy as np
import pandas as pd

from src.graph.global_heterograph_builder import build_global_graph
from src.partition.relational_partition import TABLE_NAMES


def test_sharded_global_graph_preserves_customers_masks_and_edges(tmp_path):
    tables = tmp_path / "tables"
    tables.mkdir(parents=True)
    customers = pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3, 4, 5, 6],
        "TARGET": [0, 1, 0, 1, 0, 1],
        "feature": [1., 2., 3., 4., 5., 6.],
    })
    customers.to_csv(tables / "application_train.csv", index=False)
    pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3, 4, 5, 6],
        "SK_ID_BUREAU": [11, 12, 13, 14, 15, 16],
        "feature": [1., 2., 3., 4., 5., 6.],
    }).to_csv(tables / "bureau.csv", index=False)
    pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3, 4, 5, 6],
        "SK_ID_PREV": [101, 102, 103, 104, 105, 106],
        "feature": [1., 2., 3., 4., 5., 6.],
    }).to_csv(tables / "previous_application.csv", index=False)
    for table in TABLE_NAMES[3:]:
        pd.DataFrame({
            "SK_ID_CURR": [1, 2, 3, 4, 5, 6],
            "SK_ID_PREV": [101, 102, 999, 104, 105, 106],
            "feature": [1., 2., 3., 4., 5., 6.],
        }).to_csv(tables / f"{table}.csv", index=False)

    split = pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3, 4, 5, 6],
        "TARGET": [0, 1, 0, 1, 0, 1],
        "split": ["train", "train", "validation", "validation", "test", "test"],
        "train_mask": [1, 1, 0, 0, 0, 0],
        "val_mask": [0, 0, 1, 1, 0, 0],
        "test_mask": [0, 0, 0, 0, 1, 1],
    })
    split_path = tmp_path / "customer_split.csv"
    split.to_csv(split_path, index=False)
    split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    definition = {
        "columns": [{"column": "feature", "kind": "numeric", "mean": 2.0,
                     "scale": 1.0, "count": 2, "missing_ratio": 0.0}],
        "dimension": 1,
    }
    encoder = {
        "split_hash": split_hash,
        "fingerprint": "test-encoder",
        "tables": {table: definition for table in TABLE_NAMES},
    }
    encoder_path = tmp_path / "encoder.json"
    encoder_path.write_text(json.dumps(encoder), encoding="utf-8")

    output = tmp_path / "graphs"
    manifest = build_global_graph(tables, split_path, encoder_path, output,
                                  tmp_path / "staging", num_shards=3, chunksize=2)
    assert manifest["artifact"] == "sharded_logical_global_graph"
    assert len(manifest["shards"]) == 3
    seen, mask_counts, orphan_edges = [], np.zeros(3, dtype=int), 0
    for filename in manifest["shards"]:
        with np.load(output / filename) as graph:
            seen.extend(graph["customer_ids"].tolist())
            masks = np.column_stack((graph["train_mask"], graph["val_mask"], graph["test_mask"]))
            assert np.all(masks.sum(axis=1) == 1)
            mask_counts += masks.sum(axis=0)
            orphan_edges += graph["edge__customer__has_orphan__installment"].shape[1]
    assert sorted(seen) == [1, 2, 3, 4, 5, 6]
    np.testing.assert_array_equal(mask_counts, [2, 2, 2])
    assert orphan_edges == 1
