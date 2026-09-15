import json

import numpy as np
import pandas as pd

from src.models.global_gnn import train_global_gnn


NODE_TYPES = ("customer", "bureau", "previous_application", "installment",
              "pos_cash", "credit_card")


def _write_shard(path, start_id, fingerprint):
    count = 6
    labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.float32)
    payload = {
        "y": labels,
        "train_mask": np.array([1, 1, 0, 0, 0, 0], dtype=bool),
        "val_mask": np.array([0, 0, 1, 1, 0, 0], dtype=bool),
        "test_mask": np.array([0, 0, 0, 0, 1, 1], dtype=bool),
        "customer_ids": np.arange(start_id, start_id + count, dtype=np.int64),
    }
    for offset, node in enumerate(NODE_TYPES):
        payload[f"x__{node}"] = np.column_stack((
            np.linspace(-1, 1, count), np.full(count, offset)
        )).astype(np.float32)
        payload[f"owner__{node}"] = np.arange(count, dtype=np.int64)
    edge = np.vstack((np.arange(count), np.arange(count))).astype(np.int64)
    payload["edge__customer__has__bureau"] = edge
    payload["edge__bureau__rev_has__customer"] = edge[::-1].copy()
    payload["metadata"] = np.array(json.dumps({
        "artifact": "global_graph_shard",
        "encoder_fingerprint": fingerprint,
    }))
    np.savez_compressed(path, **payload)


def test_global_gnn_end_to_end_uses_validation_checkpoint(tmp_path):
    graph_dir = tmp_path / "graphs"
    graph_dir.mkdir()
    fingerprint = "unit-test-encoder"
    shards = ["shard_000.npz", "shard_001.npz"]
    _write_shard(graph_dir / shards[0], 100, fingerprint)
    _write_shard(graph_dir / shards[1], 200, fingerprint)
    manifest = {
        "artifact": "sharded_logical_global_graph",
        "shards": shards,
        "num_shards": 2,
        "encoder_fingerprint": fingerprint,
        "graph_schema_version": "test",
    }
    (graph_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    output = tmp_path / "output"
    rows = train_global_gnn(graph_dir, output, epochs=2, batch_size=2,
                            learning_rate=1e-3, hidden=4, layers=1,
                            patience=2, seed=42, device="cpu", threads=1)
    assert (output / "global_gnn.pt").is_file()
    assert (output / "history.csv").is_file()
    assert (output / "metrics.csv").is_file()
    assert len(pd.read_csv(output / "history.csv")) == 2
    test_rows = [row for row in rows if row["subset"] == "test"]
    assert {row["threshold_rule"] for row in test_rows} == {
        "validation_f1", "fixed_0.5"
    }
    config = json.loads((output / "run_config.json").read_text())
    assert config["optimizer_scope"] == "one_shared_optimizer_across_all_shards"
    assert config["checkpoint_selection"] == "validation_pr_auc"
    assert config["threshold_scope"] == "validation_only"
    assert config["test_scope"] == "global_test_only"
    assert config["num_shards"] == 2
