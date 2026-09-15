import hashlib
import json

import numpy as np
import pandas as pd

from src.models.global_tabular import select_f1_threshold, train_global_tabular


def _fixture(root, test_shift=0.0):
    root.mkdir()
    ids = np.arange(1, 13)
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    values = np.array([-3., 3., -2., 2., -1., 1., -2.5, 2.5, -1.5, 1.5, -0.5, 0.5])
    values[8:] += test_shift
    application = root / "application_train.csv"
    pd.DataFrame({"SK_ID_CURR": ids, "TARGET": labels, "feature": values}).to_csv(
        application, index=False
    )
    names = ["train"] * 6 + ["validation"] * 2 + ["test"] * 4
    split = pd.DataFrame({"SK_ID_CURR": ids, "TARGET": labels, "split": names})
    split["train_mask"] = split["split"].eq("train")
    split["val_mask"] = split["split"].eq("validation")
    split["test_mask"] = split["split"].eq("test")
    split_path = root / "customer_split.csv"
    split.to_csv(split_path, index=False)
    encoder = {
        "fingerprint": "unit-test-encoder",
        "split_hash": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "tables": {"application_train": {
            "columns": [{"column": "feature", "kind": "numeric", "mean": 0.0,
                         "scale": 1.0, "count": 6, "missing_ratio": 0.0}],
            "dimension": 1,
        }},
    }
    encoder_path = root / "encoder.json"
    encoder_path.write_text(json.dumps(encoder), encoding="utf-8")
    return application, split_path, encoder_path


def test_threshold_selection_uses_validation_curve():
    threshold = select_f1_threshold([0, 1, 0, 1], [0.1, 0.8, 0.2, 0.7])
    assert 0.2 < threshold <= 0.8


def test_test_features_cannot_change_fit_or_validation_threshold(tmp_path):
    first = _fixture(tmp_path / "first", test_shift=0.0)
    second = _fixture(tmp_path / "second", test_shift=10000.0)
    rows_first = train_global_tabular(*first, tmp_path / "out_first", seed=42)
    rows_second = train_global_tabular(*second, tmp_path / "out_second", seed=42)
    val_first = next(r for r in rows_first if r["subset"] == "validation"
                     and r["threshold_rule"] == "validation_f1")
    val_second = next(r for r in rows_second if r["subset"] == "validation"
                      and r["threshold_rule"] == "validation_f1")
    assert val_first["threshold"] == val_second["threshold"]
    assert val_first["roc_auc"] == val_second["roc_auc"]
    config = json.loads((tmp_path / "out_first/run_config.json").read_text())
    assert config["train_scope"] == "global_train_only"
    assert config["threshold_scope"] == "global_validation_only"
    assert config["test_scope"] == "global_test_only"
