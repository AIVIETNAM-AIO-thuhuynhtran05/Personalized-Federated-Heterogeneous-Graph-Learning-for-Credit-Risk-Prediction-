"""Strict centralized tabular baselines with a global Train/Val/Test protocol."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             brier_score_loss, f1_score, log_loss,
                             precision_recall_curve, roc_auc_score)

from src.preprocessing.global_encoder import transform_table


def load_global_tabular(table_path: Path, split_path: Path, encoder_path: Path):
    encoder = json.loads(encoder_path.read_text(encoding="utf-8"))
    actual = hashlib.sha256(split_path.read_bytes()).hexdigest()
    if encoder["split_hash"] != actual:
        raise ValueError("Global split differs from the split used to fit the encoder")
    frame = pd.read_csv(table_path)
    split = pd.read_csv(split_path).set_index("SK_ID_CURR")
    if set(frame.SK_ID_CURR) != set(split.index):
        raise ValueError("Application customers and global split do not match")
    split = split.loc[frame.SK_ID_CURR]
    masks = {
        "train": split.train_mask.to_numpy(bool),
        "validation": split.val_mask.to_numpy(bool),
        "test": split.test_mask.to_numpy(bool),
    }
    if not np.all(sum(mask.astype(int) for mask in masks.values()) == 1):
        raise ValueError("Global masks must be disjoint and exhaustive")
    return {"ids": frame.SK_ID_CURR.to_numpy(np.int64),
            "y": frame.TARGET.to_numpy(np.int64),
            "x": transform_table(frame, encoder["tables"]["application_train"]),
            "masks": masks}, encoder


def make_model(name: str, seed: int):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(learning_rate=0.08, max_iter=200,
            max_leaf_nodes=31, l2_regularization=1.0, random_state=seed)
    raise ValueError(f"Unknown model: {name}")


def select_f1_threshold(labels, scores) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if not len(thresholds):
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-15)
    return float(thresholds[int(np.argmax(f1))])


def metrics(labels, scores, threshold: float) -> dict:
    labels, scores = np.asarray(labels), np.asarray(scores)
    predicted = scores >= threshold
    precision, recall, _ = precision_recall_curve(labels, scores)
    return {"n": len(labels), "n_positive": int(labels.sum()),
            "positive_rate": float(labels.mean()), "threshold": threshold,
            "accuracy": float(accuracy_score(labels, predicted)),
            "f1": float(f1_score(labels, predicted, zero_division=0)),
            "roc_auc": float(roc_auc_score(labels, scores)),
            "pr_auc": float(auc(recall, precision)),
            "average_precision": float(average_precision_score(labels, scores)),
            "brier_score": float(brier_score_loss(labels, scores)),
            "log_loss": float(log_loss(labels, scores, labels=[0, 1]))}


def train_global_tabular(table_path: Path, split_path: Path, encoder_path: Path,
                         output_dir: Path, model_names=("logistic_regression",), seed=42):
    data, encoder = load_global_tabular(table_path, split_path, encoder_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in model_names:
        print(f"Training {name}...", flush=True)
        model = make_model(name, seed)
        train = data["masks"]["train"]
        model.fit(data["x"][train], data["y"][train])
        val = data["masks"]["validation"]
        val_scores = model.predict_proba(data["x"][val])[:, 1]
        threshold = select_f1_threshold(data["y"][val], val_scores)
        for subset, mask in (("validation", val), ("test", data["masks"]["test"])):
            scores = val_scores if subset == "validation" else model.predict_proba(data["x"][mask])[:, 1]
            labels = data["y"][mask]
            rows.append({"model": name, "subset": subset,
                         "threshold_rule": "validation_f1", **metrics(labels, scores, threshold)})
            rows.append({"model": name, "subset": subset,
                         "threshold_rule": "fixed_0.5", **metrics(labels, scores, 0.5)})
            pd.DataFrame({"SK_ID_CURR": data["ids"][mask], "TARGET": labels,
                          "probability_default": scores}).to_csv(
                output_dir / f"{name}_{subset}_predictions.csv", index=False)
        joblib.dump({"model": model, "threshold": threshold}, output_dir / f"{name}.joblib")
    pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)
    config = {"protocol": "strict_global_first", "train_scope": "global_train_only",
              "threshold_scope": "global_validation_only", "test_scope": "global_test_only",
              "models": list(model_names), "seed": seed,
              "encoder_fingerprint": encoder["fingerprint"],
              "n_train": int(data["masks"]["train"].sum()), "n_features": data["x"].shape[1]}
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return rows
