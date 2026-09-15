"""Create one reproducible stratified global train/validation/test split."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _validate(data: pd.DataFrame, train_size: float, val_size: float, test_size: float) -> None:
    required = {"SK_ID_CURR", "TARGET"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("Application data is empty")
    if data.SK_ID_CURR.isna().any() or not data.SK_ID_CURR.is_unique:
        raise ValueError("SK_ID_CURR must be unique and non-null")
    if not data.TARGET.isin([0, 1]).all() or data.TARGET.nunique() != 2:
        raise ValueError("TARGET must contain both binary classes 0 and 1")
    sizes = (train_size, val_size, test_size)
    if any(size <= 0 for size in sizes) or not np.isclose(sum(sizes), 1.0):
        raise ValueError("train_size, val_size and test_size must be positive and sum to 1")


def create_global_split(
    data: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Return disjoint, exhaustive, TARGET-stratified masks for all customers."""
    _validate(data, train_size, val_size, test_size)
    indices = np.arange(len(data))
    # Chuyển tỷ lệ thành số lượng Customer nguyên trước khi split.
    # Việc này tránh lỗi sai số floating-point làm 30 thành 31.
    n_test = int(round(len(data) * test_size))
    n_val = int(round(len(data) * val_size))
    n_train = len(data) - n_test - n_val

    if min(n_train, n_val, n_test) < data.TARGET.nunique():
        raise ValueError(
            "Each split must be large enough to contain both classes"
        )

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=n_test,
        random_state=seed,
        stratify=data.TARGET.to_numpy(),
    )

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=n_val,
        random_state=seed,
        stratify=data.TARGET.to_numpy()[train_val_idx],
    )
    split = np.empty(len(data), dtype=object)
    split[train_idx] = "train"
    split[val_idx] = "validation"
    split[test_idx] = "test"
    result = data.loc[:, ["SK_ID_CURR", "TARGET"]].copy()
    result["split"] = split
    result["train_mask"] = result.split.eq("train")
    result["val_mask"] = result.split.eq("validation")
    result["test_mask"] = result.split.eq("test")
    masks = result[["train_mask", "val_mask", "test_mask"]].sum(axis=1)
    if not masks.eq(1).all():
        raise RuntimeError("Global split is not disjoint and exhaustive")
    return result


def save_global_split(
    application_path: Path,
    output_dir: Path,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> dict:
    data = pd.read_csv(application_path, usecols=["SK_ID_CURR", "TARGET"])
    split = create_global_split(data, train_size, val_size, test_size, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = output_dir / "customer_split.csv"
    split.to_csv(split_path, index=False)
    counts = {}
    for name in ("train", "validation", "test"):
        rows = split.loc[split.split == name]
        counts[name] = {
            "n_customer": int(len(rows)),
            "n_positive": int(rows.TARGET.sum()),
            "n_negative": int((rows.TARGET == 0).sum()),
            "positive_rate": float(rows.TARGET.mean()),
        }
    identity = pd.util.hash_pandas_object(
        data[["SK_ID_CURR", "TARGET"]], index=False
    ).to_numpy().tobytes()
    report = {
        "protocol": "strict_global_first",
        "strategy": "stratified_by_TARGET",
        "seed": seed,
        "ratios": {"train": train_size, "validation": val_size, "test": test_size},
        "source": str(application_path.resolve()),
        "source_rows": int(len(data)),
        "source_identity_hash": hashlib.sha256(identity).hexdigest(),
        "split_file_hash": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "counts": counts,
    }
    (output_dir / "split_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
