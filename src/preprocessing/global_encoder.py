"""Fit one centralized encoder using global-train-owned rows only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.partition.relational_partition import TABLE_NAMES


EXCLUDED = {"SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "TARGET"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _merge_numeric(state: dict, array: np.ndarray) -> None:
    array = array[np.isfinite(array)]
    n = len(array)
    if not n:
        return
    mean = float(array.mean())
    delta = mean - state["mean"]
    total = state["count"] + n
    state["m2"] += (
        float(((array - mean) ** 2).sum())
        + delta**2 * state["count"] * n / total
    )
    state["mean"] += delta * n / total
    state["count"] = total


def fit_global_encoder(
    table_dir: Path,
    split_path: Path,
    output: Path,
    missing_threshold: float = 0.80,
    chunksize: int = 200_000,
    max_categories: int = 1_000,
) -> dict:
    """Fit feature selection, imputation, scaling and vocabularies on Train only."""
    table_dir, split_path, output = map(Path, (table_dir, split_path, output))
    if not 0 <= missing_threshold <= 1:
        raise ValueError("missing_threshold must be between 0 and 1")
    if chunksize <= 0 or max_categories <= 0:
        raise ValueError("chunksize and max_categories must be positive")
    split = pd.read_csv(split_path, usecols=["SK_ID_CURR", "split"])
    if split.SK_ID_CURR.isna().any() or not split.SK_ID_CURR.is_unique:
        raise ValueError("Global split customer IDs must be unique and non-null")
    if set(split["split"]) != {"train", "validation", "test"}:
        raise ValueError("Global split must contain train, validation and test")
    train_ids = set(split.loc[split["split"].eq("train"), "SK_ID_CURR"])

    result = {
        "protocol": "strict_global_first",
        "fit_scope": "global_train_customers_only",
        "missing_threshold": missing_threshold,
        "split_file": str(split_path.resolve()),
        "split_hash": _sha256(split_path),
        "tables": {},
    }
    for table in TABLE_NAMES:
        path = table_dir / f"{table}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        schema = pd.read_csv(path, nrows=0).columns.tolist()
        if "SK_ID_CURR" not in schema:
            raise ValueError(f"{table} is missing SK_ID_CURR")
        columns = [column for column in schema if column not in EXCLUDED]
        stats = {
            column: {
                "numeric": True,
                "non_null": 0,
                "count": 0,
                "mean": 0.0,
                "m2": 0.0,
                "categories": set(),
                "overflow": False,
            }
            for column in columns
        }
        fit_rows = 0
        dtype = {column: "string" for column in columns}
        for chunk in pd.read_csv(path, chunksize=chunksize, dtype=dtype):
            train = chunk.loc[chunk.SK_ID_CURR.isin(train_ids)]
            fit_rows += len(train)
            for column, state in stats.items():
                values = train[column].dropna()
                state["non_null"] += len(values)
                numeric = pd.to_numeric(values, errors="coerce")
                if numeric.isna().any():
                    state["numeric"] = False
                if not state["overflow"]:
                    state["categories"].update(values.astype(str).unique())
                    if len(state["categories"]) > max_categories:
                        state["overflow"] = True
                        state["categories"].clear()
                _merge_numeric(state, numeric.to_numpy(dtype=float))
        if fit_rows == 0:
            raise ValueError(f"{table} has no rows owned by global Train customers")

        definitions, dropped = [], []
        missing_ratios = {}
        for column, state in stats.items():
            missing_ratio = 1.0 - state["non_null"] / fit_rows
            missing_ratios[column] = missing_ratio
            if missing_ratio > missing_threshold:
                dropped.append(column)
                continue
            if state["numeric"]:
                scale = np.sqrt(state["m2"] / max(1, state["count"]))
                definitions.append(
                    {
                        "column": column,
                        "kind": "numeric",
                        "mean": state["mean"],
                        "scale": float(scale) if scale > 1e-12 else 1.0,
                        "count": state["count"],
                        "missing_ratio": missing_ratio,
                    }
                )
            else:
                if state["overflow"]:
                    raise ValueError(
                        f"{table}.{column} exceeds max_categories={max_categories}"
                    )
                definitions.append(
                    {
                        "column": column,
                        "kind": "categorical",
                        "categories": sorted(state["categories"]),
                        "missing_ratio": missing_ratio,
                    }
                )
        dimension = sum(
            1 if spec["kind"] == "numeric" else len(spec["categories"]) + 2
            for spec in definitions
        )
        result["tables"][table] = {
            "source": str(path.resolve()),
            "source_schema": schema,
            "fit_rows": fit_rows,
            "columns_before_selection": len(columns),
            "columns_after_selection": len(definitions),
            "dropped_columns": dropped,
            "missing_ratios_train": missing_ratios,
            "columns": definitions,
            "dimension": max(1, dimension),
        }
        print(
            f"Fit {table}: {fit_rows} train-owned rows; "
            f"dropped {len(dropped)} columns; {max(1, dimension)} features",
            flush=True,
        )
    result["fingerprint"] = hashlib.sha256(
        json.dumps(result, sort_keys=True).encode()
    ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def transform_table(frame: pd.DataFrame, definition: dict) -> np.ndarray:
    """Apply a fitted table definition without learning from the input frame."""
    result = np.zeros((len(frame), definition["dimension"]), dtype=np.float32)
    offset = 0
    for spec in definition["columns"]:
        values = frame[spec["column"]]
        if spec["kind"] == "numeric":
            numeric = pd.to_numeric(values, errors="coerce").to_numpy(
                dtype=float, copy=True
            )
            numeric[~np.isfinite(numeric)] = spec["mean"]
            result[:, offset] = (numeric - spec["mean"]) / spec["scale"]
            offset += 1
        else:
            vocabulary = {
                value: index + 2 for index, value in enumerate(spec["categories"])
            }
            indices = values.astype(str).map(vocabulary).fillna(1).to_numpy(dtype=int)
            indices[values.isna().to_numpy()] = 0
            result[np.arange(len(frame)), offset + indices] = 1
            offset += len(vocabulary) + 2
    if not np.isfinite(result).all():
        raise ValueError("Non-finite encoded features")
    return result
