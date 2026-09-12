"""One shared train-only mean imputer/scaler and categorical one-hot vocabulary."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.partition.relational_partition import TABLE_NAMES

EXCLUDED = {"SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "TARGET"}


def fit_shared_encoder(client_dir: Path, output: Path, chunksize=200_000, max_categories=1000):
    report = json.loads((client_dir / "partition_report.json").read_text())
    splits = {}
    for client in report["clients"]:
        split = pd.read_csv(client_dir / client / "customer_split.csv")
        splits[client] = set(split.loc[split.train_mask, "SK_ID_CURR"])
    result = {"fit_scope": "union_of_local_train_customers_only", "tables": {},
              "split_hashes": {client: hashlib.sha256((client_dir / client / "customer_split.csv").read_bytes()).hexdigest()
                               for client in report["clients"]},
              "partition": {k: report[k] for k in ("strategy", "alpha", "seed", "num_clients")}}
    for table in TABLE_NAMES:
        columns = [c for c in report["schema"][table] if c not in EXCLUDED]
        stats = {c: {"numeric": True, "count": 0, "mean": 0., "m2": 0., "categories": set(),
                     "overflow": False} for c in columns}
        fit_rows = 0
        for client in report["clients"]:
            for chunk in pd.read_csv(client_dir / client / f"{table}.csv", chunksize=chunksize,
                                     dtype={col: "string" for col in columns}):
                train = chunk.loc[chunk.SK_ID_CURR.isin(splits[client])]
                fit_rows += len(train)
                for col, state in stats.items():
                    values = train[col].dropna()
                    numeric = pd.to_numeric(values, errors="coerce")
                    if numeric.isna().any():
                        state["numeric"] = False
                    if not state["overflow"]:
                        state["categories"].update(values.astype(str).unique())
                        if len(state["categories"]) > max_categories:
                            state["overflow"] = True
                            state["categories"].clear()
                    array = numeric.to_numpy(dtype=float)
                    array = array[np.isfinite(array)]
                    n = len(array)
                    if n:
                        mean = float(array.mean())
                        delta = mean - state["mean"]
                        total = state["count"] + n
                        state["m2"] += float(((array - mean) ** 2).sum()) + delta ** 2 * state["count"] * n / total
                        state["mean"] += delta * n / total
                        state["count"] = total
        definitions = []
        for col, state in stats.items():
            if state["numeric"]:
                scale = np.sqrt(state["m2"] / max(1, state["count"]))
                definitions.append({"column": col, "kind": "numeric", "mean": state["mean"],
                                    "scale": float(scale) if scale > 1e-12 else 1., "count": state["count"]})
            else:
                if state["overflow"]:
                    raise ValueError(f"{table}.{col} exceeds max_categories={max_categories}")
                definitions.append({"column": col, "kind": "categorical",
                                    "categories": sorted(state["categories"])})
        dimension = sum(1 if d["kind"] == "numeric" else len(d["categories"]) + 2 for d in definitions)
        result["tables"][table] = {"columns": definitions, "dimension": max(1, dimension), "fit_rows": fit_rows}
        print(f"Fit {table}: {fit_rows} train-owned rows, {max(1, dimension)} features", flush=True)
    result["fingerprint"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def transform_table(frame, definition):
    result = np.zeros((len(frame), definition["dimension"]), dtype=np.float32)
    offset = 0
    for spec in definition["columns"]:
        values = frame[spec["column"]]
        if spec["kind"] == "numeric":
            numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, copy=True)
            numeric[~np.isfinite(numeric)] = spec["mean"]
            result[:, offset] = (numeric - spec["mean"]) / spec["scale"]
            offset += 1
        else:
            vocabulary = {value: i + 2 for i, value in enumerate(spec["categories"])}
            indices = values.astype(str).map(vocabulary).fillna(1).to_numpy(dtype=int, copy=True)
            indices[values.isna().to_numpy()] = 0  # 0=missing, 1=unseen in pooled train
            result[np.arange(len(frame)), offset + indices] = 1
            offset += len(vocabulary) + 2
    if not np.isfinite(result).all():
        raise ValueError("Non-finite encoded features")
    return result
