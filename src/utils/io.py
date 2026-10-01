import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(name: str = "config.yaml") -> dict:
    with open(PROJECT_ROOT / "configs" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Giảm bộ nhớ: float64 -> float32, int64 -> int32 (trừ cột ID)."""
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == np.float64:
            df[col] = df[col].astype(np.float32)
        elif dtype == np.int64 and not col.startswith("SK_ID"):
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def read_raw(cfg: dict, table: str, **kwargs) -> pd.DataFrame:
    path = resolve(cfg["paths"]["raw_dir"]) / f"{table}.csv"
    return downcast(pd.read_csv(path, **kwargs))


def save_json(obj, path: str | Path) -> None:
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)
