# Quy tắc làm sạch application và lọc cột thiếu; tiện ích clean_application thuộc luồng application riêng.
"""Cleaning rules for the Home Credit application table."""
from __future__ import annotations
import pandas as pd

DEFAULT_PROTECTED_COLUMNS = ("SK_ID_CURR", "TARGET")

def drop_high_missing_columns(data: pd.DataFrame, threshold: float = 0.80,
                              protected_columns: tuple[str, ...] = DEFAULT_PROTECTED_COLUMNS
                              ) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns whose missing fraction is strictly greater than threshold."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    fractions = data.isna().mean()
    protected = set(protected_columns)
    # Chỉ loại khi tỷ lệ > threshold, không loại khi bằng; cột protected luôn được giữ.
    dropped = [c for c, value in fractions.items() if value > threshold and c not in protected]
    return data.drop(columns=dropped).copy(), dropped

def clean_application(data: pd.DataFrame, missing_threshold: float = 0.80
                      ) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply deterministic cleaning and return an audit report."""
    cleaned = data.copy()
    if "DAYS_EMPLOYED" in cleaned:
        cleaned["DAYS_EMPLOYED"] = cleaned["DAYS_EMPLOYED"].replace(365243, pd.NA)
    if "CODE_GENDER" in cleaned:
        cleaned = cleaned.loc[cleaned["CODE_GENDER"] != "XNA"].copy()
    cleaned, dropped = drop_high_missing_columns(cleaned, missing_threshold)
    return cleaned, {"missing_threshold": missing_threshold, "dropped_columns": dropped,
                     "rows_after_cleaning": len(cleaned),
                     "columns_after_cleaning": cleaned.shape[1]}
