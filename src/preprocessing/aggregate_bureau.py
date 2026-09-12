"""Summarize monthly bureau history without duplicating bureau nodes."""
from __future__ import annotations

import pandas as pd


def merge_bureau_balance(bureau: pd.DataFrame, balance: pd.DataFrame) -> pd.DataFrame:
    """Left join one summary per SK_ID_BUREAU; latest means largest MONTHS_BALANCE."""
    key = "SK_ID_BUREAU"
    if bureau[key].isna().any() or bureau[key].duplicated().any():
        raise ValueError("bureau must have a unique, non-null SK_ID_BUREAU")
    if balance[key].isna().any():
        raise ValueError("bureau_balance contains null SK_ID_BUREAU")
    if balance.duplicated([key, "MONTHS_BALANCE"]).any():
        raise ValueError("bureau_balance contains duplicate bureau/month pairs")

    grouped = balance.groupby(key, sort=False)
    summary = grouped.agg(
        BB_MONTH_COUNT=("MONTHS_BALANCE", "size"),
        BB_MONTH_MIN=("MONTHS_BALANCE", "min"),
        BB_MONTH_MAX=("MONTHS_BALANCE", "max"),
    )
    # Keep C (closed) and X (unknown) categorical, not numeric delinquency levels.
    statuses = balance["STATUS"].astype("string")
    for status in ("0", "1", "2", "3", "4", "5", "C", "X"):
        counts = statuses.eq(status).fillna(False).groupby(balance[key], sort=False).sum()
        summary[f"BB_STATUS_{status}_COUNT"] = counts
        summary[f"BB_STATUS_{status}_RATIO"] = counts / summary["BB_MONTH_COUNT"]
    summary["BB_OVERDUE_MONTH_COUNT"] = summary[
        [f"BB_STATUS_{s}_COUNT" for s in ("1", "2", "3", "4", "5")]
    ].sum(axis=1)
    latest = balance.loc[balance["MONTHS_BALANCE"].notna()].sort_values("MONTHS_BALANCE")
    summary["BB_LATEST_STATUS"] = latest.drop_duplicates(key, keep="last").set_index(key)["STATUS"]
    return bureau.merge(summary, on=key, how="left", validate="one_to_one", sort=False)
