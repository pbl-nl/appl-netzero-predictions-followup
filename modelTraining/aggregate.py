"""Prediction aggregation utilities for modelTraining."""

from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_predictions(
    per_run_df: pd.DataFrame,
    value_column: str = "rce_pred",
) -> pd.DataFrame:
    """Aggregate per-run predictions in original RCE space."""
    required = {"country", value_column}
    missing = required - set(per_run_df.columns)
    if missing:
        raise ValueError(f"per_run_df missing required columns: {sorted(missing)}")

    group_cols = ["country"]
    if "year" in per_run_df.columns:
        group_cols.append("year")

    grouped = per_run_df.groupby(group_cols, dropna=False)[value_column]

    result = grouped.agg(
        rce_median="median",
        rce_mean="mean",
        rce_std="std",
    ).reset_index()

    p5 = grouped.quantile(0.05).reset_index(name="rce_p5")
    p95 = grouped.quantile(0.95).reset_index(name="rce_p95")

    result = result.merge(p5, on=group_cols, how="left")
    result = result.merge(p95, on=group_cols, how="left")

    result["rce_std"] = result["rce_std"].fillna(0.0)
    result["prediction_cv"] = np.where(
        np.abs(result["rce_mean"]) > 1e-12,
        result["rce_std"] / np.abs(result["rce_mean"]),
        np.nan,
    )

    return result.sort_values(group_cols).reset_index(drop=True)
