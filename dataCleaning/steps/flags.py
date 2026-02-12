"""Issue flagging helpers for dataCleaning."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dataCleaning.config.base import CleaningConfig


def detect_missing_values(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Return issue records for missing values by country-year-column."""
    issues: list[dict[str, Any]] = []
    value_columns = [c for c in df.columns if c not in {"country", "year"}]
    for col in value_columns:
        missing = df[col].isna()
        if not missing.any():
            continue
        for row in df.loc[missing, ["country", "year"]].itertuples(index=False):
            issues.append({
                "country": row.country,
                "year": int(row.year),
                "column": col,
                "issue_type": "missing",
                "severity": "info",
                "value": None,
                "threshold": None,
                "rule_id": "flags.missing",
            })
    return issues


def detect_range_violations(df: pd.DataFrame, config: CleaningConfig) -> list[dict[str, Any]]:
    """Flag negative values on expected non-negative columns."""
    issues: list[dict[str, Any]] = []
    for col in config.non_negative_columns:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        bad = values < 0
        if not bad.any():
            continue
        for row in df.loc[bad, ["country", "year"]].itertuples(index=False):
            val = float(df.loc[(df["country"] == row.country) & (df["year"] == row.year), col].iloc[0])
            issues.append({
                "country": row.country,
                "year": int(row.year),
                "column": col,
                "issue_type": "range_violation",
                "severity": "warning",
                "value": val,
                "threshold": ">= 0",
                "rule_id": "flags.non_negative",
            })
    return issues


def detect_outliers(df: pd.DataFrame, config: CleaningConfig) -> list[dict[str, Any]]:
    """Flag outliers using MAD per country and IQR globally."""
    issues: list[dict[str, Any]] = []
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns if c != "year"
    ]

    for col in numeric_cols:
        col_values = pd.to_numeric(df[col], errors="coerce")
        if col_values.notna().sum() < 3:
            continue

        mad_flagged = pd.Series(False, index=df.index)
        for _, group in df[["country", col]].groupby("country", sort=False):
            values = pd.to_numeric(group[col], errors="coerce")
            valid = values.dropna()
            if len(valid) < 3:
                continue
            median = valid.median()
            mad = np.median(np.abs(valid - median))
            if mad == 0:
                continue
            z = 0.6745 * (values - median) / mad
            flagged = z.abs() > config.outlier_mad_threshold
            mad_flagged.loc[group.index] = mad_flagged.loc[group.index] | flagged.fillna(False)

        iqr_flagged = pd.Series(False, index=df.index)
        q1 = col_values.quantile(0.25)
        q3 = col_values.quantile(0.75)
        iqr = q3 - q1
        if pd.notna(iqr) and iqr > 0:
            low = q1 - config.outlier_iqr_multiplier * iqr
            high = q3 + config.outlier_iqr_multiplier * iqr
            iqr_flagged = (col_values < low) | (col_values > high)

        all_flags = mad_flagged | iqr_flagged
        if not all_flags.any():
            continue

        for row in df.loc[all_flags, ["country", "year"]].itertuples(index=False):
            val = float(df.loc[(df["country"] == row.country) & (df["year"] == row.year), col].iloc[0])
            issues.append({
                "country": row.country,
                "year": int(row.year),
                "column": col,
                "issue_type": "outlier_mad_iqr",
                "severity": "warning",
                "value": val,
                "threshold": (
                    f"mad>{config.outlier_mad_threshold} "
                    f"or iqr*{config.outlier_iqr_multiplier}"
                ),
                "rule_id": "flags.outlier_mad_iqr",
            })

    return issues


def detect_temporal_jumps(df: pd.DataFrame, config: CleaningConfig) -> list[dict[str, Any]]:
    """Flag large year-over-year relative jumps per country."""
    issues: list[dict[str, Any]] = []
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns if c != "year"
    ]
    ordered = df.sort_values(["country", "year"])

    for col in numeric_cols:
        for country, group in ordered.groupby("country", sort=False):
            values = pd.to_numeric(group[col], errors="coerce")
            prev = values.shift(1)
            rel_change = (values - prev).abs() / prev.abs()
            flagged = (
                rel_change > config.temporal_jump_threshold
            ) & prev.notna() & (prev != 0) & values.notna()
            if not flagged.any():
                continue
            for idx in group.index[flagged]:
                row = ordered.loc[idx]
                issues.append({
                    "country": country,
                    "year": int(row["year"]),
                    "column": col,
                    "issue_type": "temporal_jump",
                    "severity": "warning",
                    "value": float(row[col]),
                    "threshold": f"relative_change>{config.temporal_jump_threshold}",
                    "rule_id": "flags.temporal_jump",
                })

    return issues
