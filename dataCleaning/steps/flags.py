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
    """Flag negative values on non-negative columns and bounded-range violations."""
    issues: list[dict[str, Any]] = []

    # --- non-negative checks ---
    for col in config.non_negative_columns:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        bad = values < 0
        if not bad.any():
            continue
        for idx in df.index[bad]:
            issues.append({
                "country": df.at[idx, "country"],
                "year": int(df.at[idx, "year"]),
                "column": col,
                "issue_type": "range_violation",
                "severity": "warning",
                "value": float(values.at[idx]),
                "threshold": ">= 0",
                "rule_id": "flags.non_negative",
            })

    # --- bounded range checks ---
    for col, (lo, hi) in config.bounded_columns.items():
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        bad = values.notna() & ((values < lo) | (values > hi))
        if not bad.any():
            continue
        for idx in df.index[bad]:
            issues.append({
                "country": df.at[idx, "country"],
                "year": int(df.at[idx, "year"]),
                "column": col,
                "issue_type": "bounded_range_violation",
                "severity": "warning",
                "value": float(values.at[idx]),
                "threshold": f"[{lo}, {hi}]",
                "rule_id": "flags.bounded_range",
            })

    return issues


def detect_outliers(df: pd.DataFrame, config: CleaningConfig) -> list[dict[str, Any]]:
    """Flag outliers using MAD per country and IQR globally."""
    issues: list[dict[str, Any]] = []
    excluded = set(config.exclude_from_outlier_detection)
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != "year" and c not in excluded
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

        for idx in df.index[all_flags]:
            issues.append({
                "country": df.at[idx, "country"],
                "year": int(df.at[idx, "year"]),
                "column": col,
                "issue_type": "outlier_mad_iqr",
                "severity": "warning",
                "value": float(df.at[idx, col]),
                "threshold": (
                    f"mad>{config.outlier_mad_threshold} "
                    f"or iqr*{config.outlier_iqr_multiplier}"
                ),
                "rule_id": "flags.outlier_mad_iqr",
            })

    return issues


def detect_temporal_jumps(df: pd.DataFrame, config: CleaningConfig) -> list[dict[str, Any]]:
    """Flag large year-over-year relative jumps per country (gap-aware)."""
    issues: list[dict[str, Any]] = []
    excluded = set(config.exclude_from_outlier_detection)
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != "year" and c not in excluded
    ]
    ordered = df.sort_values(["country", "year"])

    for col in numeric_cols:
        threshold = config.temporal_jump_thresholds.get(
            col, config.temporal_jump_default_threshold,
        )
        for country, group in ordered.groupby("country", sort=False):
            values = pd.to_numeric(group[col], errors="coerce")
            years = group["year"]
            prev_values = values.shift(1)
            prev_years = years.shift(1)
            year_gap = years - prev_years

            rel_change = (values - prev_values).abs() / prev_values.abs()
            # Normalize by year gap so multi-year gaps aren't over-flagged
            annualized = rel_change / year_gap

            flagged = (
                annualized > threshold
            ) & prev_values.notna() & (prev_values != 0) & values.notna() & (year_gap > 0)

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
                    "threshold": f"annualized_change>{threshold}",
                    "rule_id": "flags.temporal_jump",
                })

    return issues


def detect_infinities(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Flag +inf and -inf values with critical severity."""
    issues: list[dict[str, Any]] = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(["year"])

    for col in numeric_cols:
        inf_mask = np.isinf(df[col])
        if not inf_mask.any():
            continue
        for idx in df.index[inf_mask]:
            issues.append({
                "country": df.at[idx, "country"],
                "year": int(df.at[idx, "year"]),
                "column": col,
                "issue_type": "infinity",
                "severity": "critical",
                "value": float(df.at[idx, col]),
                "threshold": "finite",
                "rule_id": "flags.infinity",
            })

    return issues


def detect_cross_column_violations(
    df: pd.DataFrame, config: CleaningConfig,
) -> list[dict[str, Any]]:
    """Flag rows where subset column > superset column."""
    issues: list[dict[str, Any]] = []

    for subset_col, superset_col in config.cross_column_rules:
        if subset_col not in df.columns or superset_col not in df.columns:
            continue
        subset = pd.to_numeric(df[subset_col], errors="coerce")
        superset = pd.to_numeric(df[superset_col], errors="coerce")
        both_present = subset.notna() & superset.notna()
        violated = both_present & (subset > superset)
        if not violated.any():
            continue
        for idx in df.index[violated]:
            issues.append({
                "country": df.at[idx, "country"],
                "year": int(df.at[idx, "year"]),
                "column": f"{subset_col} vs {superset_col}",
                "issue_type": "cross_column_violation",
                "severity": "warning",
                "value": float(subset.at[idx]),
                "threshold": f"<= {superset_col} ({float(superset.at[idx])})",
                "rule_id": "flags.cross_column",
            })

    return issues


def detect_constant_series(
    df: pd.DataFrame, config: CleaningConfig,
) -> list[dict[str, Any]]:
    """Flag country-column pairs with the same value repeated >= N consecutive years."""
    issues: list[dict[str, Any]] = []
    min_len = config.constant_series_min_length
    numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(["year"])
    ordered = df.sort_values(["country", "year"])

    for col in numeric_cols:
        for country, group in ordered.groupby("country", sort=False):
            values = pd.to_numeric(group[col], errors="coerce")
            non_null = values.dropna()
            if len(non_null) < min_len:
                continue

            # Find runs of consecutive equal values
            shifted = non_null.shift(1)
            different = non_null != shifted
            run_id = different.cumsum()
            run_lengths = run_id.map(run_id.value_counts())

            longest_run = run_lengths.max()
            if longest_run < min_len:
                continue

            # Report the first run that meets the threshold
            for rid in run_id.unique():
                mask = run_id == rid
                if mask.sum() >= min_len:
                    span_years = group.loc[non_null.index[mask], "year"]
                    repeated_val = non_null[mask].iloc[0]
                    issues.append({
                        "country": country,
                        "year": int(span_years.iloc[0]),
                        "column": col,
                        "issue_type": "constant_series",
                        "severity": "info",
                        "value": float(repeated_val),
                        "threshold": f"{int(mask.sum())} consecutive years ({int(span_years.iloc[0])}-{int(span_years.iloc[-1])})",
                        "rule_id": "flags.constant_series",
                    })
                    break  # one issue per (country, col)

    return issues
