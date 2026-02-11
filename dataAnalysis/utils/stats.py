"""Statistical utility functions for analysis steps."""

from __future__ import annotations

import numpy as np
import pandas as pd


def check_monotonicity(series: pd.Series, tolerance: float = 1e-6) -> dict:
    """Check if a series is monotonically non-increasing.

    Args:
        series: Time series (should be sorted by time).
        tolerance: Allow small increases up to this value.

    Returns:
        Dict with 'is_monotonic', 'violations_count', 'max_violation'.
    """
    diffs = series.diff().dropna()
    violations = diffs[diffs > tolerance]
    return {
        "is_monotonic": len(violations) == 0,
        "violations_count": len(violations),
        "max_violation": float(violations.max()) if len(violations) > 0 else 0.0,
    }


def variance_decomposition(
    df: pd.DataFrame, value_col: str, group_col: str = "country"
) -> dict:
    """Compute within-group vs between-group variance decomposition.

    Returns:
        Dict with 'total_var', 'between_var', 'within_var', 'icc'.
    """
    overall_mean = df[value_col].mean()
    total_var = df[value_col].var()

    group_means = df.groupby(group_col)[value_col].mean()
    group_sizes = df.groupby(group_col)[value_col].count()
    between_var = float(np.average((group_means - overall_mean) ** 2, weights=group_sizes))

    within_vars = df.groupby(group_col)[value_col].var()
    valid_within = within_vars.dropna()
    within_var = float(
        np.average(valid_within, weights=group_sizes[valid_within.index])
    )

    denom = between_var + within_var
    icc = between_var / denom if denom > 0 else np.nan

    return {
        "total_var": float(total_var),
        "between_var": between_var,
        "within_var": within_var,
        "icc": float(icc),
    }


def compute_lag_correlation(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    max_lag: int = 5,
    group_col: str = "country",
) -> pd.DataFrame:
    """Compute cross-correlation at different lags within groups.

    Args:
        df: DataFrame sorted by group and time.
        x_col: Predictor column.
        y_col: Target column.
        max_lag: Maximum lag to compute.
        group_col: Grouping column.

    Returns:
        DataFrame with lag and correlation columns.
    """
    results = []

    for lag in range(max_lag + 1):
        corrs = []
        for _, group in df.groupby(group_col):
            group = group.sort_values("year")
            if len(group) <= lag + 2:
                continue
            x = group[x_col].iloc[: len(group) - lag] if lag > 0 else group[x_col]
            y = group[y_col].iloc[lag:] if lag > 0 else group[y_col]
            x_reset = x.reset_index(drop=True)
            y_reset = y.reset_index(drop=True)
            valid = x_reset.notna() & y_reset.notna()
            if valid.sum() > 2:
                corr = x_reset[valid].corr(y_reset[valid])
                if not np.isnan(corr):
                    corrs.append(corr)

        if corrs:
            results.append({
                "lag": lag,
                "correlation": float(np.nanmean(corrs)),
                "std": float(np.nanstd(corrs)),
                "n_countries": len(corrs),
            })

    return pd.DataFrame(results)


def flag_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    """Flag outliers using the IQR method.

    Returns:
        Boolean series with True for outliers.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (series < lower) | (series > upper)


def compute_per_capita_emissions(df: pd.DataFrame) -> pd.Series:
    """Compute per-capita emissions (tonnes CO2e per person)."""
    return df["total_ghg_emissions"] / df["population_total"]
