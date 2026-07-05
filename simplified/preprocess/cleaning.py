"""Step 1: domain-range enforcement, IQR outlier detection, row-level missingness filter."""
import numpy as np
import pandas as pd

from .config import (
    CONTINUOUS_FEATURES,
    CATEGORICAL_FEATURE,
    DOMAIN_BOUNDS,
    VOLATILE_FEATURES,
    MISSINGNESS_THRESHOLD,
    IQR_MULTIPLIER,
    IQR_COUNTRY_MIN_OBS,
    IQR_YEAR_MIN_OBS,
)


def fit_cleaning(train_df):
    """
    Fit IQR bounds from training data.
    Returns params dict with country-level and year-level bounds per feature.
    """
    country_bounds = {}
    year_bounds = {}

    for feat in CONTINUOUS_FEATURES:
        if feat not in train_df.columns:
            continue

        cb = {}
        for iso3, grp in train_df.groupby("iso3"):
            vals = grp[feat].dropna()
            if len(vals) >= IQR_COUNTRY_MIN_OBS:
                q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
                iqr = q3 - q1
                cb[iso3] = (q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr)
        country_bounds[feat] = cb

        yb = {}
        for year, grp in train_df.groupby("year"):
            vals = grp[feat].dropna()
            if len(vals) >= IQR_YEAR_MIN_OBS:
                q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
                iqr = q3 - q1
                yb[year] = (q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr)
        year_bounds[feat] = yb

    return {"country_bounds": country_bounds, "year_bounds": year_bounds}


def apply_cleaning(df, params):
    """
    Apply cleaning to df using fitted params.
    Returns a copy of df with:
    - Out-of-domain values set to NaN
    - Volatile-feature outliers (flagged by both country AND year IQR) set to NaN
    - A 'ghost_row' boolean column (True = row should be dropped after rolling features)
    """
    df = df.copy()
    country_bounds = params["country_bounds"]
    year_bounds = params["year_bounds"]

    for feat in CONTINUOUS_FEATURES:
        if feat not in df.columns:
            continue

        # 1. Domain-range enforcement
        if feat in DOMAIN_BOUNDS:
            lo, hi = DOMAIN_BOUNDS[feat]
            out_of_range = df[feat].notna() & ((df[feat] < lo) | (df[feat] > hi))
            df.loc[out_of_range, feat] = np.nan

        # 2. IQR outlier coercion for volatile features only
        if feat in VOLATILE_FEATURES:
            cb = country_bounds.get(feat, {})
            yb = year_bounds.get(feat, {})

            country_lo = df["iso3"].map({k: v[0] for k, v in cb.items()})
            country_hi = df["iso3"].map({k: v[1] for k, v in cb.items()})
            year_lo = df["year"].map({k: v[0] for k, v in yb.items()})
            year_hi = df["year"].map({k: v[1] for k, v in yb.items()})

            country_flag = df[feat].notna() & (
                (df[feat] < country_lo) | (df[feat] > country_hi)
            )
            year_flag = df[feat].notna() & (
                (df[feat] < year_lo) | (df[feat] > year_hi)
            )
            # Coerce only when flagged by BOTH country-level and year-level IQR
            df.loc[country_flag & year_flag, feat] = np.nan

    # 3. Row-level missingness filter — mark as ghost rows (not dropped yet)
    # Include categorical feature in the count to match original methodology.
    all_input_features = CONTINUOUS_FEATURES + [CATEGORICAL_FEATURE]
    feature_cols = [c for c in all_input_features if c in df.columns]
    missing_frac = df[feature_cols].isna().sum(axis=1) / len(feature_cols)
    df["ghost_row"] = missing_frac > MISSINGNESS_THRESHOLD

    return df
