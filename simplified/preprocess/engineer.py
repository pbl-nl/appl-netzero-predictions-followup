"""
Step 3: feature engineering.
- 7-year backward rolling mean for each continuous feature
- 3-year and 5-year backward rolling std for each continuous feature
- All pairwise interaction terms from original continuous features
- Fill NaN at temporal boundaries with 0
- One-hot encode v2x_regime
- Z-score standardise all continuous features (fit on training data only)
"""
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import (
    CONTINUOUS_FEATURES,
    CATEGORICAL_FEATURE,
    ROLLING_MEAN_WINDOW,
    ROLLING_STD_WINDOWS,
    EXCLUDE_COLS,
)

# Synthetic columns added by embeddings.py — not interaction candidates
_SYNTHETIC_COLS = [f"pca_{i}" for i in range(5)] + ["zt", "zt_sq"]


def _feature_cols_for_scaling(df):
    """All columns that should be z-score scaled (not iso3, year, ghost_row, targets, OHE)."""
    skip = set(EXCLUDE_COLS) | {"ghost_row"} | set(_ohe_cols(df))
    return [c for c in df.columns if c not in skip and df[c].dtype != object]


def _ohe_cols(df):
    """Return OHE column names if already present."""
    return [c for c in df.columns if c.startswith(f"{CATEGORICAL_FEATURE}_")]


def fit_engineer(train_df):
    """
    Fit engineering parameters on training data.
    Returns params dict with scaler mean/std and OHE categories.
    """
    # Determine OHE categories from training data
    ohe_categories = None
    if CATEGORICAL_FEATURE in train_df.columns:
        ohe_categories = sorted(train_df[CATEGORICAL_FEATURE].dropna().unique().astype(int).tolist())

    # Apply engineering transforms to training data to get the right columns for scaler fitting
    engineered = _apply_transforms(train_df, ohe_categories)

    # Drop ghost rows before fitting the scaler
    engineered_for_fit = engineered[~engineered["ghost_row"]]

    # Fit scaler on training set (fill any remaining NaN with 0 before fitting)
    scale_cols = _feature_cols_for_scaling(engineered_for_fit)
    scaler = StandardScaler()
    scaler.fit(engineered_for_fit[scale_cols].fillna(0.0))

    return {
        "ohe_categories": ohe_categories,
        "scale_cols": scale_cols,
        "scaler": scaler,
    }


def apply_engineer(df, params, drop_ghost_rows=True):
    """
    Apply all feature engineering steps to df.

    History rows (marked `_is_history=True` by pipeline.transform) are always
    dropped after rolling features are computed — they exist only for rolling context.
    Native high-missingness rows (`ghost_row=True`) are dropped only when
    `drop_ghost_rows=True` (default). Pass `drop_ghost_rows=False` for the final
    holdout to retain all rows for lenient/all-country evaluation.
    """
    ohe_categories = params["ohe_categories"]
    scale_cols = params["scale_cols"]
    scaler = params["scaler"]

    df = _apply_transforms(df, ohe_categories)

    # Fill temporal-boundary NaNs with 0 before scaling
    for col in df.columns:
        if col in scale_cols and df[col].isna().any():
            df[col] = df[col].fillna(0.0)

    # Always drop history rows (prepended only to provide rolling context)
    if "_is_history" in df.columns:
        df = df[~df["_is_history"]].copy()
        df = df.drop(columns=["_is_history"])

    # Optionally drop high-missingness rows (strict mode only)
    if drop_ghost_rows:
        df = df[~df["ghost_row"]].copy()

    # Z-score scale
    cols_present = [c for c in scale_cols if c in df.columns]
    df[cols_present] = scaler.transform(df[cols_present].fillna(0.0))

    return df


def _apply_transforms(df, ohe_categories):
    """Internal: apply rolling, interactions, and OHE. Does not scale or drop ghost rows."""
    df = df.copy()
    feats = [c for c in CONTINUOUS_FEATURES if c in df.columns]

    # Rolling features (per country, backward-looking) — build all at once
    rolling_cols = {}
    for feat in feats:
        grouped = df.groupby("iso3")[feat]
        rolling_cols[f"{feat}_roll{ROLLING_MEAN_WINDOW}"] = grouped.transform(
            lambda x: x.rolling(ROLLING_MEAN_WINDOW, min_periods=1).mean()
        )
        for w in ROLLING_STD_WINDOWS:
            rolling_cols[f"{feat}_roll{w}std"] = grouped.transform(
                lambda x, w=w: x.rolling(w, min_periods=2).std()
            )
    if rolling_cols:
        df = pd.concat([df, pd.DataFrame(rolling_cols, index=df.index)], axis=1)

    # Pairwise interaction terms — build all at once to avoid DataFrame fragmentation
    interaction_cols = {
        f"{a}_x_{b}": df[a] * df[b]
        for a, b in combinations(feats, 2)
    }
    if interaction_cols:
        df = pd.concat([df, pd.DataFrame(interaction_cols, index=df.index)], axis=1)

    # One-hot encode v2x_regime using training-fitted categories
    if CATEGORICAL_FEATURE in df.columns and ohe_categories is not None:
        ohe_cols = {
            f"{CATEGORICAL_FEATURE}_{cat}": (df[CATEGORICAL_FEATURE] == cat).astype(float)
            for cat in ohe_categories
        }
        df = pd.concat([df.drop(columns=[CATEGORICAL_FEATURE]),
                        pd.DataFrame(ohe_cols, index=df.index)], axis=1)

    return df
