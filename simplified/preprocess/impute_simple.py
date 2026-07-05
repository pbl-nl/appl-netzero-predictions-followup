"""Step 2a: hierarchical median imputation (simple branch)."""
import pandas as pd

from .config import CONTINUOUS_FEATURES, CATEGORICAL_FEATURE


def fit_simple(train_df):
    """
    Fit hierarchical imputation statistics on training data.
    Returns params dict with per-country, per-year, and global fallbacks.
    """
    feats = [c for c in CONTINUOUS_FEATURES if c in train_df.columns]

    country_median = {}
    year_median = {}
    for feat in feats:
        country_median[feat] = train_df.groupby("iso3")[feat].median().to_dict()
        year_median[feat] = train_df.groupby("year")[feat].median().to_dict()
    global_median = {feat: train_df[feat].median() for feat in feats}

    # Categorical regime: use mode (most frequent value)
    country_mode = {}
    if CATEGORICAL_FEATURE in train_df.columns:
        for iso3, grp in train_df.groupby("iso3"):
            vals = grp[CATEGORICAL_FEATURE].dropna()
            if len(vals) > 0:
                country_mode[iso3] = vals.mode().iloc[0]

        year_mode = {}
        for year, grp in train_df.groupby("year"):
            vals = grp[CATEGORICAL_FEATURE].dropna()
            if len(vals) > 0:
                year_mode[year] = vals.mode().iloc[0]

        vals_all = train_df[CATEGORICAL_FEATURE].dropna()
        global_mode = vals_all.mode().iloc[0] if len(vals_all) > 0 else 0
    else:
        year_mode = {}
        global_mode = 0

    return {
        "country_median": country_median,
        "year_median": year_median,
        "global_median": global_median,
        "country_mode": country_mode,
        "year_mode": year_mode,
        "global_mode": global_mode,
    }


def apply_simple(df, params):
    """
    Fill missing values using three-level hierarchical fallback:
    country median → year median → global median.
    For v2x_regime: country mode → year mode → global mode.
    Returns a copy of df with NaNs filled.
    """
    df = df.copy()
    feats = [c for c in CONTINUOUS_FEATURES if c in df.columns]

    country_median = params["country_median"]
    year_median = params["year_median"]
    global_median = params["global_median"]

    for feat in feats:
        mask = df[feat].isna()
        if not mask.any():
            continue
        df.loc[mask, feat] = df.loc[mask].apply(
            lambda row: _hierarchical_fill(
                row["iso3"], row["year"],
                country_median.get(feat, {}),
                year_median.get(feat, {}),
                global_median.get(feat),
            ),
            axis=1,
        )

    # Regime variable
    if CATEGORICAL_FEATURE in df.columns:
        country_mode = params["country_mode"]
        year_mode = params["year_mode"]
        global_mode = params["global_mode"]
        mask = df[CATEGORICAL_FEATURE].isna()
        if mask.any():
            df.loc[mask, CATEGORICAL_FEATURE] = df.loc[mask].apply(
                lambda row: _hierarchical_fill(
                    row["iso3"], row["year"],
                    country_mode, year_mode, global_mode,
                ),
                axis=1,
            )

    return df


def _hierarchical_fill(iso3, year, country_lookup, year_lookup, global_val):
    val = country_lookup.get(iso3)
    if pd.notna(val):
        return val
    val = year_lookup.get(year)
    if pd.notna(val):
        return val
    return global_val
