"""Step 2b: MICE imputation (multiple imputation by chained equations)."""
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

from .config import (
    CONTINUOUS_FEATURES,
    CATEGORICAL_FEATURE,
    MICE_MAX_ITER,
    MICE_TOL,
    MICE_MISSING_FLAG_THRESHOLD,
    DOMAIN_BOUNDS,
)
from .impute_simple import fit_simple, apply_simple


def fit_mice(train_df):
    """
    Fit the MICE imputer on training data.
    Returns a params dict with the fitted IterativeImputer, scaler, column names, etc.
    """
    feats = [c for c in CONTINUOUS_FEATURES if c in train_df.columns]
    pca_cols = [c for c in train_df.columns if c.startswith("pca_")]
    year_cols = ["zt", "zt_sq"]

    # Auxiliary baseline (hierarchical median) for the full training set
    simple_params = fit_simple(train_df)

    # Determine which features need was-missing flags (>5% missing in train)
    missing_flag_feats = [
        f for f in feats
        if train_df[f].isna().mean() > MICE_MISSING_FLAG_THRESHOLD
    ]

    # Mode-impute regime for use in MICE design matrix
    regime_mode = None
    if CATEGORICAL_FEATURE in train_df.columns:
        vals = train_df[CATEGORICAL_FEATURE].dropna()
        regime_mode = int(vals.mode().iloc[0]) if len(vals) > 0 else 0

    # Build the MICE design matrix columns (in a fixed order)
    mice_feature_cols = feats + pca_cols + year_cols
    aux_cols = [f + "_aux" for f in feats + pca_cols + year_cols]
    flag_cols = [f + "_missing" for f in missing_flag_feats]
    all_mice_cols = mice_feature_cols + aux_cols + flag_cols
    if CATEGORICAL_FEATURE in train_df.columns:
        all_mice_cols = all_mice_cols + [CATEGORICAL_FEATURE]

    # Build training MICE matrix
    mice_train = _build_mice_matrix(
        train_df, simple_params, feats, pca_cols, year_cols,
        missing_flag_feats, regime_mode,
    )

    # Identify columns that are entirely NaN in training.
    # IterativeImputer drops all-NaN columns from its output, which breaks shape invariance.
    # Fill them with 0.0 so they pass through the imputer unchanged.
    mice_train_arr = mice_train[all_mice_cols].fillna(np.nan)
    all_nan_cols = [
        col for col in all_mice_cols
        if mice_train_arr[col].isna().all()
    ]
    if all_nan_cols:
        mice_train_arr = mice_train_arr.copy()
        for col in all_nan_cols:
            mice_train_arr[col] = 0.0

    # Standardise
    scaler = StandardScaler()
    mice_train_scaled = scaler.fit_transform(mice_train_arr)

    # Fit IterativeImputer
    imputer = IterativeImputer(
        max_iter=MICE_MAX_ITER,
        tol=MICE_TOL,
        random_state=42,
        skip_complete=True,
    )
    imputer.fit(mice_train_scaled)

    return {
        "imputer": imputer,
        "scaler": scaler,
        "simple_params": simple_params,
        "feats": feats,
        "pca_cols": pca_cols,
        "year_cols": year_cols,
        "missing_flag_feats": missing_flag_feats,
        "regime_mode": regime_mode,
        "all_mice_cols": all_mice_cols,
        "mice_feature_cols": mice_feature_cols,
        "all_nan_cols": all_nan_cols,
    }


def apply_mice(df, params):
    """
    Apply the fitted MICE imputer to df.
    Returns a copy of df with NaNs filled (in original continuous features).
    Post-imputation domain-range enforcement is applied to prevent extrapolation violations.
    """
    df = df.copy()

    feats = params["feats"]
    pca_cols = params["pca_cols"]
    year_cols = params["year_cols"]
    missing_flag_feats = params["missing_flag_feats"]
    regime_mode = params["regime_mode"]
    simple_params = params["simple_params"]
    all_mice_cols = params["all_mice_cols"]
    mice_feature_cols = params["mice_feature_cols"]
    scaler = params["scaler"]
    imputer = params["imputer"]

    mice_df = _build_mice_matrix(
        df, simple_params, feats, pca_cols, year_cols, missing_flag_feats, regime_mode,
    )

    arr = mice_df[all_mice_cols].fillna(np.nan)
    # Fill the same all-NaN columns that were filled during training
    all_nan_cols = params.get("all_nan_cols", [])
    if all_nan_cols:
        arr = arr.copy()
        for col in all_nan_cols:
            arr[col] = 0.0

    scaled = scaler.transform(arr)
    imputed_scaled = imputer.transform(scaled)
    imputed = scaler.inverse_transform(imputed_scaled)
    imputed_df = pd.DataFrame(imputed, columns=all_mice_cols, index=mice_df.index)

    # Write back only the original continuous feature columns
    for feat in feats:
        if feat in imputed_df.columns:
            df[feat] = imputed_df[feat].values

    # Write back regime: round, clip, cast
    if CATEGORICAL_FEATURE in df.columns and CATEGORICAL_FEATURE in imputed_df.columns:
        regime_vals = imputed_df[CATEGORICAL_FEATURE].round().clip(0, 3).astype(int)
        df[CATEGORICAL_FEATURE] = regime_vals.values

    # Global median safety fill for anything still missing
    global_median = simple_params["global_median"]
    for feat in feats:
        if df[feat].isna().any():
            df[feat] = df[feat].fillna(global_median.get(feat, 0.0))

    # Post-imputation domain-range enforcement (MICE can extrapolate)
    for feat, (lo, hi) in DOMAIN_BOUNDS.items():
        if feat in df.columns:
            df.loc[df[feat] < lo, feat] = lo
            df.loc[df[feat] > hi, feat] = hi

    return df


def _build_mice_matrix(df, simple_params, feats, pca_cols, year_cols, missing_flag_feats, regime_mode):
    """Build the full MICE design matrix for a given df."""
    result = df[feats + pca_cols + year_cols].copy() if (pca_cols + year_cols) else df[feats].copy()

    # Auxiliary baseline columns (hierarchical median fallback)
    aux_df = apply_simple(df, simple_params)
    for col in feats + pca_cols + year_cols:
        if col in aux_df.columns:
            result[col + "_aux"] = aux_df[col]
        else:
            result[col + "_aux"] = 0.0

    # Was-missing flags
    for feat in missing_flag_feats:
        result[feat + "_missing"] = df[feat].isna().astype(float)

    # Regime as ordinal numeric
    if CATEGORICAL_FEATURE in df.columns:
        regime = df[CATEGORICAL_FEATURE].copy()
        if regime_mode is not None:
            regime = regime.fillna(regime_mode)
        result[CATEGORICAL_FEATURE] = regime

    return result
