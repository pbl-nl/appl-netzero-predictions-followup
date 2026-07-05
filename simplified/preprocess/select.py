"""Step 4: Recursive Feature Elimination with ExtraTrees (k=24 features)."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import RFE

from .config import RFE_K, EXCLUDE_COLS


def _get_feature_cols(df, target_col):
    """Return all model-usable feature columns (not identifiers, targets, ghost_row)."""
    skip = set(EXCLUDE_COLS) | {"ghost_row"}
    return [c for c in df.columns if c not in skip and c != target_col and df[c].dtype != object]


def fit_rfe(train_df, target_col, common_feature_cols=None):
    """
    Fit RFE on training data for a given target.
    Only rows with non-missing target are used for fitting.

    Parameters
    ----------
    common_feature_cols : list or None
        If provided, restrict the feature universe to this intersection before
        fitting (columns missing from train_df are filled with 0.0). Matches the
        original thesis methodology where RFE operates on the common feature set
        across all inner fold training partitions.

    Returns (rfe, selected_feature_names, all_feature_cols).
    """
    if common_feature_cols is not None:
        feature_cols = [c for c in common_feature_cols
                        if c != target_col and c not in set(EXCLUDE_COLS) | {"ghost_row"}]
        # Fill any missing columns with 0 so all folds have a consistent universe
        missing = [c for c in feature_cols if c not in train_df.columns]
        if missing:
            train_df = train_df.copy()
            for c in missing:
                train_df[c] = 0.0
    else:
        feature_cols = _get_feature_cols(train_df, target_col)

    mask = train_df[target_col].notna()
    X = train_df.loc[mask, feature_cols].fillna(0.0)
    y = train_df.loc[mask, target_col]

    estimator = ExtraTreesRegressor(
        n_estimators=50,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=42,
    )
    rfe = RFE(estimator, n_features_to_select=RFE_K, step=0.5)
    rfe.fit(X, y)

    selected = [col for col, support in zip(feature_cols, rfe.support_) if support]
    return rfe, selected, feature_cols


def apply_rfe(df, rfe, selected_features, all_feature_cols):
    """
    Apply fitted RFE to df: return df with only the selected features retained
    (plus identifier and target columns from EXCLUDE_COLS).

    Any selected feature absent from df is filled with 0.0 so that val/test
    sets that are missing a column (e.g. due to a single-year rolling edge case)
    do not cause a shape mismatch at predict time.
    """
    missing = [c for c in selected_features if c not in df.columns]
    if missing:
        df = df.copy()
        for c in missing:
            df[c] = 0.0
    keep = set(EXCLUDE_COLS) | set(selected_features)
    keep_cols = [c for c in df.columns if c in keep]
    return df[keep_cols].copy()


def get_or_cache_rfe(train_eng, target_col, cache_dir, common_feature_cols=None):
    """
    Get or compute RFE for a given engineered training frame and target.

    Saves rfe.pkl, selected_features.json, and (when common_feature_cols is given)
    common_feature_columns.json under cache_dir.
    Returns (rfe, selected_features, all_feature_cols).
    """
    cache_dir = Path(cache_dir)
    rfe_path = cache_dir / "rfe.pkl"
    features_path = cache_dir / "selected_features.json"

    if rfe_path.exists() and features_path.exists():
        rfe = joblib.load(rfe_path)
        meta = json.loads(features_path.read_text())
        return rfe, meta["features"], meta["all_columns"]

    rfe, selected, all_cols = fit_rfe(train_eng, target_col,
                                      common_feature_cols=common_feature_cols)
    cache_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(rfe, rfe_path)
    features_path.write_text(json.dumps({
        "target": target_col,
        "features": selected,
        "all_columns": all_cols,
    }, indent=2))
    if common_feature_cols is not None:
        (cache_dir / "common_feature_columns.json").write_text(
            json.dumps({"common_feature_columns": list(common_feature_cols)}, indent=2)
        )
    return rfe, selected, all_cols
