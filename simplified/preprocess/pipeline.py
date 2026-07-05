"""
Preprocessing pipeline orchestrator.

Runs the 4-step pipeline (clean → embed → impute → engineer) and caches the result
so expensive steps like MICE are not repeated across runs.

The fingerprint covers: panel content, all preprocess source files, all config
constants, and the training constants that determine fold structure — so any change
that affects cached data automatically invalidates the cache.

RFE (step 4) is target-specific and handled by preprocess.select.get_or_cache_rfe.

Usage:
    pipeline = PreprocessPipeline(branch="simple", pca_mode="static")
    fitted_steps, col_report = pipeline.fit(train_df)
    train_out = pipeline.transform(train_df, fitted_steps)
    val_out   = pipeline.transform(val_df,   fitted_steps)
"""
import functools
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from .cleaning import fit_cleaning, apply_cleaning
from .embeddings import fit_country_pca, transform_country_pca
from .impute_simple import fit_simple, apply_simple
from .impute_mice import fit_mice, apply_mice
from .engineer import fit_engineer, apply_engineer
from .config import (
    MISSINGNESS_THRESHOLD, RFE_K, PCA_N_COMPONENTS, IQR_MULTIPLIER,
    IQR_COUNTRY_MIN_OBS, IQR_YEAR_MIN_OBS, MICE_MAX_ITER, MICE_TOL,
    MICE_MISSING_FLAG_THRESHOLD, ROLLING_MEAN_WINDOW, ROLLING_STD_WINDOWS,
    SLIDING_PCA_WINDOW, CONTINUOUS_FEATURES, VOLATILE_FEATURES,
    STRUCTURAL_FEATURES, CATEGORICAL_FEATURE,
)


def _file_sha256(path):
    """SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


@functools.lru_cache(maxsize=None)
def compute_fingerprint(panel_path, branch, pca_mode):
    """
    Comprehensive SHA-256 fingerprint covering all inputs that affect
    what ends up in the preprocessed cache.

    Includes: panel content, all preprocess/*.py source files, every
    config constant that affects outputs, and the training constants
    that determine fold structure (lazy import to avoid circular deps).
    """
    # Lazy import — train.config does not import from preprocess, so no cycle
    from train.config import OUTER_TEST_YEARS, N_INNER_FOLDS, FINAL_HOLDOUT_YEAR

    # Hash all preprocess source files as a single blob (sorted for stability)
    preprocess_dir = Path(__file__).parent
    src_h = hashlib.sha256()
    for src in sorted(preprocess_dir.glob("*.py")):
        src_h.update(src.read_bytes())
    sources_sha256 = src_h.hexdigest()

    cfg = {
        "panel_sha256": _file_sha256(panel_path),
        "sources_sha256": sources_sha256,
        "branch": branch,
        "pca_mode": pca_mode,
        "missingness_threshold": MISSINGNESS_THRESHOLD,
        "rfe_k": RFE_K,
        "pca_n_components": PCA_N_COMPONENTS,
        "iqr_multiplier": IQR_MULTIPLIER,
        "iqr_country_min_obs": IQR_COUNTRY_MIN_OBS,
        "iqr_year_min_obs": IQR_YEAR_MIN_OBS,
        "mice_max_iter": MICE_MAX_ITER,
        "mice_tol": MICE_TOL,
        "mice_missing_flag_threshold": MICE_MISSING_FLAG_THRESHOLD,
        "rolling_mean_window": ROLLING_MEAN_WINDOW,
        "rolling_std_windows": ROLLING_STD_WINDOWS,
        "sliding_pca_window": SLIDING_PCA_WINDOW,
        "continuous_features": sorted(CONTINUOUS_FEATURES),
        "volatile_features": sorted(VOLATILE_FEATURES),
        "structural_features": sorted(STRUCTURAL_FEATURES),
        "outer_test_years": sorted(OUTER_TEST_YEARS),
        "n_inner_folds": N_INNER_FOLDS,
        "final_holdout_year": FINAL_HOLDOUT_YEAR,
    }
    key_str = json.dumps(cfg, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


class PreprocessPipeline:
    """
    Fits and applies the preprocessing pipeline (steps 1–3, excluding RFE).

    Parameters
    ----------
    branch : "simple" or "mice"
    pca_mode : "static" or "sliding"
    """

    def __init__(self, branch="simple", pca_mode="static"):
        if branch not in ("simple", "mice"):
            raise ValueError(f"branch must be 'simple' or 'mice', got {branch!r}")
        if pca_mode not in ("static", "sliding"):
            raise ValueError(f"pca_mode must be 'static' or 'sliding', got {pca_mode!r}")
        self.branch = branch
        self.pca_mode = pca_mode

    def fit(self, train_df):
        """Fit all preprocessing steps on training data. Returns (fitted_steps, col_report)."""
        cleaning_params = fit_cleaning(train_df)
        train_cleaned = apply_cleaning(train_df, cleaning_params)

        # Forward-fill missing values by country (handles reporting lags, max 3 years)
        ffill_cols = [c for c in CONTINUOUS_FEATURES + [CATEGORICAL_FEATURE] if c in train_cleaned.columns]
        train_cleaned[ffill_cols] = train_cleaned.groupby("iso3")[ffill_cols].ffill(limit=3)

        train_good_rows = train_cleaned[~train_cleaned["ghost_row"]].copy()

        pca_params = fit_country_pca(train_good_rows, mode=self.pca_mode)
        train_embedded = transform_country_pca(train_cleaned, pca_params)

        train_embedded_good = train_embedded[~train_embedded["ghost_row"]].copy()

        if self.branch == "simple":
            impute_params = fit_simple(train_embedded_good)
            train_imputed = apply_simple(train_embedded, impute_params)
        else:
            impute_params = fit_mice(train_embedded_good)
            train_imputed = apply_mice(train_embedded, impute_params)

        engineer_params = fit_engineer(train_imputed)
        train_engineered = apply_engineer(train_imputed, engineer_params)

        fitted_steps = {
            "cleaning_params": cleaning_params,
            "pca_params": pca_params,
            "impute_params": impute_params,
            "engineer_params": engineer_params,
            "branch": self.branch,
        }
        col_report = {
            "after_cleaning": train_cleaned.columns.tolist(),
            "after_embedding": train_embedded.columns.tolist(),
            "after_imputation": train_imputed.columns.tolist(),
            "after_engineering": train_engineered.columns.tolist(),
        }
        return fitted_steps, col_report

    def transform(self, df, fitted_steps, history_df=None, drop_ghost_rows=True):
        """
        Apply fitted preprocessing to a new dataframe (e.g. validation or test split).

        Parameters
        ----------
        history_df : DataFrame or None
            Raw training data to prepend before rolling-feature computation.
            History rows are tagged `_is_history=True` and unconditionally dropped
            after engineering — they exist only to provide rolling-window context.
        drop_ghost_rows : bool, default True
            When True, high-missingness rows (ghost_row=True from cleaning) are
            dropped after rolling features are computed. Pass False for the final
            holdout to keep all rows for lenient/all-country evaluation.
        """
        branch = fitted_steps["branch"]

        df = apply_cleaning(df, fitted_steps["cleaning_params"])
        df = df.copy()
        df["_is_history"] = False

        if history_df is not None and not history_df.empty:
            hist = apply_cleaning(history_df, fitted_steps["cleaning_params"])
            hist = hist.copy()
            hist["_is_history"] = True   # marks rows to always drop after rolling
            
            df = pd.concat([hist, df], ignore_index=True).sort_values(
                ["iso3", "year"], kind="stable"
            ).reset_index(drop=True)

        # Forward-fill missing values by country (handles reporting lags, max 3 years)
        ffill_cols = [c for c in CONTINUOUS_FEATURES + [CATEGORICAL_FEATURE] if c in df.columns]
        df[ffill_cols] = df.groupby("iso3")[ffill_cols].ffill(limit=3)

        df = transform_country_pca(df, fitted_steps["pca_params"])

        if branch == "simple":
            df = apply_simple(df, fitted_steps["impute_params"])
        else:
            df = apply_mice(df, fitted_steps["impute_params"])

        df = apply_engineer(df, fitted_steps["engineer_params"],
                            drop_ghost_rows=drop_ghost_rows)
        return df

    def fit_transform(self, train_df, val_df=None):
        """
        Fit on train_df and transform both train_df and val_df.
        Returns (train_out, val_out, fitted_steps, col_report).
        val_out is None if val_df is None.
        """
        fitted_steps, col_report = self.fit(train_df)
        train_out = self.transform(train_df, fitted_steps)
        # Pass raw train_df as history so val rolling features have real context.
        val_out = (
            self.transform(val_df, fitted_steps, history_df=train_df)
            if val_df is not None else None
        )
        return train_out, val_out, fitted_steps, col_report


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _index_pairs(df):
    iso3 = df["iso3"].tolist() if "iso3" in df.columns else []
    year = df["year"].tolist() if "year" in df.columns else []
    return list(zip(iso3, year))


def _save_indices(path, train_df, other_df, other_key="test"):
    path = Path(path)
    path.write_text(json.dumps(
        {"train": _index_pairs(train_df), other_key: _index_pairs(other_df)},
        indent=2,
    ))


def _write_outer_cache(cache_dir, train_out, test_out, fitted_steps, col_report,
                       outer_train, outer_test, branch, pca_mode, test_year):
    """Write all outer-fold cache files atomically (mkdir + write)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_out.to_csv(cache_dir / "train.csv", index=False)
    test_out.to_csv(cache_dir / "test.csv", index=False)
    joblib.dump(fitted_steps, cache_dir / "preprocessor.pkl")
    (cache_dir / "column_report.json").write_text(json.dumps(col_report, indent=2))
    _save_indices(cache_dir / "indices.json", outer_train, outer_test, "test")
    (cache_dir.parent / "config.json").write_text(json.dumps({
        "branch": branch,
        "pca_mode": pca_mode,
        "missingness_threshold": MISSINGNESS_THRESHOLD,
        "test_year": test_year,
    }, indent=2))


# ---------------------------------------------------------------------------
# Public cache functions
# ---------------------------------------------------------------------------

def get_or_cache_outer_fold(
    outer_train, outer_test, test_year, branch, pca_mode, artifacts_dir, panel_path
):
    """
    Return preprocessed (outer_train_eng, outer_test_eng) for a given outer fold.
    Cached under artifacts/preprocessed/{fingerprint}/outer_{test_year}/.
    """
    fp = compute_fingerprint(panel_path, branch, pca_mode)
    cache_dir = Path(artifacts_dir) / "preprocessed" / fp / f"outer_{test_year}"

    train_path = cache_dir / "train.csv"
    test_path = cache_dir / "test.csv"

    if train_path.exists() and test_path.exists():
        print(f"  [cache hit] outer_{test_year} ({branch}, {pca_mode})")
        return pd.read_csv(train_path), pd.read_csv(test_path)

    print(f"  [preprocessing] outer_{test_year} ({branch}, {pca_mode})...")
    pipeline = PreprocessPipeline(branch=branch, pca_mode=pca_mode)
    train_out, test_out, fitted_steps, col_report = pipeline.fit_transform(outer_train, outer_test)

    _write_outer_cache(
        cache_dir, train_out, test_out, fitted_steps, col_report,
        outer_train, outer_test, branch, pca_mode, test_year,
    )
    return train_out, test_out


def get_or_cache_inner_fold(
    inner_train_raw, inner_val_raw,
    outer_test_year, strategy, fold_id,
    branch, pca_mode, panel_path, artifacts_dir,
):
    """
    Return preprocessed (inner_train_eng, inner_val_eng) for a given inner fold.
    Cached under artifacts/preprocessed/{fp}/outer_{year}/inner_{strategy}/fold_{fold_id}/.
    """
    fp = compute_fingerprint(panel_path, branch, pca_mode)
    fold_dir = (
        Path(artifacts_dir) / "preprocessed" / fp
        / f"outer_{outer_test_year}" / f"inner_{strategy}" / f"fold_{fold_id}"
    )

    train_path = fold_dir / "train.csv"
    val_path = fold_dir / "val.csv"

    if train_path.exists() and val_path.exists():
        print(f"  [cache hit] outer_{outer_test_year}/inner_{strategy}/fold_{fold_id} ({branch}, {pca_mode})")
        return pd.read_csv(train_path), pd.read_csv(val_path)

    print(f"  [preprocessing] outer_{outer_test_year}/inner_{strategy}/fold_{fold_id} ({branch}, {pca_mode})...")
    pipeline = PreprocessPipeline(branch=branch, pca_mode=pca_mode)
    train_out, val_out, fitted_steps, col_report = pipeline.fit_transform(inner_train_raw, inner_val_raw)

    fold_dir.mkdir(parents=True, exist_ok=True)
    train_out.to_csv(train_path, index=False)
    val_out.to_csv(val_path, index=False)
    joblib.dump(fitted_steps, fold_dir / "preprocessor.pkl")
    (fold_dir / "column_report.json").write_text(json.dumps(col_report, indent=2))
    _save_indices(fold_dir / "indices.json", inner_train_raw, inner_val_raw, "val")

    return train_out, val_out


def get_or_cache_final(
    final_train_raw, holdout_raw,
    branch, pca_mode, panel_path, artifacts_dir,
):
    """
    Return preprocessed (final_train_eng, holdout_all_eng) for the final training set.

    holdout_all_eng retains ALL holdout rows including high-missingness ones
    (ghost_row column is preserved). Callers split into strict/lenient/all surfaces.
    Cached under artifacts/preprocessed/{fp}/final/.
    """
    fp = compute_fingerprint(panel_path, branch, pca_mode)
    final_dir = Path(artifacts_dir) / "preprocessed" / fp / "final"

    train_path = final_dir / "train.csv"
    holdout_path = final_dir / "holdout_all.csv"

    if train_path.exists() and holdout_path.exists():
        print(f"  [cache hit] final ({branch}, {pca_mode})")
        return pd.read_csv(train_path), pd.read_csv(holdout_path)

    print(f"  [preprocessing] final ({branch}, {pca_mode})...")
    pip = PreprocessPipeline(branch=branch, pca_mode=pca_mode)
    fitted_steps, col_report = pip.fit(final_train_raw)
    train_out = pip.transform(final_train_raw, fitted_steps)
    # Retain all holdout rows (ghost rows kept) so callers can build strict/lenient/all.
    holdout_all_out = pip.transform(
        holdout_raw, fitted_steps,
        history_df=final_train_raw,
        drop_ghost_rows=False,
    )

    final_dir.mkdir(parents=True, exist_ok=True)
    train_out.to_csv(train_path, index=False)
    holdout_all_out.to_csv(holdout_path, index=False)
    joblib.dump(fitted_steps, final_dir / "preprocessor.pkl")
    (final_dir / "column_report.json").write_text(json.dumps(col_report, indent=2))
    _save_indices(final_dir / "indices.json", final_train_raw, holdout_raw, "holdout")

    return train_out, holdout_all_out
