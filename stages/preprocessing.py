"""
preprocessing.py – Preprocessor class + preprocessing cache management.

The Preprocessor class implements a three-phase pipeline:
    clean → impute → engineer → encode/scale

Two imputation branches:
    simple : Country → year → global median fallback
    mice   : Iterative imputer with panel-aware auxiliary predictors

The cache management layer preprocesses all splits/folds once and persists
them to disk, keyed by a fingerprint of the data, pipeline code, and config.

Run:  python -m stages.preprocessing          # build or reuse cache
      python -m stages.preprocessing --force   # force rebuild
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


# =============================================================================
# Configurable constants
# =============================================================================

DATA_PATH = Path("data/merged/merged.csv")

COUNTRY_COL = "country"
YEAR_COL = "year"
ID_COLS = [COUNTRY_COL, YEAR_COL]
TARGET_COLS = ["linear", "optimistic", "pessimistic"]
CATEGORICAL_FEATURES = ["v2x_regime"]

YEAR_CUTOFF = 2022
TEST_YEAR = 2022
VALIDATION_STRATEGY = "2021_holdout"  # Options: "2021_holdout", "random_pre_2022"
VALIDATION_YEAR = 2021
RANDOM_VALIDATION_SIZE = 0.20
RANDOM_STATE = 42

ROW_MISSINGNESS_THRESHOLD = 0.30
SPARSE_COLUMN_THRESHOLD = 0.95
OUTLIER_METHOD = "iqr"
IQR_MULTIPLIER = 1.5
COUNTRY_OUTLIER_MIN_OBS = 5
YEAR_OUTLIER_MIN_OBS = 20

LAG_PERIODS = [1]
ROLLING_WINDOWS = [3]
ROLLING_MIN_PERIODS = 1
STRUCTURAL_NA_FILL_VALUE = 0.0
INTERACTION_PAIRS: list[tuple[str, str]] = [
    ("total_ghg_emissions", "population_total"),
    ("oil_rents_of_gdp", "total_ghg_emissions"),
    ("renewable_energy_share_tfc_pct", "energy_intensity_mj_per_usd_ppp"),
    ("gdp_per_capita_constant_2015_us", "total_ghg_emissions"),
    ("control_of_corruption_estimate", "renewable_energy_share_tfc_pct"),
]
ENABLE_DIFF_FEATURES = True
ENABLE_PCT_CHANGE_FEATURES = True

SCALER_CHOICE = "standard"  # Options: "standard", "minmax", "robust"
MICE_MAX_ITER = 20
MICE_TOL = 1e-3
MICE_SKIP_COMPLETE = True
MICE_COUNTRY_N_COMPONENTS = 5
MICE_MISSING_THRESHOLD = 0.05  # features above this rate get was_missing flags

SHOW_SAMPLE_ROWS = 3
TOP_MISSINGNESS_ROWS = 10

RATE_FEATURES = {
    "ghg_growth",
    "annual_deforestation_of_change",
}

PCT_CHANGE_FEATURES = [
    "renewable_energy_share_tfc_pct",
    "renewable_electricity_share_pct",
    "energy_intensity_mj_per_usd_ppp",
    "inform_risk_index",
    "non_co2_emissions",
    "total_hard_to_abate_emissions",
    "rsf_press_freedom_index",
    "average_supply_of_protein_of_animal_origin_g_cap_day_3_year_average",
    "gdp_per_capita_constant_2015_us",
    "gdp_ppp_constant_2021_international",
    "population_density_people_per_sq_km_of_land_area",
    "population_total",
]

NUMERIC_RANGE_RULES: dict[str, dict[str, Any]] = {
    "renewable_energy_share_tfc_pct": {"lower": 0.0, "upper": 100.0},
    "renewable_electricity_share_pct": {"lower": 0.0, "upper": 100.0},
    "energy_intensity_mj_per_usd_ppp": {"lower": 0.0, "upper": 100.0},
    "inform_risk_index": {"lower": 0.0, "upper": 10.0},
    "ghg_growth": {"finite_only": True},
    "non_co2_emissions": {"lower": 0.0, "upper": 5000.0},
    "total_ghg_emissions": {"lower": -5000.0, "upper": 20000.0},
    "total_hard_to_abate_emissions": {"lower": 0.0, "upper": 5000.0},
    "rsf_press_freedom_index": {"lower": 0.0, "upper": 100.0},
    "agricultural_land_of_land_area": {"lower": 0.0, "upper": 100.0},
    "annual_deforestation_of_change": {"lower": -100.0, "upper": 100.0},
    "average_supply_of_protein_of_animal_origin_g_cap_day_3_year_average": {
        "lower": 0.0, "upper": 250.0,
    },
    "coal_rents_of_gdp": {"lower": 0.0, "upper": 100.0},
    "control_of_corruption_estimate": {"lower": -4.0, "upper": 4.0},
    "forest_area_of_land_area": {"lower": 0.0, "upper": 100.0},
    "gdp_per_capita_constant_2015_us": {"lower": 0.0, "upper": 500000.0},
    "gdp_ppp_constant_2021_international": {"lower": 0.0, "upper": 2.0e14},
    "natural_gas_rents_of_gdp": {"lower": 0.0, "upper": 100.0},
    "oil_rents_of_gdp": {"lower": 0.0, "upper": 100.0},
    "political_stability_and_absence_of_violence_terrorism_estimate": {
        "lower": -4.0, "upper": 4.0,
    },
    "population_density_people_per_sq_km_of_land_area": {"lower": 0.0, "upper": 100000.0},
    "population_total": {"lower": 0.0, "upper": 1.0e10},
    "regulatory_quality_estimate": {"lower": -4.0, "upper": 4.0},
    "voice_and_accountability_estimate": {"lower": -4.0, "upper": 4.0},
}

CATEGORICAL_ALLOWED_VALUES: dict[str, set[int]] = {
    "v2x_regime": {0, 1, 2, 3},
}

BRANCH_NAMES = ("simple", "mice")

CACHE_ROOT = Path("data/preprocessed")


# =============================================================================
# Utility functions
# =============================================================================


def safe_mode(series: pd.Series) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    mode = non_null.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else non_null.iloc[0]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in ID_COLS + TARGET_COLS]


def get_continuous_features(df: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    return [
        col for col in feature_columns
        if col not in CATEGORICAL_FEATURES and pd.api.types.is_numeric_dtype(df[col])
    ]


# =============================================================================
# Data loading
# =============================================================================


def load_panel_data(path: Path = DATA_PATH, verbose: bool = True) -> pd.DataFrame:
    if verbose:
        logger.info("=" * 80)
        logger.info("Load Panel Data")
        logger.info("=" * 80)

    df = pd.read_csv(path)
    if YEAR_COL not in df.columns or COUNTRY_COL not in df.columns:
        raise ValueError(f"Expected {COUNTRY_COL!r} and {YEAR_COL!r} in {path}")

    df = df[df[YEAR_COL] <= YEAR_CUTOFF].copy()
    df = df.sort_values(ID_COLS).reset_index(drop=True)

    duplicate_count = df.duplicated(ID_COLS).sum()
    if verbose:
        logger.info(f"Path: {path}")
        logger.info(f"Shape after year <= {YEAR_CUTOFF}: {df.shape}")
        logger.info(f"Duplicate country-year rows: {duplicate_count}")
    if duplicate_count:
        raise ValueError("Duplicate country-year rows detected.")

    if verbose:
        feature_columns = get_feature_columns(df)
        miss = df[feature_columns].isna().mean().sort_values(ascending=False).head(TOP_MISSINGNESS_ROWS)
        logger.info(f"Top feature missingness:\n{miss.to_string()}")
        for col in TARGET_COLS:
            logger.info(f"Target {col} missing: {int(df[col].isna().sum())}")

    return df


def split_raw_panel(
    df: pd.DataFrame,
    validation_strategy: str = VALIDATION_STRATEGY,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    test_df = df[df[YEAR_COL] == TEST_YEAR].copy()
    pre_test_df = df[df[YEAR_COL] < TEST_YEAR].copy()

    if validation_strategy == "2021_holdout":
        train_df = pre_test_df[pre_test_df[YEAR_COL] < VALIDATION_YEAR].copy()
        val_df = pre_test_df[pre_test_df[YEAR_COL] == VALIDATION_YEAR].copy()
    elif validation_strategy == "random_pre_2022":
        train_idx, val_idx = train_test_split(
            pre_test_df.index,
            test_size=RANDOM_VALIDATION_SIZE,
            random_state=RANDOM_STATE,
            shuffle=True,
        )
        train_df = pre_test_df.loc[sorted(train_idx)].copy()
        val_df = pre_test_df.loc[sorted(val_idx)].copy()
    else:
        raise ValueError(f"Unsupported validation strategy: {validation_strategy}")

    if verbose:
        for name, frame in [("train", train_df), ("val", val_df), ("test", test_df)]:
            all_missing = frame[TARGET_COLS].isna().all(axis=1).sum()
            logger.info(
                f"{name}: rows={len(frame)}, "
                f"year_range=({int(frame[YEAR_COL].min())}, {int(frame[YEAR_COL].max())}), "
                f"all_targets_missing={int(all_missing)}"
            )

    return {"train": train_df, "val": val_df, "test": test_df}


# =============================================================================
# Grouped imputation
# =============================================================================


class CountryYearFallbackImputer:
    def __init__(self, numeric_features: list[str], categorical_features: list[str]) -> None:
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features)
        self.country_stats_: dict[str, pd.Series] = {}
        self.year_stats_: dict[str, pd.Series] = {}
        self.global_stats_: dict[str, Any] = {}

    def fit(self, df: pd.DataFrame) -> "CountryYearFallbackImputer":
        for col in self.numeric_features:
            self.country_stats_[col] = df.groupby(COUNTRY_COL)[col].median()
            self.year_stats_[col] = df.groupby(YEAR_COL)[col].median()
            self.global_stats_[col] = df[col].median()
        for col in self.categorical_features:
            self.country_stats_[col] = df.groupby(COUNTRY_COL)[col].agg(safe_mode)
            self.year_stats_[col] = df.groupby(YEAR_COL)[col].agg(safe_mode)
            self.global_stats_[col] = safe_mode(df[col])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.numeric_features + self.categorical_features:
            country_fill = out[COUNTRY_COL].map(self.country_stats_.get(col, pd.Series(dtype=float)))
            year_fill = out[YEAR_COL].map(self.year_stats_.get(col, pd.Series(dtype=float)))
            global_fill = self.global_stats_.get(col, np.nan)
            out[col] = out[col].fillna(country_fill).fillna(year_fill).fillna(global_fill)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


# =============================================================================
# Preprocessor
# =============================================================================


class Preprocessor:
    """Three-phase preprocessor: clean → impute → engineer.

    Usage:
        prep = Preprocessor(active_branches=["simple"])
        train_outputs = prep.fit_transform(train_df)       # dict[branch → df]
        val_outputs = prep.transform(val_df, history_df=train_df)
    """

    def __init__(
        self,
        active_branches: list[str] | None = None,
        train_start_year: int | None = None,
    ) -> None:
        if active_branches is None:
            active_branches = list(BRANCH_NAMES)
        invalid = sorted(set(active_branches) - set(BRANCH_NAMES))
        if invalid:
            raise ValueError(f"Unsupported branches: {invalid}")

        self.active_branches_ = list(active_branches)
        self.train_start_year_: int | None = train_start_year

        # Fitted state — set during fit_transform
        self.feature_columns_: list[str] = []
        self.continuous_features_: list[str] = []
        self.retained_raw_features_: list[str] = []
        self.retained_continuous_features_: list[str] = []
        self.columns_to_drop_: set[str] = set()

        # Country PCA embeddings
        self.country_embedding_map_: dict[str, np.ndarray] = {}
        self.country_pca_global_fallback_: np.ndarray = np.zeros(MICE_COUNTRY_N_COMPONENTS)
        self.country_n_pca_components_: int = 0
        self.pca_component_names_: list[str] = []

        # Year standardization
        self.year_mean_: float = 0.0
        self.year_std_: float = 1.0

        # Names of all synthetic (non-engineerable) columns
        self.synthetic_feature_names_: list[str] = []

        # Imputers — simple_imputer_ is always fitted (used by both branches)
        self.simple_imputer_: CountryYearFallbackImputer | None = None
        self.mice_numeric_imputer_: IterativeImputer | None = None
        self.mice_categorical_imputer_: SimpleImputer | None = None
        self.numeric_global_fallback_: pd.Series | None = None

        # MICE enrichment helpers
        self.mice_scaler_means_: pd.Series | None = None
        self.mice_scaler_stds_: pd.Series | None = None
        self.mice_missing_flags_: list[str] = []       # features with was_missing indicators
        self.mice_aux_col_names_: list[str] = []       # panel-aware auxiliary column names
        self.mice_input_cols_: list[str] = []          # full ordered MICE input column set
        self.mice_regime_col_: str | None = None       # v2x_regime predictor column name

        # Encoders / scalers per branch
        self.encoder_: dict[str, OneHotEncoder] = {}
        self.scaler_: dict[str, Any] = {}
        self.final_feature_columns_: dict[str, list[str]] = {}

    # -----------------------------------------------------------------
    # Synthetic feature helpers (country PCA + year features)
    # -----------------------------------------------------------------
    def _build_country_embeddings(self, cleaned_train_df: pd.DataFrame) -> None:
        """Build country PCA embeddings from cleaned training data."""
        year_vals = cleaned_train_df[YEAR_COL].values.astype(float)
        self.year_mean_ = float(np.nanmean(year_vals))
        yr_std = float(np.nanstd(year_vals))
        self.year_std_ = yr_std if yr_std > 0 else 1.0

        feats = [c for c in self.retained_continuous_features_ if c in cleaned_train_df.columns]
        if not feats or COUNTRY_COL not in cleaned_train_df.columns:
            self.country_embedding_map_ = {}
            self.country_n_pca_components_ = 0
            self.pca_component_names_ = []
            return

        grouped = cleaned_train_df.groupby(COUNTRY_COL)[feats]
        medians = grouped.median()
        miss_rates = grouped.apply(lambda g: g.isna().mean())
        medians.columns = [f"med_{c}" for c in feats]
        miss_rates.columns = [f"miss_{c}" for c in feats]
        summary = pd.concat([medians, miss_rates], axis=1)
        countries = summary.index.tolist()

        fill_imp = SimpleImputer(strategy="median")
        summary_filled = fill_imp.fit_transform(summary.values)
        scaler = StandardScaler()
        summary_scaled = scaler.fit_transform(summary_filled)

        n_components = min(
            MICE_COUNTRY_N_COMPONENTS,
            max(len(countries) - 1, 1),
            summary_scaled.shape[1],
        )
        if n_components < 1:
            self.country_embedding_map_ = {}
            self.country_n_pca_components_ = 0
            self.pca_component_names_ = []
            return

        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        components = pca.fit_transform(summary_scaled)
        self.country_embedding_map_ = {
            country: components[i] for i, country in enumerate(countries)
        }
        self.country_n_pca_components_ = n_components
        self.country_pca_global_fallback_ = np.zeros(n_components)
        self.pca_component_names_ = [f"country_pc{i+1}" for i in range(n_components)]
        logger.info(
            f"Country embeddings: {len(countries)} countries -> "
            f"{n_components} PCA components "
            f"(explained variance: {pca.explained_variance_ratio_.sum():.1%})"
        )

    def _attach_synthetic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add permanent synthetic columns: country PCA components + year features."""
        out = df.copy()
        if self.country_n_pca_components_ > 0:
            fallback = self.country_pca_global_fallback_
            embeddings = np.array([
                self.country_embedding_map_.get(c, fallback) for c in out[COUNTRY_COL]
            ])
            for i, col_name in enumerate(self.pca_component_names_):
                out[col_name] = embeddings[:, i]
        year_std = (out[YEAR_COL].values.astype(float) - self.year_mean_) / self.year_std_
        out["year_std"] = year_std
        out["year_std_sq"] = year_std ** 2
        self.synthetic_feature_names_ = self.pca_component_names_ + ["year_std", "year_std_sq"]
        return out

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def fit_transform(self, train_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        pid = os.getpid()
        self.feature_columns_ = get_feature_columns(train_df)
        self.continuous_features_ = get_continuous_features(train_df, self.feature_columns_)

        # Phase 1: Clean
        cleaned = self._clean(train_df, history_df=None, fit=True)
        cleaned = self._fill_temporal_gaps(cleaned)
        n_good = int((~cleaned["_to_drop"]).sum())
        n_ghost = int(cleaned["_to_drop"].sum())
        self.retained_raw_features_ = [
            c for c in self.feature_columns_ if c not in self.columns_to_drop_
        ]
        self.retained_continuous_features_ = [
            c for c in self.continuous_features_ if c in self.retained_raw_features_
        ]
        print(
            f"[PID:{pid}] Cleaning: {len(train_df)}→{n_good} rows "
            f"({n_ghost} ghost rows preserved for temporal continuity), "
            f"dropped {len(self.columns_to_drop_)} columns, "
            f"{len(self.retained_continuous_features_)} continuous features retained",
            flush=True,
        )

        good_rows = cleaned.loc[~cleaned["_to_drop"]].copy()
        self._build_country_embeddings(good_rows)
        cleaned = self._attach_synthetic_features(cleaned)
        self.feature_columns_.extend(self.synthetic_feature_names_)
        self.continuous_features_.extend(self.synthetic_feature_names_)
        self.retained_raw_features_.extend(self.synthetic_feature_names_)
        self.retained_continuous_features_.extend(self.synthetic_feature_names_)
        good_rows = cleaned.loc[~cleaned["_to_drop"]].copy()

        self._fit_imputers(good_rows)

        outputs: dict[str, pd.DataFrame] = {}
        for branch in self.active_branches_:
            imputed = self._impute(cleaned, branch)
            feat_cols = get_feature_columns(imputed)
            nans = int(imputed[feat_cols].isna().sum().sum())
            print(f"[PID:{pid}] Imputation ({branch}): {nans} NaNs remaining", flush=True)

            engineered = self._engineer(imputed, history_df=None)
            engineered = (
                engineered.loc[~engineered["_to_drop"]]
                .drop(columns=["_to_drop"])
                .reset_index(drop=True)
            )

            pre_filter_rows = len(engineered)
            if self.train_start_year_ is not None:
                mask = engineered[YEAR_COL] >= self.train_start_year_
                engineered = engineered[mask].reset_index(drop=True)
                print(
                    f"[PID:{pid}] Year filter ({branch}): "
                    f"{pre_filter_rows}→{len(engineered)} rows (>={self.train_start_year_})",
                    flush=True,
                )

            final = self._encode_and_scale(engineered, branch, fit=True)
            n_feats = len(self.final_feature_columns_[branch])
            print(
                f"[PID:{pid}] Engineering ({branch}): "
                f"{len(self.retained_continuous_features_)}→{n_feats} features",
                flush=True,
            )
            outputs[branch] = final

        return outputs

    def transform(
        self,
        df: pd.DataFrame,
        history_df: pd.DataFrame | None = None,
        split_name: str = "data",
        drop_rows: bool = True,
    ) -> dict[str, pd.DataFrame]:
        cleaned = self._clean(df, history_df=history_df, fit=False, drop_rows=drop_rows)
        cleaned = self._fill_temporal_gaps(cleaned)
        cleaned = self._attach_synthetic_features(cleaned)

        history_cleaned: pd.DataFrame | None = None
        if history_df is not None and not history_df.empty:
            history_cleaned = self._clean(history_df, history_df=None, fit=False)
            current_max_year = int(cleaned[YEAR_COL].max())
            history_cleaned = self._fill_temporal_gaps(
                history_cleaned, up_to_year=current_max_year - 1
            )
            history_cleaned = self._attach_synthetic_features(history_cleaned)

        outputs: dict[str, pd.DataFrame] = {}
        for branch in self.active_branches_:
            imputed = self._impute(cleaned, branch)

            history_imputed = None
            if history_cleaned is not None:
                history_imputed = self._impute(history_cleaned, branch)

            engineered = self._engineer(imputed, history_df=history_imputed)
            engineered = (
                engineered.loc[~engineered["_to_drop"]]
                .drop(columns=["_to_drop"])
                .reset_index(drop=True)
            )
            outputs[branch] = self._encode_and_scale(engineered, branch, fit=False)

        return outputs

    # -----------------------------------------------------------------
    # Phase 1: Clean
    # -----------------------------------------------------------------
    def _clean(
        self,
        df: pd.DataFrame,
        history_df: pd.DataFrame | None,
        fit: bool,
        drop_rows: bool = True,
    ) -> pd.DataFrame:
        current = df.copy().sort_values(ID_COLS).reset_index(drop=True)
        current["_is_current"] = True

        if history_df is not None and not history_df.empty:
            history = history_df.copy().sort_values(ID_COLS).reset_index(drop=True)
            history["_is_current"] = False
            working = pd.concat([history, current], ignore_index=True, sort=False)
        else:
            working = current.copy()

        outlier_features = [c for c in self.continuous_features_ if c not in self.synthetic_feature_names_]
        self._apply_outlier_pass(working, COUNTRY_COL, outlier_features, COUNTRY_OUTLIER_MIN_OBS)
        self._apply_outlier_pass(working, YEAR_COL, outlier_features, YEAR_OUTLIER_MIN_OBS)
        self._apply_numeric_range_rules(working)
        self._apply_categorical_range_rules(working)

        cleaned = working[working["_is_current"]].copy().drop(columns=["_is_current"], errors="ignore")

        if drop_rows:
            feature_cols = get_feature_columns(cleaned)
            real_feature_cols = [c for c in feature_cols if c not in self.synthetic_feature_names_]
            row_missing = cleaned[real_feature_cols].isna().mean(axis=1) > ROW_MISSINGNESS_THRESHOLD
            all_targets_nan = cleaned[TARGET_COLS].isna().all(axis=1)
            drop_mask = all_targets_nan | row_missing
        else:
            drop_mask = pd.Series(False, index=cleaned.index)

        if fit:
            self._fit_column_drops(cleaned.loc[~drop_mask].copy())

        cleaned["_to_drop"] = drop_mask
        cleaned = cleaned.drop(columns=sorted(self.columns_to_drop_), errors="ignore")
        return cleaned.sort_values(ID_COLS).reset_index(drop=True)

    def _apply_outlier_pass(
        self, working: pd.DataFrame, group_col: str, features: list[str], min_obs: int
    ) -> None:
        for col in features:
            if col not in working.columns:
                continue
            grouped = working.groupby(group_col)[col]
            count = grouped.transform("count")
            q1 = grouped.transform(lambda s: s.quantile(0.25))
            q3 = grouped.transform(lambda s: s.quantile(0.75))
            iqr = q3 - q1
            valid = (count >= min_obs) & (iqr > 0) & working[col].notna()
            lower = q1 - IQR_MULTIPLIER * iqr
            upper = q3 + IQR_MULTIPLIER * iqr
            mask = valid & ((working[col] < lower) | (working[col] > upper)) & working["_is_current"]
            working.loc[mask, col] = np.nan

    def _apply_numeric_range_rules(self, working: pd.DataFrame) -> None:
        current_mask = working["_is_current"]
        for col, rule in NUMERIC_RANGE_RULES.items():
            if col not in working.columns:
                continue
            series = working[col]
            invalid = current_mask & series.notna()
            if rule.get("finite_only"):
                invalid &= ~np.isfinite(series)
            else:
                invalid &= (series < rule.get("lower", -np.inf)) | (series > rule.get("upper", np.inf))
            working.loc[invalid, col] = np.nan

    def _apply_categorical_range_rules(self, working: pd.DataFrame) -> None:
        current_mask = working["_is_current"]
        for col, allowed in CATEGORICAL_ALLOWED_VALUES.items():
            if col not in working.columns:
                continue
            invalid = current_mask & working[col].notna() & ~working[col].isin(sorted(allowed))
            working.loc[invalid, col] = np.nan

    def _fit_column_drops(self, train_frame: pd.DataFrame) -> None:
        """Drop sparse and zero-variance columns."""
        feature_cols = get_feature_columns(train_frame)
        sparse = train_frame[feature_cols].isna().mean()
        self.columns_to_drop_ = set(sparse[sparse >= SPARSE_COLUMN_THRESHOLD].index.tolist())
        candidates = [
            c for c in self.continuous_features_
            if c not in self.columns_to_drop_ and c in train_frame.columns
        ]
        if not candidates:
            return
        matrix = train_frame[candidates]
        zero_var = [c for c in matrix.columns if matrix[c].nunique(dropna=False) <= 1]
        self.columns_to_drop_.update(zero_var)

    def _fill_temporal_gaps(
        self, df: pd.DataFrame, up_to_year: int | None = None,
    ) -> pd.DataFrame:
        """Inject synthetic ghost rows for years absent from a country's time series."""
        id_and_meta = set([COUNTRY_COL, YEAR_COL, "_to_drop"] + TARGET_COLS)
        feature_cols = [c for c in df.columns if c not in id_and_meta]
        frames = []
        for country, grp in df.groupby(COUNTRY_COL, sort=False):
            years_present = set(int(y) for y in grp[YEAR_COL])
            if not years_present:
                frames.append(grp)
                continue
            min_year = min(years_present)
            max_year = max(max(years_present), up_to_year) if up_to_year is not None else max(years_present)
            missing_years = sorted(set(range(min_year, max_year + 1)) - years_present)
            if not missing_years:
                frames.append(grp)
                continue
            ghost = pd.DataFrame({COUNTRY_COL: country, YEAR_COL: missing_years})
            for col in feature_cols:
                ghost[col] = np.nan
            for col in TARGET_COLS:
                ghost[col] = np.nan
            ghost["_to_drop"] = True
            frames.append(pd.concat([grp, ghost], ignore_index=True))
        if not frames:
            return df.copy()
        result = pd.concat(frames, ignore_index=True)
        return result.sort_values(ID_COLS).reset_index(drop=True)

    # -----------------------------------------------------------------
    # Phase 2: Impute
    # -----------------------------------------------------------------
    def _fit_imputers(self, cleaned_train: pd.DataFrame) -> None:
        self.numeric_global_fallback_ = cleaned_train[self.retained_continuous_features_].median()
        retained_cats = [c for c in CATEGORICAL_FEATURES if c in self.retained_raw_features_]

        # Always fit simple imputer — used directly by simple branch and as
        # auxiliary context provider for the MICE branch.
        self.simple_imputer_ = CountryYearFallbackImputer(
            numeric_features=self.retained_continuous_features_,
            categorical_features=retained_cats,
        )
        self.simple_imputer_.fit(cleaned_train)

        if "mice" in self.active_branches_ and self.retained_continuous_features_:
            # Missingness flags: features above threshold get was_missing indicators
            miss_rates = cleaned_train[self.retained_continuous_features_].isna().mean()
            self.mice_missing_flags_ = [
                c for c in self.retained_continuous_features_
                if miss_rates.get(c, 0.0) > MICE_MISSING_THRESHOLD
            ]

            # Build enriched fit matrix: original features + panel-aware aux + regime
            mice_fit_df = cleaned_train[self.retained_continuous_features_].copy()

            # Panel-aware auxiliary predictors: country median per feature (no NaN)
            aux_col_names: list[str] = []
            for col in self.retained_continuous_features_:
                c_fill = cleaned_train[COUNTRY_COL].map(self.simple_imputer_.country_stats_[col])
                y_fill = cleaned_train[YEAR_COL].map(self.simple_imputer_.year_stats_[col])
                glob = self.simple_imputer_.global_stats_[col]
                aux_name = f"_aux_{col}"
                mice_fit_df[aux_name] = c_fill.fillna(y_fill).fillna(glob)
                aux_col_names.append(aux_name)
            self.mice_aux_col_names_ = aux_col_names

            # v2x_regime as ordinal predictor
            self.mice_regime_col_ = None
            regime_col = "v2x_regime"
            if regime_col in cleaned_train.columns:
                regime_raw = cleaned_train[regime_col].copy()
                c_regime = cleaned_train[COUNTRY_COL].map(
                    self.simple_imputer_.country_stats_.get(regime_col, pd.Series(dtype=float))
                )
                y_regime = cleaned_train[YEAR_COL].map(
                    self.simple_imputer_.year_stats_.get(regime_col, pd.Series(dtype=float))
                )
                glob_regime = self.simple_imputer_.global_stats_.get(regime_col, 0.0)
                mice_fit_df["_regime"] = (
                    pd.to_numeric(
                        regime_raw.fillna(c_regime).fillna(y_regime).fillna(glob_regime),
                        errors="coerce",
                    ).fillna(0.0)
                )
                self.mice_regime_col_ = "_regime"

            self.mice_input_cols_ = (
                self.retained_continuous_features_
                + aux_col_names
                + (["_regime"] if self.mice_regime_col_ else [])
            )

            self.mice_scaler_means_ = mice_fit_df[self.mice_input_cols_].mean()
            self.mice_scaler_stds_ = mice_fit_df[self.mice_input_cols_].std().replace(0, 1)
            scaled = (
                (mice_fit_df[self.mice_input_cols_] - self.mice_scaler_means_)
                / self.mice_scaler_stds_
            )
            self.mice_numeric_imputer_ = IterativeImputer(
                random_state=RANDOM_STATE,
                max_iter=MICE_MAX_ITER,
                tol=MICE_TOL,
                skip_complete=MICE_SKIP_COMPLETE,
            )
            self.mice_numeric_imputer_.fit(scaled.values)

        if "mice" in self.active_branches_ and retained_cats:
            self.mice_categorical_imputer_ = SimpleImputer(strategy="most_frequent")
            self.mice_categorical_imputer_.fit(cleaned_train[retained_cats])

    def _impute(self, df: pd.DataFrame, branch: str) -> pd.DataFrame:
        out = df.copy()
        retained_cats = [c for c in CATEGORICAL_FEATURES if c in out.columns]

        if branch == "simple":
            if self.simple_imputer_ is None:
                raise RuntimeError("Simple imputer has not been fit.")
            out = self.simple_imputer_.transform(out)

        elif branch == "mice":
            if self.mice_numeric_imputer_ is None and self.retained_continuous_features_:
                raise RuntimeError("MICE imputer has not been fit.")

            # Add binary missingness indicators before values are overwritten
            for col in self.mice_missing_flags_:
                if col in out.columns:
                    out[f"{col}_was_missing"] = out[col].isna().astype(float)

            if self.retained_continuous_features_:
                # Build enriched MICE input: original features + aux + regime
                mice_in = out[self.retained_continuous_features_].copy()

                for col, aux_name in zip(self.retained_continuous_features_, self.mice_aux_col_names_):
                    c_fill = out[COUNTRY_COL].map(self.simple_imputer_.country_stats_[col])
                    y_fill = out[YEAR_COL].map(self.simple_imputer_.year_stats_[col])
                    glob = self.simple_imputer_.global_stats_[col]
                    mice_in[aux_name] = c_fill.fillna(y_fill).fillna(glob)

                if self.mice_regime_col_ is not None:
                    regime_col = "v2x_regime"
                    if retained_cats and self.mice_categorical_imputer_ is not None:
                        cats_imp = self.mice_categorical_imputer_.transform(out[retained_cats])
                        cats_df = pd.DataFrame(cats_imp, columns=retained_cats, index=out.index)
                        regime_vals = cats_df.get(regime_col, pd.Series(np.nan, index=out.index))
                    else:
                        regime_vals = out.get(regime_col, pd.Series(np.nan, index=out.index))
                    mice_in["_regime"] = pd.to_numeric(regime_vals, errors="coerce").fillna(0.0)

                scaled = (
                    (mice_in[self.mice_input_cols_] - self.mice_scaler_means_)
                    / self.mice_scaler_stds_
                )
                imputed_scaled = self.mice_numeric_imputer_.transform(scaled.values)

                # Extract only original continuous features (first n_orig columns)
                n_orig = len(self.retained_continuous_features_)
                orig_means = self.mice_scaler_means_.values[:n_orig]
                orig_stds = self.mice_scaler_stds_.values[:n_orig]
                out[self.retained_continuous_features_] = (
                    imputed_scaled[:, :n_orig] * orig_stds + orig_means
                )
                if self.numeric_global_fallback_ is not None:
                    out[self.retained_continuous_features_] = out[
                        self.retained_continuous_features_
                    ].fillna(self.numeric_global_fallback_)

            if retained_cats and self.mice_categorical_imputer_ is not None:
                out[retained_cats] = self.mice_categorical_imputer_.transform(out[retained_cats])

        else:
            raise ValueError(f"Unsupported branch: {branch}")

        # Enforce categorical ranges after imputation
        for col, allowed in CATEGORICAL_ALLOWED_VALUES.items():
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").round().clip(
                    lower=min(allowed), upper=max(allowed),
                )

        return out

    # -----------------------------------------------------------------
    # Phase 3: Engineer
    # -----------------------------------------------------------------
    def _engineer(self, current_df: pd.DataFrame, history_df: pd.DataFrame | None) -> pd.DataFrame:
        current = current_df.copy().sort_values(ID_COLS).reset_index(drop=True)
        current["_is_current"] = True

        if history_df is not None and not history_df.empty:
            history = history_df.copy().sort_values(ID_COLS).reset_index(drop=True)
            history["_is_current"] = False
            working = pd.concat([history, current], ignore_index=True, sort=False)
        else:
            working = current.copy()

        working = working.sort_values(ID_COLS).reset_index(drop=True)
        group = working.groupby(COUNTRY_COL, sort=False)
        new_cols: dict[str, pd.Series] = {}
        col_names: list[str] = []

        engineerable = [
            c for c in self.retained_continuous_features_
            if c not in self.synthetic_feature_names_
        ]

        for lag in LAG_PERIODS:
            for col in engineerable:
                name = f"{col}_lag{lag}"
                new_cols[name] = group[col].shift(lag)
                col_names.append(name)

        if ENABLE_DIFF_FEATURES:
            for col in engineerable:
                if col not in RATE_FEATURES:
                    name = f"{col}_diff1"
                    new_cols[name] = group[col].diff(1)
                    col_names.append(name)

        if ENABLE_PCT_CHANGE_FEATURES:
            for col in PCT_CHANGE_FEATURES:
                if col in engineerable:
                    name = f"{col}_pct_change1"
                    new_cols[name] = group[col].pct_change().replace([np.inf, -np.inf], np.nan)
                    col_names.append(name)

        for window in ROLLING_WINDOWS:
            for col in engineerable:
                name = f"{col}_rollmean{window}"
                new_cols[name] = group[col].transform(
                    lambda s: s.rolling(window=window, min_periods=ROLLING_MIN_PERIODS).mean()
                )
                col_names.append(name)

        for left, right in INTERACTION_PAIRS:
            if left in working.columns and right in working.columns:
                name = f"{left}_x_{right}"
                new_cols[name] = working[left] * working[right]
                col_names.append(name)

        if new_cols:
            working = pd.concat([working, pd.DataFrame(new_cols, index=working.index)], axis=1)

        result = working[working["_is_current"]].copy().drop(columns=["_is_current"], errors="ignore")
        if col_names:
            result[col_names] = result[col_names].replace([np.inf, -np.inf], np.nan)
            result[col_names] = result[col_names].fillna(STRUCTURAL_NA_FILL_VALUE)

        return result

    def _encode_and_scale(self, df: pd.DataFrame, branch: str, fit: bool) -> pd.DataFrame:
        feature_cols = get_feature_columns(df)
        raw_cats = [c for c in CATEGORICAL_FEATURES if c in df.columns]
        numeric_feats = [c for c in feature_cols if c not in raw_cats]

        numeric_df = df[numeric_feats].copy()
        cat_df = df[raw_cats].copy()
        if raw_cats:
            cat_df = cat_df.apply(pd.to_numeric, errors="coerce").round().astype("Int64").astype(str)

        if fit:
            if raw_cats:
                encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                encoded_array = encoder.fit_transform(cat_df)
                self.encoder_[branch] = encoder
                encoded_cols = encoder.get_feature_names_out(raw_cats).tolist()
            else:
                encoded_array = np.empty((len(df), 0))
                encoded_cols = []
            scaler = self._make_scaler()
            if numeric_feats:
                scaler.fit(numeric_df)
            self.scaler_[branch] = scaler
            self.final_feature_columns_[branch] = numeric_feats + encoded_cols
        else:
            encoder = self.encoder_.get(branch)
            if raw_cats and encoder is not None:
                encoded_array = encoder.transform(cat_df)
                encoded_cols = encoder.get_feature_names_out(raw_cats).tolist()
            else:
                encoded_array = np.empty((len(df), 0))
                encoded_cols = []
            scaler = self.scaler_[branch]

        if numeric_feats:
            scaled_df = pd.DataFrame(
                scaler.transform(numeric_df), columns=numeric_feats, index=df.index,
            )
        else:
            scaled_df = pd.DataFrame(index=df.index)

        encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)
        final_features = pd.concat([scaled_df, encoded_df], axis=1)

        if not fit:
            expected = self.final_feature_columns_[branch]
            final_features = final_features.reindex(columns=expected, fill_value=0.0)

        result = pd.concat(
            [df[ID_COLS + TARGET_COLS].reset_index(drop=True), final_features.reset_index(drop=True)],
            axis=1,
        )
        return result

    @staticmethod
    def _make_scaler() -> Any:
        if SCALER_CHOICE == "standard":
            return StandardScaler()
        if SCALER_CHOICE == "minmax":
            return MinMaxScaler()
        if SCALER_CHOICE == "robust":
            return RobustScaler()
        raise ValueError(f"Unsupported scaler: {SCALER_CHOICE}")


# =============================================================================
# Validation helpers
# =============================================================================


def validate_branch_outputs(
    train_outputs: dict[str, pd.DataFrame],
    val_outputs: dict[str, pd.DataFrame],
    test_outputs: dict[str, pd.DataFrame],
) -> None:
    for branch in train_outputs:
        train_df = train_outputs[branch]
        val_df = val_outputs[branch]
        test_df = test_outputs[branch]
        train_feats = [c for c in train_df.columns if c not in ID_COLS + TARGET_COLS]
        val_feats = [c for c in val_df.columns if c not in ID_COLS + TARGET_COLS]
        test_feats = [c for c in test_df.columns if c not in ID_COLS + TARGET_COLS]
        if train_feats != val_feats or train_feats != test_feats:
            raise AssertionError(f"Feature columns mismatch across splits for branch {branch}.")
        for name, frame, feats in [("train", train_df, train_feats),
                                    ("val", val_df, val_feats),
                                    ("test", test_df, test_feats)]:
            nan_count = int(frame[feats].isna().sum().sum())
            logger.info(f"{branch}/{name}: rows={len(frame)}, features={len(feats)}, NaNs={nan_count}")
            if nan_count:
                raise AssertionError(f"NaNs remain in final features for {branch}/{name}.")


def summarize_target_shapes(outputs: dict[str, pd.DataFrame], split_name: str) -> None:
    for branch, frame in outputs.items():
        for target in TARGET_COLS:
            usable = int(frame[target].notna().sum())
            logger.info(f"{split_name}/{branch}/{target}: usable rows={usable}")


def run_pipeline_for_strategy(df: pd.DataFrame, validation_strategy: str) -> dict[str, Any]:
    splits = split_raw_panel(df, validation_strategy=validation_strategy)
    preprocessor = Preprocessor()
    train_outputs = preprocessor.fit_transform(splits["train"])
    val_history = splits["train"]
    test_history = pd.concat([splits["train"], splits["val"]], ignore_index=True)
    val_outputs = preprocessor.transform(splits["val"], history_df=val_history, split_name="val")
    test_outputs = preprocessor.transform(splits["test"], history_df=test_history, split_name="test")
    validate_branch_outputs(train_outputs, val_outputs, test_outputs)
    summarize_target_shapes(train_outputs, "train")
    summarize_target_shapes(val_outputs, "val")
    summarize_target_shapes(test_outputs, "test")
    return {
        "pipeline": preprocessor,
        "splits": splits,
        "train_outputs": train_outputs,
        "val_outputs": val_outputs,
        "test_outputs": test_outputs,
    }


# =============================================================================
# Cache I/O helpers
# =============================================================================


def write_frame(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def read_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


# =============================================================================
# Cache fingerprint
# =============================================================================


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_fingerprint_config() -> dict[str, Any]:
    from stages.training.config import (
        ACTIVE_BRANCHES,
        ACTIVE_OUTER_STRATEGIES,
        N_TEMPORAL_CV_FOLDS,
        OUTER_TEST_YEARS,
        TEST_YEAR as _TEST_YEAR,
        TRAIN_START_YEAR,
    )
    return {
        "data_sha256": _file_sha256(DATA_PATH),
        "pipeline_sha256": _file_sha256(Path("stages/preprocessing.py")),
        "core_sha256": _file_sha256(Path("stages/training/core.py")),
        "branches": sorted(ACTIVE_BRANCHES),
        "strategies": sorted(ACTIVE_OUTER_STRATEGIES),
        "outer_test_years": sorted(OUTER_TEST_YEARS),
        "test_year": _TEST_YEAR,
        "train_start_year": TRAIN_START_YEAR,
        "n_temporal_cv_folds": N_TEMPORAL_CV_FOLDS,
        "random_state": RANDOM_STATE,
        "row_missingness_threshold": ROW_MISSINGNESS_THRESHOLD,
        "sparse_column_threshold": SPARSE_COLUMN_THRESHOLD,
        "outlier_method": OUTLIER_METHOD,
        "iqr_multiplier": IQR_MULTIPLIER,
        "country_outlier_min_obs": COUNTRY_OUTLIER_MIN_OBS,
        "year_outlier_min_obs": YEAR_OUTLIER_MIN_OBS,
        "lag_periods": LAG_PERIODS,
        "rolling_windows": ROLLING_WINDOWS,
        "rolling_min_periods": ROLLING_MIN_PERIODS,
        "structural_na_fill_value": STRUCTURAL_NA_FILL_VALUE,
        "enable_diff_features": ENABLE_DIFF_FEATURES,
        "enable_pct_change_features": ENABLE_PCT_CHANGE_FEATURES,
        "scaler_choice": SCALER_CHOICE,
        "mice_max_iter": MICE_MAX_ITER,
        "mice_tol": MICE_TOL,
        "mice_skip_complete": MICE_SKIP_COMPLETE,
        "mice_country_n_components": MICE_COUNTRY_N_COMPONENTS,
        "mice_missing_threshold": MICE_MISSING_THRESHOLD,
        "interaction_pairs": sorted(INTERACTION_PAIRS),
    }


def compute_fingerprint() -> str:
    config = _build_fingerprint_config()
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


# =============================================================================
# Cache validation
# =============================================================================


def enumerate_cache_units(
    branches: list[str],
    strategies: list[str],
    outer_test_years: list[int],
) -> dict[str, list]:
    return {
        "outer": [(b, y) for b in branches for y in outer_test_years],
        "fold_sets": [
            (s, b, y) for s in strategies for b in branches for y in outer_test_years
        ],
        "final": list(branches),
        "final_fold_sets": [(s, b) for s in strategies for b in branches],
    }


def _expected_files_for_unit(
    cache_dir: Path, unit_type: str, unit_key: tuple | str,
) -> list[Path]:
    if unit_type == "outer":
        branch, year = unit_key
        base = cache_dir / "outer" / branch / str(year)
        return [base / "train.csv", base / "test.csv", base / "preprocessor.joblib"]
    if unit_type == "fold_sets":
        strategy, branch, year = unit_key
        base = cache_dir / "folds" / strategy / branch / str(year)
        return [base]
    if unit_type == "final":
        branch = unit_key
        base = cache_dir / "final" / branch
        return [
            base / "train.csv",
            base / "test_scored.csv",
            base / "test_all.csv",
            base / "preprocessor.joblib",
        ]
    if unit_type == "final_fold_sets":
        strategy, branch = unit_key
        base = cache_dir / "final" / "folds" / strategy / branch
        return [base]
    return []


def validate_cache(cache_dir: Path) -> bool:
    from stages.training.config import ACTIVE_BRANCHES, ACTIVE_OUTER_STRATEGIES, OUTER_TEST_YEARS

    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    if not manifest.get("complete"):
        return False
    manifest_fp = manifest.get("fingerprint")
    if manifest_fp != cache_dir.name:
        logger.info(f"Cache dir/manifest mismatch: dir={cache_dir.name}, manifest={manifest_fp}")
        return False
    current_fp = compute_fingerprint()
    if manifest_fp != current_fp:
        logger.info(f"Cache fingerprint mismatch: cached={manifest_fp}, current={current_fp}")
        return False
    units = enumerate_cache_units(ACTIVE_BRANCHES, ACTIVE_OUTER_STRATEGIES, OUTER_TEST_YEARS)
    for unit_type, unit_list in units.items():
        for unit_key in unit_list:
            for path in _expected_files_for_unit(cache_dir, unit_type, unit_key):
                if not path.exists():
                    logger.info(f"Cache missing: {path}")
                    return False
                if path.is_dir() and not any(path.iterdir()):
                    logger.info(f"Cache empty dir: {path}")
                    return False
    return True


# =============================================================================
# Preprocessing workers
# =============================================================================


def _preprocess_outer_unit(
    raw_df: pd.DataFrame,
    branch: str,
    outer_test_year: int,
    base_dir: Path,
) -> None:
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message=".*Early stopping criterion not reached.*")

    from stages.training.config import TEST_YEAR as _TEST_YEAR, TRAIN_START_YEAR, YEAR_COL as _YEAR_COL
    from stages.training.core import make_nested_outer_split, prepare_branch_outer_frames

    t0 = time.monotonic()
    pid = os.getpid()
    tag = f"outer/{branch}/{outer_test_year}"

    raw_pre_test = raw_df[raw_df[_YEAR_COL] < _TEST_YEAR].copy()
    split = make_nested_outer_split(raw_pre_test, outer_test_year)
    result = prepare_branch_outer_frames(split, branch, train_start_year=TRAIN_START_YEAR)

    out_dir = base_dir / "outer" / branch / str(outer_test_year)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_frame(result["train_frame"], out_dir / "train.csv")
    write_frame(result["test_frame"], out_dir / "test.csv")
    joblib.dump(result["preprocessor"], out_dir / "preprocessor.joblib")

    elapsed = time.monotonic() - t0
    print(f"[PID:{pid}] {tag} done in {elapsed:.1f}s", flush=True)


def _preprocess_fold_set(
    raw_df: pd.DataFrame,
    strategy: str,
    branch: str,
    outer_test_year: int,
    base_dir: Path,
) -> None:
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message=".*Early stopping criterion not reached.*")

    from stages.training.config import TEST_YEAR as _TEST_YEAR, TRAIN_START_YEAR, YEAR_COL as _YEAR_COL
    from stages.training.core import (
        make_cv_folds, make_nested_outer_split, prepare_branch_fold_frames,
    )

    t0 = time.monotonic()
    pid = os.getpid()
    tag = f"folds/{strategy}/{branch}/{outer_test_year}"

    raw_pre_test = raw_df[raw_df[_YEAR_COL] < _TEST_YEAR].copy()
    split = make_nested_outer_split(raw_pre_test, outer_test_year)
    raw_folds = make_cv_folds(split["outer_train"], strategy)
    fold_frames = prepare_branch_fold_frames(raw_folds, branch, train_start_year=TRAIN_START_YEAR)

    out_dir = base_dir / "folds" / strategy / branch / str(outer_test_year)
    for fold in fold_frames:
        fold_dir = out_dir / f"fold_{fold['fold_id']}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        write_frame(fold["train_frame"], fold_dir / "train.csv")
        write_frame(fold["val_frame"], fold_dir / "val.csv")

    elapsed = time.monotonic() - t0
    print(f"[PID:{pid}] {tag} done in {elapsed:.1f}s ({len(fold_frames)} folds)", flush=True)


def _preprocess_final_unit(
    raw_df: pd.DataFrame,
    branch: str,
    base_dir: Path,
) -> None:
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message=".*Early stopping criterion not reached.*")

    from stages.training.config import TEST_YEAR as _TEST_YEAR, TRAIN_START_YEAR, YEAR_COL as _YEAR_COL

    t0 = time.monotonic()
    pid = os.getpid()
    tag = f"final/{branch}"

    final_train_raw = raw_df[raw_df[_YEAR_COL] < _TEST_YEAR].copy()
    final_test_raw = raw_df[raw_df[_YEAR_COL] == _TEST_YEAR].copy()

    preprocessor = Preprocessor(active_branches=[branch], train_start_year=TRAIN_START_YEAR)
    train_outputs = preprocessor.fit_transform(final_train_raw)
    test_outputs_scored = preprocessor.transform(
        final_test_raw, history_df=final_train_raw, drop_rows=True,
        split_name=f"final_test_{branch}",
    )
    test_outputs_all = preprocessor.transform(
        final_test_raw, history_df=final_train_raw, drop_rows=False,
        split_name=f"final_test_all_{branch}",
    )

    out_dir = base_dir / "final" / branch
    out_dir.mkdir(parents=True, exist_ok=True)
    write_frame(train_outputs[branch], out_dir / "train.csv")
    write_frame(test_outputs_scored[branch], out_dir / "test_scored.csv")
    write_frame(test_outputs_all[branch], out_dir / "test_all.csv")
    joblib.dump(preprocessor, out_dir / "preprocessor.joblib")

    elapsed = time.monotonic() - t0
    print(f"[PID:{pid}] {tag} done in {elapsed:.1f}s", flush=True)


def _preprocess_final_fold_set(
    raw_df: pd.DataFrame,
    strategy: str,
    branch: str,
    base_dir: Path,
) -> None:
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message=".*Early stopping criterion not reached.*")

    from stages.training.config import TEST_YEAR as _TEST_YEAR, TRAIN_START_YEAR, YEAR_COL as _YEAR_COL
    from stages.training.core import make_cv_folds, prepare_branch_fold_frames

    t0 = time.monotonic()
    pid = os.getpid()
    tag = f"final/folds/{strategy}/{branch}"

    final_train_raw = raw_df[raw_df[_YEAR_COL] < _TEST_YEAR].copy()
    raw_folds = make_cv_folds(final_train_raw, strategy)
    fold_frames = prepare_branch_fold_frames(raw_folds, branch, train_start_year=TRAIN_START_YEAR)

    out_dir = base_dir / "final" / "folds" / strategy / branch
    for fold in fold_frames:
        fold_dir = out_dir / f"fold_{fold['fold_id']}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        write_frame(fold["train_frame"], fold_dir / "train.csv")
        write_frame(fold["val_frame"], fold_dir / "val.csv")

    elapsed = time.monotonic() - t0
    print(f"[PID:{pid}] {tag} done in {elapsed:.1f}s ({len(fold_frames)} folds)", flush=True)


# =============================================================================
# Cache orchestration
# =============================================================================


def _build_cache(base_dir: Path, fingerprint: str) -> None:
    """Build all preprocessed artifacts into base_dir."""
    from stages.training.config import (
        ACTIVE_BRANCHES,
        ACTIVE_OUTER_STRATEGIES,
        DEFAULT_CPU_WORKERS,
        OUTER_TEST_YEARS,
    )

    stage_t0 = time.monotonic()
    raw_df = load_panel_data(verbose=True)
    units = enumerate_cache_units(ACTIVE_BRANCHES, ACTIVE_OUTER_STRATEGIES, OUTER_TEST_YEARS)

    tasks: list[tuple[Any, ...]] = []
    for branch, year in units["outer"]:
        tasks.append((_preprocess_outer_unit, raw_df, branch, year, base_dir))
    for strategy, branch, year in units["fold_sets"]:
        tasks.append((_preprocess_fold_set, raw_df, strategy, branch, year, base_dir))
    for branch in units["final"]:
        tasks.append((_preprocess_final_unit, raw_df, branch, base_dir))
    for strategy, branch in units["final_fold_sets"]:
        tasks.append((_preprocess_final_fold_set, raw_df, strategy, branch, base_dir))

    logger.info(f"Dispatching {len(tasks)} preprocessing units across {DEFAULT_CPU_WORKERS} workers")
    max_workers = min(len(tasks), DEFAULT_CPU_WORKERS)

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(fn, *args): idx for idx, (fn, *args) in enumerate(tasks)
            }
            for future in as_completed(futures):
                future.result()
    except PermissionError:
        logger.warning("Process pool unavailable. Running sequentially.")
        for fn, *args in tasks:
            fn(*args)

    stage_elapsed = time.monotonic() - stage_t0

    manifest = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "complete": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": _build_fingerprint_config(),
        "expected_units": {
            k: [list(u) if isinstance(u, tuple) else u for u in v]
            for k, v in units.items()
        },
        "total_elapsed_seconds": stage_elapsed,
    }
    with open(base_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    logger.info(f"Preprocessing stage complete in {stage_elapsed:.1f}s")


def run_preprocessing_stage(force: bool = False) -> Path:
    """Run preprocessing stage. Returns path to valid cache directory."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    fingerprint = compute_fingerprint()
    cache_dir = CACHE_ROOT / fingerprint
    lock_path = CACHE_ROOT / f"{fingerprint}.lock"

    if not force and validate_cache(cache_dir):
        logger.info(f"Valid preprocessing cache found: {cache_dir}")
        return cache_dir

    logger.info(f"Building preprocessing cache: {cache_dir}")
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    lock_fd = open(lock_path, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        if not force and validate_cache(cache_dir):
            logger.info(f"Cache built by another process: {cache_dir}")
            return cache_dir
        tmp_dir = CACHE_ROOT / f"{fingerprint}.tmp"
        if tmp_dir.exists():
            logger.info(f"Removing stale temp dir: {tmp_dir}")
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)
        _build_cache(tmp_dir, fingerprint)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        tmp_dir.rename(cache_dir)
        logger.info(f"Preprocessing cache published: {cache_dir}")
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass

    return cache_dir


# =============================================================================
# CLI entry point
# =============================================================================


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    cache_path = run_preprocessing_stage(force=force)
    print(f"Cache directory: {cache_path}")
