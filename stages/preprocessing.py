"""
preprocessing.py - reusable preprocessing classes and constants.

The Preprocessor class implements a three-phase pipeline:
    clean -> impute -> engineer -> encode/scale

Two imputation branches:
    simple : Country -> year -> global median fallback
    mice   : Iterative imputer with panel-aware auxiliary predictors
"""

from __future__ import annotations

import logging
import os
from itertools import combinations
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


# --- Configurable constants ---

DATA_PATH = Path("data/merged/merged.csv")

COUNTRY_COL = "country"
YEAR_COL = "year"
ID_COLS = [COUNTRY_COL, YEAR_COL]
TARGET_COLS = ["linear", "optimistic", "pessimistic"]
CATEGORICAL_FEATURES = ["v2x_regime"]
NON_FEATURE_COLS = [
    "non_co2_emissions_mt_co2e",
    "total_ghg_emissions_mt_co2e",
    "total_hard_to_abate_emissions_mt_co2e",
    "population_total",
]

TEST_YEAR = 2022
RANDOM_STATE = 42

ROW_MISSINGNESS_THRESHOLD = 0.30
OUTLIER_METHOD = "iqr"
IQR_MULTIPLIER = 1.5
COUNTRY_OUTLIER_MIN_OBS = 5
YEAR_OUTLIER_MIN_OBS = 20
OUTLIER_WIPEOUT_WARNING_THRESHOLD = 0.80

LAG_PERIODS = []#[1, 2, 3]
# Max number of consecutive years a last-valid value may be carried forward (per country)
# before the cross-sectional imputer takes over. Bounds reporting-lag fill (e.g. 2021→2022).
FORWARD_FILL_LIMIT = 3
ROLLING_WINDOWS = [7]#[3, 5, 7]
ROLLING_STD_WINDOWS = [3,5]
ROLLING_MIN_PERIODS = 1
STRUCTURAL_NA_FILL_VALUE = 0.0
ALL_PAIRWISE_INTERACTIONS = True  # generates all C(n,2) pairs from engineerable features
INTERACTION_PAIRS: list[tuple[str, str]] = []  # used only when ALL_PAIRWISE_INTERACTIONS=False
# When False, country_pc1–5, year_std, year_std_sq are NOT added as model input columns.
# PCA is still computed internally for MICE auxiliary features.
INCLUDE_SYNTHETIC_FEATURES = True
ENABLE_DIFF_FEATURES = False #True
ENABLE_PCT_CHANGE_FEATURES = False #True

SCALER_CHOICE = "standard"  # Options: "standard", "minmax", "robust"
MICE_MAX_ITER = 20
MICE_TOL = 1e-3
MICE_SKIP_COMPLETE = True
MICE_COUNTRY_N_COMPONENTS = 5
MICE_MISSING_THRESHOLD = 0.05  # features above this rate get was_missing flags

RATE_FEATURES = {
    "ghg_growth",
    "annual_deforestation_of_change",
}

# Pct-change is applied to all engineerable features except RATE_FEATURES (already rates).

# Lower bound for features that cannot logically be zero (denominators / fundamental
# magnitudes). The strict `<` check in _numeric_range_invalid_mask then flags exact 0 or
# fabricated near-zero values (e.g. an all-NaN IEA group that summed to 0.0) and converts
# them to NaN at clean time. A single tiny epsilon is safe: the realistic minimum of every
# feature it guards is orders of magnitude larger.
POSITIVE_LOWER_EPSILON = 1e-6

# Columns to drop before any preprocessing step (cleaning, imputation, engineering).
# Override via EXCLUDE_COLUMNS in root config.py.
EXCLUDE_COLUMNS: list[str] = []

# Canonical list of constants whose runtime values affect the preprocessor output.
# data_preparation.py reads this list to build the cache fingerprint automatically.
# Add the name of any new preprocessing toggle here — no other fingerprint code changes needed.
PREPROCESSING_CACHE_KEYS: list[str] = [
    "ROW_MISSINGNESS_THRESHOLD",
    "OUTLIER_METHOD",
    "IQR_MULTIPLIER",
    "COUNTRY_OUTLIER_MIN_OBS",
    "YEAR_OUTLIER_MIN_OBS",
    "LAG_PERIODS",
    "FORWARD_FILL_LIMIT",
    "ROLLING_WINDOWS",
    "ROLLING_STD_WINDOWS",
    "ROLLING_MIN_PERIODS",
    "STRUCTURAL_NA_FILL_VALUE",
    "ENABLE_DIFF_FEATURES",
    "ENABLE_PCT_CHANGE_FEATURES",
    "ALL_PAIRWISE_INTERACTIONS",
    "INTERACTION_PAIRS",
    "INCLUDE_SYNTHETIC_FEATURES",
    "SCALER_CHOICE",
    "MICE_MAX_ITER",
    "MICE_TOL",
    "MICE_SKIP_COMPLETE",
    "MICE_COUNTRY_N_COMPONENTS",
    "MICE_MISSING_THRESHOLD",
    "RANDOM_STATE",
    "EXCLUDE_COLUMNS",
]

NUMERIC_RANGE_RULES: dict[str, dict[str, Any]] = {
    "renewable_energy_share_tfc_pct": {"lower": 0.0, "upper": 100.0},
    "renewable_electricity_share_pct": {"lower": 0.0, "upper": 100.0},
    "energy_intensity_mj_per_usd_ppp": {"lower": POSITIVE_LOWER_EPSILON, "upper": 100.0},
    "inform_risk_index": {"lower": 0.0, "upper": 10.0},
    "ghg_growth": {"finite_only": True},
    "non_co2_emissions_per_capita": {"lower": 0.0, "upper": 500.0},
    "total_ghg_emissions_per_capita": {"lower": -500.0, "upper": 500.0},
    "total_hard_to_abate_emissions_per_capita": {"lower": 0.0, "upper": 500.0},
    "rsf_press_freedom_index": {"lower": 0.0, "upper": 100.0},
    "agricultural_land_of_land_area": {"lower": 0.0, "upper": 100.0},
    "annual_deforestation_of_change": {"lower": -100.0, "upper": 100.0},
    "average_supply_of_protein_of_animal_origin_g_cap_day_3_year_average": {
        "lower": 0.0, "upper": 250.0,
    },
    "coal_rents_of_gdp": {"lower": 0.0, "upper": 100.0},
    "control_of_corruption_estimate": {"lower": -4.0, "upper": 4.0},
    "forest_area_of_land_area": {"lower": 0.0, "upper": 100.0},
    "gdp_per_capita_constant_2015_us": {"lower": POSITIVE_LOWER_EPSILON, "upper": 500000.0},
    "gdp_ppp_constant_2021_international": {"lower": POSITIVE_LOWER_EPSILON, "upper": 2.0e14},
    "natural_gas_rents_of_gdp": {"lower": 0.0, "upper": 100.0},
    "oil_rents_of_gdp": {"lower": 0.0, "upper": 100.0},
    "political_stability_and_absence_of_violence_terrorism_estimate": {
        "lower": -4.0, "upper": 4.0,
    },
    "population_density_people_per_sq_km_of_land_area": {
        "lower": POSITIVE_LOWER_EPSILON, "upper": 100000.0,
    },
    "population_total": {"lower": POSITIVE_LOWER_EPSILON, "upper": 1.0e10},
    "regulatory_quality_estimate": {"lower": -4.0, "upper": 4.0},
    "voice_and_accountability_estimate": {"lower": -4.0, "upper": 4.0},
}

CATEGORICAL_ALLOWED_VALUES: dict[str, set[int]] = {
    "v2x_regime": {0, 1, 2, 3},
}

OUTLIER_POLICY_RANGE_ONLY = "range_only"
OUTLIER_POLICY_COUNTRY_IQR = "country_iqr"
OUTLIER_POLICY_COUNTRY_AND_YEAR_IQR = "country_and_year_iqr"

STRUCTURAL_IDENTITY_FEATURES = {
    "renewable_energy_share_tfc_pct",
    "renewable_electricity_share_pct",
    "energy_intensity_mj_per_usd_ppp",
    "non_co2_emissions_per_capita",
    "total_ghg_emissions_per_capita",
    "total_hard_to_abate_emissions_per_capita",
    "agricultural_land_of_land_area",
    "average_supply_of_protein_of_animal_origin_g_cap_day_3_year_average",
    "coal_rents_of_gdp",
    "forest_area_of_land_area",
    "gdp_per_capita_constant_2015_us",
    "gdp_ppp_constant_2021_international",
    "natural_gas_rents_of_gdp",
    "oil_rents_of_gdp",
    "population_density_people_per_sq_km_of_land_area",
}

YEAR_CONFIRMED_OUTLIER_FEATURES = {
    "annual_deforestation_of_change",
    "control_of_corruption_estimate",
    "ghg_growth",
    "inform_risk_index",
    "political_stability_and_absence_of_violence_terrorism_estimate",
    "regulatory_quality_estimate",
    "rsf_press_freedom_index",
    "voice_and_accountability_estimate",
}

OUTLIER_POLICY_BY_FEATURE: dict[str, str] = {
    **{col: OUTLIER_POLICY_RANGE_ONLY for col in STRUCTURAL_IDENTITY_FEATURES},
    **{col: OUTLIER_POLICY_COUNTRY_AND_YEAR_IQR for col in YEAR_CONFIRMED_OUTLIER_FEATURES},
}

BRANCH_NAMES = ("simple", "mice")

# --- Utility functions ---


def safe_mode(series: pd.Series) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    mode = non_null.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else non_null.iloc[0]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in ID_COLS + TARGET_COLS + NON_FEATURE_COLS]


def get_continuous_features(df: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    return [
        col for col in feature_columns
        if col not in CATEGORICAL_FEATURES and pd.api.types.is_numeric_dtype(df[col])
    ]


# --- Grouped imputation ---


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
            med = df[col].median()
            self.global_stats_[col] = 0.0 if pd.isna(med) else med
        for col in self.categorical_features:
            self.country_stats_[col] = df.groupby(COUNTRY_COL)[col].agg(safe_mode)
            self.year_stats_[col] = df.groupby(YEAR_COL)[col].agg(safe_mode)
            mode = safe_mode(df[col])
            self.global_stats_[col] = 0.0 if pd.isna(mode) or mode is None else mode
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


# --- Preprocessor ---


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

        # Column metadata
        self.feature_columns_: list[str] = []
        self.continuous_features_: list[str] = []
        self.retained_raw_features_: list[str] = []
        self.retained_continuous_features_: list[str] = []

        # Country PCA embeddings
        self.country_embedding_map_: dict[str, np.ndarray] = {}
        self.country_pca_global_fallback_: np.ndarray = np.zeros(MICE_COUNTRY_N_COMPONENTS)
        self.country_n_pca_components_: int = 0
        self.pca_component_names_: list[str] = []

        # Year standardization
        self.year_mean_: float = 0.0
        self.year_std_: float = 1.0

        # Synthetic feature names (non-engineerable columns)
        self.synthetic_feature_names_: list[str] = []
        self._last_clean_missingness_: pd.DataFrame = pd.DataFrame()
        self.last_transform_missingness_: pd.DataFrame = pd.DataFrame()
        self.outlier_diagnostics_: pd.DataFrame = pd.DataFrame()
        self.outlier_wipeout_diagnostics_: pd.DataFrame = pd.DataFrame()
        self.last_outlier_diagnostics_: pd.DataFrame = pd.DataFrame()
        self.last_outlier_wipeout_diagnostics_: pd.DataFrame = pd.DataFrame()

        # Imputers
        self.simple_imputer_: CountryYearFallbackImputer | None = None
        self.mice_numeric_imputer_: IterativeImputer | None = None
        self.mice_categorical_imputer_: SimpleImputer | None = None
        self.numeric_global_fallback_: pd.Series | None = None

        # MICE enrichment state
        self.mice_scaler_means_: pd.Series | None = None
        self.mice_scaler_stds_: pd.Series | None = None
        self.mice_missing_flags_: list[str] = []
        self.mice_aux_col_names_: list[str] = []
        self.mice_input_cols_: list[str] = []
        self.mice_regime_col_: str | None = None

        # Outlier bounds fitted on training data only (country-level and year-level)
        self.outlier_lower_country_: dict[str, pd.Series] = {}
        self.outlier_upper_country_: dict[str, pd.Series] = {}
        self.outlier_lower_year_: dict[str, pd.Series] = {}
        self.outlier_upper_year_: dict[str, pd.Series] = {}

        # Encoders / scalers per branch
        self.encoder_: dict[str, OneHotEncoder] = {}
        self.scaler_: dict[str, Any] = {}
        self.final_feature_columns_: dict[str, list[str]] = {}

    # -----------------------------------------------------------------
    # Synthetic feature helpers (country PCA + year features)
    # -----------------------------------------------------------------

    def _reset_country_embeddings(self) -> None:
        self.country_embedding_map_ = {}
        self.country_n_pca_components_ = 0
        self.pca_component_names_ = []

    def _build_country_embeddings(self, cleaned_train_df: pd.DataFrame) -> None:
        """Build country PCA embeddings from cleaned training data."""
        year_vals = cleaned_train_df[YEAR_COL].values.astype(float)
        self.year_mean_ = float(np.nanmean(year_vals))
        yr_std = float(np.nanstd(year_vals))
        self.year_std_ = yr_std if yr_std > 0 else 1.0

        feats = [c for c in self.retained_continuous_features_ if c in cleaned_train_df.columns]
        if not feats or COUNTRY_COL not in cleaned_train_df.columns:
            self._reset_country_embeddings()
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
            self._reset_country_embeddings()
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
        """Add permanent synthetic columns: country PCA components + year features.

        When INCLUDE_SYNTHETIC_FEATURES is False, no columns are attached and
        synthetic_feature_names_ is set to an empty list.
        """
        if not INCLUDE_SYNTHETIC_FEATURES:
            self.synthetic_feature_names_ = []
            return df.copy()
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
        if EXCLUDE_COLUMNS:
            cols_to_drop = [c for c in EXCLUDE_COLUMNS if c in train_df.columns]
            if cols_to_drop:
                logger.info("Excluding %d columns before preprocessing: %s", len(cols_to_drop), cols_to_drop)
                train_df = train_df.drop(columns=cols_to_drop)
        pid = os.getpid()
        self.feature_columns_ = get_feature_columns(train_df)
        self.continuous_features_ = get_continuous_features(train_df, self.feature_columns_)

        # Phase 1: Clean
        cleaned = self._clean(train_df, history_df=None, fit=True)
        cleaned = self._fill_temporal_gaps(cleaned)
        n_good = int((~cleaned["_to_drop"]).sum())
        n_ghost = int(cleaned["_to_drop"].sum())
        self.retained_raw_features_ = list(self.feature_columns_)
        self.retained_continuous_features_ = list(self.continuous_features_)
        print(
            f"[PID:{pid}] Cleaning: {len(train_df)}→{n_good} rows "
            f"({n_ghost} ghost rows preserved for temporal continuity), "
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

        # Forward-fill reporting-lag gaps (per country, <= FORWARD_FILL_LIMIT yrs) before the
        # cross-sectional imputer runs, then re-evaluate the missingness floor on the filled
        # values. Imputers are already fit on the pre-ffill observed rows above.
        cleaned = self._forward_fill_features(cleaned)
        cleaned, _ = self._rescue_dropped_rows(cleaned, drop_rows=True)

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
        drop_rows: bool | str = True,
    ) -> dict[str, pd.DataFrame]:
        if EXCLUDE_COLUMNS:
            cols_to_drop = [c for c in EXCLUDE_COLUMNS if c in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
        cleaned = self._clean(df, history_df=history_df, fit=False, drop_rows=drop_rows)
        # Capture this split's diagnostic now: the history _clean() call below clobbers
        # self._last_clean_missingness_ with the training rows.
        split_missingness = self._last_clean_missingness_.copy()
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

        # Carry the most recent valid value (incl. the last training year) forward into the
        # current frame, then re-evaluate the missingness floor so rows rescued by the fill
        # survive, and refresh the diagnostic to match the values actually used.
        cleaned, history_cleaned = self._forward_fill_with_history(cleaned, history_cleaned)
        cleaned, post_missing = self._rescue_dropped_rows(cleaned, drop_rows)
        if post_missing is not None:
            split_missingness = self._refresh_missingness_diagnostic(
                split_missingness, cleaned, post_missing
            )
        self.last_transform_missingness_ = split_missingness

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
        drop_rows: bool | str = True,
    ) -> pd.DataFrame:
        current = df.copy().sort_values(ID_COLS).reset_index(drop=True)
        current["_is_current"] = True  # kept for _apply_*_range_rules compatibility

        outlier_features = [c for c in self.continuous_features_ if c not in self.synthetic_feature_names_]
        if fit:
            self._fit_outlier_bounds(current, outlier_features)
        self._apply_feature_outlier_policy(current, outlier_features)
        self._apply_categorical_range_rules(current)

        cleaned = current.drop(columns=["_is_current"], errors="ignore")
        feature_cols = get_feature_columns(cleaned)
        real_feature_cols = [c for c in feature_cols if c not in self.synthetic_feature_names_]
        if real_feature_cols:
            row_missing_fraction = cleaned[real_feature_cols].isna().mean(axis=1)
        else:
            row_missing_fraction = pd.Series(0.0, index=cleaned.index)

        if drop_rows is True:
            row_missing = row_missing_fraction > ROW_MISSINGNESS_THRESHOLD
            all_targets_nan = cleaned[TARGET_COLS].isna().all(axis=1)
            drop_mask = all_targets_nan | row_missing
        elif drop_rows == "targets_only":
            # Evaluation mode: keep rows with high feature missingness (imputation handles them);
            # only drop rows where every target is NaN (no metric can be computed).
            all_targets_nan = cleaned[TARGET_COLS].isna().all(axis=1)
            drop_mask = all_targets_nan
        else:
            drop_mask = pd.Series(False, index=cleaned.index)

        self._last_clean_missingness_ = cleaned[[COUNTRY_COL, YEAR_COL]].copy()
        self._last_clean_missingness_["row_missingness_fraction"] = row_missing_fraction.to_numpy()
        self._last_clean_missingness_["included"] = (~drop_mask).to_numpy()

        cleaned["_to_drop"] = drop_mask
        return cleaned.sort_values(ID_COLS).reset_index(drop=True)

    def _fit_outlier_bounds(
        self, df: pd.DataFrame, outlier_features: list[str]
    ) -> None:
        for group_col, min_obs, lower_store, upper_store in [
            (COUNTRY_COL, COUNTRY_OUTLIER_MIN_OBS,
             self.outlier_lower_country_, self.outlier_upper_country_),
            (YEAR_COL, YEAR_OUTLIER_MIN_OBS,
             self.outlier_lower_year_, self.outlier_upper_year_),
        ]:
            for col in outlier_features:
                if col not in df.columns:
                    continue
                grouped = df.groupby(group_col)[col]
                count = grouped.count()
                q1 = grouped.quantile(0.25)
                q3 = grouped.quantile(0.75)
                iqr = q3 - q1
                valid = (count >= min_obs) & (iqr > 0)
                lower = (q1 - IQR_MULTIPLIER * iqr).where(valid)
                upper = (q3 + IQR_MULTIPLIER * iqr).where(valid)
                lower_store[col] = lower.dropna()
                upper_store[col] = upper.dropna()

    def _apply_feature_outlier_policy(
        self, df: pd.DataFrame, outlier_features: list[str]
    ) -> None:
        diag_rows: list[dict[str, Any]] = []
        wipeout_rows: list[dict[str, Any]] = []

        for col in outlier_features:
            if col not in df.columns:
                continue

            original_notna = df[col].notna()
            country_flag = self._outlier_mask(
                df,
                col,
                group_col=COUNTRY_COL,
                lower_store=self.outlier_lower_country_,
                upper_store=self.outlier_upper_country_,
            )
            year_audit_flag = self._outlier_mask(
                df,
                col,
                group_col=YEAR_COL,
                lower_store=self.outlier_lower_year_,
                upper_store=self.outlier_upper_year_,
            )
            range_flag = self._numeric_range_invalid_mask(df, col)

            policy = self._outlier_policy_for_feature(col)
            if policy == OUTLIER_POLICY_RANGE_ONLY:
                iqr_delete = pd.Series(False, index=df.index)
            elif policy == OUTLIER_POLICY_COUNTRY_AND_YEAR_IQR:
                iqr_delete = country_flag & year_audit_flag
            elif policy == OUTLIER_POLICY_COUNTRY_IQR:
                iqr_delete = country_flag
            else:
                raise ValueError(f"Unsupported outlier policy for {col}: {policy}")

            final_delete = (iqr_delete | range_flag) & original_notna
            df.loc[final_delete, col] = np.nan

            diag_rows.append({
                "feature": col,
                "policy": policy,
                "valid_values": int(original_notna.sum()),
                "country_iqr_flags": int((country_flag & original_notna).sum()),
                "year_iqr_audit_flags": int((year_audit_flag & original_notna).sum()),
                "range_invalid_flags": int((range_flag & original_notna).sum()),
                "final_deleted_values": int(final_delete.sum()),
            })

            if original_notna.any():
                country_summary = pd.DataFrame({
                    COUNTRY_COL: df[COUNTRY_COL],
                    "valid_values": original_notna.astype(int),
                    "deleted_values": final_delete.astype(int),
                }).groupby(COUNTRY_COL).sum()
                country_summary = country_summary[country_summary["valid_values"] >= COUNTRY_OUTLIER_MIN_OBS]
                if not country_summary.empty:
                    country_summary["deleted_fraction"] = (
                        country_summary["deleted_values"] / country_summary["valid_values"]
                    )
                    flagged = country_summary[
                        country_summary["deleted_fraction"] >= OUTLIER_WIPEOUT_WARNING_THRESHOLD
                    ]
                    for country, row in flagged.iterrows():
                        wipeout_rows.append({
                            COUNTRY_COL: country,
                            "feature": col,
                            "policy": policy,
                            "valid_values": int(row["valid_values"]),
                            "deleted_values": int(row["deleted_values"]),
                            "deleted_fraction": float(row["deleted_fraction"]),
                        })

        self.last_outlier_diagnostics_ = pd.DataFrame(diag_rows)
        self.last_outlier_wipeout_diagnostics_ = pd.DataFrame(wipeout_rows)
        if not self.last_outlier_diagnostics_.empty:
            self.outlier_diagnostics_ = pd.concat(
                [self.outlier_diagnostics_, self.last_outlier_diagnostics_],
                ignore_index=True,
            )
        if not self.last_outlier_wipeout_diagnostics_.empty:
            self.outlier_wipeout_diagnostics_ = pd.concat(
                [self.outlier_wipeout_diagnostics_, self.last_outlier_wipeout_diagnostics_],
                ignore_index=True,
            )
            logger.warning(
                "Outlier cleaning deleted >=%.0f%% of valid history for %d country-feature pairs.",
                OUTLIER_WIPEOUT_WARNING_THRESHOLD * 100,
                len(self.last_outlier_wipeout_diagnostics_),
            )

    def _outlier_mask(
        self,
        df: pd.DataFrame,
        col: str,
        group_col: str,
        lower_store: dict[str, pd.Series],
        upper_store: dict[str, pd.Series],
    ) -> pd.Series:
        lower_bounds = lower_store.get(col)
        upper_bounds = upper_store.get(col)
        if lower_bounds is None or upper_bounds is None:
            return pd.Series(False, index=df.index)
        lower = df[group_col].map(lower_bounds)
        upper = df[group_col].map(upper_bounds)
        valid = df[col].notna() & lower.notna() & upper.notna()
        return valid & ((df[col] < lower) | (df[col] > upper))

    def _numeric_range_invalid_mask(self, working: pd.DataFrame, col: str) -> pd.Series:
        rule = NUMERIC_RANGE_RULES.get(col)
        if rule is None or col not in working.columns:
            return pd.Series(False, index=working.index)
        current_mask = working["_is_current"]
        series = working[col]
        invalid = current_mask & series.notna()
        if rule.get("finite_only"):
            invalid &= ~np.isfinite(series)
        else:
            invalid &= (series < rule.get("lower", -np.inf)) | (series > rule.get("upper", np.inf))
        return invalid

    @staticmethod
    def _outlier_policy_for_feature(col: str) -> str:
        return OUTLIER_POLICY_BY_FEATURE.get(col, OUTLIER_POLICY_COUNTRY_IQR)

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

    def _fill_temporal_gaps(
        self, df: pd.DataFrame, up_to_year: int | None = None,
    ) -> pd.DataFrame:
        """Inject synthetic ghost rows for years absent from a country's time series."""
        ghost_pieces: list[pd.DataFrame] = []
        for country, grp in df.groupby(COUNTRY_COL, sort=False):
            years_present = set(int(y) for y in grp[YEAR_COL])
            if not years_present:
                continue
            min_year = min(years_present)
            max_year = max(max(years_present), up_to_year) if up_to_year is not None else max(years_present)
            missing_years = sorted(set(range(min_year, max_year + 1)) - years_present)
            if missing_years:
                ghost_pieces.append(pd.DataFrame({
                    COUNTRY_COL: country,
                    YEAR_COL: missing_years,
                    "_to_drop": True,
                }))

        if not ghost_pieces:
            return df.copy()

        ghosts = pd.concat(ghost_pieces, ignore_index=True)
        result = pd.concat([df, ghosts], ignore_index=True)
        return result.sort_values(ID_COLS).reset_index(drop=True)

    # -----------------------------------------------------------------
    # Phase 1b: Forward-fill (reporting-lag handling, before imputation)
    # -----------------------------------------------------------------
    def _forward_fill_columns(self) -> list[str]:
        """Raw continuous + categorical features eligible for forward-fill.

        Excludes synthetic (deterministic) features, targets, non-features and IDs."""
        cont = [c for c in self.retained_continuous_features_ if c not in self.synthetic_feature_names_]
        cats = [c for c in CATEGORICAL_FEATURES if c in self.retained_raw_features_]
        return cont + cats

    def _forward_fill_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Carry each country's last valid value forward (<= FORWARD_FILL_LIMIT years).

        Grouping by COUNTRY_COL guarantees values never cross country boundaries."""
        cols = [c for c in self._forward_fill_columns() if c in df.columns]
        if not cols:
            return df
        out = df.sort_values(ID_COLS).reset_index(drop=True)
        out[cols] = out.groupby(COUNTRY_COL)[cols].ffill(limit=FORWARD_FILL_LIMIT)
        return out

    def _forward_fill_with_history(
        self, current: pd.DataFrame, history: pd.DataFrame | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Forward-fill across the history->current boundary (e.g. 2021 -> 2022 test).

        Concatenates history + current per country so the most recent training-year value
        can be carried into the current frame, then splits the frames back apart."""
        cols = [c for c in self._forward_fill_columns() if c in current.columns]
        if not cols:
            return current, history
        if history is None or history.empty:
            return self._forward_fill_features(current), history
        marker = "_ff_is_current"
        cur = current.copy()
        cur[marker] = True
        hist = history.copy()
        hist[marker] = False
        combined = (
            pd.concat([hist, cur], ignore_index=True)
            .sort_values(ID_COLS)
            .reset_index(drop=True)
        )
        combined[cols] = combined.groupby(COUNTRY_COL)[cols].ffill(limit=FORWARD_FILL_LIMIT)
        new_current = combined[combined[marker]].drop(columns=[marker]).reset_index(drop=True)
        new_history = combined[~combined[marker]].drop(columns=[marker]).reset_index(drop=True)
        return new_current, new_history

    def _rescue_dropped_rows(
        self, cleaned: pd.DataFrame, drop_rows: bool | str,
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        """Re-evaluate the row-missingness floor after forward-fill.

        Forward-fill only removes NaNs, so post-ffill missingness is always <= pre-ffill:
        no row that passed the floor can begin failing. We therefore only flip previously
        dropped rows back in (rows ffill rescued). Ghost / target-less rows keep NaN targets
        and are never rescued. Returns the updated frame plus the post-ffill missingness
        fraction (per row) so callers can refresh their diagnostic frame; None when there
        are no real feature columns to score."""
        real_feats = [
            c for c in get_feature_columns(cleaned)
            if c not in self.synthetic_feature_names_ and c != "_to_drop"
        ]
        if not real_feats:
            return cleaned, None
        post_missing = cleaned[real_feats].isna().mean(axis=1)
        cleaned = cleaned.copy()
        if drop_rows is True:
            all_targets_nan = cleaned[TARGET_COLS].isna().all(axis=1)
            rescued = (
                cleaned["_to_drop"]
                & ~all_targets_nan
                & (post_missing <= ROW_MISSINGNESS_THRESHOLD)
            )
            if rescued.any():
                cleaned.loc[rescued, "_to_drop"] = False
        return cleaned, post_missing

    def _refresh_missingness_diagnostic(
        self, diag: pd.DataFrame | None, cleaned: pd.DataFrame, post_missing: pd.Series,
    ) -> pd.DataFrame | None:
        """Return `diag` updated to post-ffill missingness / inclusion.

        Pure function (does not mutate self): callers pass the diagnostic frame captured
        for *this* split, because `_last_clean_missingness_` is clobbered by the separate
        history `_clean` call inside transform(). Aligned by (country, year) so ghost rows
        (absent from the diagnostic) are ignored."""
        if diag is None or diag.empty:
            return diag
        upd = pd.DataFrame({
            COUNTRY_COL: cleaned[COUNTRY_COL].to_numpy(),
            YEAR_COL: cleaned[YEAR_COL].to_numpy(),
            "_post_missing": post_missing.to_numpy(),
            "_post_included": (~cleaned["_to_drop"]).to_numpy(),
        })
        merged = diag.merge(upd, on=[COUNTRY_COL, YEAR_COL], how="left")
        # Where a post-ffill update exists, replace missingness/inclusion; else keep the
        # pre-ffill value. Built via np.where to avoid partial-assignment dtype issues.
        has = merged["_post_missing"].notna().to_numpy()
        merged["row_missingness_fraction"] = np.where(
            has,
            merged["_post_missing"].to_numpy(dtype=float),
            merged["row_missingness_fraction"].to_numpy(dtype=float),
        )
        merged["included"] = np.where(
            has,
            (merged["_post_included"] == True).to_numpy(),  # noqa: E712 (NaN -> False)
            merged["included"].to_numpy(dtype=bool),
        )
        return merged[
            [COUNTRY_COL, YEAR_COL, "row_missingness_fraction", "included"]
        ].copy()

    # -----------------------------------------------------------------
    # Phase 2: Impute
    # -----------------------------------------------------------------
    def _fit_imputers(self, cleaned_train: pd.DataFrame) -> None:
        self.numeric_global_fallback_ = cleaned_train[self.retained_continuous_features_].median().fillna(0.0)
        retained_cats = [c for c in CATEGORICAL_FEATURES if c in self.retained_raw_features_]
        self._fit_simple_imputer(cleaned_train, retained_cats)
        if "mice" in self.active_branches_:
            self._fit_mice_imputer(cleaned_train, retained_cats)

    def _fit_simple_imputer(
        self, cleaned_train: pd.DataFrame, retained_cats: list[str],
    ) -> None:
        self.simple_imputer_ = CountryYearFallbackImputer(
            numeric_features=self.retained_continuous_features_,
            categorical_features=retained_cats,
        )
        self.simple_imputer_.fit(cleaned_train)

    def _fit_mice_imputer(
        self, cleaned_train: pd.DataFrame, retained_cats: list[str],
    ) -> None:
        if self.retained_continuous_features_:
            miss_rates = cleaned_train[self.retained_continuous_features_].isna().mean()
            self.mice_missing_flags_ = [
                c for c in self.retained_continuous_features_
                if miss_rates.get(c, 0.0) > MICE_MISSING_THRESHOLD
            ]

            # Panel-aware auxiliary predictors
            mice_fit_df = cleaned_train[self.retained_continuous_features_].copy()
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

        if retained_cats:
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

            for col in self.mice_missing_flags_:
                if col in out.columns:
                    out[f"{col}_was_missing"] = out[col].isna().astype(float)

            if self.retained_continuous_features_:
                mice_in = self._build_mice_input(out)
                self._attach_regime_to_mice_input(mice_in, out, retained_cats)
                out = self._apply_mice_transform(out, mice_in)

            if retained_cats and self.mice_categorical_imputer_ is not None:
                out[retained_cats] = self.mice_categorical_imputer_.transform(out[retained_cats])

            # Drop missingness indicators — MICE preprocessing internals, not features.
            was_missing_cols = [
                f"{c}_was_missing" for c in self.mice_missing_flags_
                if f"{c}_was_missing" in out.columns
            ]
            if was_missing_cols:
                out = out.drop(columns=was_missing_cols)

        else:
            raise ValueError(f"Unsupported branch: {branch}")

        # Enforce categorical ranges after imputation
        for col, allowed in CATEGORICAL_ALLOWED_VALUES.items():
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").round().clip(
                    lower=min(allowed), upper=max(allowed),
                )

        return out

    def _build_mice_input(self, out: pd.DataFrame) -> pd.DataFrame:
        """Build enriched MICE input: original features + panel-aware aux columns."""
        mice_in = out[self.retained_continuous_features_].copy()
        for col, aux_name in zip(self.retained_continuous_features_, self.mice_aux_col_names_):
            c_fill = out[COUNTRY_COL].map(self.simple_imputer_.country_stats_[col])
            y_fill = out[YEAR_COL].map(self.simple_imputer_.year_stats_[col])
            glob = self.simple_imputer_.global_stats_[col]
            mice_in[aux_name] = c_fill.fillna(y_fill).fillna(glob)
        return mice_in

    def _attach_regime_to_mice_input(
        self, mice_in: pd.DataFrame, out: pd.DataFrame, retained_cats: list[str],
    ) -> None:
        """Attach v2x_regime as a numeric column to the MICE input matrix."""
        if self.mice_regime_col_ is None:
            return
        regime_col = "v2x_regime"
        if retained_cats and self.mice_categorical_imputer_ is not None:
            cats_imp = self.mice_categorical_imputer_.transform(out[retained_cats])
            cats_df = pd.DataFrame(cats_imp, columns=retained_cats, index=out.index)
            regime_vals = cats_df.get(regime_col, pd.Series(np.nan, index=out.index))
        else:
            regime_vals = out.get(regime_col, pd.Series(np.nan, index=out.index))
        mice_in["_regime"] = pd.to_numeric(regime_vals, errors="coerce").fillna(0.0)

    def _apply_mice_transform(self, out: pd.DataFrame, mice_in: pd.DataFrame) -> pd.DataFrame:
        """Scale, apply MICE imputer, inverse-scale, and write back to out."""
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
            for col in engineerable:
                if col not in RATE_FEATURES:
                    name = f"{col}_pct_change1"
                    new_cols[name] = group[col].pct_change().replace([np.inf, -np.inf], np.nan)
                    col_names.append(name)

        for window in ROLLING_WINDOWS:
            for col in engineerable:
                w = window  # capture loop variable for lambda
                name = f"{col}_rollmean{w}"
                new_cols[name] = group[col].transform(
                    lambda s, _w=w: s.rolling(window=_w, min_periods=ROLLING_MIN_PERIODS).mean()
                )
                col_names.append(name)

        for window in ROLLING_STD_WINDOWS:
            for col in engineerable:
                w = window
                name = f"{col}_rollstd{w}"
                new_cols[name] = group[col].transform(
                    lambda s, _w=w: s.rolling(window=_w, min_periods=ROLLING_MIN_PERIODS).std()
                )
                col_names.append(name)

        pairs = combinations(engineerable, 2) if ALL_PAIRWISE_INTERACTIONS else iter(INTERACTION_PAIRS)
        for left, right in pairs:
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


CountryYearFallbackImputer.__module__ = "stages.preprocessing"
Preprocessor.__module__ = "stages.preprocessing"


# =============================================================================
# Root config override
# =============================================================================

def _apply_root_config_overrides() -> None:
    """Load project-root config.py and override module constants when not None."""
    import importlib.util
    from pathlib import Path as _Path
    root_path = _Path(__file__).resolve().parent.parent / "config.py"
    if not root_path.exists():
        return
    spec = importlib.util.spec_from_file_location("_root_cfg", root_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    g = globals()
    for name in (
        "INCLUDE_SYNTHETIC_FEATURES",
        "ALL_PAIRWISE_INTERACTIONS",
        "INTERACTION_PAIRS",
        "ROW_MISSINGNESS_THRESHOLD",
        "IQR_MULTIPLIER",
        "LAG_PERIODS",
        "ROLLING_WINDOWS",
        "ROLLING_STD_WINDOWS",
        "FORWARD_FILL_LIMIT",
        "ENABLE_DIFF_FEATURES",
        "ENABLE_PCT_CHANGE_FEATURES",
        "SCALER_CHOICE",
        "MICE_COUNTRY_N_COMPONENTS",
        "MICE_MAX_ITER",
        "EXCLUDE_COLUMNS",
    ):
        val = getattr(m, name, None)
        if val is not None:
            g[name] = val


_apply_root_config_overrides()


if __name__ == "__main__":
    raise SystemExit(
        "Use `python -m stages.training.data_preparation` to build the preprocessing cache."
    )
