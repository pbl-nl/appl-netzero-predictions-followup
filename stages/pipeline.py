from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
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
VIF_CUTOFF = 10.0
MAX_VIF_ITERATIONS = 100

LAG_PERIODS = [1]
ROLLING_WINDOWS = [3]
ROLLING_MIN_PERIODS = 1
STRUCTURAL_NA_FILL_VALUE = 0.0
INTERACTION_PAIRS: list[tuple[str, str]] = []
ENABLE_DIFF_FEATURES = True
ENABLE_PCT_CHANGE_FEATURES = True

SCALER_CHOICE = "standard"  # Options: "standard", "minmax", "robust"
MICE_MAX_ITER = 20
MICE_TOL = 1e-3
MICE_SKIP_COMPLETE = True

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
        "lower": 0.0,
        "upper": 250.0,
    },
    "coal_rents_of_gdp": {"lower": 0.0, "upper": 100.0},
    "control_of_corruption_estimate": {"lower": -4.0, "upper": 4.0},
    "forest_area_of_land_area": {"lower": 0.0, "upper": 100.0},
    "gdp_per_capita_constant_2015_us": {"lower": 0.0, "upper": 500000.0},
    "gdp_ppp_constant_2021_international": {"lower": 0.0, "upper": 2.0e14},
    "natural_gas_rents_of_gdp": {"lower": 0.0, "upper": 100.0},
    "oil_rents_of_gdp": {"lower": 0.0, "upper": 100.0},
    "political_stability_and_absence_of_violence_terrorism_estimate": {
        "lower": -4.0,
        "upper": 4.0,
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


# =============================================================================
# Utility functions
# =============================================================================


def log_transform(y: np.ndarray | pd.Series) -> np.ndarray:
    return np.log1p(np.asarray(y, dtype=float))


_MAX_LOG_PRED = float(np.log1p(1e6))


def inverse_log_transform(y: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.clip(np.asarray(y, dtype=float), 0.0, _MAX_LOG_PRED)
    return np.expm1(arr)


def safe_mode(series: pd.Series) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    mode = non_null.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else non_null.iloc[0]


def compute_vif_values(matrix: pd.DataFrame) -> pd.DataFrame:
    if matrix.shape[1] <= 1:
        return pd.DataFrame(
            [{"feature": col, "vif": 1.0} for col in matrix.columns]
        ).sort_values("vif", ascending=False).reset_index(drop=True)

    corr = matrix.corr().values
    try:
        inv = np.linalg.inv(corr)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(corr)

    vifs = np.diag(inv)
    records = [
        {"feature": col, "vif": float(max(v, 1.0))}
        for col, v in zip(matrix.columns, vifs)
    ]
    return pd.DataFrame(records).sort_values("vif", ascending=False).reset_index(drop=True)


# =============================================================================
# Loading and splitting
# =============================================================================


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in ID_COLS + TARGET_COLS]


def get_continuous_features(df: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    return [
        col for col in feature_columns
        if col not in CATEGORICAL_FEATURES and pd.api.types.is_numeric_dtype(df[col])
    ]


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

    def __init__(self, active_branches: list[str] | None = None) -> None:
        if active_branches is None:
            active_branches = list(BRANCH_NAMES)
        invalid = sorted(set(active_branches) - set(BRANCH_NAMES))
        if invalid:
            raise ValueError(f"Unsupported branches: {invalid}")

        self.active_branches_ = list(active_branches)
        self.timing_records_: list[dict[str, Any]] = []

        # Fitted state — set during fit_transform
        self.feature_columns_: list[str] = []
        self.continuous_features_: list[str] = []
        self.retained_raw_features_: list[str] = []
        self.retained_continuous_features_: list[str] = []
        self.columns_to_drop_: set[str] = set()

        # Imputers
        self.simple_imputer_: CountryYearFallbackImputer | None = None
        self.mice_numeric_imputer_: IterativeImputer | None = None
        self.mice_categorical_imputer_: SimpleImputer | None = None
        self.numeric_global_fallback_: pd.Series | None = None

        # Encoders / scalers per branch
        self.encoder_: dict[str, OneHotEncoder] = {}
        self.scaler_: dict[str, Any] = {}
        self.final_feature_columns_: dict[str, list[str]] = {}

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def fit_transform(self, train_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        pid = os.getpid()
        self.timing_records_ = []
        self.feature_columns_ = get_feature_columns(train_df)
        self.continuous_features_ = get_continuous_features(train_df, self.feature_columns_)

        # Phase 1: Clean
        t0 = time.monotonic()
        cleaned = self._clean(train_df, history_df=None, fit=True)
        self.timing_records_.append({
            "operation": "preprocess_sub_phase", "sub_operation": "clean",
            "elapsed_seconds": time.monotonic() - t0,
            "rows_in": len(train_df), "rows_out": len(cleaned),
            "detail": "fit_transform",
        })
        self.retained_raw_features_ = [
            c for c in self.feature_columns_ if c not in self.columns_to_drop_
        ]
        self.retained_continuous_features_ = [
            c for c in self.continuous_features_ if c in self.retained_raw_features_
        ]
        print(
            f"[PID:{pid}] Cleaning: {len(train_df)}→{len(cleaned)} rows, "
            f"dropped {len(self.columns_to_drop_)} columns, "
            f"{len(self.retained_continuous_features_)} continuous features retained",
            flush=True,
        )

        # Fit imputers on cleaned training data
        t0 = time.monotonic()
        self._fit_imputers(cleaned)
        self.timing_records_.append({
            "operation": "preprocess_sub_phase", "sub_operation": "fit_imputers",
            "elapsed_seconds": time.monotonic() - t0,
            "rows_in": len(cleaned), "rows_out": len(cleaned),
            "detail": "fit_transform",
        })

        # Phase 2 + 3 per branch: Impute → Engineer
        outputs: dict[str, pd.DataFrame] = {}
        for branch in self.active_branches_:
            t0 = time.monotonic()
            imputed = self._impute(cleaned, branch)
            self.timing_records_.append({
                "operation": "preprocess_sub_phase", "sub_operation": "impute",
                "elapsed_seconds": time.monotonic() - t0,
                "rows_in": len(cleaned), "rows_out": len(imputed),
                "detail": f"fit_transform/{branch}",
            })
            feat_cols = get_feature_columns(imputed)
            nans = int(imputed[feat_cols].isna().sum().sum())
            print(f"[PID:{pid}] Imputation ({branch}): {nans} NaNs remaining", flush=True)

            t0 = time.monotonic()
            engineered = self._engineer(imputed, history_df=None)
            self.timing_records_.append({
                "operation": "preprocess_sub_phase", "sub_operation": "engineer",
                "elapsed_seconds": time.monotonic() - t0,
                "rows_in": len(imputed), "rows_out": len(engineered),
                "n_features": len(get_feature_columns(engineered)),
                "detail": f"fit_transform/{branch}",
            })

            t0 = time.monotonic()
            final = self._encode_and_scale(engineered, branch, fit=True)
            n_feats = len(self.final_feature_columns_[branch])
            self.timing_records_.append({
                "operation": "preprocess_sub_phase", "sub_operation": "encode_and_scale",
                "elapsed_seconds": time.monotonic() - t0,
                "rows_in": len(engineered), "rows_out": len(final),
                "n_features": n_feats,
                "detail": f"fit_transform/{branch}",
            })
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
        self.timing_records_ = []

        t0 = time.monotonic()
        cleaned = self._clean(df, history_df=history_df, fit=False, drop_rows=drop_rows)
        self.timing_records_.append({
            "operation": "preprocess_sub_phase", "sub_operation": "clean",
            "elapsed_seconds": time.monotonic() - t0,
            "rows_in": len(df), "rows_out": len(cleaned),
            "detail": f"transform/{split_name}",
        })

        history_cleaned: pd.DataFrame | None = None
        if history_df is not None and not history_df.empty:
            t0 = time.monotonic()
            history_cleaned = self._clean(history_df, history_df=None, fit=False)
            self.timing_records_.append({
                "operation": "preprocess_sub_phase", "sub_operation": "clean_history",
                "elapsed_seconds": time.monotonic() - t0,
                "rows_in": len(history_df), "rows_out": len(history_cleaned),
                "detail": f"transform/{split_name}",
            })

        outputs: dict[str, pd.DataFrame] = {}
        for branch in self.active_branches_:
            t0 = time.monotonic()
            imputed = self._impute(cleaned, branch)
            self.timing_records_.append({
                "operation": "preprocess_sub_phase", "sub_operation": "impute",
                "elapsed_seconds": time.monotonic() - t0,
                "rows_in": len(cleaned), "rows_out": len(imputed),
                "detail": f"transform/{split_name}/{branch}",
            })

            history_imputed = None
            if history_cleaned is not None:
                t0 = time.monotonic()
                history_imputed = self._impute(history_cleaned, branch)
                self.timing_records_.append({
                    "operation": "preprocess_sub_phase", "sub_operation": "impute_history",
                    "elapsed_seconds": time.monotonic() - t0,
                    "rows_in": len(history_cleaned), "rows_out": len(history_imputed),
                    "detail": f"transform/{split_name}/{branch}",
                })

            t0 = time.monotonic()
            engineered = self._engineer(imputed, history_df=history_imputed)
            self.timing_records_.append({
                "operation": "preprocess_sub_phase", "sub_operation": "engineer",
                "elapsed_seconds": time.monotonic() - t0,
                "rows_in": len(imputed), "rows_out": len(engineered),
                "n_features": len(get_feature_columns(engineered)),
                "detail": f"transform/{split_name}/{branch}",
            })

            t0 = time.monotonic()
            result = self._encode_and_scale(engineered, branch, fit=False)
            self.timing_records_.append({
                "operation": "preprocess_sub_phase", "sub_operation": "encode_and_scale",
                "elapsed_seconds": time.monotonic() - t0,
                "rows_in": len(engineered), "rows_out": len(result),
                "n_features": len(self.final_feature_columns_.get(branch, [])),
                "detail": f"transform/{split_name}/{branch}",
            })
            outputs[branch] = result

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

        # Outlier replacement (IQR, computed on merged data, applied to current only)
        self._apply_outlier_pass(working, COUNTRY_COL, self.continuous_features_, COUNTRY_OUTLIER_MIN_OBS)
        self._apply_outlier_pass(working, YEAR_COL, self.continuous_features_, YEAR_OUTLIER_MIN_OBS)

        # Range validation
        self._apply_numeric_range_rules(working)
        self._apply_categorical_range_rules(working)

        # Extract current rows
        cleaned = working[working["_is_current"]].copy().drop(columns=["_is_current"], errors="ignore")

        # Row drops: all targets missing OR excessive feature missingness
        if drop_rows:
            feature_cols = get_feature_columns(cleaned)
            row_missing = cleaned[feature_cols].isna().mean(axis=1) > ROW_MISSINGNESS_THRESHOLD
            all_targets_nan = cleaned[TARGET_COLS].isna().all(axis=1)
            drop_mask = all_targets_nan | row_missing
        else:
            drop_mask = pd.Series(False, index=cleaned.index)

        # Column drops + VIF (only during fit, computed on retained rows)
        if fit:
            self._fit_column_drops_and_vif(cleaned.loc[~drop_mask].copy())

        cleaned = cleaned.loc[~drop_mask].copy()
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

    def _fit_column_drops_and_vif(self, train_frame: pd.DataFrame) -> None:
        feature_cols = get_feature_columns(train_frame)
        sparse = train_frame[feature_cols].isna().mean()
        self.columns_to_drop_ = set(sparse[sparse >= SPARSE_COLUMN_THRESHOLD].index.tolist())

        vif_candidates = [
            c for c in self.continuous_features_
            if c not in self.columns_to_drop_ and c in train_frame.columns
        ]
        if not vif_candidates:
            return

        # Temp impute for VIF computation
        temp = train_frame[ID_COLS + vif_candidates].copy()
        temp_imputer = CountryYearFallbackImputer(numeric_features=vif_candidates, categorical_features=[])
        temp = temp_imputer.fit_transform(temp)

        matrix = temp[vif_candidates].copy()
        zero_var = [c for c in matrix.columns if matrix[c].nunique(dropna=False) <= 1]
        self.columns_to_drop_.update(zero_var)
        matrix = matrix.drop(columns=zero_var, errors="ignore")

        selected = matrix.columns.tolist()
        for _ in range(MAX_VIF_ITERATIONS):
            if len(selected) <= 1:
                break
            vif_df = compute_vif_values(matrix[selected])
            if vif_df.empty:
                break
            max_row = vif_df.iloc[0]
            if float(max_row["vif"]) <= VIF_CUTOFF:
                break
            selected.remove(str(max_row["feature"]))

        vif_dropped = [c for c in matrix.columns.tolist() if c not in selected]
        self.columns_to_drop_.update(vif_dropped)

    # -----------------------------------------------------------------
    # Phase 2: Impute
    # -----------------------------------------------------------------
    def _fit_imputers(self, cleaned_train: pd.DataFrame) -> None:
        self.numeric_global_fallback_ = cleaned_train[self.retained_continuous_features_].median()
        retained_cats = [c for c in CATEGORICAL_FEATURES if c in self.retained_raw_features_]

        if "simple" in self.active_branches_:
            self.simple_imputer_ = CountryYearFallbackImputer(
                numeric_features=self.retained_continuous_features_,
                categorical_features=retained_cats,
            )
            self.simple_imputer_.fit(cleaned_train)

        if "mice" in self.active_branches_ and self.retained_continuous_features_:
            self.mice_numeric_imputer_ = IterativeImputer(
                random_state=RANDOM_STATE,
                max_iter=MICE_MAX_ITER,
                tol=MICE_TOL,
                skip_complete=MICE_SKIP_COMPLETE,
            )
            self.mice_numeric_imputer_.fit(cleaned_train[self.retained_continuous_features_])

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
            if self.retained_continuous_features_:
                numeric_vals = self.mice_numeric_imputer_.transform(out[self.retained_continuous_features_])
                out[self.retained_continuous_features_] = numeric_vals
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
    # Phase 3: Engineer (feature engineering + encode + scale)
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

        # Lags
        for lag in LAG_PERIODS:
            for col in self.retained_continuous_features_:
                name = f"{col}_lag{lag}"
                new_cols[name] = group[col].shift(lag)
                col_names.append(name)

        # Diffs (skip rate features)
        if ENABLE_DIFF_FEATURES:
            for col in self.retained_continuous_features_:
                if col not in RATE_FEATURES:
                    name = f"{col}_diff1"
                    new_cols[name] = group[col].diff(1)
                    col_names.append(name)

        # Pct changes
        if ENABLE_PCT_CHANGE_FEATURES:
            for col in PCT_CHANGE_FEATURES:
                if col in self.retained_continuous_features_:
                    name = f"{col}_pct_change1"
                    new_cols[name] = group[col].pct_change().replace([np.inf, -np.inf], np.nan)
                    col_names.append(name)

        # Rolling means
        for window in ROLLING_WINDOWS:
            for col in self.retained_continuous_features_:
                name = f"{col}_rollmean{window}"
                new_cols[name] = group[col].transform(
                    lambda s: s.rolling(window=window, min_periods=ROLLING_MIN_PERIODS).mean()
                )
                col_names.append(name)

        # Interactions
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
            # Fit encoder
            if raw_cats:
                encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                encoded_array = encoder.fit_transform(cat_df)
                self.encoder_[branch] = encoder
                encoded_cols = encoder.get_feature_names_out(raw_cats).tolist()
            else:
                encoded_array = np.empty((len(df), 0))
                encoded_cols = []

            # Fit scaler
            scaler = self._make_scaler()
            if numeric_feats:
                scaler.fit(numeric_df)
            self.scaler_[branch] = scaler
            self.final_feature_columns_[branch] = numeric_feats + encoded_cols
        else:
            # Transform with fitted encoder/scaler
            encoder = self.encoder_.get(branch)
            if raw_cats and encoder is not None:
                encoded_array = encoder.transform(cat_df)
                encoded_cols = encoder.get_feature_names_out(raw_cats).tolist()
            else:
                encoded_array = np.empty((len(df), 0))
                encoded_cols = []
            scaler = self.scaler_[branch]

        # Scale numeric features
        if numeric_feats:
            scaled_df = pd.DataFrame(
                scaler.transform(numeric_df), columns=numeric_feats, index=df.index,
            )
        else:
            scaled_df = pd.DataFrame(index=df.index)

        encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)
        final_features = pd.concat([scaled_df, encoded_df], axis=1)

        # Align columns for transform mode
        if not fit:
            expected = self.final_feature_columns_[branch]
            final_features = final_features.reindex(columns=expected, fill_value=0.0)

        # Combine ID/target columns with features
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


# =============================================================================
# Pipeline orchestration
# =============================================================================


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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    df = load_panel_data(DATA_PATH)
    result = run_pipeline_for_strategy(df, VALIDATION_STRATEGY)
    train_rows = len(result["train_outputs"]["simple"])
    val_rows = len(result["val_outputs"]["simple"])
    test_rows = len(result["test_outputs"]["simple"])
    logger.info(f"Done: train={train_rows}, val={val_rows}, test={test_rows}")
