"""Per-fold preprocessing pipeline for modelPreparation.

Merges V1's country-aware imputation logic with V2's sklearn compatibility
and multi-target support.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from modelPreparation.config.base import ModelPreparationConfig

logger = logging.getLogger(__name__)


class ModelPreparationPipeline(BaseEstimator, TransformerMixin):
    """Fold-contained preprocessing pipeline.

    Fit this class on training data only. All learned parameters are derived
    only from data passed to ``fit``.

    Multi-target aware: y is expected as a DataFrame with one column per
    target scenario. Each target gets its own shift constant.
    """

    def __init__(self, config: ModelPreparationConfig | None = None):
        self.config = config or ModelPreparationConfig()

        self.fitted_: bool = False

        self.feature_columns_: list[str] = []
        self.selected_features_: list[str] = []

        self.train_medians_: dict[str, float] = {}
        self.country_last_values_: dict[str, dict[str, float]] = {}

        self.target_shift_constants_: dict[str, float] = {}

        self.scaler_means_: dict[str, float] = {}
        self.scaler_stds_: dict[str, float] = {}

        self.imputation_coverage_: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_feature_columns(self, X: pd.DataFrame) -> list[str]:
        key_cols = set(self.config.key_columns)
        return [c for c in X.columns if c not in key_cols]

    def _to_numeric_features(self, X: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        out = X.copy()
        for feature in features:
            out[feature] = pd.to_numeric(out[feature], errors="coerce")
        return out

    def _sorted_index(self, X: pd.DataFrame) -> pd.Index:
        country_col = self.config.key_columns[0]
        if country_col not in X.columns:
            return X.index

        sort_cols = [country_col]
        year_col = self.config.key_columns[1]
        if year_col in X.columns:
            sort_cols.append(year_col)
        return X.sort_values(sort_cols).index

    def _learn_imputation_params(self, X: pd.DataFrame) -> None:
        """Learn per-feature imputation parameters from training data.

        Strategy: forward-fill within country -> country last-value map -> median.
        """
        ordered_idx = self._sorted_index(X)
        country_col = self.config.key_columns[0]

        for feature in self.feature_columns_:
            series = pd.to_numeric(X[feature], errors="coerce")

            if country_col in X.columns:
                country_series = X.loc[ordered_idx, country_col]
                ordered_values = series.loc[ordered_idx]
                ffilled = ordered_values.groupby(country_series, sort=False).ffill()

                df_last = pd.DataFrame({
                    "country": country_series.values,
                    "value": ffilled.values,
                })
                country_last: dict[str, float] = {}
                for country, group in df_last.groupby("country", sort=False):
                    non_null = group["value"].dropna()
                    if not non_null.empty:
                        country_last[str(country)] = float(non_null.iloc[-1])
            else:
                ffilled = series
                country_last = {}

            median = float(pd.Series(ffilled).median(skipna=True))
            if np.isnan(median):
                median = 0.0

            self.country_last_values_[feature] = country_last
            self.train_medians_[feature] = median

    def _impute_features(self, X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
        """Apply country-aware imputation: ffill -> country-last -> median."""
        out = X.copy()
        ordered_idx = self._sorted_index(out)
        country_col = self.config.key_columns[0]

        imputation_rates: dict[str, float] = {}

        for feature in self.feature_columns_:
            series = pd.to_numeric(out[feature], errors="coerce")
            missing_before = series.isna()

            if country_col in out.columns:
                country_series = out.loc[ordered_idx, country_col]
                ordered_values = series.loc[ordered_idx]
                ffilled = ordered_values.groupby(country_series, sort=False).ffill()
                series = ffilled.reindex(out.index)

                if feature in self.country_last_values_:
                    mapped_last = out[country_col].astype(str).map(
                        self.country_last_values_[feature]
                    )
                    series = series.fillna(mapped_last)

            series = series.fillna(self.train_medians_.get(feature, 0.0))
            out[feature] = series

            imputed = missing_before & series.notna()
            rate = float(imputed.mean())
            imputation_rates[feature] = rate
            if rate > 0.5:
                logger.warning(
                    "High imputation rate for %s: %.1f%%", feature, rate * 100
                )

        return out, imputation_rates

    def _apply_feature_transforms(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()

        for feature in self.config.log_features:
            if feature in out.columns:
                out[feature] = np.log(np.clip(out[feature].astype(float), a_min=1e-9, a_max=None))

        for feature in self.config.log1p_features:
            if feature in out.columns:
                out[feature] = np.log1p(np.clip(out[feature].astype(float), a_min=0.0, a_max=None))

        return out

    def _transform_targets(self, y: pd.DataFrame) -> pd.DataFrame:
        """Apply log(y + c) per target column."""
        y_out = y.copy()
        for col in y_out.columns:
            y_num = pd.to_numeric(y_out[col], errors="coerce")
            if y_num.isna().any():
                raise ValueError(f"Target column '{col}' contains NaN or non-numeric values")
            c = self.target_shift_constants_.get(col, 1.0)
            transformed = np.log(y_num + c)
            if not np.isfinite(transformed).all():
                raise ValueError(f"Target transform for '{col}' generated non-finite values")
            y_out[col] = transformed
        return y_out

    def _ensure_y_dataframe(self, y: pd.DataFrame | pd.Series) -> pd.DataFrame:
        if isinstance(y, pd.Series):
            return y.to_frame()
        return y

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame | pd.Series) -> ModelPreparationPipeline:
        """Fit imputation, target transform, optional selector, and optional scaler."""
        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train must be provided")
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Cannot fit pipeline on empty data")
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must have the same length")

        X = X_train.copy()
        y_df = self._ensure_y_dataframe(y_train).copy()

        # 1. Resolve feature columns (exclude key columns)
        self.feature_columns_ = self._resolve_feature_columns(X)

        # 2. Coerce to numeric
        X = self._to_numeric_features(X, self.feature_columns_)

        # 3. Learn imputation params
        self._learn_imputation_params(X)

        # 4. Learn per-target shift constants: c = max(1.0, -min(y) + 1.0)
        #    Override with pre-computed values if provided (for stable shifts
        #    across bootstrap runs).
        self.target_shift_constants_ = {}
        precomputed = self.config.precomputed_target_shift_constants or {}
        for col in y_df.columns:
            if col in precomputed:
                self.target_shift_constants_[col] = precomputed[col]
            else:
                y_num = pd.to_numeric(y_df[col], errors="coerce")
                if y_num.isna().any():
                    raise ValueError(f"y_train column '{col}' contains NaN or non-numeric values")
                y_min = float(y_num.min())
                self.target_shift_constants_[col] = max(1.0, -y_min + 1.0)

        # 5. Apply imputation + transforms, then fit scaler if needed
        X_imputed, fit_imputation_rates = self._impute_features(X)
        self.imputation_coverage_["fit"] = fit_imputation_rates

        transformed_full = self._apply_feature_transforms(X_imputed[self.feature_columns_])

        y_transformed = self._transform_targets(y_df)

        # 6. Optional feature selection
        if self.config.enable_feature_selection and self.config.feature_selector is not None:
            selected = self.config.feature_selector(transformed_full.copy(), y_transformed.copy())
            selected = [f for f in selected if f in transformed_full.columns]
            if not selected:
                raise ValueError("Feature selector returned no valid features")
            self.selected_features_ = selected
        else:
            self.selected_features_ = list(self.feature_columns_)

        transformed = transformed_full[self.selected_features_].copy()

        # 7. Fit scaler if configured
        if self.config.scaling_mode == "standard":
            means = transformed.mean(axis=0)
            stds = transformed.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
            self.scaler_means_ = {k: float(v) for k, v in means.to_dict().items()}
            self.scaler_stds_ = {k: float(v) for k, v in stds.to_dict().items()}

        self.fitted_ = True
        return self

    def transform(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | pd.Series | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Transform features (and optional targets) with fit-time parameters."""
        if not self.fitted_:
            raise ValueError("Pipeline must be fit before transform")
        if X is None or len(X) == 0:
            raise ValueError("X must be non-empty")

        work = X.copy()

        missing = [c for c in self.feature_columns_ if c not in work.columns]
        if missing:
            raise ValueError(f"Transform input is missing feature columns: {missing}")

        work = self._to_numeric_features(work, self.feature_columns_)

        imputed, imputation_rates = self._impute_features(work)
        self.imputation_coverage_["transform"] = imputation_rates

        transformed = self._apply_feature_transforms(imputed[self.feature_columns_])
        transformed = transformed[self.selected_features_].copy()

        if self.config.scaling_mode == "standard":
            means = pd.Series(self.scaler_means_)
            stds = pd.Series(self.scaler_stds_)
            transformed = (transformed - means) / stds

        if not np.isfinite(transformed.to_numpy(dtype=float)).all():
            raise ValueError("Feature transform generated non-finite values")

        y_transformed = None
        if y is not None:
            y_df = self._ensure_y_dataframe(y).copy()
            y_transformed = self._transform_targets(y_df)

        return transformed, y_transformed

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame | pd.Series,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fit pipeline and transform training data."""
        self.fit(X_train, y_train)
        X_transformed, y_transformed = self.transform(X_train, y_train)
        if y_transformed is None:
            raise RuntimeError("Expected transformed target after fit_transform")
        return X_transformed, y_transformed

    def inverse_transform_target(
        self,
        y_pred: np.ndarray | pd.Series,
        target_name: str,
    ) -> np.ndarray:
        """Map log-shifted target predictions back to original RCE space."""
        if not self.fitted_:
            raise ValueError("Pipeline must be fit before inverse_transform_target")
        if target_name not in self.target_shift_constants_:
            raise ValueError(
                f"Unknown target '{target_name}'. "
                f"Known targets: {sorted(self.target_shift_constants_.keys())}"
            )

        values = np.asarray(y_pred, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("y_pred contains non-finite values")

        c = self.target_shift_constants_[target_name]
        return np.exp(values) - c

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return learned parameters for serialization."""
        if not self.fitted_:
            raise ValueError("Pipeline must be fit before get_params")

        return {
            "feature_columns": list(self.feature_columns_),
            "selected_features": list(self.selected_features_),
            "train_medians": dict(self.train_medians_),
            "country_last_values": dict(self.country_last_values_),
            "target_shift_constants": dict(self.target_shift_constants_),
            "scaling_mode": self.config.scaling_mode,
            "scaler_means": dict(self.scaler_means_),
            "scaler_stds": dict(self.scaler_stds_),
            "log_features": list(self.config.log_features),
            "log1p_features": list(self.config.log1p_features),
            "imputation_coverage": dict(self.imputation_coverage_),
        }
