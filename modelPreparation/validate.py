"""Post-transform validation utilities for modelPreparation."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from modelPreparation.core.protocols import FoldValidationReport


def _non_finite_locations(df: pd.DataFrame) -> list[str]:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return []

    bad_mask = ~np.isfinite(numeric.to_numpy())
    if not bad_mask.any():
        return []

    rows, cols = np.where(bad_mask)
    bad_entries: list[str] = []
    for r, c in zip(rows, cols):
        bad_entries.append(f"{numeric.index[r]}:{numeric.columns[c]}")
    return bad_entries


def _imputation_coverage(
    raw_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, float]:
    coverage: dict[str, float] = {}
    for feature in feature_columns:
        if feature not in raw_df.columns or feature not in processed_df.columns:
            continue
        raw_missing = raw_df[feature].isna()
        now_present = processed_df[feature].notna()
        coverage[feature] = float((raw_missing & now_present).mean())
    return coverage


def validate_fold_output(
    X_train_raw: pd.DataFrame,
    X_train_processed: pd.DataFrame,
    X_valid_raw: pd.DataFrame | None = None,
    X_valid_processed: pd.DataFrame | None = None,
    y_train_processed: pd.Series | pd.DataFrame | None = None,
    danger_features: Iterable[str] | None = None,
    expected_counts: dict[str, int] | None = None,
    train_year_max: int | None = None,
    valid_year_range: tuple[int, int] | None = None,
    skewness_threshold: float = 2.0,
    high_imputation_threshold: float = 0.50,
) -> FoldValidationReport:
    """Validate fold outputs and return diagnostics report."""
    errors: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, object] = {}

    # 1) Finite checks
    bad_train = _non_finite_locations(X_train_processed)
    if bad_train:
        errors.append(f"Non-finite values in processed train matrix (sample): {bad_train[:5]}")

    if X_valid_processed is not None:
        bad_valid = _non_finite_locations(X_valid_processed)
        if bad_valid:
            errors.append(f"Non-finite values in processed valid matrix (sample): {bad_valid[:5]}")

    if y_train_processed is not None:
        if isinstance(y_train_processed, pd.DataFrame):
            for col in y_train_processed.columns:
                y_values = np.asarray(pd.to_numeric(y_train_processed[col], errors="coerce"), dtype=float)
                if not np.isfinite(y_values).all():
                    errors.append(f"Non-finite values found in transformed training target '{col}'")
                elif len(y_values) >= 3:
                    skewness = float(pd.Series(y_values).skew())
                    metrics[f"target_skewness_{col}"] = skewness
                    if abs(skewness) > skewness_threshold:
                        warnings.append(
                            f"Transformed target '{col}' skewness={skewness:.3f} "
                            f"exceeds threshold {skewness_threshold:.3f}"
                        )
        else:
            y_values = np.asarray(pd.to_numeric(y_train_processed, errors="coerce"), dtype=float)
            if not np.isfinite(y_values).all():
                errors.append("Non-finite values found in transformed training target")
            elif len(y_values) >= 3:
                skewness = float(pd.Series(y_values).skew())
                metrics["target_skewness"] = skewness
                if abs(skewness) > skewness_threshold:
                    warnings.append(
                        f"Transformed target skewness={skewness:.3f} exceeds threshold {skewness_threshold:.3f}"
                    )

    # 2) DANGER features absent
    if danger_features is None:
        danger_features = []
    present_danger = [f for f in danger_features if f in X_train_processed.columns]
    if X_valid_processed is not None:
        present_danger.extend([f for f in danger_features if f in X_valid_processed.columns])
    present_danger = sorted(set(present_danger))
    if present_danger:
        errors.append(f"DANGER features present after preprocessing: {present_danger}")

    # 3) Split integrity checks
    if train_year_max is not None and "year" in X_train_raw.columns:
        if int(pd.to_numeric(X_train_raw["year"], errors="coerce").max()) > train_year_max:
            errors.append("Train split contains year(s) beyond configured train boundary")

    if valid_year_range is not None and X_valid_raw is not None and "year" in X_valid_raw.columns:
        lo, hi = valid_year_range
        years = pd.to_numeric(X_valid_raw["year"], errors="coerce")
        if ((years < lo) | (years > hi)).any():
            errors.append("Valid split contains year(s) outside configured validation range")

    # 4) Row count checks
    if expected_counts is not None:
        if "train" in expected_counts and len(X_train_raw) != expected_counts["train"]:
            errors.append(
                f"Train row count mismatch: expected {expected_counts['train']}, got {len(X_train_raw)}"
            )
        if X_valid_raw is not None and "valid" in expected_counts and len(X_valid_raw) != expected_counts["valid"]:
            errors.append(
                f"Valid row count mismatch: expected {expected_counts['valid']}, got {len(X_valid_raw)}"
            )

    # 5) Imputation coverage diagnostics
    feature_columns = [
        c for c in X_train_processed.columns
        if c not in {"country", "year"}
    ]
    train_imputation = _imputation_coverage(X_train_raw, X_train_processed, feature_columns)
    metrics["train_imputation_coverage"] = train_imputation

    if X_valid_raw is not None and X_valid_processed is not None:
        valid_imputation = _imputation_coverage(X_valid_raw, X_valid_processed, feature_columns)
        metrics["valid_imputation_coverage"] = valid_imputation

        high_valid = [
            f for f, rate in valid_imputation.items() if rate > high_imputation_threshold
        ]
        if high_valid:
            warnings.append(
                "High validation imputation coverage (>"
                f"{high_imputation_threshold:.2f}) for features: {sorted(high_valid)}"
            )

    return FoldValidationReport(
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        metrics=metrics,
    )
