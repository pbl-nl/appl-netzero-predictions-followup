"""Globally safe feature selection (Step 1)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from modelPreparation.config.base import ModelPreparationConfig


def _replace_leading_zeros_with_nan(
    df: pd.DataFrame,
    feature: str,
    country_col: str,
    year_col: str,
) -> tuple[pd.DataFrame, int]:
    """Replace only leading zeros per country with NaN for one feature."""
    if feature not in df.columns:
        return df, 0

    ordered = df.sort_values([country_col, year_col]).copy()
    converted = 0

    for _, group in ordered.groupby(country_col, sort=False):
        series = group[feature]

        non_zero_mask = series.notna() & (series != 0)
        if non_zero_mask.any():
            first_non_zero_index = non_zero_mask[non_zero_mask].index[0]
            first_position = group.index.get_loc(first_non_zero_index)
            leading_idx = group.index[:first_position]
        else:
            leading_idx = group.index

        zero_leading_idx = [idx for idx in leading_idx if ordered.at[idx, feature] == 0]
        if zero_leading_idx:
            ordered.loc[zero_leading_idx, feature] = np.nan
            converted += len(zero_leading_idx)

    ordered = ordered.sort_index()
    return ordered, converted


def run_select(
    df: pd.DataFrame,
    config: ModelPreparationConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply globally safe exclusions and structural feature drops."""
    work = df.copy()

    excluded_rows = 0
    excluded_countries = 0
    if config.exclude_negative_rce_countries:
        before_rows = len(work)
        before_countries = work[config.key_columns[0]].nunique()
        work = work[~work[config.key_columns[0]].isin(config.negative_rce_countries)].copy()
        excluded_rows = before_rows - len(work)
        excluded_countries = before_countries - work[config.key_columns[0]].nunique()

    dropped_global_features: list[str] = []
    for feature in config.danger_features + config.global_structural_drop_features:
        if feature in work.columns:
            work = work.drop(columns=[feature])
            dropped_global_features.append(feature)

    leading_zero_converted: dict[str, int] = {}
    country_col, year_col = config.key_columns
    for feature in config.leading_zero_as_missing_cols:
        work, converted = _replace_leading_zeros_with_nan(
            work,
            feature=feature,
            country_col=country_col,
            year_col=year_col,
        )
        if converted > 0:
            leading_zero_converted[feature] = converted

    metadata = {
        "excluded_negative_rce_rows": excluded_rows,
        "excluded_negative_rce_countries": excluded_countries,
        "dropped_global_features": dropped_global_features,
        "leading_zero_converted": leading_zero_converted,
    }
    return work, metadata
