"""Deterministic lagging for CAUTION features (Step 2).

Uses V2's no-row-drop approach: lagged NaN values are left in place for
downstream imputation to handle, preserving data in small-data regimes.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from modelPreparation.config.base import ModelPreparationConfig


def run_lag(
    df: pd.DataFrame,
    config: ModelPreparationConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Lag configured features by one year without dropping rows."""
    country_col, year_col = config.key_columns

    work = df.sort_values([country_col, year_col]).copy()

    lagged_features = [f for f in config.caution_features_to_lag if f in work.columns]
    for feature in lagged_features:
        work[feature] = work.groupby(country_col, sort=False)[feature].shift(1)

    metadata = {
        "lagged_features_in_place": lagged_features,
    }
    return work, metadata
