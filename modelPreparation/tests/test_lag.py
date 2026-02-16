"""Tests for Step 2 lagging logic (no row drop)."""

from __future__ import annotations

import pandas as pd

from modelPreparation.config.base import ModelPreparationConfig
from modelPreparation.steps.lag import run_lag


def test_lag_shifts_caution_features_without_dropping_rows(tmp_path):
    df = pd.DataFrame(
        {
            "country": ["USA", "USA", "USA", "DEU", "DEU", "DEU"],
            "year": [2000, 2001, 2002, 2000, 2001, 2002],
            "population_total": [100, 110, 120, 80, 81, 82],
            "agricultural_land_of_land_area": [50.0, 51.0, 52.0, 40.0, 40.5, 41.0],
            "optimistic": [1, 2, 3, 1, 2, 3],
            "linear": [1, 2, 3, 1, 2, 3],
            "pessimistic": [1, 2, 3, 1, 2, 3],
        }
    )

    config = ModelPreparationConfig(data_dir=tmp_path, output_dir=tmp_path / "out")

    lagged, meta = run_lag(df, config)

    # No rows dropped (V2 behavior)
    assert len(lagged) == 6
    assert meta["lagged_features_in_place"] == ["population_total"]

    # Column names unchanged (no _lag1 suffix)
    assert all(not col.endswith("_lag1") for col in lagged.columns)

    lagged = lagged.sort_values(["country", "year"]).reset_index(drop=True)

    # First row per country: lagged feature is NaN
    usa_2000 = lagged[(lagged["country"] == "USA") & (lagged["year"] == 2000)].iloc[0]
    deu_2000 = lagged[(lagged["country"] == "DEU") & (lagged["year"] == 2000)].iloc[0]
    assert pd.isna(usa_2000["population_total"])
    assert pd.isna(deu_2000["population_total"])

    # Caution feature lagged in place for subsequent rows
    usa_2001 = lagged[(lagged["country"] == "USA") & (lagged["year"] == 2001)].iloc[0]
    usa_2002 = lagged[(lagged["country"] == "USA") & (lagged["year"] == 2002)].iloc[0]
    deu_2001 = lagged[(lagged["country"] == "DEU") & (lagged["year"] == 2001)].iloc[0]

    assert usa_2001["population_total"] == 100
    assert usa_2002["population_total"] == 110
    assert deu_2001["population_total"] == 80

    # SAFE feature is unchanged
    assert usa_2001["agricultural_land_of_land_area"] == 51.0
    assert usa_2002["agricultural_land_of_land_area"] == 52.0
