"""Tests for Step 1 selection logic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from modelPreparation.config.base import ModelPreparationConfig
from modelPreparation.steps.select import run_select


def test_select_drops_danger_and_structural_features_and_excludes_countries(tmp_path):
    df = pd.DataFrame(
        {
            "country": ["USA", "USA", "USA", "USA", "BTN", "BTN"],
            "year": [2000, 2001, 2002, 2003, 2000, 2001],
            "coal_rents_of_gdp": [0.0, 0.0, 4.0, 0.0, 1.0, 2.0],
            "total_ghg_emissions": [10, 11, 12, 13, 1, 1],
            "non_co2_emissions": [4, 4, 4, 4, 1, 1],
            "total_hard_to_abate_emissions": [3, 3, 3, 3, 1, 1],
            "ghg_growth": [0.1, 0.2, 0.1, 0.0, 0.1, 0.2],
            "annual_deforestation_of_change": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "optimistic": [1.0, 2.0, 3.0, 4.0, 0.1, 0.2],
            "linear": [1.0, 2.0, 3.0, 4.0, 0.1, 0.2],
            "pessimistic": [1.0, 2.0, 3.0, 4.0, 0.1, 0.2],
        }
    )

    config = ModelPreparationConfig(
        data_dir=tmp_path,
        output_dir=tmp_path / "out",
        leading_zero_as_missing_cols=["coal_rents_of_gdp"],
    )

    selected, meta = run_select(df, config)

    assert selected["country"].nunique() == 1
    assert selected["country"].unique().tolist() == ["USA"]

    for col in [
        "total_ghg_emissions",
        "non_co2_emissions",
        "total_hard_to_abate_emissions",
        "ghg_growth",
        "annual_deforestation_of_change",
    ]:
        assert col not in selected.columns

    usa = selected.sort_values(["country", "year"]).reset_index(drop=True)
    assert np.isnan(usa.loc[0, "coal_rents_of_gdp"])
    assert np.isnan(usa.loc[1, "coal_rents_of_gdp"])
    assert usa.loc[2, "coal_rents_of_gdp"] == 4.0
    assert usa.loc[3, "coal_rents_of_gdp"] == 0.0

    assert meta["excluded_negative_rce_countries"] == 1
    assert "total_ghg_emissions" in meta["dropped_global_features"]
    assert meta["leading_zero_converted"]["coal_rents_of_gdp"] == 2
