"""Tests for enforced country-year panel behavior in merge step."""

from __future__ import annotations

import pandas as pd

from dataPreparation.config.merge import MergeConfig
from dataPreparation.steps.merge import run_merge


def test_run_merge_enforces_full_panel_and_preserves_nans(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    worldbank = pd.DataFrame({
        "country.value": [
            "United States",
            "United States",
            "United States",
            "Germany",
            "Germany",
            "Germany",
        ],
        "countryiso3code": ["USA", "USA", "USA", "DEU", "DEU", "DEU"],
        "date": [1959, 1960, 1962, 1961, 1962, 1963],
        "indicator.value": ["Total GHG Emissions"] * 6,
        "value": [90.0, 100.0, 110.0, 80.0, 85.0, 88.0],
    })
    worldbank.to_csv(data_dir / "worldbank_indicators.csv", index=False)

    config = MergeConfig(
        data_dir=data_dir,
        panel_year_start=1960,
        panel_year_end=1962,
        enforce_full_panel=True,
        drop_outside_panel_years=True,
    )
    result = run_merge(config)
    assert result.success

    merged = pd.read_csv(data_dir / "merged_indicators.csv")
    assert not merged.duplicated(["country", "year"]).any()
    assert merged["year"].min() == 1960
    assert merged["year"].max() == 1962
    assert len(merged) == 6  # 2 countries * 3 years

    usa_1961 = merged[(merged["country"] == "USA") & (merged["year"] == 1961)]
    assert len(usa_1961) == 1
    assert pd.isna(usa_1961.iloc[0]["total_ghg_emissions"])
