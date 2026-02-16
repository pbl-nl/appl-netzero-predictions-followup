"""Tests for Step 3 split and feature-list logic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from modelPreparation.config.base import ModelPreparationConfig
from modelPreparation.steps.split import run_split


def _build_df() -> pd.DataFrame:
    rows = []
    for country in ["USA", "DEU", "FRA", "JPN"]:
        for year in range(2018, 2025):
            rows.append(
                {
                    "country": country,
                    "year": year,
                    "gdp_per_capita_constant_2015_us": float(100 + year),
                    "coal_rents_of_gdp": np.nan if year in {2021, 2022, 2023} else 2.0,
                    "inform_risk_index": float(5 + (year - 2018) * 0.1),
                    "feature_all_missing_2024": np.nan if year == 2024 else 1.0,
                    "optimistic": float(year - 2000),
                    "linear": float(year - 2000),
                    "pessimistic": float(year - 2000),
                }
            )
    return pd.DataFrame(rows)


def test_panel_split_partitions_and_feature_drops(tmp_path):
    df = _build_df()
    config = ModelPreparationConfig(data_dir=tmp_path, output_dir=tmp_path / "out", random_seed=7)

    split = run_split(df, split_name="1990_2024", config=config)

    assert split.row_counts == {"train": 12, "valid": 12, "forecast": 4}

    train_years = {year for _, year in split.partitions["train"]}
    valid_years = {year for _, year in split.partitions["valid"]}
    forecast_years = {year for _, year in split.partitions["forecast"]}

    assert train_years == {2018, 2019, 2020}
    assert valid_years == {2021, 2022, 2023}
    assert forecast_years == {2024}

    train_keys = {tuple(x) for x in split.partitions["train"]}
    valid_keys = {tuple(x) for x in split.partitions["valid"]}
    forecast_keys = {tuple(x) for x in split.partitions["forecast"]}

    assert train_keys.isdisjoint(valid_keys)
    assert train_keys.isdisjoint(forecast_keys)
    assert valid_keys.isdisjoint(forecast_keys)

    assert "coal_rents_of_gdp" in split.dropped_split_features
    assert "inform_risk_index" in split.dropped_split_features
    assert "gdp_per_capita_constant_2015_us" in split.feature_columns


def test_snapshot_split_seed_reproducibility_and_missing_feature_drop(tmp_path):
    df = _build_df()
    config = ModelPreparationConfig(data_dir=tmp_path, output_dir=tmp_path / "out", random_seed=13)

    split_a = run_split(df, split_name="2024", config=config)
    split_b = run_split(df, split_name="2024", config=config)

    assert split_a.partitions == split_b.partitions
    assert split_a.row_counts["forecast"] == 0
    assert split_a.row_counts["train"] + split_a.row_counts["valid"] == 4

    assert "feature_all_missing_2024" in split_a.dropped_split_features
    assert "inform_risk_index" not in split_a.dropped_split_features
