"""Tests for ModelPreparationPipeline (multi-target, sklearn-compatible)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modelPreparation.config.base import ModelPreparationConfig
from modelPreparation.pipeline import ModelPreparationPipeline


def test_pipeline_fit_transform_and_inverse_target_roundtrip():
    X_train = pd.DataFrame(
        {
            "country": ["USA", "USA", "DEU", "DEU"],
            "year": [2018, 2019, 2018, 2019],
            "gdp_per_capita_constant_2015_us": [10000.0, np.nan, 9000.0, 9100.0],
            "population_total": [300.0, 301.0, np.nan, 82.0],
            "population_density_people_per_sq_km_of_land_area": [10.0, np.nan, 20.0, 21.0],
            "energy_intensity_mj_per_usd_ppp": [5.0, 5.1, np.nan, 4.0],
            "coal_rents_of_gdp": [0.1, np.nan, 0.2, 0.3],
            "average_supply_of_protein_of_animal_origin_g_cap_day_3_year_average": [20.0, 21.0, np.nan, 19.0],
            "rsf_press_freedom_index": [80.0, 79.5, 75.0, np.nan],
        }
    )
    y_train = pd.DataFrame({
        "optimistic": [0.0, 1.0, 2.0, 3.0],
        "linear": [-2.0, 0.0, 1.0, 3.0],
        "pessimistic": [-5.0, -1.0, 0.0, 2.0],
    })

    X_valid = pd.DataFrame(
        {
            "country": ["USA", "DEU", "FRA"],
            "year": [2021, 2021, 2021],
            "gdp_per_capita_constant_2015_us": [np.nan, 9300.0, np.nan],
            "population_total": [302.0, np.nan, np.nan],
            "population_density_people_per_sq_km_of_land_area": [11.0, np.nan, np.nan],
            "energy_intensity_mj_per_usd_ppp": [5.2, np.nan, np.nan],
            "coal_rents_of_gdp": [np.nan, 0.4, np.nan],
            "average_supply_of_protein_of_animal_origin_g_cap_day_3_year_average": [22.0, np.nan, np.nan],
            "rsf_press_freedom_index": [78.0, np.nan, np.nan],
        }
    )
    y_valid = pd.DataFrame({
        "optimistic": [1.0, 2.0, 3.0],
        "linear": [1.0, 2.0, 3.0],
        "pessimistic": [1.0, 2.0, 3.0],
    })

    config = ModelPreparationConfig(scaling_mode="standard")
    pipeline = ModelPreparationPipeline(config)

    X_train_proc, y_train_proc = pipeline.fit_transform(X_train, y_train)
    X_valid_proc, y_valid_proc = pipeline.transform(X_valid, y_valid)

    # Per-target shift constants exist
    assert len(pipeline.target_shift_constants_) == 3
    assert all(c > 0 for c in pipeline.target_shift_constants_.values())

    assert X_train_proc.shape[0] == 4
    assert X_valid_proc.shape[0] == 3

    assert np.isfinite(X_train_proc.to_numpy(dtype=float)).all()
    assert np.isfinite(X_valid_proc.to_numpy(dtype=float)).all()
    assert np.isfinite(y_train_proc.to_numpy(dtype=float)).all()
    assert np.isfinite(y_valid_proc.to_numpy(dtype=float)).all()

    # Inverse roundtrip per target
    for col in y_train.columns:
        y_roundtrip = pipeline.inverse_transform_target(y_train_proc[col], target_name=col)
        np.testing.assert_allclose(
            y_roundtrip, y_train[col].to_numpy(dtype=float), rtol=1e-6, atol=1e-6
        )


def test_pipeline_uses_train_only_median_for_unseen_country_missing_values():
    X_train = pd.DataFrame(
        {
            "country": ["USA", "USA", "DEU", "DEU"],
            "year": [2018, 2019, 2018, 2019],
            "test_feature": [1.0, np.nan, 2.0, np.nan],
        }
    )
    y_train = pd.DataFrame({"target": [1.0, 2.0, 3.0, 4.0]})

    X_valid = pd.DataFrame(
        {
            "country": ["NEW", "NEW"],
            "year": [2021, 2022],
            "test_feature": [np.nan, 999.0],
        }
    )

    config = ModelPreparationConfig(
        scaling_mode="none",
        log_features=[],
        log1p_features=[],
    )
    pipeline = ModelPreparationPipeline(config)

    pipeline.fit(X_train, y_train)
    X_valid_proc, _ = pipeline.transform(X_valid)

    # Median from train values [1.0, 2.0] => 1.5 should fill first NEW row
    assert X_valid_proc.iloc[0]["test_feature"] == pytest.approx(1.5)
    assert X_valid_proc.iloc[1]["test_feature"] == pytest.approx(999.0)


def test_pipeline_multi_target_shift_constants():
    X_train = pd.DataFrame(
        {
            "country": ["USA", "USA"],
            "year": [2018, 2019],
            "feat": [1.0, 2.0],
        }
    )
    y_train = pd.DataFrame({
        "optimistic": [5.0, 6.0],
        "pessimistic": [-10.0, -8.0],
    })

    config = ModelPreparationConfig(
        scaling_mode="none",
        log_features=[],
        log1p_features=[],
    )
    pipeline = ModelPreparationPipeline(config)
    pipeline.fit(X_train, y_train)

    # optimistic min=5.0, c = max(1.0, -5.0+1.0) = 1.0
    assert pipeline.target_shift_constants_["optimistic"] == pytest.approx(1.0)
    # pessimistic min=-10.0, c = max(1.0, 10.0+1.0) = 11.0
    assert pipeline.target_shift_constants_["pessimistic"] == pytest.approx(11.0)


def test_pipeline_fit_on_empty_data_raises_clean_error():
    pipeline = ModelPreparationPipeline()
    with pytest.raises(ValueError, match="empty"):
        pipeline.fit(pd.DataFrame(), pd.DataFrame())


def test_pipeline_inverse_transform_unknown_target_raises():
    X_train = pd.DataFrame({"country": ["USA"], "year": [2020], "feat": [1.0]})
    y_train = pd.DataFrame({"optimistic": [1.0]})

    config = ModelPreparationConfig(
        scaling_mode="none", log_features=[], log1p_features=[]
    )
    pipeline = ModelPreparationPipeline(config)
    pipeline.fit(X_train, y_train)

    with pytest.raises(ValueError, match="Unknown target"):
        pipeline.inverse_transform_target(np.array([0.5]), target_name="nonexistent")
