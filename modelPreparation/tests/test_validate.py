"""Tests for fold output validation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from modelPreparation.validate import validate_fold_output


def test_validate_catches_injected_failures():
    X_train_raw = pd.DataFrame(
        {
            "country": ["USA", "USA"],
            "year": [2020, 2021],
            "f1": [1.0, np.nan],
        }
    )
    X_train_processed = pd.DataFrame(
        {
            "f1": [0.0, np.nan],
            "total_ghg_emissions": [1.0, 2.0],
        }
    )

    X_valid_raw = pd.DataFrame({"country": ["USA"], "year": [2025], "f1": [np.nan]})
    X_valid_processed = pd.DataFrame({"f1": [1.5], "total_ghg_emissions": [3.0]})

    report = validate_fold_output(
        X_train_raw=X_train_raw,
        X_train_processed=X_train_processed,
        X_valid_raw=X_valid_raw,
        X_valid_processed=X_valid_processed,
        y_train_processed=pd.Series([0.0, 0.5]),
        danger_features=["total_ghg_emissions"],
        expected_counts={"train": 2, "valid": 1},
        train_year_max=2020,
        valid_year_range=(2021, 2023),
    )

    assert report.passed is False
    assert any("Non-finite values" in e for e in report.errors)
    assert any("DANGER features" in e for e in report.errors)
    assert any("Train split contains year" in e for e in report.errors)
    assert any("Valid split contains year" in e for e in report.errors)


def test_validate_passes_clean_data():
    X_train_raw = pd.DataFrame(
        {
            "country": ["USA", "USA"],
            "year": [2019, 2020],
            "f1": [1.0, 2.0],
        }
    )
    X_train_processed = pd.DataFrame({"f1": [0.0, 1.0]})

    X_valid_raw = pd.DataFrame({"country": ["USA"], "year": [2021], "f1": [2.5]})
    X_valid_processed = pd.DataFrame({"f1": [1.5]})

    report = validate_fold_output(
        X_train_raw=X_train_raw,
        X_train_processed=X_train_processed,
        X_valid_raw=X_valid_raw,
        X_valid_processed=X_valid_processed,
        y_train_processed=pd.Series([-0.2, 0.2]),
        danger_features=["total_ghg_emissions"],
        expected_counts={"train": 2, "valid": 1},
        train_year_max=2020,
        valid_year_range=(2021, 2023),
    )

    assert report.passed is True
    assert report.errors == []


def test_validate_warns_on_high_validation_imputation():
    X_train_raw = pd.DataFrame(
        {
            "country": ["USA", "USA"],
            "year": [2019, 2020],
            "f1": [1.0, np.nan],
        }
    )
    X_train_processed = pd.DataFrame({"f1": [1.0, 1.0]})

    X_valid_raw = pd.DataFrame(
        {
            "country": ["USA", "USA"],
            "year": [2021, 2022],
            "f1": [np.nan, np.nan],
        }
    )
    X_valid_processed = pd.DataFrame({"f1": [1.0, 1.0]})

    report = validate_fold_output(
        X_train_raw=X_train_raw,
        X_train_processed=X_train_processed,
        X_valid_raw=X_valid_raw,
        X_valid_processed=X_valid_processed,
        y_train_processed=pd.Series([0.0, 0.1]),
        high_imputation_threshold=0.50,
    )

    assert report.passed is True
    assert any("High validation imputation coverage" in w for w in report.warnings)


def test_validate_handles_multi_target_dataframe():
    X_train_raw = pd.DataFrame(
        {"country": ["USA", "USA", "USA"], "year": [2019, 2020, 2021], "f1": [1.0, 2.0, 3.0]}
    )
    X_train_processed = pd.DataFrame({"f1": [0.0, 1.0, 2.0]})

    y_train = pd.DataFrame({
        "optimistic": [0.5, 1.0, 1.5],
        "pessimistic": [0.3, 0.8, 1.3],
    })

    report = validate_fold_output(
        X_train_raw=X_train_raw,
        X_train_processed=X_train_processed,
        y_train_processed=y_train,
    )

    assert report.passed is True
    assert "target_skewness_optimistic" in report.metrics
    assert "target_skewness_pessimistic" in report.metrics
