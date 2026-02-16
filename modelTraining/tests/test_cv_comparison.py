"""Tests for outer CV comparison module."""

from __future__ import annotations

import pandas as pd

from modelTraining.config.base import ModelTrainingConfig
from modelTraining.cv_comparison import run_outer_cv_comparison


def _panel_fixture():
    rows = []
    for country, base in [("AAA", 10.0), ("BBB", 20.0), ("CCC", 15.0)]:
        for year in [2018, 2019, 2020, 2021]:
            rows.append({
                "country": country,
                "year": year,
                "feat": float(year - 2017) + base,
                "linear": base + float(year - 2018),
                "optimistic": base + float(year - 2018),
                "pessimistic": base + float(year - 2018),
            })
    df = pd.DataFrame(rows)
    partitions = {
        "train": [[c, y] for c in ["AAA", "BBB", "CCC"] for y in [2018, 2019, 2020]],
        "valid": [[c, 2021] for c in ["AAA", "BBB", "CCC"]],
        "forecast": [],
    }
    return df, partitions, ["feat"]


def _snapshot_fixture():
    rows = []
    countries = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    for i, country in enumerate(countries):
        rows.append({
            "country": country,
            "year": 2024,
            "feat": float(i),
            "linear": float(10 + i),
            "optimistic": float(10 + i),
            "pessimistic": float(10 + i),
        })
    df = pd.DataFrame(rows)
    partitions = {
        "train": [[c, 2024] for c in countries[:4]],
        "valid": [[c, 2024] for c in countries[4:]],
        "forecast": [],
    }
    return df, partitions, ["feat"]


def test_panel_outer_cv_produces_results(tmp_path):
    df, partitions, features = _panel_fixture()
    config = ModelTrainingConfig(
        output_dir=tmp_path / "out",
        model_ready_dir=tmp_path / "mr",
        data_dir=tmp_path / "data",
        algorithms=["LinearRegression"],
        splits=["1990_2024"],
        scenarios=["linear"],
        n_runs=1,
        n_jobs=1,
        run_outer_cv=True,
        n_outer_cv_folds=2,
    )

    result = run_outer_cv_comparison(
        split="1990_2024",
        scenario="linear",
        split_df=df,
        partitions=partitions,
        feature_columns=features,
        config=config,
    )

    assert not result.empty
    assert "Algorithm" in result.columns
    assert "fold" in result.columns
    assert "RMSE_log" in result.columns


def test_snapshot_outer_cv_produces_results(tmp_path):
    df, partitions, features = _snapshot_fixture()
    config = ModelTrainingConfig(
        output_dir=tmp_path / "out",
        model_ready_dir=tmp_path / "mr",
        data_dir=tmp_path / "data",
        algorithms=["LinearRegression"],
        splits=["2024"],
        scenarios=["linear"],
        n_runs=1,
        n_jobs=1,
        run_outer_cv=True,
        n_outer_cv_folds=3,
    )

    result = run_outer_cv_comparison(
        split="2024",
        scenario="linear",
        split_df=df,
        partitions=partitions,
        feature_columns=features,
        config=config,
    )

    assert not result.empty
    assert set(result["Algorithm"].unique()) == {"LinearRegression"}
