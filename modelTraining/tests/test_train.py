"""Tests for training loops."""

from __future__ import annotations

import pandas as pd

from modelTraining.config.base import ModelTrainingConfig
from modelTraining.train_panel import train_panel_combo
from modelTraining.train_snapshot import _train_snapshot_run, train_snapshot_combo
from modelTraining.utils import bootstrap_rows


def _snapshot_fixture():
    rows = []
    for i, country in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]):
        rows.append(
            {
                "country": country,
                "year": 2024,
                "feat": float(i),
                "linear": float(10 + i),
                "optimistic": float(10 + i),
                "pessimistic": float(10 + i),
            }
        )
    df = pd.DataFrame(rows)
    partitions = {
        "train": [["AAA", 2024], ["BBB", 2024], ["CCC", 2024], ["DDD", 2024]],
        "valid": [["EEE", 2024], ["FFF", 2024]],
        "forecast": [],
    }
    return df, partitions, ["feat"]


def _panel_fixture():
    rows = []
    for country, base in [("AAA", 10.0), ("BBB", 20.0)]:
        for year, value in [(2019, base), (2020, base + 1), (2021, base + 2), (2024, base + 3)]:
            rows.append(
                {
                    "country": country,
                    "year": year,
                    "feat": float(year - 2018),
                    "linear": value,
                    "optimistic": value,
                    "pessimistic": value,
                }
            )
    df = pd.DataFrame(rows)
    partitions = {
        "train": [["AAA", 2019], ["AAA", 2020], ["BBB", 2019], ["BBB", 2020]],
        "valid": [["AAA", 2021], ["BBB", 2021]],
        "forecast": [["AAA", 2024], ["BBB", 2024]],
    }
    return df, partitions, ["feat"]


def test_snapshot_training_runs_and_is_reproducible(tmp_path):
    df, partitions, features = _snapshot_fixture()
    config = ModelTrainingConfig(
        output_dir=tmp_path / "out",
        model_ready_dir=tmp_path / "mr",
        data_dir=tmp_path / "data",
        algorithms=["RandomForest"],
        splits=["2024"],
        scenarios=["linear"],
        n_runs=2,
        n_jobs=1,
    )
    config.param_grids["RandomForest"] = {
        "n_estimators": [10],
        "max_depth": [3],
        "min_samples_leaf": [1],
    }

    out = train_snapshot_combo(
        algorithm="RandomForest",
        split="2024",
        scenario="linear",
        split_df=df,
        partitions=partitions,
        feature_columns=features,
        config=config,
        combo_dir=tmp_path / "combo",
    )

    assert len(out.run_results) == 2
    assert all(r.success for r in out.run_results)
    assert not out.inference_predictions.empty

    run_a, _ = _train_snapshot_run(
        algorithm="RandomForest",
        split="2024",
        scenario="linear",
        run_id=0,
        split_df=df,
        partitions=partitions,
        feature_columns=features,
        config=config,
        combo_dir=tmp_path / "combo2",
    )
    run_b, _ = _train_snapshot_run(
        algorithm="RandomForest",
        split="2024",
        scenario="linear",
        run_id=0,
        split_df=df,
        partitions=partitions,
        feature_columns=features,
        config=config,
        combo_dir=tmp_path / "combo3",
    )
    assert run_a.success and run_b.success
    assert run_a.inference_predictions == run_b.inference_predictions


def test_lstm_is_skipped_on_snapshot(tmp_path):
    df, partitions, features = _snapshot_fixture()
    config = ModelTrainingConfig(
        output_dir=tmp_path / "out",
        model_ready_dir=tmp_path / "mr",
        data_dir=tmp_path / "data",
        algorithms=["LSTM"],
        splits=["2024"],
        scenarios=["linear"],
        n_runs=1,
        n_jobs=1,
    )

    result, _ = _train_snapshot_run(
        algorithm="LSTM",
        split="2024",
        scenario="linear",
        run_id=0,
        split_df=df,
        partitions=partitions,
        feature_columns=features,
        config=config,
        combo_dir=tmp_path / "combo",
    )

    assert not result.success
    assert "not supported" in (result.error or "")


def test_panel_linear_regression_training_smoke(tmp_path):
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
    )

    out = train_panel_combo(
        algorithm="LinearRegression",
        split="1990_2024",
        scenario="linear",
        split_df=df,
        partitions=partitions,
        feature_columns=features,
        config=config,
        combo_dir=tmp_path / "panel_combo",
    )

    assert len(out.run_results) == 1
    assert out.run_results[0].success
    assert not out.inference_predictions.empty


def test_snapshot_inference_contains_only_validation_countries(tmp_path):
    df, partitions, features = _snapshot_fixture()
    config = ModelTrainingConfig(
        output_dir=tmp_path / "out",
        model_ready_dir=tmp_path / "mr",
        data_dir=tmp_path / "data",
        algorithms=["RandomForest"],
        splits=["2024"],
        scenarios=["linear"],
        n_runs=1,
        n_jobs=1,
    )
    config.param_grids["RandomForest"] = {
        "n_estimators": [10],
        "max_depth": [3],
        "min_samples_leaf": [1],
    }

    out = train_snapshot_combo(
        algorithm="RandomForest",
        split="2024",
        scenario="linear",
        split_df=df,
        partitions=partitions,
        feature_columns=features,
        config=config,
        combo_dir=tmp_path / "combo_infer",
    )

    # Inference predictions should only contain validation countries (EEE, FFF).
    valid_countries = {"EEE", "FFF"}
    infer_countries = set(out.inference_predictions["country"].unique())
    assert infer_countries == valid_countries


def test_bootstrap_changes_with_seed():
    X = pd.DataFrame(
        {
            "country": ["A", "A", "B", "B"],
            "year": [2019, 2020, 2019, 2020],
            "feat": [1.0, 2.0, 3.0, 4.0],
        }
    )
    y = pd.Series([1.0, 2.0, 3.0, 4.0])

    X_a, _ = bootstrap_rows(X, y, seed=1)
    X_b, _ = bootstrap_rows(X, y, seed=2)
    assert not X_a.equals(X_b)
