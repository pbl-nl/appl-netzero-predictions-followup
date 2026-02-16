"""Tests for baseline models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from modelTraining.baselines import run_baselines
from modelTraining.config.base import ModelTrainingConfig


def _build_split_df() -> tuple[pd.DataFrame, dict[str, list[list[object]]], list[str]]:
    rows = []
    for country, base in [("AAA", 10.0), ("BBB", 20.0)]:
        rows.append(
            {
                "country": country,
                "year": 2020,
                "feat": base,
                "linear": base,
                "optimistic": base,
                "pessimistic": base,
            }
        )
        rows.append(
            {
                "country": country,
                "year": 2021,
                "feat": base + 1.0,
                "linear": base,
                "optimistic": base,
                "pessimistic": base,
            }
        )
        rows.append(
            {
                "country": country,
                "year": 2024,
                "feat": base + 2.0,
                "linear": base,
                "optimistic": base,
                "pessimistic": base,
            }
        )

    df = pd.DataFrame(rows)
    partitions = {
        "train": [["AAA", 2020], ["BBB", 2020]],
        "valid": [["AAA", 2021], ["BBB", 2021]],
        "forecast": [["AAA", 2024], ["BBB", 2024]],
    }
    return df, partitions, ["feat"]


def test_baseline_metrics_and_alpha_selection(tmp_path):
    df, partitions, features = _build_split_df()
    config = ModelTrainingConfig(
        output_dir=tmp_path / "out",
        model_ready_dir=tmp_path / "mr",
        data_dir=tmp_path / "data",
        algorithms=["RandomForest"],
        splits=["1990_2024"],
        scenarios=["linear"],
        n_runs=1,
        n_jobs=1,
    )

    result = run_baselines(
        split_df=df,
        partitions=partitions,
        feature_columns=features,
        split="1990_2024",
        scenario="linear",
        config=config,
    )

    got = set(result.metrics["Algorithm"].tolist())
    assert got == {"country_mean", "global_mean", "ridge", "lasso"}

    country_rmse = float(result.metrics[result.metrics["Algorithm"] == "country_mean"]["RMSE_log"].iloc[0])
    global_rmse = float(result.metrics[result.metrics["Algorithm"] == "global_mean"]["RMSE_log"].iloc[0])
    assert np.isclose(country_rmse, 0.0)
    assert global_rmse > 0.0

    ridge_alpha = float(result.metrics[result.metrics["Algorithm"] == "ridge"]["best_alpha"].iloc[0])
    lasso_alpha = float(result.metrics[result.metrics["Algorithm"] == "lasso"]["best_alpha"].iloc[0])
    assert ridge_alpha in config.alpha_grid
    assert lasso_alpha in config.alpha_grid
