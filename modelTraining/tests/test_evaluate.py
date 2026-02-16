"""Tests for evaluation summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from modelTraining.config.base import ModelTrainingConfig
from modelTraining.evaluate import build_algorithm_comparison, build_evaluation_row, summarize_run_metrics


def test_summarize_run_metrics_outputs_expected_fields():
    run_df = pd.DataFrame(
        {
            "run_id": [0, 1, 2],
            "RMSE_log": [1.0, 2.0, 3.0],
            "MAE_log": [0.5, 1.0, 1.5],
            "R2": [0.1, 0.2, 0.3],
            "Within_RMSE": [0.2, 0.3, 0.4],
            "NZ_MAE": [5.0, 6.0, 7.0],
        }
    )
    summary = summarize_run_metrics(run_df, confidence_level=0.90)
    assert np.isclose(summary["RMSE_log_mean"], 2.0)
    assert "RMSE_log_ci_low" in summary
    assert "NZ_MAE_std" in summary


def test_build_algorithm_comparison_uses_ridge_reference(tmp_path):
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

    run_df = pd.DataFrame(
        {
            "run_id": [0, 1],
            "RMSE_log": [0.8, 1.0],
            "MAE_log": [0.6, 0.7],
            "R2": [0.5, 0.6],
            "Within_RMSE": [0.4, 0.5],
            "NZ_MAE": [4.0, 5.0],
        }
    )
    eval_row = build_evaluation_row(
        algorithm="RandomForest",
        split="1990_2024",
        scenario="linear",
        run_metrics_df=run_df,
        config=config,
    )

    ridge = pd.DataFrame(
        {
            "Algorithm": ["ridge"],
            "Split": ["1990_2024"],
            "Scenario": ["linear"],
            "RMSE_log": [1.0],
        }
    )

    comp = build_algorithm_comparison(eval_row, ridge)
    assert comp.shape[0] == 1
    assert np.isclose(float(comp.iloc[0]["vs_Ridge"]), (0.9 - 1.0) / 1.0)
