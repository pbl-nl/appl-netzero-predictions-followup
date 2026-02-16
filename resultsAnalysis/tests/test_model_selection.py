from __future__ import annotations

import json

import pandas as pd

from resultsAnalysis.config.base import ResultsAnalysisConfig
from resultsAnalysis.core.protocols import RankingOutputs
from resultsAnalysis.steps.s06_model_selection import _choose_combo_for_scenario, _train_and_predict


def _ranking(df: pd.DataFrame) -> RankingOutputs:
    return RankingOutputs(
        algorithm_ranking=df,
        ranking_by_split=pd.DataFrame(),
        ranking_by_scenario=pd.DataFrame(),
        best_combo_by_scenario=pd.DataFrame(),
    )


def test_simplicity_thresholds_prefer_ridge_then_rf(tmp_path):
    cfg = ResultsAnalysisConfig(project_root=tmp_path)

    df_ridge = pd.DataFrame(
        [
            {"Algorithm": "XGBoost", "Split": "2024", "Scenario": "linear", "RMSE_mean": 1.0, "Within_RMSE": 1.0, "NZ_MAE": 1.0},
            {"Algorithm": "RandomForest", "Split": "2024", "Scenario": "linear", "RMSE_mean": 1.01, "Within_RMSE": 1.1, "NZ_MAE": 1.0},
            {"Algorithm": "ridge", "Split": "2024", "Scenario": "linear", "RMSE_mean": 1.04, "Within_RMSE": 1.2, "NZ_MAE": 1.0},
        ]
    )
    chosen = _choose_combo_for_scenario(cfg, _ranking(df_ridge), "linear")
    assert chosen.algorithm == "ridge"

    df_rf = pd.DataFrame(
        [
            {"Algorithm": "XGBoost", "Split": "2024", "Scenario": "optimistic", "RMSE_mean": 1.0, "Within_RMSE": 1.0, "NZ_MAE": 1.0},
            {"Algorithm": "RandomForest", "Split": "2024", "Scenario": "optimistic", "RMSE_mean": 1.01, "Within_RMSE": 1.2, "NZ_MAE": 1.0},
            {"Algorithm": "ridge", "Split": "2024", "Scenario": "optimistic", "RMSE_mean": 1.3, "Within_RMSE": 1.2, "NZ_MAE": 1.0},
        ]
    )
    chosen_rf = _choose_combo_for_scenario(cfg, _ranking(df_rf), "optimistic")
    assert chosen_rf.algorithm == "RandomForest"


def test_retrain_smoke_linear_regression(tmp_path):
    project = tmp_path
    cfg = ResultsAnalysisConfig(project_root=project)

    split_dir = project / "data" / "model_ready" / "2024"
    split_dir.mkdir(parents=True)

    df = pd.DataFrame(
        {
            "country": ["AAA", "BBB", "CCC", "DDD"],
            "year": [2024, 2024, 2024, 2024],
            "feat": [1.0, 2.0, 3.0, 4.0],
            "linear": [10.0, 11.0, 12.0, 13.0],
            "optimistic": [10.0, 11.0, 12.0, 13.0],
            "pessimistic": [10.0, 11.0, 12.0, 13.0],
        }
    )
    df.to_csv(split_dir / "pre_split_data.csv", index=False)

    payload_idx = {
        "partitions": {
            "train": [["AAA", 2024], ["BBB", 2024]],
            "valid": [["CCC", 2024]],
            "forecast": [["DDD", 2024]],
        }
    }
    (split_dir / "split_indices.json").write_text(json.dumps(payload_idx), encoding="utf-8")
    (split_dir / "feature_list.json").write_text(json.dumps({"feature_columns": ["feat"]}), encoding="utf-8")

    model_payload, per_run = _train_and_predict(
        config=cfg,
        split="2024",
        scenario="linear",
        algorithm="LinearRegression",
        best_params={},
    )

    assert "model" in model_payload
    assert not per_run.empty
    assert {"run_id", "country", "year", "rce_pred"}.issubset(set(per_run.columns))
