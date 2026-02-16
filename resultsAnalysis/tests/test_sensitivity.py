from __future__ import annotations

import pandas as pd

from resultsAnalysis.config.base import ResultsAnalysisConfig
from resultsAnalysis.core.protocols import RankingOutputs
from resultsAnalysis.steps.s03_sensitivity import analyze_sensitivity


def test_sensitivity_spreads(tmp_path):
    project = tmp_path
    mt = project / "data" / "model_training"
    mt.mkdir(parents=True)

    nz = pd.DataFrame(
        [
            # Scenario references (best combo per scenario)
            {"Split": "2024", "Scenario": "optimistic", "Algorithm": "A", "country": "USA", "nz_year_median": 2040},
            {"Split": "2024", "Scenario": "linear", "Algorithm": "A", "country": "USA", "nz_year_median": 2050},
            {"Split": "2024", "Scenario": "pessimistic", "Algorithm": "A", "country": "USA", "nz_year_median": 2060},
            # split spread data for linear using best algo A
            {"Split": "1990_2024", "Scenario": "linear", "Algorithm": "A", "country": "USA", "nz_year_median": 2048},
            {"Split": "2010_2024", "Scenario": "linear", "Algorithm": "A", "country": "USA", "nz_year_median": 2052},
            # algorithm spread data for linear, fixed split=2024
            {"Split": "2024", "Scenario": "linear", "Algorithm": "B", "country": "USA", "nz_year_median": 2044},
        ]
    )
    nz.to_csv(mt / "netzero_predictions.csv", index=False)

    algo_rank = pd.DataFrame(
        [
            {"Algorithm": "A", "Split": "2024", "Scenario": "optimistic", "RMSE_mean": 1.0, "Within_RMSE": 1.0, "NZ_MAE": 1.0},
            {"Algorithm": "A", "Split": "2024", "Scenario": "linear", "RMSE_mean": 1.0, "Within_RMSE": 1.0, "NZ_MAE": 1.0},
            {"Algorithm": "A", "Split": "2024", "Scenario": "pessimistic", "RMSE_mean": 1.0, "Within_RMSE": 1.0, "NZ_MAE": 1.0},
        ]
    )
    best_combo = pd.DataFrame(
        [
            {"Scenario": "optimistic", "Split": "2024", "Algorithm": "A", "RMSE_mean": 1.0, "Within_RMSE": 1.0, "NZ_MAE": 1.0},
            {"Scenario": "linear", "Split": "2024", "Algorithm": "A", "RMSE_mean": 1.0, "Within_RMSE": 1.0, "NZ_MAE": 1.0},
            {"Scenario": "pessimistic", "Split": "2024", "Algorithm": "A", "RMSE_mean": 1.0, "Within_RMSE": 1.0, "NZ_MAE": 1.0},
        ]
    )
    ranking = RankingOutputs(
        algorithm_ranking=algo_rank,
        ranking_by_split=pd.DataFrame(),
        ranking_by_scenario=pd.DataFrame(),
        best_combo_by_scenario=best_combo,
    )

    cfg = ResultsAnalysisConfig(project_root=project)
    out = analyze_sensitivity(cfg, ranking)

    scen_spread = float(out.sensitivity_scenario[out.sensitivity_scenario["country"] == "USA"]["scenario_spread"].iloc[0])
    assert scen_spread == 20.0

    split_spread = float(out.sensitivity_split[out.sensitivity_split["country"] == "USA"]["split_spread"].iloc[0])
    assert split_spread == 4.0

    algo_spread = float(
        out.sensitivity_algorithm[
            (out.sensitivity_algorithm["country"] == "USA")
            & (out.sensitivity_algorithm["scenario"] == "linear")
        ]["algo_spread"].iloc[0]
    )
    assert algo_spread == 6.0
