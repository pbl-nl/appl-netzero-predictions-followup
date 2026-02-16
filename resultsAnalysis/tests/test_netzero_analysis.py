from __future__ import annotations

import pandas as pd

from resultsAnalysis.config.base import ResultsAnalysisConfig
from resultsAnalysis.core.protocols import RankingOutputs
from resultsAnalysis.steps.s02_netzero_analysis import analyze_netzero


def _ranking_outputs() -> RankingOutputs:
    empty = pd.DataFrame()
    best = pd.DataFrame(
        [{"Scenario": "linear", "Split": "2024", "Algorithm": "AlgoX", "RMSE_mean": 1.0, "Within_RMSE": 1.0, "NZ_MAE": 1.0}]
    )
    return RankingOutputs(
        algorithm_ranking=empty,
        ranking_by_split=empty,
        ranking_by_scenario=empty,
        best_combo_by_scenario=best,
    )


def test_netzero_categories_with_status_precedence(tmp_path):
    project = tmp_path
    mt = project / "data" / "model_training"
    (mt / "2024" / "linear" / "AlgoX" / "predictions").mkdir(parents=True)

    nz = pd.DataFrame(
        [
            {"Split": "2024", "Scenario": "linear", "Algorithm": "AlgoX", "country": "USA", "nz_year_median": 2050, "pledge_year": 2050, "nz_year_p5": 2048, "nz_year_p95": 2052, "nz_vs_pledge_diff": 0},
            {"Split": "2024", "Scenario": "linear", "Algorithm": "AlgoX", "country": "CAN", "nz_year_median": 2040, "pledge_year": 2050, "nz_year_p5": 2038, "nz_year_p95": 2043, "nz_vs_pledge_diff": -10},
            {"Split": "2024", "Scenario": "linear", "Algorithm": "AlgoX", "country": "MEX", "nz_year_median": 2060, "pledge_year": 2050, "nz_year_p5": 2058, "nz_year_p95": 2064, "nz_vs_pledge_diff": 10},
            {"Split": "2024", "Scenario": "linear", "Algorithm": "AlgoX", "country": "BRA", "nz_year_median": 2070, "pledge_year": 2050, "nz_year_p5": 2068, "nz_year_p95": 2075, "nz_vs_pledge_diff": 20},
            {"Split": "2024", "Scenario": "linear", "Algorithm": "AlgoX", "country": "ARG", "nz_year_median": 2020, "pledge_year": 2050, "nz_year_p5": 2018, "nz_year_p95": 2022, "nz_vs_pledge_diff": -30},
            {"Split": "2024", "Scenario": "linear", "Algorithm": "AlgoX", "country": "CHL", "nz_year_median": 2200, "pledge_year": 2050, "nz_year_p5": 2190, "nz_year_p95": 2200, "nz_vs_pledge_diff": 150},
        ]
    )
    nz.to_csv(mt / "netzero_predictions.csv", index=False)

    per_run = pd.DataFrame(
        [
            {"country": "USA", "status": "ok"},
            {"country": "CAN", "status": "ok"},
            {"country": "MEX", "status": "ok"},
            {"country": "BRA", "status": "ok"},
            {"country": "ARG", "status": "past"},
            {"country": "CHL", "status": "not_converging"},
        ]
    )
    per_run.to_csv(mt / "2024" / "linear" / "AlgoX" / "predictions" / "netzero_per_run.csv", index=False)

    cfg = ResultsAnalysisConfig(project_root=project)
    out = analyze_netzero(cfg, _ranking_outputs())

    cats = out.netzero_country_summary.set_index("country")["category"].astype(str).to_dict()
    assert cats["USA"] == "On track"
    assert cats["CAN"] == "Ahead of pledge"
    assert cats["MEX"] == "Behind pledge"
    assert cats["BRA"] == "Far behind pledge"
    assert cats["ARG"] == "Already past due"
    assert cats["CHL"] == "Not converging"
