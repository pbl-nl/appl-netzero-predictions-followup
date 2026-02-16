from __future__ import annotations

import pandas as pd

from resultsAnalysis.config.base import ResultsAnalysisConfig
from resultsAnalysis.core.protocols import RankingOutputs
from resultsAnalysis.steps.s05_uncertainty import analyze_uncertainty


def test_uncertainty_calibration_and_flags(tmp_path):
    project = tmp_path
    mt = project / "data" / "model_training"
    mr = project / "data" / "model_ready" / "2024"
    combo = mt / "2024" / "linear" / "AlgoA" / "predictions"
    combo.mkdir(parents=True)
    mr.mkdir(parents=True)

    pd.DataFrame(
        {
            "country": ["USA", "CAN"],
            "year": [2024, 2024],
            "linear": [10.0, 20.0],
        }
    ).to_csv(mr / "pre_split_data.csv", index=False)

    per_run = pd.DataFrame(
        [
            {"run_id": 0, "country": "USA", "year": 2024, "rce_pred": 9.0},
            {"run_id": 1, "country": "USA", "year": 2024, "rce_pred": 11.0},
            {"run_id": 0, "country": "CAN", "year": 2024, "rce_pred": 18.0},
            {"run_id": 1, "country": "CAN", "year": 2024, "rce_pred": 22.0},
        ]
    )
    per_run.to_csv(combo / "per_run.csv", index=False)

    nz = pd.DataFrame(
        [
            {"country": "USA", "nz_year_pred": 2050},
            {"country": "USA", "nz_year_pred": 2085},
            {"country": "CAN", "nz_year_pred": 2060},
            {"country": "CAN", "nz_year_pred": 2062},
        ]
    )
    nz.to_csv(combo / "netzero_per_run.csv", index=False)

    ranking = RankingOutputs(
        algorithm_ranking=pd.DataFrame(
            [{"Algorithm": "AlgoA", "Split": "2024", "Scenario": "linear"}]
        ),
        ranking_by_split=pd.DataFrame(),
        ranking_by_scenario=pd.DataFrame(),
        best_combo_by_scenario=pd.DataFrame(
            [{"Scenario": "linear", "Split": "2024", "Algorithm": "AlgoA"}]
        ),
    )

    cfg = ResultsAnalysisConfig(project_root=project)
    out = analyze_uncertainty(cfg, ranking)

    assert not out.calibration_report.empty
    cov = float(out.calibration_report.iloc[0]["coverage_90"])
    assert 0.0 <= cov <= 1.0

    assert not out.stability_by_country.empty
    usa_flag = bool(out.stability_by_country[out.stability_by_country["country"] == "USA"]["high_uncertainty"].iloc[0])
    assert usa_flag is True
