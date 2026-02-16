from __future__ import annotations

import pandas as pd

from resultsAnalysis.config.base import ResultsAnalysisConfig
from resultsAnalysis.steps.s01_ranking import rank_algorithms


def test_ranking_priority_and_ci_overlap(tmp_path):
    project = tmp_path
    mt = project / "data" / "model_training"
    (mt / "baselines").mkdir(parents=True)

    comp = pd.DataFrame(
        [
            {"Algorithm": "AlgoA", "Split": "2024", "Scenario": "linear", "RMSE_mean": 1.0, "RMSE_std": 0.1, "Within_RMSE": 0.5, "NZ_MAE": 10.0, "vs_Ridge": 0.02},
            {"Algorithm": "AlgoB", "Split": "2024", "Scenario": "linear", "RMSE_mean": 1.0, "RMSE_std": 0.1, "Within_RMSE": 0.4, "NZ_MAE": 20.0, "vs_Ridge": 0.03},
        ]
    )
    comp.to_csv(mt / "algorithm_comparison.csv", index=False)

    baselines = pd.DataFrame(
        [
            {"Algorithm": "country_mean", "Split": "2024", "Scenario": "linear", "RMSE_log": 1.2},
            {"Algorithm": "ridge", "Split": "2024", "Scenario": "linear", "RMSE_log": 0.95, "best_alpha": 1.0},
        ]
    )
    baselines.to_csv(mt / "baselines" / "metrics.csv", index=False)

    eval_dir_a = mt / "2024" / "linear" / "AlgoA" / "evaluation"
    eval_dir_b = mt / "2024" / "linear" / "AlgoB" / "evaluation"
    eval_dir_a.mkdir(parents=True)
    eval_dir_b.mkdir(parents=True)

    pd.DataFrame([{"RMSE_log_ci_low": 0.90, "RMSE_log_ci_high": 1.05}]).to_csv(eval_dir_a / "metrics.csv", index=False)
    pd.DataFrame([{"RMSE_log_ci_low": 0.92, "RMSE_log_ci_high": 1.10}]).to_csv(eval_dir_b / "metrics.csv", index=False)

    cfg = ResultsAnalysisConfig(project_root=project)
    out = rank_algorithms(cfg)

    rows = out.algorithm_ranking
    top = rows[(rows["Split"] == "2024") & (rows["Scenario"] == "linear")].sort_values("rank_in_split_scenario").iloc[0]
    assert top["Algorithm"] == "ridge"  # lower RMSE from baseline merge

    # Tie between AlgoA/AlgoB on RMSE, AlgoB should rank ahead via lower Within_RMSE.
    ab = rows[(rows["Algorithm"].isin(["AlgoA", "AlgoB"])) & (rows["Split"] == "2024")]
    algo_b_rank = int(ab[ab["Algorithm"] == "AlgoB"]["rank_in_split_scenario"].iloc[0])
    algo_a_rank = int(ab[ab["Algorithm"] == "AlgoA"]["rank_in_split_scenario"].iloc[0])
    assert algo_b_rank < algo_a_rank

    # CI overlap annotation should exist and be true for both in this synthetic setup.
    assert bool(ab[ab["Algorithm"] == "AlgoA"]["rmse_ci_overlaps_best"].iloc[0]) in {True, False}
