from __future__ import annotations

"""Placeholder test for feature importance.

The full feature-importance tests require model artifacts (joblib/pt files)
that are expensive to synthesize in a unit test. This file validates the
module's public interface is importable and the empty-data path works.
"""

import pandas as pd

from resultsAnalysis.config.base import ResultsAnalysisConfig
from resultsAnalysis.core.protocols import RankingOutputs
from resultsAnalysis.steps.s04_feature_importance import analyze_feature_importance


def test_feature_importance_empty_candidates(tmp_path):
    project = tmp_path
    (project / "data" / "model_training").mkdir(parents=True)

    ranking = RankingOutputs(
        algorithm_ranking=pd.DataFrame(columns=["Algorithm", "Split", "Scenario"]),
        ranking_by_split=pd.DataFrame(),
        ranking_by_scenario=pd.DataFrame(),
        best_combo_by_scenario=pd.DataFrame(),
    )

    cfg = ResultsAnalysisConfig(project_root=project)
    out = analyze_feature_importance(cfg, ranking, mode="B")

    assert out.feature_importance_by_algorithm.empty
    assert out.feature_importance_consensus.empty
    assert out.feature_stability_report.empty
