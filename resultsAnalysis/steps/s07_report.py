"""Standalone Markdown report generator for resultsAnalysis."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from resultsAnalysis.config.base import ResultsAnalysisConfig
from resultsAnalysis.core.protocols import (
    FeatureImportanceOutputs,
    ModelSelectionOutputs,
    NetZeroOutputs,
    RankingOutputs,
    SensitivityOutputs,
    UncertaintyOutputs,
    VisualizationOutputs,
)

STEP_NAME = "S07_report"


def _table_block(df: pd.DataFrame, n: int = 20) -> str:
    if df is None or df.empty:
        return "_No data available._\n"
    show = df.head(n).copy()
    return "```text\n" + show.to_string(index=False) + "\n```\n"


def generate_report(
    config: ResultsAnalysisConfig,
    ranking: RankingOutputs,
    netzero: NetZeroOutputs,
    sensitivity: SensitivityOutputs,
    feature: FeatureImportanceOutputs,
    uncertainty: UncertaintyOutputs,
    model_sel: ModelSelectionOutputs,
    visuals: VisualizationOutputs,
) -> Path:
    report_path = config.output_dir / "report.md"

    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("# Net-Zero Predictions: Results Report\n\n")
        fh.write(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n")

        fh.write("## 1. Algorithm Performance\n\n")
        fh.write("### Ranking Table\n")
        fh.write(_table_block(ranking.algorithm_ranking[["Algorithm", "Split", "Scenario", "RMSE_mean", "Within_RMSE", "NZ_MAE", "vs_Ridge_pct", "rank_in_split_scenario"]], n=30))
        fh.write("![RMSE grouped bars](figures/ranking_grouped_ci.png)\n\n")
        fh.write("![RMSE heatmap](figures/ranking_heatmap.png)\n\n")
        fh.write("[Interactive RMSE scatter](figures/ranking_scatter.html)\n\n")

        fh.write("## 2. Predicted Net-Zero Years\n\n")
        fh.write("### Country Summary (top 20)\n")
        fh.write(_table_block(netzero.netzero_country_summary[["scenario", "country", "region", "nz_year_median", "pledge_year", "nz_vs_pledge_diff", "category"]], n=20))
        fh.write("### Category Distribution\n")
        fh.write(_table_block(netzero.netzero_by_category, n=30))
        fh.write("![Pledge gap histogram](figures/netzero_gap_hist.png)\n\n")
        fh.write("![Top 10 ahead/behind](figures/netzero_top10_ahead_behind.png)\n\n")
        fh.write("![Region pledge gap](figures/region_pledge_gap.png)\n\n")
        fh.write("![World map](figures/world_choropleth.png)\n\n")
        fh.write("[Interactive world map](figures/world_choropleth.html)\n\n")
        fh.write("[Interactive country dot plot](figures/netzero_dot_ci.html)\n\n")

        fh.write("## 3. Sensitivity Analysis\n\n")
        fh.write("### Summary Statistics\n")
        fh.write(_table_block(sensitivity.sensitivity_summary, n=20))
        fh.write("![Sensitivity spread](figures/sensitivity_violin.png)\n\n")
        fh.write("[Interactive sensitivity overlay](figures/sensitivity_overlay.html)\n\n")

        fh.write("## 4. Feature Importance\n\n")
        fh.write("### Consensus Ranking\n")
        fh.write(_table_block(feature.feature_importance_consensus, n=25))
        fh.write("### Stability\n")
        fh.write(_table_block(feature.feature_stability_report, n=20))
        fh.write("![Feature bars](figures/feature_top15_by_algo.png)\n\n")
        fh.write("![Feature rank heatmap](figures/feature_heatmap.png)\n\n")
        if "feature_boxplot" in visuals.interactive_html_paths:
            fh.write("[Interactive feature boxplot](figures/feature_boxplot.html)\n\n")

        fh.write("## 5. Uncertainty & Calibration\n\n")
        fh.write("### Calibration Coverage\n")
        fh.write(_table_block(uncertainty.calibration_report, n=40))
        fh.write("### High Uncertainty Countries\n")
        fh.write(_table_block(uncertainty.high_uncertainty_flags, n=40))
        fh.write("![Calibration bars](figures/calibration_bars.png)\n\n")
        fh.write("![Interval width histogram](figures/interval_width_hist.png)\n\n")
        fh.write("[Interactive stability scatter](figures/stability_scatter.html)\n\n")

        fh.write("## 6. Final Model\n\n")
        for selected in model_sel.selected_models:
            fh.write(f"### Scenario: {selected.scenario}\n")
            fh.write(f"- Algorithm: `{selected.algorithm}`\n")
            fh.write(f"- Split: `{selected.split}`\n")
            fh.write(f"- Model artifact: `{selected.model_path.relative_to(config.output_dir.parent)}`\n")
            fh.write(f"- Predictions: `{selected.final_predictions_path.relative_to(config.output_dir.parent)}`\n")
            fh.write(f"- Config: `{selected.final_config_path.relative_to(config.output_dir.parent)}`\n")
            fh.write(f"- Rationale: `{selected.rationale_path.relative_to(config.output_dir.parent)}`\n\n")

        if not model_sel.stacked_vs_final.empty:
            fh.write("### Stacked Ensemble Comparison\n")
            fh.write(_table_block(model_sel.stacked_vs_final, n=30))
            fh.write("![Stacked vs final](figures/stacked_vs_final.png)\n\n")

    return report_path
