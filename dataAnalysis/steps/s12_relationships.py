"""S12: Relationship Checks — Features vs Emissions vs Targets."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dataAnalysis.config.base import AnalysisConfig
from dataAnalysis.core.protocols import AnalysisResult
from dataAnalysis.utils.interpret import describe_correlation
from dataAnalysis.utils.io import save_figure, save_table

STEP_NAME = "S12_relationships"


def run(df: pd.DataFrame, config: AnalysisConfig) -> AnalysisResult:
    findings: list[str] = []
    tables: list = []
    figures: list = []

    features = [c for c in config.ALL_FEATURE_COLS if c in df.columns]
    targets = [c for c in config.TARGET_COLS if c in df.columns]

    if not features or not targets:
        findings.append("Insufficient feature/target columns for relationship analysis.")
        return AnalysisResult(
            success=True, step_name=STEP_NAME, split_name=config.split_name,
            findings=findings,
        )

    # --- Raw correlations ---
    corr_cols = targets + features
    raw_corr = df[corr_cols].corr(method="spearman")
    # Extract feature-target block
    raw_ft = raw_corr.loc[features, targets]
    raw_ft_melted = raw_ft.reset_index().melt(
        id_vars="index", var_name="target", value_name="correlation"
    )
    raw_ft_melted.columns = ["feature", "target", "correlation"]
    tables.append(save_table(raw_ft_melted, "correlations_raw", config))

    # Report strongest raw correlations
    strongest = raw_ft_melted.reindex(
        raw_ft_melted["correlation"].abs().sort_values(ascending=False).index
    ).head(5)
    findings.append("Strongest raw (Spearman) feature-target correlations:")
    for _, row in strongest.iterrows():
        findings.append(
            f"  - {row['feature']} vs {row['target']}: "
            f"r = {row['correlation']:.3f} ({describe_correlation(row['correlation'])})"
        )

    # --- Within-country demeaned correlations (panel only) ---
    if not config.is_cross_sectional:
        demeaned = df[["country"] + corr_cols].copy()
        for col in corr_cols:
            group_means = demeaned.groupby("country")[col].transform("mean")
            demeaned[col] = demeaned[col] - group_means

        dem_corr = demeaned[corr_cols].corr(method="spearman")
        dem_ft = dem_corr.loc[features, targets]
        dem_ft_melted = dem_ft.reset_index().melt(
            id_vars="index", var_name="target", value_name="correlation"
        )
        dem_ft_melted.columns = ["feature", "target", "correlation"]
        tables.append(save_table(dem_ft_melted, "correlations_demeaned", config))

        strongest_dem = dem_ft_melted.reindex(
            dem_ft_melted["correlation"].abs().sort_values(ascending=False).index
        ).head(5)
        findings.append("Strongest demeaned (within-country) feature-target correlations:")
        for _, row in strongest_dem.iterrows():
            findings.append(
                f"  - {row['feature']} vs {row['target']}: "
                f"r = {row['correlation']:.3f} ({describe_correlation(row['correlation'])})"
            )

        # Identify features where raw >> demeaned (confounded by country identity)
        comparison = raw_ft_melted.merge(
            dem_ft_melted, on=["feature", "target"], suffixes=("_raw", "_demeaned")
        )
        comparison["drop"] = (
            comparison["correlation_raw"].abs() - comparison["correlation_demeaned"].abs()
        )
        big_drops = comparison[comparison["drop"] > 0.3]
        if len(big_drops) > 0:
            findings.append(
                f"{len(big_drops)} feature-target pairs lose >0.3 correlation "
                "after demeaning — likely confounded by country identity."
            )
    else:
        findings.append(
            "Demeaned correlations skipped for cross-sectional split "
            "(no within-country variation)."
        )

    # Correlation heatmap figure
    try:
        fig, ax = plt.subplots(figsize=(12, 10))
        # Use top features by abs correlation with linear target
        if "linear" in raw_ft.columns:
            top_feat = (
                raw_ft["linear"].abs().sort_values(ascending=False).head(15).index.tolist()
            )
        else:
            top_feat = features[:15]

        plot_cols = targets + top_feat
        plot_corr = df[plot_cols].corr(method="spearman")
        mask = np.triu(np.ones_like(plot_corr, dtype=bool), k=1)
        sns.heatmap(
            plot_corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, ax=ax, vmin=-1, vmax=1,
        )
        ax.set_title(f"Feature-Target Correlations ({config.split_name})")
        plt.tight_layout()
        figures.append(save_figure(fig, "correlation_heatmap", config))
    except Exception:
        findings.append("Correlation heatmap skipped (rendering issue).")

    return AnalysisResult(
        success=True, step_name=STEP_NAME, split_name=config.split_name,
        findings=findings, tables_saved=tables, figures_saved=figures,
    )
