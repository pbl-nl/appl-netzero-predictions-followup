"""S06: Missingness Analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dataAnalysis.config.base import AnalysisConfig
from dataAnalysis.core.protocols import AnalysisResult
from dataAnalysis.utils.interpret import describe_missingness
from dataAnalysis.utils.io import save_figure, save_table

STEP_NAME = "S06_missingness"


def run(df: pd.DataFrame, config: AnalysisConfig) -> AnalysisResult:
    findings: list[str] = []
    tables: list = []
    figures: list = []

    all_cols = [c for c in config.ALL_FEATURE_COLS + config.TARGET_COLS if c in df.columns]

    # --- Overall missingness per feature ---
    overall = df[all_cols].isna().mean() * 100
    overall_df = overall.reset_index()
    overall_df.columns = ["feature", "missing_pct"]
    overall_df = overall_df.sort_values("missing_pct", ascending=False)
    tables.append(save_table(overall_df, "missingness_overall", config))

    high_miss = overall_df[overall_df["missing_pct"] > 50]
    moderate_miss = overall_df[
        (overall_df["missing_pct"] > 20) & (overall_df["missing_pct"] <= 50)
    ]
    low_miss = overall_df[overall_df["missing_pct"] <= 20]

    findings.append(
        f"Feature missingness tiers: "
        f"{len(high_miss)} high (>50%), "
        f"{len(moderate_miss)} moderate (20-50%), "
        f"{len(low_miss)} low (<20%)."
    )
    if len(high_miss) > 0:
        findings.append(
            f"High-risk features (>50% missing): "
            f"{high_miss['feature'].tolist()} — "
            f"{describe_missingness(high_miss['missing_pct'].max())}."
        )

    # --- Missingness by year ---
    if not config.is_cross_sectional:
        by_year = df.groupby("year")[all_cols].apply(
            lambda x: x.isna().mean() * 100
        )
        by_year_melted = by_year.reset_index().melt(
            id_vars="year", var_name="feature", value_name="missing_pct"
        )
        tables.append(save_table(by_year_melted, "missingness_by_year", config))

        # Heatmap
        try:
            years = sorted(df["year"].unique())
            step = max(1, len(years) // 20)
            sample_years = years[::step]
            by_year_sample = by_year.loc[by_year.index.isin(sample_years)]

            fig, ax = plt.subplots(figsize=(14, 10))
            sns.heatmap(
                by_year_sample.T, cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "Missing %"}, vmin=0, vmax=100,
            )
            ax.set_title(f"Missingness by Year and Feature ({config.split_name})")
            figures.append(save_figure(fig, "missingness_heatmap", config))
        except Exception:
            findings.append("Missingness heatmap skipped (rendering issue).")

    # --- Missingness by country ---
    by_country = df.groupby("country")[all_cols].apply(
        lambda x: x.isna().mean() * 100
    )
    country_avg = by_country.mean(axis=1).sort_values(ascending=False)
    country_miss_df = country_avg.reset_index()
    country_miss_df.columns = ["country", "avg_missing_pct"]
    tables.append(save_table(country_miss_df, "missingness_by_country", config))

    high_miss_countries = (country_avg > 80).sum()
    low_miss_countries = (country_avg < 20).sum()
    findings.append(
        f"Country-level missingness: {high_miss_countries} countries with >80% avg missing, "
        f"{low_miss_countries} countries with <20%."
    )

    # --- Row-wise missingness ---
    row_miss = df[all_cols].isna().sum(axis=1)
    findings.append(
        f"Row-wise missingness: median {row_miss.median():.0f}/{len(all_cols)} features missing, "
        f"max {row_miss.max()}/{len(all_cols)}."
    )

    return AnalysisResult(
        success=True, step_name=STEP_NAME, split_name=config.split_name,
        findings=findings, tables_saved=tables, figures_saved=figures,
    )
