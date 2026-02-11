"""S08: Outliers & Structural Breaks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataAnalysis.config.base import AnalysisConfig
from dataAnalysis.core.protocols import AnalysisResult
from dataAnalysis.utils.io import save_figure, save_table
from dataAnalysis.utils.stats import flag_outliers_iqr

STEP_NAME = "S08_outliers"

_KEY_SERIES = [
    "total_ghg_emissions",
    "non_co2_emissions",
    "gdp_per_capita_constant_2015_us",
    "population_total",
]


def run(df: pd.DataFrame, config: AnalysisConfig) -> AnalysisResult:
    findings: list[str] = []
    tables: list = []
    figures: list = []

    available = [c for c in _KEY_SERIES if c in df.columns]

    if config.is_cross_sectional:
        # Cross-sectional: IQR-based outliers only
        outlier_rows = []
        for col in available:
            series = df[col].dropna()
            if len(series) < 10:
                continue
            is_outlier = flag_outliers_iqr(series, k=3.0)
            n_outliers = is_outlier.sum()
            if n_outliers > 0:
                flagged = df.loc[series.index[is_outlier], ["country", "year", col]]
                for _, row in flagged.iterrows():
                    outlier_rows.append({
                        "country": row["country"],
                        "year": int(row["year"]),
                        "feature": col,
                        "value": float(row[col]),
                        "type": "cross_sectional_iqr",
                    })
            findings.append(
                f"{col}: {n_outliers} extreme outliers (3*IQR) "
                f"out of {len(series)} observations."
            )

        if outlier_rows:
            outlier_df = pd.DataFrame(outlier_rows)
            tables.append(save_table(outlier_df, "outlier_jumps", config))

        return AnalysisResult(
            success=True, step_name=STEP_NAME, split_name=config.split_name,
            findings=findings, tables_saved=tables,
        )

    # Panel: year-over-year within-country jumps
    df_sorted = df.sort_values(["country", "year"])
    jump_rows = []

    for col in available:
        for country, grp in df_sorted.groupby("country"):
            series = grp[col].dropna()
            if len(series) < 2:
                continue
            abs_changes = series.diff().abs().dropna()
            if len(abs_changes) == 0:
                continue
            pct_changes = series.pct_change().abs().dropna()

            # Flag top 1% absolute jumps globally is hard per-country,
            # so flag changes > 100% (doubling/halving) as suspicious
            big_jumps = pct_changes[pct_changes > 1.0]
            for idx in big_jumps.index:
                jump_rows.append({
                    "country": country,
                    "year": int(grp.loc[idx, "year"]),
                    "feature": col,
                    "value": float(series.loc[idx]),
                    "prev_value": float(series.shift(1).loc[idx]),
                    "pct_change": float(pct_changes.loc[idx]),
                    "type": "temporal_jump",
                })

    if jump_rows:
        jump_df = pd.DataFrame(jump_rows)
        jump_df = jump_df.sort_values("pct_change", ascending=False)
        tables.append(save_table(jump_df, "outlier_jumps", config))

        total_jumps = len(jump_df)
        findings.append(
            f"{total_jumps} temporal jumps (>100% YoY change) detected "
            f"across {len(available)} features."
        )

        # Per-feature breakdown
        by_feature = jump_df.groupby("feature").size()
        for feat, count in by_feature.items():
            findings.append(f"  - {feat}: {count} jumps")

        # Top 10 most extreme
        top10 = jump_df.head(10)
        findings.append(
            f"Most extreme jump: {top10.iloc[0]['country']} "
            f"{top10.iloc[0]['feature']} in {int(top10.iloc[0]['year'])} "
            f"({top10.iloc[0]['pct_change']:.0%} change)."
        )

        # Box plot of jump severity by feature
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            features_with_jumps = sorted(jump_df["feature"].unique())
            data_for_plot = [
                jump_df[jump_df["feature"] == f]["pct_change"].clip(upper=10).values
                for f in features_with_jumps
            ]
            ax.boxplot(data_for_plot, labels=features_with_jumps, vert=True)
            ax.set_ylabel("Absolute % change (clipped at 10x)")
            ax.set_title(f"Jump Severity by Feature ({config.split_name})")
            ax.tick_params(axis="x", rotation=45)
            plt.tight_layout()
            figures.append(save_figure(fig, "outlier_jump_severity", config))
        except Exception:
            findings.append("Jump severity plot skipped (rendering issue).")
    else:
        findings.append("No temporal jumps > 100% detected.")

    return AnalysisResult(
        success=True, step_name=STEP_NAME, split_name=config.split_name,
        findings=findings, tables_saved=tables, figures_saved=figures,
    )
