"""S09: Variance Decomposition (Within vs Between)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataAnalysis.config.base import AnalysisConfig
from dataAnalysis.core.protocols import AnalysisResult
from dataAnalysis.utils.interpret import describe_icc
from dataAnalysis.utils.io import save_figure, save_table
from dataAnalysis.utils.stats import variance_decomposition

STEP_NAME = "S09_variance_decomp"


def run(df: pd.DataFrame, config: AnalysisConfig) -> AnalysisResult:
    findings: list[str] = []
    tables: list = []
    figures: list = []

    cols_to_analyze = config.TARGET_COLS + config.ALL_FEATURE_COLS
    cols_to_analyze = [c for c in cols_to_analyze if c in df.columns]

    if config.is_cross_sectional:
        # Single year: only cross-sectional spread (no within-country variance)
        spread_rows = []
        for col in cols_to_analyze:
            series = df[col].dropna()
            if len(series) < 10:
                continue
            spread_rows.append({
                "variable": col,
                "n_obs": len(series),
                "std": float(series.std()),
                "cv": float(series.std() / series.mean()) if series.mean() != 0 else np.nan,
                "iqr": float(series.quantile(0.75) - series.quantile(0.25)),
            })

        spread_df = pd.DataFrame(spread_rows).sort_values("cv", ascending=False)
        tables.append(save_table(spread_df, "variance_decomposition", config))

        findings.append(
            f"Cross-sectional spread computed for {len(spread_df)} features. "
            "No within-country variance with a single year."
        )
        if len(spread_df) > 0:
            top_cv = spread_df.head(3)
            findings.append(
                f"Highest cross-sectional variation (CV): "
                + ", ".join(
                    f"{r['variable']} (CV={r['cv']:.2f})"
                    for _, r in top_cv.iterrows()
                )
            )

        return AnalysisResult(
            success=True, step_name=STEP_NAME, split_name=config.split_name,
            findings=findings, tables_saved=tables,
        )

    # Panel: full variance decomposition
    results = []
    for col in cols_to_analyze:
        valid_df = df[df[col].notna()]
        if len(valid_df) < 100 or valid_df["country"].nunique() < 10:
            continue
        try:
            decomp = variance_decomposition(valid_df, col)
            decomp["variable"] = col
            results.append(decomp)
        except Exception:
            continue

    if not results:
        findings.append("Insufficient data for variance decomposition.")
        return AnalysisResult(
            success=True, step_name=STEP_NAME, split_name=config.split_name,
            findings=findings,
        )

    var_df = pd.DataFrame(results).sort_values("icc", ascending=False)
    tables.append(save_table(var_df, "variance_decomposition", config))

    # Top/bottom by ICC
    for scenario in config.TARGET_COLS:
        row = var_df[var_df["variable"] == scenario]
        if len(row) > 0:
            icc_val = row.iloc[0]["icc"]
            findings.append(
                f"ICC for '{scenario}' target = {icc_val:.2f} — "
                f"{icc_val * 100:.0f}% of variance is between-country. "
                f"{describe_icc(icc_val).capitalize()}."
            )

    # Features ranked by ICC
    feature_icc = var_df[~var_df["variable"].isin(config.TARGET_COLS)]
    if len(feature_icc) > 0:
        top_icc = feature_icc.head(3)
        bottom_icc = feature_icc.tail(3)
        findings.append(
            "Features with highest ICC (stable across time, vary by country): "
            + ", ".join(
                f"{r['variable']} ({r['icc']:.2f})"
                for _, r in top_icc.iterrows()
            )
        )
        findings.append(
            "Features with lowest ICC (temporally dynamic): "
            + ", ".join(
                f"{r['variable']} ({r['icc']:.2f})"
                for _, r in bottom_icc.iterrows()
            )
        )

    # ICC bar chart
    try:
        plot_df = var_df.head(20).sort_values("icc")
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = [
            "steelblue" if v in config.TARGET_COLS else "coral"
            for v in plot_df["variable"]
        ]
        ax.barh(plot_df["variable"], plot_df["icc"], color=colors, alpha=0.7)
        ax.set_xlabel("Intraclass Correlation (ICC)")
        ax.set_title(f"Top 20 Features by ICC ({config.split_name})")
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
        plt.tight_layout()
        figures.append(save_figure(fig, "icc_bar_chart", config))
    except Exception:
        findings.append("ICC bar chart skipped (rendering issue).")

    return AnalysisResult(
        success=True, step_name=STEP_NAME, split_name=config.split_name,
        findings=findings, tables_saved=tables, figures_saved=figures,
    )
