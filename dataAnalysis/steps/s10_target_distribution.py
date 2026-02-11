"""S10: Target Distribution & Scaling."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataAnalysis.config.base import AnalysisConfig
from dataAnalysis.core.protocols import AnalysisResult
from dataAnalysis.utils.interpret import describe_skew
from dataAnalysis.utils.io import save_figure, save_table

STEP_NAME = "S10_target_distribution"


def run(df: pd.DataFrame, config: AnalysisConfig) -> AnalysisResult:
    findings: list[str] = []
    tables: list = []
    figures: list = []

    dist_rows = []
    for scenario in config.TARGET_COLS:
        if scenario not in df.columns:
            continue
        series = df[scenario].dropna()
        if len(series) < 10:
            continue

        n_zero = int((series == 0).sum())
        n_negative = int((series < 0).sum())
        min_positive = float(series[series > 0].min()) if (series > 0).any() else np.nan
        skew = float(series.skew())
        p50 = float(series.quantile(0.50))
        p99 = float(series.quantile(0.99))
        tail_ratio = p99 / p50 if p50 != 0 else np.nan

        dist_rows.append({
            "scenario": scenario,
            "n_obs": len(series),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "skew": skew,
            "p50": p50,
            "p99": p99,
            "tail_ratio_p99_p50": tail_ratio,
            "n_zero": n_zero,
            "n_negative": n_negative,
            "min_positive": min_positive,
        })

        # Transform recommendation
        if n_negative > 0:
            shift = abs(series.min()) + 1
            rec = f"Negatives present — use log(x + {shift:.1f}) or shifted transform"
        elif n_zero > 0:
            rec = f"log1p(x) recommended ({n_zero} zeros)"
        else:
            rec = "log(x) safe — strictly positive"

        findings.append(
            f"'{scenario}': skew = {skew:.2f} ({describe_skew(skew)}), "
            f"{n_zero} zeros, {n_negative} negatives, "
            f"tail ratio (p99/p50) = {tail_ratio:.1f}. "
            f"Recommended transform: {rec}."
        )

    if dist_rows:
        dist_df = pd.DataFrame(dist_rows)
        tables.append(save_table(dist_df, "target_distribution", config))

    # Distribution plots
    try:
        n_scenarios = len([s for s in config.TARGET_COLS if s in df.columns])
        fig, axes = plt.subplots(1, n_scenarios, figsize=(5 * n_scenarios, 5))
        if n_scenarios == 1:
            axes = [axes]

        for idx, scenario in enumerate(config.TARGET_COLS):
            if scenario not in df.columns:
                continue
            ax = axes[idx]
            series = df[scenario].dropna()
            ax.hist(series, bins=50, edgecolor="black", alpha=0.7)
            ax.axvline(series.median(), color="red", linestyle="--",
                       label=f"Median: {series.median():.1f}")
            ax.axvline(0, color="green", linestyle="-", alpha=0.5)
            ax.set_title(f"{scenario} (skew={series.skew():.2f})")
            ax.set_xlabel("RCE")
            ax.set_ylabel("Frequency")
            ax.legend(fontsize=8)

        plt.suptitle(f"Target Distributions ({config.split_name})")
        plt.tight_layout()
        figures.append(save_figure(fig, "target_distributions", config))
    except Exception:
        findings.append("Target distribution plots skipped (rendering issue).")

    return AnalysisResult(
        success=True, step_name=STEP_NAME, split_name=config.split_name,
        findings=findings, tables_saved=tables, figures_saved=figures,
    )
