"""S04: Target-Emissions Identity Check — delta_rce vs per-capita emissions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataAnalysis.config.base import AnalysisConfig
from dataAnalysis.core.protocols import AnalysisResult
from dataAnalysis.utils.io import save_figure, save_table

STEP_NAME = "S04_target_identity"


def run(df: pd.DataFrame, config: AnalysisConfig) -> AnalysisResult:
    findings: list[str] = []
    tables: list = []
    figures: list = []

    if config.is_cross_sectional:
        findings.append(
            "Requires consecutive years — skipped for cross-sectional split "
            f"({config.split_name})."
        )
        return AnalysisResult(
            success=True, step_name=STEP_NAME, split_name=config.split_name,
            findings=findings,
        )

    residuals_all = []

    for country in df["country"].unique():
        cdata = df[df["country"] == country].sort_values("year")
        if len(cdata) < 2:
            continue

        for scenario in config.TARGET_COLS:
            rce = cdata[["year", scenario, "total_ghg_emissions", "population_total"]].dropna()
            if len(rce) < 2:
                continue

            delta_rce = rce[scenario].values[:-1] - rce[scenario].values[1:]
            pc_em = (
                rce["total_ghg_emissions"].values[:-1]
                / rce["population_total"].values[:-1]
            )
            residual = delta_rce - pc_em

            for i, r in enumerate(residual):
                residuals_all.append({
                    "country": country,
                    "scenario": scenario,
                    "year": int(rce["year"].values[i]),
                    "delta_rce": float(delta_rce[i]),
                    "per_capita_emissions": float(pc_em[i]),
                    "residual": float(r),
                })

    if not residuals_all:
        findings.append("Insufficient data for identity check.")
        return AnalysisResult(
            success=True, step_name=STEP_NAME, split_name=config.split_name,
            findings=findings,
        )

    res_df = pd.DataFrame(residuals_all)
    tables.append(save_table(res_df, "identity_residuals", config))

    # Summarize residuals (linear scenario)
    linear_res = res_df[res_df["scenario"] == "linear"]["residual"]
    median_res = linear_res.median()
    iqr_res = linear_res.quantile(0.75) - linear_res.quantile(0.25)
    within_tol = (linear_res.abs() < 1.0).mean() * 100

    if abs(median_res) < 0.1:
        findings.append(
            f"Identity approximately holds: median residual = {median_res:.4f}, "
            f"IQR = {iqr_res:.4f}, {within_tol:.1f}% within ±1.0 tolerance."
        )
    else:
        findings.append(
            f"Systematic residual detected: median = {median_res:.4f}, "
            f"IQR = {iqr_res:.4f}. "
            "Likely unit or population series mismatch."
        )

    # Histogram figure
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(linear_res.clip(-50, 50), bins=80, edgecolor="black", alpha=0.7)
        ax.axvline(0, color="red", linestyle="--", linewidth=2)
        ax.axvline(median_res, color="orange", linestyle="--",
                   label=f"Median: {median_res:.3f}")
        ax.set_xlabel("Residual (delta_RCE - per_capita_emissions)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Identity Residual Distribution ({config.split_name})")
        ax.legend()
        figures.append(save_figure(fig, "identity_residuals", config))
    except Exception:
        findings.append("Residual histogram skipped (rendering issue).")

    return AnalysisResult(
        success=True, step_name=STEP_NAME, split_name=config.split_name,
        findings=findings, tables_saved=tables, figures_saved=figures,
    )
