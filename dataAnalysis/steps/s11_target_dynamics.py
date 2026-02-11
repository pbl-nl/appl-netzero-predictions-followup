"""S11: Target Time Dynamics — curvature analysis of RCE trajectories."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataAnalysis.config.base import AnalysisConfig
from dataAnalysis.core.protocols import AnalysisResult
from dataAnalysis.utils.io import save_figure, save_table

STEP_NAME = "S11_target_dynamics"

_SAMPLE_COUNTRIES = ["USA", "CHN", "DEU", "IND", "BRA", "JPN", "GBR", "FRA"]


def run(df: pd.DataFrame, config: AnalysisConfig) -> AnalysisResult:
    findings: list[str] = []
    tables: list = []
    figures: list = []

    if config.is_cross_sectional:
        findings.append("Requires time series — skipped for cross-sectional split.")
        return AnalysisResult(
            success=True, step_name=STEP_NAME, split_name=config.split_name,
            findings=findings,
        )

    # Curvature analysis: second derivative of RCE over time
    curvature_rows = []

    for country in df["country"].unique():
        cdata = df[df["country"] == country].sort_values("year")

        for scenario in config.TARGET_COLS:
            series = cdata[["year", scenario]].dropna()
            if len(series) < 5:
                continue

            y = series[scenario].values
            # Second derivative (approximation using second differences)
            d2 = np.diff(y, n=2)
            mean_curvature = float(np.mean(d2))

            curvature_rows.append({
                "country": country,
                "scenario": scenario,
                "n_years": len(series),
                "mean_curvature": mean_curvature,
                "curvature_sign": "concave" if mean_curvature < 0 else "convex",
            })

    if not curvature_rows:
        findings.append("Insufficient data for curvature analysis.")
        return AnalysisResult(
            success=True, step_name=STEP_NAME, split_name=config.split_name,
            findings=findings,
        )

    curv_df = pd.DataFrame(curvature_rows)
    tables.append(save_table(curv_df, "target_curvature", config))

    # Check expected curvature patterns:
    # optimistic should be concave (accelerating decline)
    # pessimistic should be convex (decelerating decline)
    for scenario, expected_sign in [
        ("optimistic", "concave"),
        ("pessimistic", "convex"),
    ]:
        sdata = curv_df[curv_df["scenario"] == scenario]
        if len(sdata) == 0:
            continue
        matching = (sdata["curvature_sign"] == expected_sign).mean() * 100
        findings.append(
            f"'{scenario}' curvature: {matching:.1f}% of countries show "
            f"expected {expected_sign} shape. "
            f"{'Consistent with scenario design.' if matching > 60 else 'Some countries deviate from expected shape.'}"
        )

    # Linear scenario — check for predominantly negative curvature (concave=faster decline)
    linear = curv_df[curv_df["scenario"] == "linear"]
    if len(linear) > 0:
        pct_concave = (linear["mean_curvature"] < 0).mean() * 100
        findings.append(
            f"'linear' curvature: {pct_concave:.1f}% concave (accelerating decline), "
            f"{100 - pct_concave:.1f}% convex (decelerating decline)."
        )

    # Sample country trajectory plots
    available_sample = [c for c in _SAMPLE_COUNTRIES if c in df["country"].unique()]
    if available_sample:
        try:
            n_plots = min(len(available_sample), 8)
            ncols = 4
            nrows = (n_plots + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
            axes = np.array(axes).flatten()

            for idx, country in enumerate(available_sample[:n_plots]):
                ax = axes[idx]
                cdata = df[df["country"] == country].sort_values("year")
                for scenario in config.TARGET_COLS:
                    valid = cdata[["year", scenario]].dropna()
                    if len(valid) > 0:
                        ax.plot(valid["year"], valid[scenario],
                                label=scenario, marker=".", markersize=2)
                ax.set_title(country)
                ax.set_xlabel("Year")
                ax.set_ylabel("RCE")
                ax.grid(True, alpha=0.3)
                if idx == 0:
                    ax.legend(fontsize=7)

            for idx in range(n_plots, len(axes)):
                axes[idx].axis("off")

            plt.suptitle(f"Sample Country Trajectories ({config.split_name})")
            plt.tight_layout()
            figures.append(save_figure(fig, "target_trajectories", config))
        except Exception:
            findings.append("Trajectory plots skipped (rendering issue).")

    return AnalysisResult(
        success=True, step_name=STEP_NAME, split_name=config.split_name,
        findings=findings, tables_saved=tables, figures_saved=figures,
    )
