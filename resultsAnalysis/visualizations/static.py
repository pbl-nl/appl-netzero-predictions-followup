"""Static matplotlib/seaborn visualizations for resultsAnalysis."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from resultsAnalysis.config.base import ResultsAnalysisConfig
from resultsAnalysis.core.protocols import (
    FeatureImportanceOutputs,
    ModelSelectionOutputs,
    NetZeroOutputs,
    RankingOutputs,
    SensitivityOutputs,
    UncertaintyOutputs,
)


def _ensure_mpl_env() -> None:
    """Set matplotlib config directory to avoid home-dir writes.

    Called at function entry rather than module level to avoid import-time
    side effects.
    """
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def _get_plt_and_sns():
    """Lazy import matplotlib and seaborn after env setup."""
    _ensure_mpl_env()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    return plt, sns


def _save(fig, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def plot_ranking_grouped_bars(config: ResultsAnalysisConfig, ranking: RankingOutputs, path: Path) -> None:
    plt, sns = _get_plt_and_sns()
    df = ranking.algorithm_ranking.copy()
    df = df[~df["Algorithm"].isin(["country_mean", "global_mean"])].copy()
    if df.empty:
        return

    splits = sorted(df["Split"].dropna().unique().tolist())
    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 5), squeeze=False)

    for i, split in enumerate(splits):
        ax = axes[0, i]
        part = df[df["Split"] == split].copy().sort_values(["RMSE_mean", "Algorithm"])
        if part.empty:
            continue

        keep_algorithms = (
            part.groupby("Algorithm", as_index=False)["RMSE_mean"]
            .mean()
            .sort_values("RMSE_mean")
            .head(8)["Algorithm"]
            .tolist()
        )
        part = part[part["Algorithm"].isin(keep_algorithms)].copy()

        scenarios = sorted(part["Scenario"].astype(str).unique().tolist())
        algorithms = sorted(part["Algorithm"].astype(str).unique().tolist())
        x = np.arange(len(scenarios), dtype=float)
        width = 0.8 / max(1, len(algorithms))

        for j, algo in enumerate(algorithms):
            y_vals: list[float] = []
            y_errs: list[float] = []
            for scenario in scenarios:
                row = part[(part["Algorithm"] == algo) & (part["Scenario"] == scenario)]
                if row.empty:
                    y_vals.append(np.nan)
                    y_errs.append(0.0)
                    continue
                rmse = float(row.iloc[0]["RMSE_mean"])
                y_vals.append(rmse)
                ci_low = pd.to_numeric(pd.Series([row.iloc[0].get("RMSE_ci_low")]), errors="coerce").iloc[0]
                ci_high = pd.to_numeric(pd.Series([row.iloc[0].get("RMSE_ci_high")]), errors="coerce").iloc[0]
                if np.isfinite(ci_low) and np.isfinite(ci_high):
                    y_errs.append(float(max(rmse - ci_low, ci_high - rmse)))
                else:
                    std = pd.to_numeric(pd.Series([row.iloc[0].get("RMSE_std")]), errors="coerce").iloc[0]
                    y_errs.append(float(std) if np.isfinite(std) else 0.0)

            ax.bar(x - 0.4 + width / 2 + j * width, y_vals, width=width, yerr=y_errs, label=algo, capsize=2)

        ax.set_title(f"RMSE by Scenario ({split})")
        ax.set_xlabel("Scenario")
        ax.set_ylabel("RMSE (log)")
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios)
        if i > 0:
            ax.legend_.remove()

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="Algorithm", loc="upper center", ncol=min(6, len(labels)))
    _save(fig, path, config.figure_dpi)


def plot_ranking_heatmap(config: ResultsAnalysisConfig, ranking: RankingOutputs, path: Path) -> None:
    plt, sns = _get_plt_and_sns()
    df = ranking.algorithm_ranking.copy()
    if df.empty:
        return
    df["split_scenario"] = df["Split"].astype(str) + " | " + df["Scenario"].astype(str)
    pivot = df.pivot_table(index="Algorithm", columns="split_scenario", values="RMSE_mean", aggfunc="mean")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 1.2), max(4, pivot.shape[0] * 0.5)))
    sns.heatmap(pivot, cmap="YlOrRd_r", annot=True, fmt=".2f", ax=ax)
    ax.set_title("RMSE Heatmap: Algorithm x Split/Scenario")
    _save(fig, path, config.figure_dpi)


def plot_netzero_histogram(config: ResultsAnalysisConfig, netzero: NetZeroOutputs, path: Path) -> None:
    plt, sns = _get_plt_and_sns()
    df = netzero.netzero_country_summary.copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="nz_vs_pledge_diff", hue="scenario", bins=30, element="step", ax=ax)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Distribution of Pledge Gap (Predicted - Pledged)")
    ax.set_xlabel("Years")
    _save(fig, path, config.figure_dpi)


def plot_top_ahead_behind(config: ResultsAnalysisConfig, netzero: NetZeroOutputs, path: Path) -> None:
    plt, sns = _get_plt_and_sns()
    df = netzero.netzero_country_summary.copy()
    if df.empty:
        return

    ahead = df.sort_values("nz_vs_pledge_diff", ascending=True).head(10)
    behind = df.sort_values("nz_vs_pledge_diff", ascending=False).head(10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(data=ahead, y="country", x="nz_vs_pledge_diff", hue="scenario", dodge=False, ax=axes[0])
    axes[0].set_title("Top 10 Furthest Ahead")
    axes[0].set_xlabel("Years vs pledge (negative=better)")

    sns.barplot(data=behind, y="country", x="nz_vs_pledge_diff", hue="scenario", dodge=False, ax=axes[1])
    axes[1].set_title("Top 10 Furthest Behind")
    axes[1].set_xlabel("Years vs pledge")

    axes[0].legend_.remove()
    _save(fig, path, config.figure_dpi)


def plot_region_gap_bar(config: ResultsAnalysisConfig, netzero: NetZeroOutputs, path: Path) -> None:
    plt, sns = _get_plt_and_sns()
    df = netzero.pledge_gap_by_region.copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x="region", y="mean_pledge_gap", hue="scenario", ax=ax)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Mean Pledge Gap by Region")
    ax.set_xlabel("Region")
    ax.set_ylabel("Mean years vs pledge")
    ax.tick_params(axis="x", rotation=25)
    _save(fig, path, config.figure_dpi)


def plot_sensitivity_violin(config: ResultsAnalysisConfig, sensitivity: SensitivityOutputs, path: Path) -> None:
    plt, sns = _get_plt_and_sns()
    a = sensitivity.sensitivity_scenario[["country", "scenario_spread"]].copy() if not sensitivity.sensitivity_scenario.empty else pd.DataFrame(columns=["country", "scenario_spread"])
    b = sensitivity.sensitivity_split[["country", "split_spread"]].copy() if not sensitivity.sensitivity_split.empty else pd.DataFrame(columns=["country", "split_spread"])
    c = sensitivity.sensitivity_algorithm[["country", "algo_spread"]].copy() if not sensitivity.sensitivity_algorithm.empty else pd.DataFrame(columns=["country", "algo_spread"])

    merged = a.merge(b, on="country", how="outer").merge(c, on="country", how="outer")
    if merged.empty:
        return

    long_df = merged.melt(
        id_vars=["country"],
        value_vars=["scenario_spread", "split_spread", "algo_spread"],
        var_name="dimension",
        value_name="spread_years",
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.violinplot(data=long_df, x="dimension", y="spread_years", inner="quartile", ax=ax)
    ax.set_title("Sensitivity Spread Across Dimensions")
    ax.set_xlabel("")
    ax.set_ylabel("Spread (years)")
    _save(fig, path, config.figure_dpi)


def plot_feature_bars(config: ResultsAnalysisConfig, feature: FeatureImportanceOutputs, path: Path) -> None:
    plt, sns = _get_plt_and_sns()
    df = feature.feature_importance_by_algorithm.copy()
    if df.empty:
        return

    algos = sorted(df["algorithm"].unique().tolist())
    fig, axes = plt.subplots(len(algos), 1, figsize=(10, max(4, 2.4 * len(algos))), squeeze=False)
    for i, algo in enumerate(algos):
        ax = axes[i, 0]
        part = df[df["algorithm"] == algo].sort_values("mean_importance", ascending=False).head(15)
        sns.barplot(data=part, y="feature", x="mean_importance", ax=ax, color="#336699")
        ax.set_title(f"Top Features: {algo}")
        ax.set_xlabel("Mean importance")
        ax.set_ylabel("")
    _save(fig, path, config.figure_dpi)


def plot_feature_heatmap(config: ResultsAnalysisConfig, feature: FeatureImportanceOutputs, path: Path) -> None:
    plt, sns = _get_plt_and_sns()
    df = feature.feature_importance_by_algorithm.copy()
    if df.empty:
        return

    pivot = df.pivot_table(index="feature", columns="algorithm", values="mean_rank", aggfunc="mean")
    pivot = pivot.sort_values(by=list(pivot.columns), na_position="last")
    pivot = pivot.head(25)
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(8, max(4, pivot.shape[0] * 0.35)))
    sns.heatmap(pivot, cmap="mako_r", annot=True, fmt=".1f", ax=ax)
    ax.set_title("Feature Rank Heatmap (Lower Rank = More Important)")
    _save(fig, path, config.figure_dpi)


def plot_calibration_bars(config: ResultsAnalysisConfig, uncertainty: UncertaintyOutputs, path: Path) -> None:
    plt, sns = _get_plt_and_sns()
    df = uncertainty.calibration_report.copy()
    if df.empty:
        return

    agg = df.groupby("Scenario", as_index=False).agg(coverage_90=("coverage_90", "mean"), coverage_50=("coverage_50", "mean"))
    long_df = agg.melt(id_vars="Scenario", value_vars=["coverage_90", "coverage_50"], var_name="interval", value_name="coverage")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=long_df, x="Scenario", y="coverage", hue="interval", ax=ax)
    ax.axhline(config.calibration_expected_90, color="#b02a37", linestyle="--", linewidth=1)
    ax.axhline(config.calibration_expected_50, color="#0b7285", linestyle="--", linewidth=1)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Calibration Coverage (Actual vs Expected)")
    _save(fig, path, config.figure_dpi)


def plot_interval_width_hist(config: ResultsAnalysisConfig, uncertainty: UncertaintyOutputs, path: Path) -> None:
    plt, sns = _get_plt_and_sns()
    df = uncertainty.stability_by_country.copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="nz_year_range", bins=30, ax=ax)
    ax.axvline(config.high_uncertainty_nz_range_threshold, color="red", linestyle="--")
    ax.set_title("Net-Zero Interval Width Distribution")
    ax.set_xlabel("NZ year range (p95 - p5)")
    _save(fig, path, config.figure_dpi)


def plot_stacked_vs_final(config: ResultsAnalysisConfig, model_sel: ModelSelectionOutputs, path: Path) -> None:
    plt, sns = _get_plt_and_sns()
    df = model_sel.stacked_vs_final.copy()
    if df.empty:
        return

    agg = df.groupby("scenario", as_index=False).agg(mean_gap=("final_minus_stacked_years", "mean"))
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=agg, x="scenario", y="mean_gap", ax=ax, color="#6c757d")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Final Model vs Stacked Ensemble (Mean NZ-year gap)")
    ax.set_ylabel("Final - Stacked (years)")
    _save(fig, path, config.figure_dpi)


def generate_static_visuals(
    config: ResultsAnalysisConfig,
    ranking: RankingOutputs,
    netzero: NetZeroOutputs,
    sensitivity: SensitivityOutputs,
    feature: FeatureImportanceOutputs,
    uncertainty: UncertaintyOutputs,
    model_sel: ModelSelectionOutputs,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}

    path = config.figures_dir / "ranking_grouped_ci.png"
    plot_ranking_grouped_bars(config, ranking, path)
    outputs["ranking_grouped_ci"] = path

    path = config.figures_dir / "ranking_heatmap.png"
    plot_ranking_heatmap(config, ranking, path)
    outputs["ranking_heatmap"] = path

    path = config.figures_dir / "netzero_gap_hist.png"
    plot_netzero_histogram(config, netzero, path)
    outputs["netzero_gap_hist"] = path

    path = config.figures_dir / "netzero_top10_ahead_behind.png"
    plot_top_ahead_behind(config, netzero, path)
    outputs["netzero_top10_ahead_behind"] = path

    path = config.figures_dir / "region_pledge_gap.png"
    plot_region_gap_bar(config, netzero, path)
    outputs["region_pledge_gap"] = path

    path = config.figures_dir / "sensitivity_violin.png"
    plot_sensitivity_violin(config, sensitivity, path)
    outputs["sensitivity_violin"] = path

    path = config.figures_dir / "feature_top15_by_algo.png"
    plot_feature_bars(config, feature, path)
    outputs["feature_top15_by_algo"] = path

    path = config.figures_dir / "feature_heatmap.png"
    plot_feature_heatmap(config, feature, path)
    outputs["feature_heatmap"] = path

    path = config.figures_dir / "calibration_bars.png"
    plot_calibration_bars(config, uncertainty, path)
    outputs["calibration_bars"] = path

    path = config.figures_dir / "interval_width_hist.png"
    plot_interval_width_hist(config, uncertainty, path)
    outputs["interval_width_hist"] = path

    path = config.figures_dir / "stacked_vs_final.png"
    plot_stacked_vs_final(config, model_sel, path)
    outputs["stacked_vs_final"] = path

    return outputs
