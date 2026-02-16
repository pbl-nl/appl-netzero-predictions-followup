"""Interactive plotly visualizations for resultsAnalysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from resultsAnalysis.config.base import ResultsAnalysisConfig
from resultsAnalysis.core.protocols import (
    FeatureImportanceOutputs,
    NetZeroOutputs,
    RankingOutputs,
    SensitivityOutputs,
    UncertaintyOutputs,
)


def _write_fig(
    fig: go.Figure,
    html_path: Path,
    png_path: Path | None = None,
    *,
    width: int = 1200,
    height: int = 700,
    scale: int = 2,
) -> tuple[Path, Path | None]:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    if png_path is not None:
        try:
            fig.write_image(str(png_path), width=width, height=height, scale=scale)
        except Exception:
            pass
    return html_path, png_path


def plot_ranking_scatter(config: ResultsAnalysisConfig, ranking: RankingOutputs) -> tuple[Path, Path]:
    df = ranking.algorithm_ranking.copy()
    df = df[~df["Algorithm"].isin(["country_mean", "global_mean"])].copy()

    fig = px.scatter(
        df,
        x="RMSE_mean",
        y="Within_RMSE",
        color="Algorithm",
        symbol="Scenario",
        hover_data=["Split", "NZ_MAE", "vs_Ridge_pct"],
        title="Within-RMSE vs Overall RMSE",
    )
    fig.update_layout(template="plotly_white")

    html = config.figures_dir / "ranking_scatter.html"
    png = config.figures_dir / "ranking_scatter.png"
    _write_fig(fig, html, png, width=config.plotly_width, height=config.plotly_height, scale=config.plotly_scale)
    return html, png


def plot_world_choropleth(config: ResultsAnalysisConfig, netzero: NetZeroOutputs) -> tuple[Path, Path]:
    df = netzero.netzero_country_summary.copy()
    color_scale = [
        [0.0, "#1d4f91"],
        [0.5, "#f1f3f5"],
        [1.0, "#a61e4d"],
    ]

    fig = px.choropleth(
        df,
        locations="country",
        locationmode="ISO-3",
        color="nz_vs_pledge_diff",
        hover_name="country",
        hover_data=["scenario", "split", "algorithm", "nz_year_median", "pledge_year", "category"],
        color_continuous_scale=color_scale,
        title="World Map: Predicted NZ Year vs Pledge Gap",
    )
    fig.update_geos(showframe=False, showcoastlines=False)
    fig.update_layout(template="plotly_white")

    html = config.figures_dir / "world_choropleth.html"
    png = config.figures_dir / "world_choropleth.png"
    _write_fig(fig, html, png, width=config.plotly_width, height=config.plotly_height, scale=config.plotly_scale)
    return html, png


def plot_netzero_dot_ci(config: ResultsAnalysisConfig, netzero: NetZeroOutputs) -> tuple[Path, Path]:
    df = netzero.netzero_country_summary.copy().dropna(subset=["nz_vs_pledge_diff"]).copy()
    df = df.sort_values("nz_vs_pledge_diff")

    fig = px.scatter(
        df,
        x="nz_vs_pledge_diff",
        y="country",
        color="scenario",
        error_x=(df["nz_year_p95"] - df["nz_year_median"]).clip(lower=0),
        error_x_minus=(df["nz_year_median"] - df["nz_year_p5"]).clip(lower=0),
        hover_data=["split", "algorithm", "category", "pledge_year"],
        title="Country-level Pledge Gap with NZ-year Uncertainty",
        height=max(700, int(len(df) * 8)),
    )
    fig.update_layout(template="plotly_white")

    html = config.figures_dir / "netzero_dot_ci.html"
    png = config.figures_dir / "netzero_dot_ci.png"
    _write_fig(fig, html, png, width=config.plotly_width, height=config.plotly_height, scale=config.plotly_scale)
    return html, png


def plot_sensitivity_overlay(config: ResultsAnalysisConfig, sensitivity: SensitivityOutputs) -> tuple[Path, Path]:
    traces = []
    if not sensitivity.sensitivity_scenario.empty:
        traces.append(go.Violin(y=sensitivity.sensitivity_scenario["scenario_spread"], name="Scenario spread", box_visible=True, meanline_visible=True))
    if not sensitivity.sensitivity_split.empty:
        traces.append(go.Violin(y=sensitivity.sensitivity_split["split_spread"], name="Split spread", box_visible=True, meanline_visible=True))
    if not sensitivity.sensitivity_algorithm.empty:
        traces.append(go.Violin(y=sensitivity.sensitivity_algorithm["algo_spread"], name="Algorithm spread", box_visible=True, meanline_visible=True))

    fig = go.Figure(data=traces)
    fig.update_layout(template="plotly_white", title="Sensitivity Distribution Overlay", yaxis_title="Spread (years)")

    html = config.figures_dir / "sensitivity_overlay.html"
    png = config.figures_dir / "sensitivity_overlay.png"
    _write_fig(fig, html, png, width=config.plotly_width, height=config.plotly_height, scale=config.plotly_scale)
    return html, png


def plot_feature_boxplot(config: ResultsAnalysisConfig, feature: FeatureImportanceOutputs) -> tuple[Path, Path]:
    df = feature.feature_importance_by_algorithm.copy()
    top = (
        df.groupby("feature", as_index=False)["mean_importance"]
        .mean()
        .sort_values("mean_importance", ascending=False)
        .head(15)
    )
    show_df = df[df["feature"].isin(top["feature"])].copy()

    fig = px.box(
        show_df,
        x="feature",
        y="mean_importance",
        color="algorithm",
        points="all",
        title="Feature Importance Distribution (Top 15 Features)",
    )
    fig.update_layout(template="plotly_white", xaxis_tickangle=45)

    html = config.figures_dir / "feature_boxplot.html"
    png = config.figures_dir / "feature_boxplot.png"
    _write_fig(fig, html, png, width=config.plotly_width, height=config.plotly_height, scale=config.plotly_scale)
    return html, png


def plot_stability_scatter(config: ResultsAnalysisConfig, uncertainty: UncertaintyOutputs) -> tuple[Path, Path]:
    df = uncertainty.stability_by_country.copy()

    fig = px.scatter(
        df,
        x="prediction_cv",
        y="nz_year_range",
        color="scenario",
        hover_name="country",
        hover_data=["split", "algorithm", "high_uncertainty"],
        title="Stability Scatter: Prediction CV vs NZ-year Range",
    )
    fig.add_vline(x=config.high_uncertainty_cv_threshold, line_dash="dash", line_color="red")
    fig.add_hline(y=config.high_uncertainty_nz_range_threshold, line_dash="dash", line_color="red")
    fig.update_layout(template="plotly_white")

    html = config.figures_dir / "stability_scatter.html"
    png = config.figures_dir / "stability_scatter.png"
    _write_fig(fig, html, png, width=config.plotly_width, height=config.plotly_height, scale=config.plotly_scale)
    return html, png


def generate_interactive_visuals(
    config: ResultsAnalysisConfig,
    ranking: RankingOutputs,
    netzero: NetZeroOutputs,
    sensitivity: SensitivityOutputs,
    feature: FeatureImportanceOutputs,
    uncertainty: UncertaintyOutputs,
) -> tuple[dict[str, Path], dict[str, Path]]:
    html_paths: dict[str, Path] = {}
    png_paths: dict[str, Path] = {}

    html, png = plot_ranking_scatter(config, ranking)
    html_paths["ranking_scatter"] = html
    png_paths["ranking_scatter"] = png

    html, png = plot_world_choropleth(config, netzero)
    html_paths["world_choropleth"] = html
    png_paths["world_choropleth"] = png

    html, png = plot_netzero_dot_ci(config, netzero)
    html_paths["netzero_dot_ci"] = html
    png_paths["netzero_dot_ci"] = png

    html, png = plot_sensitivity_overlay(config, sensitivity)
    html_paths["sensitivity_overlay"] = html
    png_paths["sensitivity_overlay"] = png

    if not feature.feature_importance_by_algorithm.empty:
        html, png = plot_feature_boxplot(config, feature)
        html_paths["feature_boxplot"] = html
        png_paths["feature_boxplot"] = png

    html, png = plot_stability_scatter(config, uncertainty)
    html_paths["stability_scatter"] = html
    png_paths["stability_scatter"] = png

    return html_paths, png_paths
