"""
model_comparison.py – Cross-model performance analysis.

Produces tables and figures comparing all evaluated models across targets,
strategies, and imputation branches. Not limited to winners — works with
the full outer_results.csv and final_results.csv.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from stages.post_analysis.config import SCENARIO_COLORS, STRATEGY_COLORS
from stages.post_analysis.plot_utils import (
    apply_thesis_style,
    model_color_map,
    save_fig,
    scenario_label,
    strategy_label,
    add_diagonal,
)

log = logging.getLogger(__name__)


def run_model_comparison(
    final_results: pd.DataFrame,
    outer_results: pd.DataFrame,
    out_dir: Path,
) -> dict[str, pd.DataFrame]:
    """
    Orchestrate all model comparison outputs.

    Returns a dict of produced DataFrames (for later Excel export).
    """
    apply_thesis_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: dict[str, pd.DataFrame] = {}

    if final_results.empty:
        log.warning("model_comparison: final_results is empty — skipping.")
        return produced

    perf_df = table_performance_summary(final_results, out_dir)
    produced["performance_summary"] = perf_df

    for metric in ("final_test_mae", "final_test_rmse", "final_test_r2"):
        if metric in final_results.columns:
            figure_performance_heatmap(final_results, metric, out_dir)

    figure_model_ranking(final_results, out_dir)
    figure_train_test_gap_scatter(final_results, out_dir)

    if not outer_results.empty:
        figure_outer_cv_boxplots(outer_results, out_dir)
        produced["outer_cv_results"] = outer_results

    return produced


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def table_performance_summary(
    final_results: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    """
    Build and save a performance summary table.

    Adds ``train_test_gap_mae = final_test_mae - final_train_mae`` and sorts by
    ``(target, outer_strategy_source, final_test_mae)``.
    """
    df = final_results.copy()

    if "final_train_mae" in df.columns and "final_test_mae" in df.columns:
        df["train_test_gap_mae"] = df["final_test_mae"] - df["final_train_mae"]

    keep_cols = [
        c for c in [
            "target", "outer_strategy_source", "branch", "model_name",
            "candidate_rank", "candidate_role",
            "final_test_mae", "final_test_rows",
            "final_test_mae_lenient", "final_test_rows_lenient",
            "final_test_rmse", "final_test_r2",
            "final_test_mse", "final_test_explained_variance",
            "final_train_mae", "train_test_gap_mae",
            "feature_count", "bootstrap_iterations",
            "final_model_svr_converged", "bootstrap_svr_converged_share",
        ] if c in df.columns
    ]
    df = df[keep_cols].copy()

    sort_cols = [c for c in ["target", "outer_strategy_source", "final_test_mae"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    path = out_dir / "performance_summary.csv"
    df.to_csv(path, index=False)
    log.info("Saved: %s", path)
    return df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def figure_performance_heatmap(
    final_results: pd.DataFrame,
    metric: str,
    out_dir: Path,
) -> None:
    """
    Heatmap of *metric* with rows = model×branch and columns = target×strategy.

    A separate figure is produced for each metric passed.
    """
    df = final_results.copy()
    if metric not in df.columns:
        log.warning("Metric '%s' not in final_results — skipping heatmap.", metric)
        return

    # Build readable row/column labels
    strategy_col = "outer_strategy_source" if "outer_strategy_source" in df.columns else "outer_strategy"
    if strategy_col not in df.columns:
        log.warning("No strategy column found — skipping heatmap for %s.", metric)
        return

    df["_row"] = df["model_name"] + " / " + df["branch"]
    df["_col"] = df["target"].map(scenario_label) + "\n" + df[strategy_col].map(strategy_label)

    pivot = df.pivot_table(index="_row", columns="_col", values=metric, aggfunc="mean")

    fig_h = max(4, len(pivot) * 0.5)
    fig_w = max(6, len(pivot.columns) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # For R², higher is better → use a reversed colormap
    cmap = "RdYlGn" if metric == "final_test_r2" else "RdYlGn_r"
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=cmap,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        cbar_kws={"label": metric.replace("_", " ").upper()},
    )
    ax.set_title(f"Model Performance — {metric.replace('_', ' ').upper()}", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)

    save_fig(out_dir / f"heatmap_{metric}.png")
    log.info("Saved heatmap: %s", metric)


def figure_model_ranking(
    final_results: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Horizontal bar chart of mean final_test_mae per model_name averaged across all
    (target, strategy, branch) combinations. Error bars show std deviation.
    """
    if "final_test_mae" not in final_results.columns or "model_name" not in final_results.columns:
        log.warning("Cannot produce model ranking — required columns missing.")
        return

    agg = (
        final_results.groupby("model_name")["final_test_mae"]
        .agg(mean_mae="mean", std_mae="std")
        .sort_values("mean_mae")
        .reset_index()
    )

    colors = model_color_map(agg["model_name"].tolist())
    bar_colors = [colors[m] for m in agg["model_name"]]

    fig, ax = plt.subplots(figsize=(8, max(3, len(agg) * 0.5)))
    ax.barh(
        agg["model_name"], agg["mean_mae"],
        xerr=agg["std_mae"].fillna(0),
        color=bar_colors, edgecolor="white", height=0.6, capsize=4,
    )
    ax.set_xlabel("Mean Test MAE (across targets / strategies / branches)")
    ax.set_title("Model Ranking by Test MAE")
    ax.axvline(agg["mean_mae"].min(), color="grey", linestyle=":", linewidth=0.8)

    save_fig(out_dir / "model_ranking_by_mae.png")
    log.info("Saved model ranking figure.")


def figure_train_test_gap_scatter(
    final_results: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Scatter plot: x = final_train_mae, y = final_test_mae.

    Points coloured by model_name. A y=x reference line shows the overfitting
    boundary (points above = test > train = overfitting).
    """
    if not {"final_train_mae", "final_test_mae", "model_name"}.issubset(final_results.columns):
        log.warning("Cannot produce gap scatter — required columns missing.")
        return

    df = final_results.dropna(subset=["final_train_mae", "final_test_mae"]).copy()
    colors = model_color_map(df["model_name"].unique().tolist())

    fig, ax = plt.subplots(figsize=(7, 6))
    for model, grp in df.groupby("model_name"):
        ax.scatter(
            grp["final_train_mae"], grp["final_test_mae"],
            label=model, color=colors[model], alpha=0.75, s=60, edgecolors="white",
        )

    add_diagonal(ax)
    ax.set_xlabel("Final Train MAE")
    ax.set_ylabel("Test MAE (2022 holdout)")
    ax.set_title("Train vs. Test MAE — Overfitting Diagnostic")
    ax.legend(fontsize=7, ncol=2)

    save_fig(out_dir / "train_test_gap_scatter.png")
    log.info("Saved train/test gap scatter.")


def figure_outer_cv_boxplots(
    outer_results: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Box plots of outer-fold test MAE per model and strategy, faceted by target.
    """
    required = {"outer_test_mae", "model_name", "target"}
    if not required.issubset(outer_results.columns):
        log.warning("Cannot produce outer CV boxplots — 'outer_test_mae' column missing.")
        return

    df = outer_results.dropna(subset=["outer_test_mae"]).copy()
    targets = sorted(df["target"].unique())
    if not targets:
        return

    strategy_col = next(
        (c for c in ["outer_strategy", "outer_strategy_source"] if c in df.columns), None
    )

    n_targets = len(targets)
    fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 5), sharey=False)
    if n_targets == 1:
        axes = [axes]

    for ax, target in zip(axes, targets):
        subset = df[df["target"] == target]
        order = (
            subset.groupby("model_name")["outer_test_mae"]
            .median()
            .sort_values()
            .index.tolist()
        )

        hue = strategy_col if strategy_col and subset[strategy_col].nunique() > 1 else None
        if hue:
            palette = {k: v for k, v in STRATEGY_COLORS.items() if k in subset[hue].unique()}
            sns.boxplot(
                data=subset, x="outer_test_mae", y="model_name",
                hue=hue, order=order, ax=ax, palette=palette,
                linewidth=0.8,
            )
        else:
            sns.boxplot(
                data=subset, x="outer_test_mae", y="model_name",
                order=order, ax=ax, color="#4393c3", linewidth=0.8,
            )
        ax.set_title(f"Target: {scenario_label(target)}")
        ax.set_xlabel("Outer Test MAE")
        ax.set_ylabel("")

    fig.suptitle("Outer CV MAE Distribution per Model", y=1.02, fontsize=13)

    save_fig(out_dir / "outer_cv_boxplots.png")
    log.info("Saved outer CV boxplots.")
