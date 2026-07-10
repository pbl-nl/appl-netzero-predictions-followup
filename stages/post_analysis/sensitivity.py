"""
sensitivity.py – Sensitivity analysis across branches, strategies, and targets.

Uses fold-level ``outer_results.csv`` rather than winner-only ``final_results``
so matched comparisons are populated even when the final winners differ.
"""

from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from stages.post_analysis.plot_utils import apply_thesis_style, branch_label, save_fig

log = logging.getLogger(__name__)


def _get_strategy_col(df: pd.DataFrame) -> str | None:
    for col in ("outer_strategy_source", "outer_strategy"):
        if col in df.columns:
            return col
    return None


def _outer_metric_frame(outer_results: pd.DataFrame) -> pd.DataFrame:
    """Return a normalised fold-level frame with ``outer_test_mae`` present."""
    return outer_results.copy()


def _coverage_status(pair_count: int, expected_count: int) -> str:
    if pair_count <= 0:
        return "no_overlap"
    if pair_count >= expected_count:
        return "full_overlap"
    return "partial_overlap"


def _save_table(df: pd.DataFrame, out_dir: Path, name: str) -> pd.DataFrame:
    path = out_dir / name
    df.to_csv(path, index=False)
    log.info("Saved: %s", path)
    return df


def run_sensitivity(
    outer_results: pd.DataFrame,
    final_prediction_summary: pd.DataFrame,
    out_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Orchestrate all sensitivity analysis outputs."""
    apply_thesis_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: dict[str, pd.DataFrame] = {}

    if outer_results.empty:
        log.warning("sensitivity: outer_results is empty — skipping.")
        return produced

    outer_df = _outer_metric_frame(outer_results)
    branch_df = table_branch_sensitivity(outer_df, out_dir)
    strategy_df = table_strategy_sensitivity(outer_df, out_dir)
    scenario_df = table_scenario_sensitivity(outer_df, out_dir)
    produced["branch_sensitivity"] = branch_df
    produced["strategy_sensitivity"] = strategy_df
    produced["scenario_sensitivity"] = scenario_df

    figure_sensitivity_heatmaps(branch_df, strategy_df, scenario_df, out_dir)

    if not final_prediction_summary.empty:
        _figure_prediction_agreements(final_prediction_summary, out_dir)

    return produced


def table_branch_sensitivity(
    outer_results: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    """
    Compare every pair of active imputation branches on matched outer folds for each
    ``(outer_strategy, target, model_name)`` combination.

    One row per (outer_strategy, target, model_name, branch_a, branch_b) — branch_a/
    branch_b range over all ``C(n, 2)`` pairs of branches present in the data, so this
    scales from two branches (simple/mice) to any number (e.g. + interpolation).
    """
    strategy_col = _get_strategy_col(outer_results)
    required = {"branch", "outer_test_mae", "model_name", "target", "outer_test_year"}
    if strategy_col is None or not required.issubset(outer_results.columns):
        log.warning("branch_sensitivity: required columns missing.")
        return pd.DataFrame()

    idx_cols = [strategy_col, "target", "model_name", "outer_test_year"]
    pivot = (
        outer_results[idx_cols + ["branch", "outer_test_mae"]]
        .pivot_table(index=idx_cols, columns="branch", values="outer_test_mae", aggfunc="mean")
        .reset_index()
    )
    pivot = pivot.rename(columns={strategy_col: "outer_strategy_source"})

    branches = sorted(outer_results["branch"].dropna().unique())
    for branch in branches:
        if branch not in pivot.columns:
            pivot[branch] = np.nan

    rows: list[dict[str, Any]] = []
    for branch_a, branch_b in combinations(branches, 2):
        pair = pivot.copy()
        pair["paired"] = pair[[branch_a, branch_b]].notna().all(axis=1)
        pair["delta_mae"] = pair[branch_b] - pair[branch_a]

        for keys, grp in pair.groupby(["outer_strategy_source", "target", "model_name"], dropna=False):
            pair_count = int(grp["paired"].sum())
            expected_count = int(len(grp))
            paired = grp[grp["paired"]]
            rows.append(
                {
                    "outer_strategy_source": keys[0],
                    "target": keys[1],
                    "model_name": keys[2],
                    "branch_a": branch_a,
                    "branch_b": branch_b,
                    "branch_a_mae_mean": float(paired[branch_a].mean()) if not paired.empty else float("nan"),
                    "branch_b_mae_mean": float(paired[branch_b].mean()) if not paired.empty else float("nan"),
                    "delta_mae_mean": float(paired["delta_mae"].mean()) if not paired.empty else float("nan"),
                    "delta_mae_std": float(paired["delta_mae"].std()) if len(paired) > 1 else float("nan"),
                    "pair_count": pair_count,
                    "expected_pair_count": expected_count,
                    "coverage_status": _coverage_status(pair_count, expected_count),
                }
            )

    return _save_table(pd.DataFrame(rows), out_dir, "branch_sensitivity.csv")


def table_strategy_sensitivity(
    outer_results: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    """
    Compare temporal_cv vs random_cv on matched outer folds for each
    ``(target, branch, model_name)`` combination.
    """
    strategy_col = _get_strategy_col(outer_results)
    required = {"outer_test_mae", "model_name", "target", "branch", "outer_test_year"}
    if strategy_col is None or not required.issubset(outer_results.columns):
        log.warning("strategy_sensitivity: required columns missing.")
        return pd.DataFrame()

    idx_cols = ["target", "branch", "model_name", "outer_test_year"]
    pivot = (
        outer_results[idx_cols + [strategy_col, "outer_test_mae"]]
        .pivot_table(index=idx_cols, columns=strategy_col, values="outer_test_mae", aggfunc="mean")
        .reset_index()
    )
    if "temporal_cv" not in pivot.columns:
        pivot["temporal_cv"] = np.nan
    if "random_cv" not in pivot.columns:
        pivot["random_cv"] = np.nan

    pivot["paired"] = pivot[["temporal_cv", "random_cv"]].notna().all(axis=1)
    pivot["delta_mae"] = pivot["random_cv"] - pivot["temporal_cv"]

    rows: list[dict[str, Any]] = []
    for keys, grp in pivot.groupby(["target", "branch", "model_name"], dropna=False):
        pair_count = int(grp["paired"].sum())
        expected_count = int(len(grp))
        paired = grp[grp["paired"]]
        rows.append(
            {
                "target": keys[0],
                "branch": keys[1],
                "model_name": keys[2],
                "temporal_cv_mae_mean": float(paired["temporal_cv"].mean()) if not paired.empty else float("nan"),
                "random_cv_mae_mean": float(paired["random_cv"].mean()) if not paired.empty else float("nan"),
                "delta_mae_mean": float(paired["delta_mae"].mean()) if not paired.empty else float("nan"),
                "delta_mae_std": float(paired["delta_mae"].std()) if len(paired) > 1 else float("nan"),
                "pair_count": pair_count,
                "expected_pair_count": expected_count,
                "coverage_status": _coverage_status(pair_count, expected_count),
            }
        )

    return _save_table(pd.DataFrame(rows), out_dir, "strategy_sensitivity.csv")


def table_scenario_sensitivity(
    outer_results: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    """
    Compare the three targets on matched outer folds for each
    ``(outer_strategy, branch, model_name)`` combination.
    """
    strategy_col = _get_strategy_col(outer_results)
    required = {"outer_test_mae", "model_name", "target", "branch", "outer_test_year"}
    if strategy_col is None or not required.issubset(outer_results.columns):
        log.warning("scenario_sensitivity: required columns missing.")
        return pd.DataFrame()

    idx_cols = [strategy_col, "branch", "model_name", "outer_test_year"]
    pivot = (
        outer_results[idx_cols + ["target", "outer_test_mae"]]
        .pivot_table(index=idx_cols, columns="target", values="outer_test_mae", aggfunc="mean")
        .reset_index()
    )
    pivot = pivot.rename(columns={strategy_col: "outer_strategy_source"})
    for target in ["linear", "optimistic", "pessimistic"]:
        if target not in pivot.columns:
            pivot[target] = np.nan

    paired_mask = pivot[["linear", "optimistic", "pessimistic"]].notna().all(axis=1)
    pivot["paired"] = paired_mask
    pivot["scenario_range_mae"] = pivot[["linear", "optimistic", "pessimistic"]].max(axis=1) - pivot[
        ["linear", "optimistic", "pessimistic"]
    ].min(axis=1)

    rows: list[dict[str, Any]] = []
    for keys, grp in pivot.groupby(["outer_strategy_source", "branch", "model_name"], dropna=False):
        pair_count = int(grp["paired"].sum())
        expected_count = int(len(grp))
        paired = grp[grp["paired"]]
        rows.append(
            {
                "outer_strategy_source": keys[0],
                "branch": keys[1],
                "model_name": keys[2],
                "linear_mae_mean": float(paired["linear"].mean()) if not paired.empty else float("nan"),
                "optimistic_mae_mean": float(paired["optimistic"].mean()) if not paired.empty else float("nan"),
                "pessimistic_mae_mean": float(paired["pessimistic"].mean()) if not paired.empty else float("nan"),
                "scenario_range_mae_mean": float(paired["scenario_range_mae"].mean()) if not paired.empty else float("nan"),
                "pair_count": pair_count,
                "expected_pair_count": expected_count,
                "coverage_status": _coverage_status(pair_count, expected_count),
            }
        )

    return _save_table(pd.DataFrame(rows), out_dir, "scenario_sensitivity.csv")


def _heatmap_or_message(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    index_col: str,
    column_col: str,
    value_col: str,
    title: str,
    cbar_label: str,
    cmap: str = "RdBu_r",
    center: float | None = 0.0,
) -> None:
    if df.empty or value_col not in df.columns:
        ax.axis("off")
        ax.text(0.5, 0.5, "Insufficient matched pairs", ha="center", va="center", fontsize=11)
        ax.set_title(title)
        return

    usable = df[df["pair_count"] > 0].copy() if "pair_count" in df.columns else df.copy()
    if usable.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "Insufficient matched pairs", ha="center", va="center", fontsize=11)
        ax.set_title(title)
        return

    pivot = usable.pivot_table(index=index_col, columns=column_col, values=value_col, aggfunc="mean")
    if pivot.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "Insufficient matched pairs", ha="center", va="center", fontsize=11)
        ax.set_title(title)
        return

    vmax = np.nanmax(np.abs(pivot.values))
    vmax = max(float(vmax) if np.isfinite(vmax) else 0.0, 0.01)
    heatmap_kwargs: dict[str, Any] = {
        "data": pivot,
        "ax": ax,
        "cmap": cmap,
        "annot": True,
        "fmt": ".3f",
        "linewidths": 0.5,
        "cbar_kws": {"label": cbar_label},
    }
    if center is None:
        heatmap_kwargs["vmin"] = 0
        heatmap_kwargs["vmax"] = vmax
    else:
        heatmap_kwargs["center"] = center
        heatmap_kwargs["vmin"] = -vmax
        heatmap_kwargs["vmax"] = vmax
    sns.heatmap(**heatmap_kwargs)
    ax.set_title(title, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.sca(ax)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)


def figure_sensitivity_heatmaps(
    branch_df: pd.DataFrame,
    strategy_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Heatmap figure summarising matched sensitivity comparisons.

    One panel per branch pair (``branch_a``/``branch_b`` in ``branch_df``, i.e. every
    ``C(n, 2)`` combination of active branches), plus one panel each for strategy and
    scenario sensitivity."""
    branch_pairs: list[tuple[str, str]] = []
    if not branch_df.empty and {"branch_a", "branch_b"}.issubset(branch_df.columns):
        branch_pairs = sorted(set(zip(branch_df["branch_a"], branch_df["branch_b"])))
    n_branch_panels = max(len(branch_pairs), 1)

    total_panels = n_branch_panels + 2
    fig, axes = plt.subplots(1, total_panels, figsize=(6 * total_panels, 6))
    if total_panels == 1:
        axes = [axes]

    if branch_pairs:
        for i, (branch_a, branch_b) in enumerate(branch_pairs):
            pair_df = branch_df[
                (branch_df["branch_a"] == branch_a) & (branch_df["branch_b"] == branch_b)
            ]
            _heatmap_or_message(
                axes[i],
                pair_df,
                index_col="model_name",
                column_col="target",
                value_col="delta_mae_mean",
                title=f"Branch ∆ MAE\n({branch_label(branch_b)} − {branch_label(branch_a)})",
                cbar_label="∆ MAE",
            )
    else:
        _heatmap_or_message(
            axes[0],
            branch_df,
            index_col="model_name",
            column_col="target",
            value_col="delta_mae_mean",
            title="Branch ∆ MAE",
            cbar_label="∆ MAE",
        )

    _heatmap_or_message(
        axes[n_branch_panels],
        strategy_df,
        index_col="model_name",
        column_col="target",
        value_col="delta_mae_mean",
        title="Strategy ∆ MAE\n(Random − Temporal CV)",
        cbar_label="∆ MAE",
    )
    _heatmap_or_message(
        axes[n_branch_panels + 1],
        scenario_df,
        index_col="model_name",
        column_col="outer_strategy_source",
        value_col="scenario_range_mae_mean",
        title="Scenario Spread\n(max − min outer MAE)",
        cbar_label="MAE Range",
        cmap="YlOrRd",
        center=None,
    )

    fig.suptitle("Sensitivity Analysis: Matched Fold Comparisons", fontsize=13)
    save_fig(out_dir / "sensitivity_heatmaps.png")
    log.info("Saved sensitivity heatmaps.")


def _figure_prediction_agreements(
    prediction_summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Country-level strategy agreement using mean predictions from
    ``final_prediction_summary.csv``.
    """
    if "prediction_mean" not in prediction_summary.columns:
        return

    strategy_col = _get_strategy_col(prediction_summary)
    if strategy_col is None or "country" not in prediction_summary.columns:
        return

    targets = sorted(prediction_summary["target"].dropna().unique()) if "target" in prediction_summary.columns else []
    for target in targets:
        df_t = prediction_summary[prediction_summary["target"] == target].copy()
        group_cols = [c for c in ["country", strategy_col, "branch"] if c in df_t.columns]
        agg = df_t.groupby(group_cols)["prediction_mean"].mean().reset_index()
        pivot = agg.pivot_table(
            index=[c for c in ["country", "branch"] if c in agg.columns],
            columns=strategy_col,
            values="prediction_mean",
        ).reset_index()
        if "temporal_cv" not in pivot.columns or "random_cv" not in pivot.columns:
            continue

        fig, ax = plt.subplots(figsize=(6, 5))
        if "branch" in pivot.columns:
            for branch, grp in pivot.groupby("branch"):
                ax.scatter(
                    grp["temporal_cv"],
                    grp["random_cv"],
                    label=str(branch),
                    alpha=0.65,
                    s=30,
                    edgecolors="white",
                )
        else:
            ax.scatter(
                pivot["temporal_cv"],
                pivot["random_cv"],
                alpha=0.65,
                s=30,
                edgecolors="white",
            )

        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, zorder=0)
        ax.set_xlabel("Temporal CV — mean prediction")
        ax.set_ylabel("Random CV — mean prediction")
        ax.set_title(f"Strategy Agreement: {target}")
        if "branch" in pivot.columns:
            ax.legend(title="Branch")

        save_fig(out_dir / f"prediction_agreement_{target}.png")
        log.info("Saved prediction agreement plot: %s", target)
