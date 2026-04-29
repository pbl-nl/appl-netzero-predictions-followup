"""
scenarios.py — Section F: Scenario comparison (linear/optimistic/pessimistic).

Source: outer_results.csv, final_prediction_summary.csv
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from stages.post_analysis.matplotlib_setup import configure_matplotlib_environment
configure_matplotlib_environment()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stages.post_analysis.plot_utils import (
    apply_thesis_style, model_color_map, strategy_label,
    scenario_label, branch_label, annotate_n,
)
from stages.post_analysis.config import FIGURE_DPI, SCENARIO_COLORS, BRANCH_COLORS

from stages.thesis_analysis._shared import save_thesis_figure, save_thesis_table, strategy_col

log = logging.getLogger(__name__)

_TARGETS = ["linear", "optimistic", "pessimistic"]
_TARGET_PAIRS = [("linear", "optimistic"), ("linear", "pessimistic"), ("optimistic", "pessimistic")]


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def table_scenario_mae_by_model(outer_results: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Per (strategy, branch, model): MAE per scenario and scenario_spread."""
    try:
        if outer_results.empty or "outer_test_mae" not in outer_results.columns:
            return pd.DataFrame()

        ok = outer_results[outer_results["status"] == "ok"].copy() if "status" in outer_results.columns else outer_results.copy()
        scol = strategy_col(ok) or "outer_strategy"
        group_cols = [c for c in [scol, "target", "branch", "model_name"] if c in ok.columns]

        agg = ok.groupby(group_cols).agg(
            mean_outer_test_mae=("outer_test_mae", "mean"),
            std_outer_test_mae=("outer_test_mae", "std"),
            n_folds=("outer_test_mae", "count"),
        ).reset_index()

        # Pivot wide on target
        pivot_cols = [c for c in [scol, "branch", "model_name"] if c in agg.columns]
        if "target" not in agg.columns:
            return save_thesis_table(agg, out_dir, "scenario_mae_by_model",
                                     caption="Mean outer test MAE by scenario per model.")

        wide = agg.pivot_table(index=pivot_cols, columns="target",
                               values="mean_outer_test_mae").reset_index()
        wide.columns.name = None

        present_targets = [t for t in _TARGETS if t in wide.columns]
        if len(present_targets) > 1:
            wide["scenario_spread"] = wide[present_targets].max(axis=1) - wide[present_targets].min(axis=1)

        return save_thesis_table(wide, out_dir, "scenario_mae_by_model",
                                 caption="Mean outer test MAE by scenario (wide format) — reveals scenario difficulty differences.")
    except Exception as exc:
        log.warning("table_scenario_mae_by_model failed: %s", exc)
        return pd.DataFrame()


def table_scenario_error_correlation(prediction_summary: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    Pearson correlation of prediction errors across scenarios for countries with 2022 actuals.

    Only rows with actual.notna() are used (genuine held-out actuals, ~123 countries).
    """
    try:
        if prediction_summary.empty or "actual" not in prediction_summary.columns:
            return pd.DataFrame()

        df = prediction_summary[prediction_summary["actual"].notna()].copy()
        if df.empty or "prediction_mean" not in df.columns:
            return pd.DataFrame()

        df["error"] = df["actual"] - df["prediction_mean"]

        scol = strategy_col(df) or "outer_strategy_source"
        group_cols = [c for c in [scol, "branch", "model_name", "candidate_rank"] if c in df.columns]

        rows = []
        for keys, grp in df.groupby(group_cols):
            if isinstance(keys, (str, int, float)):
                keys = (keys,)
            kv = dict(zip(group_cols, keys))

            # Pivot on target for this candidate
            if "target" not in grp.columns or "country" not in grp.columns:
                continue
            pivot = grp.pivot_table(index="country", columns="target", values="error")

            row = dict(kv)
            n = len(pivot.dropna())
            row["n_countries"] = n

            for t1, t2 in _TARGET_PAIRS:
                key = f"corr_{t1}_{t2}"
                if t1 in pivot.columns and t2 in pivot.columns:
                    both = pivot[[t1, t2]].dropna()
                    if len(both) >= 3:
                        r, _ = scipy_stats.pearsonr(both[t1], both[t2])
                        row[key] = r
                    else:
                        row[key] = np.nan
                else:
                    row[key] = np.nan
            rows.append(row)

        if not rows:
            return pd.DataFrame()
        result = pd.DataFrame(rows)
        return save_thesis_table(result, out_dir, "scenario_error_correlation",
                                 caption="Pearson correlation of country-level prediction errors across target scenarios (countries with 2022 actuals only).")
    except Exception as exc:
        log.warning("table_scenario_error_correlation failed: %s", exc)
        return pd.DataFrame()


def table_scenario_country_rank_stability(prediction_summary: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Spearman rho of country error-rank consistency across scenarios."""
    try:
        if prediction_summary.empty or "actual" not in prediction_summary.columns:
            return pd.DataFrame()

        df = prediction_summary[prediction_summary["actual"].notna()].copy()
        if df.empty or "prediction_mean" not in df.columns:
            return pd.DataFrame()

        df["abs_error"] = (df["actual"] - df["prediction_mean"]).abs()

        scol = strategy_col(df) or "outer_strategy_source"
        group_cols = [c for c in [scol, "branch", "model_name", "candidate_rank"] if c in df.columns]

        rows = []
        for keys, grp in df.groupby(group_cols):
            if isinstance(keys, (str, int, float)):
                keys = (keys,)
            kv = dict(zip(group_cols, keys))

            if "target" not in grp.columns or "country" not in grp.columns:
                continue
            pivot = grp.pivot_table(index="country", columns="target", values="abs_error")

            row = dict(kv)
            row["n_countries"] = len(pivot.dropna())

            for t1, t2 in _TARGET_PAIRS:
                key = f"spearman_{t1}_{t2}"
                if t1 in pivot.columns and t2 in pivot.columns:
                    both = pivot[[t1, t2]].dropna()
                    if len(both) >= 3:
                        r, _ = scipy_stats.spearmanr(both[t1], both[t2])
                        row[key] = r
                    else:
                        row[key] = np.nan
                else:
                    row[key] = np.nan
            rows.append(row)

        if not rows:
            return pd.DataFrame()
        result = pd.DataFrame(rows)
        return save_thesis_table(result, out_dir, "scenario_country_rank_stability",
                                 caption="Spearman rank correlation of country error ranks across target scenarios — measures whether the same countries are hardest across all scenarios.")
    except Exception as exc:
        log.warning("table_scenario_country_rank_stability failed: %s", exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_scenario_mae_comparison_bar(scenario_mae: pd.DataFrame, outer_results: pd.DataFrame, out_dir: Path) -> None:
    """
    Grouped bar chart: X=model, three bars per group (one per scenario).
    Faceted by branch. One file per strategy.
    """
    try:
        if outer_results.empty or "outer_test_mae" not in outer_results.columns:
            return
        apply_thesis_style()

        ok = outer_results[outer_results["status"] == "ok"].copy() if "status" in outer_results.columns else outer_results.copy()
        scol = strategy_col(ok) or "outer_strategy"

        for strat in ok[scol].unique() if scol in ok.columns else ["all"]:
            sub = ok[ok[scol] == strat] if scol in ok.columns else ok
            branches = sorted(sub["branch"].unique()) if "branch" in sub.columns else ["all"]
            n_branches = len(branches)

            fig, axes = plt.subplots(1, n_branches, figsize=(max(6, n_branches * 6), 5), sharey=True)
            if n_branches == 1:
                axes = [axes]

            targets_present = [t for t in _TARGETS if t in sub.get("target", pd.Series()).unique()] \
                if "target" in sub.columns else []
            x_width = 0.8 / max(len(targets_present), 1)

            for ax, br in zip(axes, branches):
                br_sub = sub[sub["branch"] == br] if "branch" in sub.columns else sub
                models = sorted(br_sub["model_name"].unique()) if "model_name" in br_sub.columns else []
                x = np.arange(len(models))

                for i, tgt in enumerate(targets_present):
                    tgt_sub = br_sub[br_sub["target"] == tgt] if "target" in br_sub.columns else br_sub
                    means = tgt_sub.groupby("model_name")["outer_test_mae"].mean().reindex(models)
                    stds = tgt_sub.groupby("model_name")["outer_test_mae"].std().reindex(models).fillna(0)
                    offset = (i - len(targets_present) / 2 + 0.5) * x_width
                    color = SCENARIO_COLORS.get(tgt, "steelblue")
                    ax.bar(x + offset, means.fillna(0), width=x_width * 0.9,
                           color=color, alpha=0.8, label=scenario_label(tgt),
                           yerr=stds, capsize=3, edgecolor="white")

                ax.set_xticks(x)
                ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
                ax.set_title(f"{branch_label(br)}", fontsize=9)
                ax.set_ylabel("Mean outer test MAE" if br == branches[0] else "")
                ax.set_ylim(bottom=0)

            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper right", fontsize=8, bbox_to_anchor=(1, 1))
            fig.suptitle(f"Scenario difficulty comparison — {strategy_label(strat)}", fontsize=11)

            caption = (
                f"Mean outer test MAE by scenario and model for {strategy_label(strat)}. "
                "Groups = models; bars = scenarios (linear/optimistic/pessimistic); panels = imputation branch. "
                "Error bars = std across outer folds. Shows which target scenario is hardest to predict."
            )
            save_thesis_figure(out_dir / f"scenario_mae_comparison_bar_{strat}.png", caption)
    except Exception as exc:
        log.warning("figure_scenario_mae_comparison_bar failed: %s", exc)


def figure_error_correlation_heatmap(corr_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Heatmap of Pearson r for each scenario pair per model, one per (strategy, branch).
    """
    try:
        if corr_df.empty:
            return
        apply_thesis_style()

        scol = strategy_col(corr_df) or "outer_strategy_source"
        group_cols = [c for c in [scol, "branch"] if c in corr_df.columns]

        corr_pair_cols = [c for c in corr_df.columns if c.startswith("corr_")]
        if not corr_pair_cols:
            return

        for keys, grp in corr_df.groupby(group_cols):
            if isinstance(keys, (str, int, float)):
                keys = (keys,)
            kv = dict(zip(group_cols, keys))
            strat = kv.get(scol, "")
            br = kv.get("branch", "")

            models = sorted(grp["model_name"].unique()) if "model_name" in grp.columns else []
            if not models:
                continue

            # Aggregate across candidate ranks
            model_agg = grp.groupby("model_name")[corr_pair_cols].mean()
            matrix = model_agg.reindex(models).values

            n = int(grp["n_countries"].max()) if "n_countries" in grp.columns else "?"

            fig, ax = plt.subplots(figsize=(max(4, len(corr_pair_cols) * 1.5), max(3, len(models) * 0.8) + 1))
            im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax, label="Pearson r")
            ax.set_xticks(range(len(corr_pair_cols)))
            ax.set_xticklabels([c.replace("corr_", "").replace("_", " vs ") for c in corr_pair_cols],
                               rotation=20, ha="right", fontsize=8)
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels(models, fontsize=8)
            ax.set_title(f"Scenario error correlation — {strategy_label(strat)} | {branch_label(br)}", fontsize=10)

            for i in range(len(models)):
                for j in range(len(corr_pair_cols)):
                    val = matrix[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                fontsize=8, color="white" if abs(val) > 0.6 else "black")

            ax.text(0.98, 0.02, f"N = {n} countries with actuals",
                    transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

            caption = (
                f"Country-level prediction error correlation across scenarios for "
                f"{strategy_label(strat)} | {branch_label(br)} (N = {n} countries with 2022 actuals). "
                "Red = high correlation (model makes systematic errors across all scenarios); "
                "blue = scenarios diverge."
            )
            save_thesis_figure(out_dir / f"scenario_error_correlation_heatmap_{strat}_{br}.png", caption)
    except Exception as exc:
        log.warning("figure_error_correlation_heatmap failed: %s", exc)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_scenarios(outer_results: pd.DataFrame, prediction_summary: pd.DataFrame,
                  out_dir: Path, cfg=None) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}

    scenario_mae = table_scenario_mae_by_model(outer_results, out_dir)
    tables["scenario_mae_by_model"] = scenario_mae

    error_corr = table_scenario_error_correlation(prediction_summary, out_dir)
    tables["scenario_error_correlation"] = error_corr

    rank_stability = table_scenario_country_rank_stability(prediction_summary, out_dir)
    tables["scenario_country_rank_stability"] = rank_stability

    figure_scenario_mae_comparison_bar(scenario_mae, outer_results, out_dir)
    figure_error_correlation_heatmap(error_corr, out_dir)

    return tables
