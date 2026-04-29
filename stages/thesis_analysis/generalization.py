"""
generalization.py — Section D: Cross-fold generalization analysis.

Source: outer_results.csv, final_results.csv
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from stages.post_analysis.matplotlib_setup import configure_matplotlib_environment
configure_matplotlib_environment()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stages.post_analysis.plot_utils import (
    apply_thesis_style, model_color_map, strategy_label,
    scenario_label, branch_label, add_diagonal, annotate_n,
)
from stages.post_analysis.config import FIGURE_DPI

from stages.thesis_analysis._shared import save_thesis_figure, save_thesis_table, strategy_col

log = logging.getLogger(__name__)

_BRANCH_LINESTYLES = {"simple": "-", "mice": "--"}
_TARGET_MARKERS = {"linear": "o", "optimistic": "^", "pessimistic": "s"}


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def table_inner_to_outer_degradation(outer_results: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Per (strategy, target, branch, model): mean/std degradation = outer_test_mae - inner_val_mean_mae."""
    try:
        if outer_results.empty:
            return pd.DataFrame()

        ok = outer_results[outer_results["status"] == "ok"].copy() if "status" in outer_results.columns else outer_results.copy()
        if "outer_test_mae" not in ok.columns or "inner_val_mean_mae" not in ok.columns:
            return pd.DataFrame()

        ok["degradation"] = ok["outer_test_mae"] - ok["inner_val_mean_mae"]
        scol = strategy_col(ok) or "outer_strategy"
        group_cols = [c for c in [scol, "target", "branch", "model_name"] if c in ok.columns]

        agg = ok.groupby(group_cols).agg(
            mean_inner_val_mae=("inner_val_mean_mae", "mean"),
            mean_outer_test_mae=("outer_test_mae", "mean"),
            mean_degradation=("degradation", "mean"),
            std_degradation=("degradation", "std"),
            n_folds=("degradation", "count"),
        ).reset_index().sort_values("mean_degradation", ascending=False)

        return save_thesis_table(agg, out_dir, "general_inner_to_outer_degradation",
                                 caption="Inner-to-outer MAE degradation per candidate: mean outer_test_mae - inner_val_mean_mae.")
    except Exception as exc:
        log.warning("table_inner_to_outer_degradation failed: %s", exc)
        return pd.DataFrame()


def table_year_difficulty(outer_results: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Per (strategy, target, outer_test_year): mean/std outer_test_mae and year difficulty rank."""
    try:
        if outer_results.empty or "outer_test_mae" not in outer_results.columns:
            return pd.DataFrame()

        ok = outer_results[outer_results["status"] == "ok"].copy() if "status" in outer_results.columns else outer_results.copy()
        scol = strategy_col(ok) or "outer_strategy"
        group_cols = [c for c in [scol, "target", "outer_test_year"] if c in ok.columns]

        agg = ok.groupby(group_cols).agg(
            mean_outer_test_mae=("outer_test_mae", "mean"),
            std_outer_test_mae=("outer_test_mae", "std"),
            n_candidates=("outer_test_mae", "count"),
        ).reset_index()

        strat_tgt = [c for c in [scol, "target"] if c in agg.columns]
        agg["year_difficulty_rank"] = agg.groupby(strat_tgt)["mean_outer_test_mae"].rank(method="min", ascending=False)

        return save_thesis_table(agg, out_dir, "general_year_difficulty",
                                 caption="Year difficulty ranking: mean outer test MAE per (strategy, target, outer_test_year).")
    except Exception as exc:
        log.warning("table_year_difficulty failed: %s", exc)
        return pd.DataFrame()


def table_strict_vs_lenient(outer_results: pd.DataFrame, final_results: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Per candidate: outer and final strict vs lenient MAE comparison."""
    try:
        rows = []

        if not outer_results.empty and "outer_test_mae" in outer_results.columns and "outer_test_mae_lenient" in outer_results.columns:
            ok = outer_results[outer_results["status"] == "ok"].copy() if "status" in outer_results.columns else outer_results.copy()
            scol = strategy_col(ok) or "outer_strategy"
            group_cols = [c for c in [scol, "target", "branch", "model_name"] if c in ok.columns]
            outer_agg = ok.groupby(group_cols).agg(
                outer_mean_mae_strict=("outer_test_mae", "mean"),
                outer_mean_mae_lenient=("outer_test_mae_lenient", "mean"),
            ).reset_index()
            outer_agg["outer_mean_mae_gap"] = outer_agg["outer_mean_mae_lenient"] - outer_agg["outer_mean_mae_strict"]
            rows.append(("outer", outer_agg, scol))

        if not final_results.empty and "final_model_test_mae" in final_results.columns and "final_model_test_mae_lenient" in final_results.columns:
            scol_f = strategy_col(final_results) or "outer_strategy_source"
            group_cols_f = [c for c in [scol_f, "target", "branch", "model_name", "candidate_rank", "candidate_role"] if c in final_results.columns]
            final_sub = final_results[group_cols_f + ["final_model_test_mae", "final_model_test_mae_lenient"]].copy()
            final_sub["final_mae_gap"] = final_sub["final_model_test_mae_lenient"] - final_sub["final_model_test_mae"]
            rows.append(("final", final_sub, scol_f))

        if not rows:
            return pd.DataFrame()

        # Combine outer and final on common keys
        if len(rows) == 2:
            _, outer_df, scol_o = rows[0]
            _, final_df, scol_f = rows[1]
            # Align strategy col name
            final_df = final_df.rename(columns={scol_f: scol_o})
            merge_cols = [c for c in [scol_o, "target", "branch", "model_name"] if c in outer_df.columns and c in final_df.columns]
            result = outer_df.merge(final_df, on=merge_cols, how="outer")
        else:
            _, result, _ = rows[0]

        return save_thesis_table(result, out_dir, "general_strict_vs_lenient",
                                 caption="Strict vs lenient test set MAE comparison at outer CV and final holdout levels.")
    except Exception as exc:
        log.warning("table_strict_vs_lenient failed: %s", exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_performance_by_year(outer_results: pd.DataFrame, out_dir: Path) -> None:
    """Line plot per model (solid=simple, dashed=mice) vs outer_test_year."""
    try:
        if outer_results.empty or "outer_test_mae" not in outer_results.columns:
            return
        apply_thesis_style()

        ok = outer_results[outer_results["status"] == "ok"].copy() if "status" in outer_results.columns else outer_results.copy()
        scol = strategy_col(ok) or "outer_strategy"
        group_cols = [c for c in [scol, "target"] if c in ok.columns]
        models = ok["model_name"].unique().tolist() if "model_name" in ok.columns else []
        colors = model_color_map(models)

        for keys, grp in ok.groupby(group_cols):
            if isinstance(keys, (str, int, float)):
                keys = (keys,)
            kv = dict(zip(group_cols, keys))
            strat = kv.get(scol, "")
            tgt = kv.get("target", "")

            years = sorted(grp["outer_test_year"].unique()) if "outer_test_year" in grp.columns else []
            if not years:
                continue

            fig, ax = plt.subplots(figsize=(7, 4))

            for model, mgrp in grp.groupby("model_name") if "model_name" in grp.columns else [("all", grp)]:
                for branch, bgrp in mgrp.groupby("branch") if "branch" in mgrp.columns else [("all", mgrp)]:
                    by_year = bgrp.groupby("outer_test_year")["outer_test_mae"].mean()
                    ls = _BRANCH_LINESTYLES.get(str(branch), "-")
                    color = colors.get(model, "steelblue")
                    ax.plot(by_year.index, by_year.values, color=color, linestyle=ls,
                            marker="o", markersize=5, label=f"{model}/{branch_label(str(branch))}")
                    for x, y in zip(by_year.index, by_year.values):
                        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(4, 3), fontsize=6)

            ax.set_xlabel("Outer test year")
            ax.set_ylabel("Outer test MAE")
            ax.set_title(f"Performance by holdout year — {strategy_label(strat)} | {scenario_label(tgt)}", fontsize=10)
            ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
            ax.set_ylim(bottom=0)
            ax.set_xticks(years)

            parts = [str(kv.get(c, "")) for c in group_cols]
            fname = "general_performance_by_year_" + "_".join(parts) + ".png"
            caption = (
                f"Outer test MAE by holdout year for {strategy_label(strat)} | {scenario_label(tgt)}. "
                "Solid lines = simple imputation, dashed = MICE. "
                "Reveals whether certain years are systematically more difficult to predict."
            )
            save_thesis_figure(out_dir / fname, caption)
    except Exception as exc:
        log.warning("figure_performance_by_year failed: %s", exc)


def figure_inner_to_outer_scatter(outer_results: pd.DataFrame, out_dir: Path) -> None:
    """Scatter of inner_val_mean_mae vs outer_test_mae; y=x diagonal; colored by model."""
    try:
        if outer_results.empty or "inner_val_mean_mae" not in outer_results.columns or "outer_test_mae" not in outer_results.columns:
            return
        apply_thesis_style()

        ok = outer_results[outer_results["status"] == "ok"].copy() if "status" in outer_results.columns else outer_results.copy()
        scol = strategy_col(ok) or "outer_strategy"
        models = ok["model_name"].unique().tolist() if "model_name" in ok.columns else []
        colors = model_color_map(models)

        for strat in ok[scol].unique() if scol in ok.columns else ["all"]:
            sub = ok[ok[scol] == strat] if scol in ok.columns else ok
            if sub.empty:
                continue

            fig, ax = plt.subplots(figsize=(7, 6))

            for model, mgrp in sub.groupby("model_name") if "model_name" in sub.columns else [("all", sub)]:
                for tgt, tgrp in mgrp.groupby("target") if "target" in mgrp.columns else [("all", mgrp)]:
                    marker = _TARGET_MARKERS.get(str(tgt), "o")
                    col = colors.get(model, "steelblue")
                    ax.scatter(tgrp["inner_val_mean_mae"], tgrp["outer_test_mae"],
                               color=col, marker=marker, alpha=0.6, s=30,
                               label=f"{model}/{scenario_label(str(tgt))}")

            add_diagonal(ax)
            ax.set_xlabel("Inner validation MAE (mean across inner folds)")
            ax.set_ylabel("Outer test MAE")
            ax.set_title(f"Inner-to-outer degradation — {strategy_label(strat)}", fontsize=10)
            ax.legend(fontsize=6, bbox_to_anchor=(1.01, 1), loc="upper left", ncol=1)
            annotate_n(ax, len(sub))

            caption = (
                f"Inner-to-outer degradation scatter for {strategy_label(strat)}. "
                "X = mean inner validation MAE; Y = outer test MAE on held-out year. "
                "Points above the y=x diagonal degraded more than inner CV predicted. "
                "Marker shapes distinguish target scenarios."
            )
            save_thesis_figure(out_dir / f"general_inner_to_outer_degradation_scatter_{strat}.png", caption)
    except Exception as exc:
        log.warning("figure_inner_to_outer_scatter failed: %s", exc)


def figure_year_difficulty_heatmap(year_difficulty: pd.DataFrame, out_dir: Path) -> None:
    """Heatmap: rows=targets, cols=years; cell=mean outer_test_mae."""
    try:
        if year_difficulty.empty or "mean_outer_test_mae" not in year_difficulty.columns:
            return
        apply_thesis_style()

        scol = strategy_col(year_difficulty) or "outer_strategy"
        strategies = year_difficulty[scol].unique() if scol in year_difficulty.columns else []
        targets = sorted(year_difficulty["target"].unique()) if "target" in year_difficulty.columns else []
        years = sorted(year_difficulty["outer_test_year"].unique()) if "outer_test_year" in year_difficulty.columns else []

        for strat in strategies:
            sub = year_difficulty[year_difficulty[scol] == strat] if scol in year_difficulty.columns else year_difficulty

            matrix = np.full((len(targets), len(years)), np.nan)
            for i, tgt in enumerate(targets):
                for j, yr in enumerate(years):
                    cell = sub[(sub["target"] == tgt) & (sub["outer_test_year"] == yr)]
                    if not cell.empty:
                        matrix[i, j] = cell["mean_outer_test_mae"].iloc[0]

            fig, ax = plt.subplots(figsize=(max(4, len(years) * 1.5), max(2, len(targets) * 1.2) + 1))
            im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
            plt.colorbar(im, ax=ax, label="Mean outer test MAE")
            ax.set_xticks(range(len(years)))
            ax.set_xticklabels([str(y) for y in years])
            ax.set_yticks(range(len(targets)))
            ax.set_yticklabels([scenario_label(t) for t in targets])
            ax.set_xlabel("Outer test year")
            ax.set_ylabel("Target scenario")
            ax.set_title(f"Year difficulty heatmap — {strategy_label(strat)}", fontsize=10)

            for i in range(len(targets)):
                for j in range(len(years)):
                    if not np.isnan(matrix[i, j]):
                        ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)

            caption = (
                f"Year difficulty heatmap for {strategy_label(strat)}: mean outer test MAE "
                "across all models per target scenario and holdout year. "
                "Darker = higher MAE = harder year."
            )
            save_thesis_figure(out_dir / f"general_year_difficulty_heatmap_{strat}.png", caption)
    except Exception as exc:
        log.warning("figure_year_difficulty_heatmap failed: %s", exc)


def figure_strict_vs_lenient_scatter(strict_lenient_df: pd.DataFrame, out_dir: Path) -> None:
    """Scatter of outer_mean_mae_strict vs outer_mean_mae_lenient; y=x diagonal."""
    try:
        if strict_lenient_df.empty:
            return
        xcol = next((c for c in ["outer_mean_mae_strict", "final_model_test_mae"] if c in strict_lenient_df.columns), None)
        ycol = next((c for c in ["outer_mean_mae_lenient", "final_model_test_mae_lenient"] if c in strict_lenient_df.columns), None)
        if xcol is None or ycol is None:
            return
        apply_thesis_style()

        scol = strategy_col(strict_lenient_df) or "outer_strategy"
        models = strict_lenient_df["model_name"].unique().tolist() if "model_name" in strict_lenient_df.columns else []
        colors = model_color_map(models)

        for strat in (strict_lenient_df[scol].unique() if scol in strict_lenient_df.columns else ["all"]):
            sub = strict_lenient_df[strict_lenient_df[scol] == strat] if scol in strict_lenient_df.columns else strict_lenient_df
            sub = sub.dropna(subset=[xcol, ycol])
            if sub.empty:
                continue

            fig, ax = plt.subplots(figsize=(6, 5))
            for model, mgrp in sub.groupby("model_name") if "model_name" in sub.columns else [("all", sub)]:
                ax.scatter(mgrp[xcol], mgrp[ycol], color=colors.get(model, "steelblue"),
                           alpha=0.7, s=40, label=model)
            add_diagonal(ax)
            ax.set_xlabel("MAE (strict test set)")
            ax.set_ylabel("MAE (lenient test set)")
            ax.set_title(f"Strict vs lenient MAE — {strategy_label(strat)}", fontsize=10)
            ax.legend(fontsize=8)
            annotate_n(ax, len(sub))

            caption = (
                f"Strict vs lenient test set MAE for {strategy_label(strat)}. "
                "X = outer mean MAE on strict set (drop_rows=True); Y = outer mean MAE on lenient set (drop_rows='targets_only'). "
                "Points below the y=x line: lenient evaluation is easier (fewer strict constraints). "
                "Points above: lenient set includes harder-to-predict countries."
            )
            save_thesis_figure(out_dir / f"general_strict_vs_lenient_scatter_{strat}.png", caption)
    except Exception as exc:
        log.warning("figure_strict_vs_lenient_scatter failed: %s", exc)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_generalization(outer_results: pd.DataFrame, final_results: pd.DataFrame,
                       out_dir: Path, cfg=None) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}

    deg = table_inner_to_outer_degradation(outer_results, out_dir)
    tables["general_inner_to_outer_degradation"] = deg

    yd = table_year_difficulty(outer_results, out_dir)
    tables["general_year_difficulty"] = yd

    sl = table_strict_vs_lenient(outer_results, final_results, out_dir)
    tables["general_strict_vs_lenient"] = sl

    figure_performance_by_year(outer_results, out_dir)
    figure_inner_to_outer_scatter(outer_results, out_dir)
    figure_year_difficulty_heatmap(yd, out_dir)
    figure_strict_vs_lenient_scatter(sl, out_dir)

    return tables
