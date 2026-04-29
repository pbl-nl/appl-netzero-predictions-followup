"""
bootstrap.py — Section E: Bootstrap stability analysis.

Source: final_bootstrap_predictions.csv, final_prediction_summary.csv, final_results.csv

Constraint: 2022 actuals are present for ~123 of ~280 countries.
All MAE computations filter to actual.notna() and annotate figures with N.
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

from stages.thesis_analysis._shared import (
    compute_bootstrap_iteration_mae,
    save_thesis_figure,
    save_thesis_table,
    strategy_col,
)
from stages.thesis_analysis.config import BOOTSTRAP_CI_TOP_N

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def table_running_mean_convergence(bootstrap_preds: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Running mean MAE over bootstrap iterations per candidate."""
    try:
        iter_mae = compute_bootstrap_iteration_mae(bootstrap_preds)
        if iter_mae.empty:
            return pd.DataFrame()

        scol = strategy_col(iter_mae) or "outer_strategy_source"
        id_cols = [c for c in [scol, "target", "branch", "model_name", "candidate_rank", "candidate_role"]
                   if c in iter_mae.columns]

        rows = []
        for keys, grp in iter_mae.groupby(id_cols):
            grp = grp.sort_values("iteration")
            mae_vals = grp["mae_on_actuals"].values
            n = grp["n_rows_with_actual"].values[0] if "n_rows_with_actual" in grp.columns else np.nan
            running_mean = np.cumsum(mae_vals) / np.arange(1, len(mae_vals) + 1)
            running_std = np.array([mae_vals[:i+1].std() for i in range(len(mae_vals))])
            running_range = np.array([mae_vals[:i+1].max() - mae_vals[:i+1].min() for i in range(len(mae_vals))])
            for it_idx, (it, mv, rm, rs, rr) in enumerate(zip(
                grp["iteration"].values, mae_vals, running_mean, running_std, running_range
            )):
                row = dict(zip(id_cols, keys if isinstance(keys, tuple) else (keys,)))
                row.update({
                    "iteration": it,
                    "mae_on_actuals": mv,
                    "running_mean_mae": rm,
                    "running_std_mae": rs,
                    "running_range": rr,
                    "n_rows_with_actual": n,
                })
                rows.append(row)

        result = pd.DataFrame(rows)
        return save_thesis_table(result, out_dir, "bootstrap_running_mean_convergence",
                                 caption="Running mean bootstrap test MAE per iteration per candidate.")
    except Exception as exc:
        log.warning("table_running_mean_convergence failed: %s", exc)
        return pd.DataFrame()


def table_country_ci_width(prediction_summary: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Per (strategy, target, country): CI width (p95-p05) and rank."""
    try:
        if prediction_summary.empty:
            return pd.DataFrame()

        df = prediction_summary.copy()
        if "prediction_p95" not in df.columns or "prediction_p05" not in df.columns:
            return pd.DataFrame()

        df["ci_width"] = df["prediction_p95"] - df["prediction_p05"]
        df["has_actual"] = df["actual"].notna() if "actual" in df.columns else False

        scol = strategy_col(df) or "outer_strategy_source"
        group_cols = [c for c in [scol, "target"] if c in df.columns]

        country_id = [c for c in [scol, "target", "branch", "model_name", "candidate_rank", "country"] if c in df.columns]
        agg = df.groupby(country_id).agg(
            mean_ci_width=("ci_width", "mean"),
            mean_prediction_std=("prediction_std", "mean") if "prediction_std" in df.columns else ("ci_width", "mean"),
            has_actual=("has_actual", "max"),
        ).reset_index()

        # Rank within (strategy, target)
        rank_cols = [c for c in group_cols if c in agg.columns]
        if rank_cols:
            agg["ci_width_rank"] = agg.groupby(rank_cols)["mean_ci_width"].rank(ascending=False, method="min")

        return save_thesis_table(agg, out_dir, "bootstrap_country_ci_width",
                                 caption="Country-level bootstrap CI width (p95−p05) with ranking.")
    except Exception as exc:
        log.warning("table_country_ci_width failed: %s", exc)
        return pd.DataFrame()


def table_point_vs_bootstrap_mean(final_results: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Per winner: deterministic final_model_test_mae vs bootstrap_test_mae_mean."""
    try:
        if final_results.empty:
            return pd.DataFrame()

        cols = [c for c in [
            "target", "branch", "model_name", "candidate_rank", "candidate_role",
            "final_model_test_mae", "bootstrap_test_mae_mean",
            "bootstrap_test_mae_p05", "bootstrap_test_mae_p95",
        ] if c in final_results.columns]
        scol = strategy_col(final_results)
        if scol:
            cols = [scol] + cols

        result = final_results[cols].copy()
        if "final_model_test_mae" in result.columns and "bootstrap_test_mae_mean" in result.columns:
            result["alignment_gap"] = result["bootstrap_test_mae_mean"] - result["final_model_test_mae"]

        return save_thesis_table(result, out_dir, "bootstrap_point_vs_mean_comparison",
                                 caption="Single-fit final test MAE vs bootstrap mean — alignment between deterministic and bootstrap evaluations.")
    except Exception as exc:
        log.warning("table_point_vs_bootstrap_mean failed: %s", exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_running_mean_convergence(convergence_df: pd.DataFrame, out_dir: Path) -> None:
    """Running mean MAE + ±1 std band per candidate."""
    try:
        if convergence_df.empty:
            return
        apply_thesis_style()

        scol = strategy_col(convergence_df) or "outer_strategy_source"
        id_cols = [c for c in [scol, "target", "branch", "model_name", "candidate_rank", "candidate_role"]
                   if c in convergence_df.columns]

        for keys, grp in convergence_df.groupby(id_cols):
            if isinstance(keys, (str, int, float)):
                keys = (keys,)
            kv = dict(zip(id_cols, keys))
            grp = grp.sort_values("iteration")

            n_actuals = int(grp["n_rows_with_actual"].iloc[0]) if "n_rows_with_actual" in grp.columns else "?"
            final_val = grp["running_mean_mae"].iloc[-1]

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(grp["iteration"], grp["running_mean_mae"], color="steelblue", linewidth=2, label="Running mean MAE")
            if "running_std_mae" in grp.columns:
                ax.fill_between(grp["iteration"],
                                grp["running_mean_mae"] - grp["running_std_mae"],
                                grp["running_mean_mae"] + grp["running_std_mae"],
                                alpha=0.2, color="steelblue", label="±1 std")
            ax.axhline(final_val, color="steelblue", linestyle=":", linewidth=1, label=f"Final value ({final_val:.4f})")
            ax.set_xlabel("Bootstrap iteration")
            ax.set_ylabel("Running mean MAE")
            strat = kv.get(scol, "")
            tgt = kv.get("target", "")
            br = kv.get("branch", "")
            model = kv.get("model_name", "")
            rank = kv.get("candidate_rank", "")
            ax.set_title(f"Bootstrap convergence — {model} | {strategy_label(strat)} | {scenario_label(tgt)} | {branch_label(br)} | rank {rank}",
                         fontsize=9)
            ax.legend(fontsize=8)
            ax.set_xlim(left=1)

            ax.text(0.97, 0.05, f"N = {n_actuals} countries with 2022 actuals",
                    transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

            parts = [str(kv.get(c, "")) for c in id_cols]
            fname = "bootstrap_running_mean_" + "_".join(parts) + ".png"
            caption = (
                f"Bootstrap running-mean MAE convergence for {model} | {strategy_label(strat)} | "
                f"{scenario_label(tgt)} | {branch_label(br)} | rank {rank}. "
                "Shaded band = ±1 std. Horizontal dotted line = final value. "
                f"N = {n_actuals} countries with 2022 actuals used for MAE computation."
            )
            save_thesis_figure(out_dir / fname, caption)
    except Exception as exc:
        log.warning("figure_running_mean_convergence failed: %s", exc)


def figure_bootstrap_mae_distribution(bootstrap_preds: pd.DataFrame, out_dir: Path) -> None:
    """Overlaid histograms of per-iteration MAE per candidate, one per (strategy, target)."""
    try:
        iter_mae = compute_bootstrap_iteration_mae(bootstrap_preds)
        if iter_mae.empty:
            return
        apply_thesis_style()

        scol = strategy_col(iter_mae) or "outer_strategy_source"
        group_cols = [c for c in [scol, "target"] if c in iter_mae.columns]
        id_cols = [c for c in [scol, "target", "branch", "model_name", "candidate_rank", "candidate_role"]
                   if c in iter_mae.columns]

        for keys, grp in iter_mae.groupby(group_cols):
            if isinstance(keys, (str, int, float)):
                keys = (keys,)
            kv = dict(zip(group_cols, keys))
            strat = kv.get(scol, "")
            tgt = kv.get("target", "")

            n_actuals = int(grp["n_rows_with_actual"].mode().iloc[0]) if "n_rows_with_actual" in grp.columns else "?"

            fig, ax = plt.subplots(figsize=(8, 4))
            cand_id_cols = [c for c in id_cols if c in grp.columns and c not in group_cols]

            for cand_keys, cand_grp in grp.groupby(cand_id_cols) if cand_id_cols else [("all", grp)]:
                if isinstance(cand_keys, (str, int, float)):
                    cand_keys = (cand_keys,)
                ckv = dict(zip(cand_id_cols, cand_keys))
                role = ckv.get("candidate_role", "primary")
                model = ckv.get("model_name", "")
                br = ckv.get("branch", "")
                rank = ckv.get("candidate_rank", "")
                color = "steelblue" if role == "primary" else "darkorange"
                label = f"{model}/{branch_label(str(br))} r{rank} ({role})"
                mae_vals = cand_grp["mae_on_actuals"].dropna()
                if len(mae_vals) < 2:
                    continue
                ax.hist(mae_vals, bins=20, alpha=0.5, color=color, label=label, edgecolor="white")
                ax.axvline(mae_vals.mean(), color=color, linewidth=1.5, linestyle="--")

            ax.set_xlabel("Per-iteration MAE")
            ax.set_ylabel("Count of iterations")
            ax.set_title(f"Bootstrap MAE distribution — {strategy_label(strat)} | {scenario_label(tgt)}", fontsize=10)
            ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
            ax.text(0.02, 0.97, f"N = {n_actuals} countries with actuals",
                    transform=ax.transAxes, fontsize=7, va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

            parts = [str(kv.get(c, "")) for c in group_cols]
            fname = "bootstrap_mae_distribution_" + "_".join(parts) + ".png"
            caption = (
                f"Distribution of per-iteration bootstrap MAE for {strategy_label(strat)} | {scenario_label(tgt)}. "
                f"Each distribution = 200 resampling iterations on N = {n_actuals} countries with 2022 actuals. "
                "Blue = primary candidates; orange = backup. Dashed vertical lines = means."
            )
            save_thesis_figure(out_dir / fname, caption)
    except Exception as exc:
        log.warning("figure_bootstrap_mae_distribution failed: %s", exc)


def figure_country_ci_width_top_n(ci_width_df: pd.DataFrame, out_dir: Path) -> None:
    """Horizontal bar chart: top-N countries by CI width, one per (strategy, target)."""
    try:
        if ci_width_df.empty or "mean_ci_width" not in ci_width_df.columns:
            return
        apply_thesis_style()

        scol = strategy_col(ci_width_df) or "outer_strategy_source"
        group_cols = [c for c in [scol, "target"] if c in ci_width_df.columns]

        for keys, grp in ci_width_df.groupby(group_cols):
            if isinstance(keys, (str, int, float)):
                keys = (keys,)
            kv = dict(zip(group_cols, keys))
            strat = kv.get(scol, "")
            tgt = kv.get("target", "")

            # Aggregate across candidates for same country
            country_agg = grp.groupby("country")["mean_ci_width"].mean().reset_index() if "country" in grp.columns else grp
            has_actual = grp.groupby("country")["has_actual"].max().reset_index() if "has_actual" in grp.columns and "country" in grp.columns else pd.DataFrame()
            if not has_actual.empty:
                country_agg = country_agg.merge(has_actual, on="country", how="left")

            top = country_agg.nlargest(BOOTSTRAP_CI_TOP_N, "mean_ci_width")
            if top.empty:
                continue

            top = top.sort_values("mean_ci_width", ascending=True)
            colors = []
            for _, row in top.iterrows():
                ha = row.get("has_actual", False)
                colors.append("steelblue" if ha else "lightgrey")

            fig, ax = plt.subplots(figsize=(7, max(4, len(top) * 0.4 + 1)))
            ax.barh(range(len(top)), top["mean_ci_width"], color=colors, height=0.6, edgecolor="white")
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(top["country"].tolist(), fontsize=8)
            ax.set_xlabel("Mean bootstrap CI width (p95−p05)")
            ax.set_title(f"Top {BOOTSTRAP_CI_TOP_N} countries by CI width — {strategy_label(strat)} | {scenario_label(tgt)}",
                         fontsize=9)

            import matplotlib.patches as mpatches
            patches = [
                mpatches.Patch(color="steelblue", label="Has 2022 actual"),
                mpatches.Patch(color="lightgrey", label="Prediction-only country"),
            ]
            ax.legend(handles=patches, fontsize=8, loc="lower right")

            parts = [str(kv.get(c, "")) for c in group_cols]
            fname = f"bootstrap_ci_width_top{BOOTSTRAP_CI_TOP_N}_" + "_".join(parts) + ".png"
            caption = (
                f"Top {BOOTSTRAP_CI_TOP_N} countries by mean bootstrap CI width (p95−p05) for "
                f"{strategy_label(strat)} | {scenario_label(tgt)}. "
                "Blue = country has 2022 actual; grey = prediction-only (no ground truth available)."
            )
            save_thesis_figure(out_dir / fname, caption)
    except Exception as exc:
        log.warning("figure_country_ci_width_top_n failed: %s", exc)


def figure_point_vs_bootstrap_mean_scatter(point_vs_mean: pd.DataFrame, out_dir: Path) -> None:
    """Scatter of final_model_test_mae vs bootstrap_test_mae_mean with CI error bars."""
    try:
        if point_vs_mean.empty:
            return
        xcol = "final_model_test_mae"
        ycol = "bootstrap_test_mae_mean"
        if xcol not in point_vs_mean.columns or ycol not in point_vs_mean.columns:
            return
        apply_thesis_style()

        scol = strategy_col(point_vs_mean) or "outer_strategy_source"
        models = point_vs_mean["model_name"].unique().tolist() if "model_name" in point_vs_mean.columns else []
        colors = model_color_map(models)

        for strat in (point_vs_mean[scol].unique() if scol in point_vs_mean.columns else ["all"]):
            sub = point_vs_mean[point_vs_mean[scol] == strat] if scol in point_vs_mean.columns else point_vs_mean
            sub = sub.dropna(subset=[xcol, ycol])
            if sub.empty:
                continue

            fig, ax = plt.subplots(figsize=(6, 5))
            for model, mgrp in sub.groupby("model_name") if "model_name" in sub.columns else [("all", sub)]:
                col = colors.get(model, "steelblue")
                xerr = None
                if "bootstrap_test_mae_p05" in mgrp.columns and "bootstrap_test_mae_p95" in mgrp.columns:
                    y_lo = mgrp[ycol] - mgrp["bootstrap_test_mae_p05"]
                    y_hi = mgrp["bootstrap_test_mae_p95"] - mgrp[ycol]
                    yerr = np.array([y_lo.clip(lower=0).values, y_hi.clip(lower=0).values])
                else:
                    yerr = None

                ax.errorbar(mgrp[xcol], mgrp[ycol],
                            yerr=yerr, fmt="o", color=col, alpha=0.8,
                            capsize=4, label=model, markersize=6)

                # Annotate rank
                for _, row in mgrp.iterrows():
                    rank = row.get("candidate_rank", "")
                    br = row.get("branch", "")
                    ax.annotate(f"r{rank}/{branch_label(str(br))}", (row[xcol], row[ycol]),
                                textcoords="offset points", xytext=(5, 3), fontsize=6)

            add_diagonal(ax)
            ax.set_xlabel("Final model test MAE (single fit)")
            ax.set_ylabel("Bootstrap mean test MAE (200 iterations)")
            ax.set_title(f"Point estimate vs bootstrap mean — {strategy_label(strat)}", fontsize=10)
            ax.legend(fontsize=8)
            annotate_n(ax, len(sub))

            caption = (
                f"Point estimate vs bootstrap mean MAE for {strategy_label(strat)}. "
                "X = final single-model test MAE on 2022 holdout. Y = mean over 200 bootstrap iterations. "
                "Error bars = p05–p95 CI. Points above diagonal: bootstrap mean is pessimistic relative to single-fit."
            )
            save_thesis_figure(out_dir / f"bootstrap_point_vs_mean_scatter_{strat}.png", caption)
    except Exception as exc:
        log.warning("figure_point_vs_bootstrap_mean_scatter failed: %s", exc)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_bootstrap(bootstrap_preds: pd.DataFrame, prediction_summary: pd.DataFrame,
                  final_results: pd.DataFrame, out_dir: Path, cfg=None) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}

    convergence = table_running_mean_convergence(bootstrap_preds, out_dir)
    tables["bootstrap_running_mean_convergence"] = convergence

    ci_width = table_country_ci_width(prediction_summary, out_dir)
    tables["bootstrap_country_ci_width"] = ci_width

    point_vs_mean = table_point_vs_bootstrap_mean(final_results, out_dir)
    tables["bootstrap_point_vs_mean_comparison"] = point_vs_mean

    figure_running_mean_convergence(convergence, out_dir)
    figure_bootstrap_mae_distribution(bootstrap_preds, out_dir)
    figure_country_ci_width_top_n(ci_width, out_dir)
    figure_point_vs_bootstrap_mean_scatter(point_vs_mean, out_dir)

    return tables
