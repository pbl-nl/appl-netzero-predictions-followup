"""
overfitting.py — Section C: Overfitting audit deep dive.

Source: overfitting_audit.csv, trials.csv, outer_results.csv
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

from stages.post_analysis.plot_utils import apply_thesis_style, model_color_map, strategy_label, scenario_label
from stages.post_analysis.config import FIGURE_DPI

from stages.thesis_analysis._shared import save_thesis_figure, save_thesis_table, strategy_col
from stages.thesis_analysis.config import OUTER_GAP_ABS_THRESHOLD, OVERFIT_INNER_GAP_RATIO_THRESHOLD

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def table_gap_summary_by_model(overfitting_audit: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Per (strategy, target, model): mean/std gaps and flag rates."""
    try:
        if overfitting_audit.empty:
            return pd.DataFrame()

        ok = overfitting_audit[overfitting_audit["status"] == "ok"].copy() \
            if "status" in overfitting_audit.columns else overfitting_audit.copy()

        scol = strategy_col(ok) or "outer_strategy"
        group_cols = [c for c in [scol, "target", "model_name"] if c in ok.columns]

        agg_dict: dict[str, object] = {"n_folds": ("outer_test_mae", "count")} \
            if "outer_test_mae" in ok.columns else {}

        for metric_col, agg_name in [
            ("inner_train_val_gap_rel_mae", "inner_gap_rel_mae"),
            ("outer_generalization_gap_abs_mae", "outer_gap_abs_mae"),
            ("outer_generalization_gap_rel_mae", "outer_gap_rel_mae"),
        ]:
            if metric_col in ok.columns:
                agg_dict[f"mean_{agg_name}"] = (metric_col, "mean")
                agg_dict[f"std_{agg_name}"] = (metric_col, "std")

        for flag_col in ["inner_gap_flagged", "outer_gap_flagged"]:
            if flag_col in ok.columns:
                agg_dict[f"{flag_col}_rate"] = (flag_col, "mean")

        if not agg_dict:
            return pd.DataFrame()

        result = ok.groupby(group_cols).agg(**agg_dict).reset_index()

        return save_thesis_table(result, out_dir, "overfit_gap_summary_by_model",
                                 caption="Overfitting gap summary per (strategy, target, model): mean/std gap metrics and flag rates.")
    except Exception as exc:
        log.warning("table_gap_summary_by_model failed: %s", exc)
        return pd.DataFrame()


def table_flagged_candidates(overfitting_audit: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Candidates flagged across all folds for outer gap."""
    try:
        if overfitting_audit.empty or "candidate_all_folds_outer_gap_flagged" not in overfitting_audit.columns:
            return pd.DataFrame()

        flagged = overfitting_audit[overfitting_audit["candidate_all_folds_outer_gap_flagged"] == True].copy()
        if flagged.empty:
            return pd.DataFrame()

        scol = strategy_col(flagged) or "outer_strategy"
        group_cols = [c for c in [scol, "target", "branch", "model_name"] if c in flagged.columns]

        agg: dict[str, object] = {}
        for col in ["candidate_all_folds_inner_gap_flagged", "selected_as_primary", "selected_as_backup"]:
            if col in flagged.columns:
                agg[col] = (col, "max")
        for col in ["outer_generalization_gap_abs_mae", "outer_generalization_gap_rel_mae"]:
            if col in flagged.columns:
                agg[f"mean_{col}"] = (col, "mean")

        if not agg:
            result = flagged[group_cols].drop_duplicates()
        else:
            result = flagged.groupby(group_cols).agg(**agg).reset_index()

        return save_thesis_table(result, out_dir, "overfit_flagged_candidates",
                                 caption="Candidates flagged on all outer folds for outer generalization gap.")
    except Exception as exc:
        log.warning("table_flagged_candidates failed: %s", exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_inner_vs_outer_gap_scatter(audit_df: pd.DataFrame, out_dir: Path) -> None:
    """Inner vs outer gap scatter per strategy: quadrant view with flag markers."""
    try:
        if audit_df.empty:
            return
        apply_thesis_style()

        ok = audit_df[audit_df["status"] == "ok"].copy() if "status" in audit_df.columns else audit_df.copy()
        if "inner_train_val_gap_rel_mae" not in ok.columns or "outer_generalization_gap_abs_mae" not in ok.columns:
            return

        scol = strategy_col(ok) or "outer_strategy"
        strategies = ok[scol].unique() if scol in ok.columns else ["all"]
        models = ok["model_name"].unique().tolist() if "model_name" in ok.columns else []
        colors = model_color_map(models)

        for strat in strategies:
            sub = ok[ok[scol] == strat] if scol in ok.columns else ok
            if sub.empty:
                continue

            fig, ax = plt.subplots(figsize=(7, 6))

            for model, grp in sub.groupby("model_name") if "model_name" in sub.columns else [("all", sub)]:
                flagged = grp["outer_gap_flagged"].astype(bool) if "outer_gap_flagged" in grp.columns else pd.Series(False, index=grp.index)
                col = colors.get(model, "steelblue")
                ax.scatter(
                    grp.loc[~flagged, "inner_train_val_gap_rel_mae"],
                    grp.loc[~flagged, "outer_generalization_gap_abs_mae"],
                    color=col, marker="o", alpha=0.6, s=30, label=model if not flagged.all() else None,
                )
                ax.scatter(
                    grp.loc[flagged, "inner_train_val_gap_rel_mae"],
                    grp.loc[flagged, "outer_generalization_gap_abs_mae"],
                    color=col, marker="x", alpha=0.9, s=60, linewidths=1.5, label=f"{model} (flagged)",
                )

            ax.axhline(OUTER_GAP_ABS_THRESHOLD, color="red", linestyle="--", linewidth=1,
                       label=f"Outer gap threshold ({OUTER_GAP_ABS_THRESHOLD})")
            ax.axvline(OVERFIT_INNER_GAP_RATIO_THRESHOLD, color="orange", linestyle="--", linewidth=1,
                       label=f"Inner gap threshold ({OVERFIT_INNER_GAP_RATIO_THRESHOLD})")
            ax.set_xlabel("Inner train-validation relative MAE gap")
            ax.set_ylabel("Outer generalization absolute MAE gap")
            ax.set_title(f"Inner vs outer gap — {strategy_label(strat)}", fontsize=10)
            ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")

            caption = (
                f"Inner vs outer gap scatter for {strategy_label(strat)}. "
                f"X = inner train-validation relative MAE gap (flag threshold {OVERFIT_INNER_GAP_RATIO_THRESHOLD}). "
                f"Y = outer generalization absolute MAE gap (flag threshold {OUTER_GAP_ABS_THRESHOLD}). "
                "Crosses = outer-gap-flagged folds. Quadrant: top-right = flagged on both."
            )
            save_thesis_figure(out_dir / f"overfit_inner_vs_outer_gap_scatter_{strat}.png", caption)
    except Exception as exc:
        log.warning("figure_inner_vs_outer_gap_scatter failed: %s", exc)


def figure_penalty_vs_outer_flag(trials: pd.DataFrame, overfitting_audit: pd.DataFrame, out_dir: Path) -> None:
    """Scatter of best-trial gap_penalty vs outer_generalization_gap per model."""
    try:
        if trials.empty or overfitting_audit.empty:
            return
        if "gap_penalty" not in trials.columns or "penalized_cv_loss" not in trials.columns:
            return
        apply_thesis_style()

        ok_trials = trials[trials["status"] == "ok"].copy() if "status" in trials.columns else trials.copy()
        scol_t = strategy_col(ok_trials) or "outer_strategy"
        id_cols = [c for c in [scol_t, "target", "branch", "model_name", "outer_test_year"] if c in ok_trials.columns]

        # Best trial per context = minimum penalized_cv_loss
        best_trials = ok_trials.loc[ok_trials.groupby(id_cols)["penalized_cv_loss"].idxmin()][
            id_cols + ["gap_penalty"]].copy()

        scol_a = strategy_col(overfitting_audit) or "outer_strategy"
        audit_cols = [c for c in [scol_a, "target", "branch", "model_name", "outer_test_year"] if c in overfitting_audit.columns]
        audit_sub = overfitting_audit[["status"] + audit_cols + ["outer_generalization_gap_abs_mae", "outer_gap_flagged"]].copy() \
            if "outer_generalization_gap_abs_mae" in overfitting_audit.columns else pd.DataFrame()
        if audit_sub.empty:
            return
        if "status" in audit_sub.columns:
            audit_sub = audit_sub[audit_sub["status"] == "ok"]

        # Rename strategy col to match best_trials
        audit_sub = audit_sub.rename(columns={scol_a: scol_t})
        merge_cols = [c for c in id_cols if c in audit_sub.columns]
        merged = best_trials.merge(audit_sub[merge_cols + ["outer_generalization_gap_abs_mae", "outer_gap_flagged"]],
                                   on=merge_cols, how="inner")

        if merged.empty or "model_name" not in merged.columns:
            return

        counts = merged.groupby("model_name").size()
        eligible = counts[counts > 1].index.tolist()

        for model in sorted(eligible):
            if model == "dummy_mean":
                continue
            sub = merged[merged["model_name"] == model].dropna(subset=["gap_penalty", "outer_generalization_gap_abs_mae"])
            if len(sub) < 2:
                continue

            fig, ax = plt.subplots(figsize=(6, 5))
            flagged = sub["outer_gap_flagged"].astype(bool) if "outer_gap_flagged" in sub.columns else pd.Series(False, index=sub.index)
            ax.scatter(sub.loc[~flagged, "gap_penalty"],
                       sub.loc[~flagged, "outer_generalization_gap_abs_mae"],
                       color="steelblue", alpha=0.7, s=40, label="Not outer-flagged")
            ax.scatter(sub.loc[flagged, "gap_penalty"],
                       sub.loc[flagged, "outer_generalization_gap_abs_mae"],
                       color="crimson", alpha=0.9, s=60, marker="x", linewidths=1.5, label="Outer-gap-flagged")
            ax.axhline(OUTER_GAP_ABS_THRESHOLD, color="red", linestyle="--", linewidth=1,
                       label=f"Outer threshold ({OUTER_GAP_ABS_THRESHOLD})")
            ax.set_xlabel("Gap penalty (best trial)")
            ax.set_ylabel("Outer generalization gap abs MAE")
            ax.set_title(f"Penalty vs outer gap — {model}")
            ax.legend(fontsize=8)

            caption = (
                f"Gap penalty vs outer generalization gap for {model}. "
                "X = gap_penalty applied to the best (minimum penalized-loss) trial per context. "
                "Y = outer generalization absolute MAE gap. Red crosses = outer-gap-flagged folds. "
                "Answers: did higher tuning-time penalty reduce outer generalization failure?"
            )
            save_thesis_figure(out_dir / f"overfit_gap_penalty_outer_flag_scatter_{model}.png", caption)
    except Exception as exc:
        log.warning("figure_penalty_vs_outer_flag failed: %s", exc)


def figure_outer_gap_violin(audit_df: pd.DataFrame, out_dir: Path) -> None:
    """Violin + strip of outer_generalization_gap_abs_mae per model, one per (strategy, target)."""
    try:
        if audit_df.empty or "outer_generalization_gap_abs_mae" not in audit_df.columns:
            return
        apply_thesis_style()

        ok = audit_df[audit_df["status"] == "ok"].copy() if "status" in audit_df.columns else audit_df.copy()
        scol = strategy_col(ok) or "outer_strategy"
        group_cols = [c for c in [scol, "target"] if c in ok.columns]

        for keys, grp in ok.groupby(group_cols):
            if isinstance(keys, (str, int, float)):
                keys = (keys,)
            kv = dict(zip(group_cols, keys))
            strat = kv.get(scol, "")
            tgt = kv.get("target", "")

            models = sorted(grp["model_name"].unique()) if "model_name" in grp.columns else []
            if not models:
                continue

            fig, ax = plt.subplots(figsize=(max(5, len(models) * 1.2), 5))
            data_by_model = [grp[grp["model_name"] == m]["outer_generalization_gap_abs_mae"].dropna().values
                             for m in models]
            # Only plot models with data
            valid = [(m, d) for m, d in zip(models, data_by_model) if len(d) > 0]
            if not valid:
                plt.close()
                continue
            models_v, data_v = zip(*valid)

            parts = ax.violinplot(data_v, positions=range(len(models_v)), showmedians=True)
            for pc in parts.get("bodies", []):
                pc.set_alpha(0.5)
                pc.set_facecolor("steelblue")

            # Strip overlay
            for i, (m, d) in enumerate(zip(models_v, data_v)):
                jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=len(d))
                flagged_mask = (grp[grp["model_name"] == m]["outer_gap_flagged"].astype(bool).values
                                if "outer_gap_flagged" in grp.columns else np.zeros(len(d), dtype=bool))
                if len(flagged_mask) != len(d):
                    flagged_mask = np.zeros(len(d), dtype=bool)
                ax.scatter(i + jitter[~flagged_mask], d[~flagged_mask], color="steelblue", alpha=0.6, s=20, zorder=3)
                ax.scatter(i + jitter[flagged_mask], d[flagged_mask], color="crimson", alpha=0.9, s=30,
                           marker="x", linewidths=1.5, zorder=4)

            ax.axhline(OUTER_GAP_ABS_THRESHOLD, color="red", linestyle="--", linewidth=1,
                       label=f"Flag threshold ({OUTER_GAP_ABS_THRESHOLD})")
            ax.set_xticks(range(len(models_v)))
            ax.set_xticklabels(list(models_v), rotation=30, ha="right")
            ax.set_ylabel("Outer generalization gap (abs MAE)")
            ax.set_title(f"Outer gap distribution — {strategy_label(strat)} | {scenario_label(tgt)}", fontsize=10)
            ax.legend(fontsize=8)

            parts_str = [str(kv.get(c, "")) for c in group_cols]
            fname = "overfit_outer_gap_violin_" + "_".join(parts_str) + ".png"
            caption = (
                f"Distribution of outer generalization gap (absolute MAE) per model for "
                f"{strategy_label(strat)} | {scenario_label(tgt)}. "
                f"Dashed line = flagging threshold ({OUTER_GAP_ABS_THRESHOLD}). "
                "Red crosses = outer-gap-flagged individual folds. Points = individual outer-fold observations."
            )
            save_thesis_figure(out_dir / fname, caption)
    except Exception as exc:
        log.warning("figure_outer_gap_violin failed: %s", exc)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_overfitting(overfitting_audit: pd.DataFrame, trials: pd.DataFrame,
                    outer_results: pd.DataFrame, out_dir: Path, cfg=None) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}

    gap_summary = table_gap_summary_by_model(overfitting_audit, out_dir)
    tables["overfit_gap_summary_by_model"] = gap_summary

    flagged = table_flagged_candidates(overfitting_audit, out_dir)
    tables["overfit_flagged_candidates"] = flagged

    figure_inner_vs_outer_gap_scatter(overfitting_audit, out_dir)
    figure_penalty_vs_outer_flag(trials, overfitting_audit, out_dir)
    figure_outer_gap_violin(overfitting_audit, out_dir)

    return tables
