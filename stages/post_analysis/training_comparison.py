"""
training_comparison.py – Compare training-stage model outputs beyond final winners.

Builds candidate-level summaries, outer-year breakdowns, trial progression, and
winner context from ``outer_results.csv``, ``trials.csv``, ``winners.csv``, and
``overfitting_audit.csv``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from stages.post_analysis.plot_utils import apply_thesis_style, save_fig, scenario_label, strategy_label

log = logging.getLogger(__name__)


def _strategy_col(df: pd.DataFrame) -> str | None:
    for col in ("outer_strategy_source", "outer_strategy"):
        if col in df.columns:
            return col
    return None


def _outer_metric_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()


def _save_table(df: pd.DataFrame, out_dir: Path, filename: str) -> pd.DataFrame:
    path = out_dir / filename
    df.to_csv(path, index=False)
    log.info("Saved: %s", path)
    return df


def table_outer_candidate_summary(
    outer_results: pd.DataFrame,
    overfitting_audit: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    """Aggregate fold-level outer performance per candidate."""
    df = _outer_metric_frame(outer_results)
    strategy_col = _strategy_col(df)
    if strategy_col is None or "outer_test_mae" not in df.columns:
        return pd.DataFrame()

    audit = _outer_metric_frame(overfitting_audit) if not overfitting_audit.empty else df.copy()
    if strategy_col not in audit.columns and "outer_strategy" in audit.columns:
        audit = audit.rename(columns={"outer_strategy": strategy_col})

    group_cols = [strategy_col, "target", "branch", "model_name"]
    summary = (
        df.groupby(group_cols)
        .agg(
            mean_outer_test_mae=("outer_test_mae", "mean"),
            std_outer_test_mae=("outer_test_mae", "std"),
            mean_outer_train_mae=("outer_train_mae", "mean"),
            std_outer_train_mae=("outer_train_mae", "std"),
            mean_inner_val_mae=("inner_val_mean_mae", "mean"),
            mean_inner_train_mae=("inner_train_mean_mae", "mean"),
            mean_inner_gap_rel=("inner_train_val_gap_rel_mae", "mean"),
            mean_outer_gap_abs=("outer_generalization_gap_abs_mae", "mean"),
            mean_outer_gap_rel=("outer_generalization_gap_rel_mae", "mean"),
            mean_penalized_cv_loss=("penalized_cv_loss", "mean"),
            n_outer_folds=("outer_test_year", "nunique"),
        )
        .reset_index()
        .rename(columns={strategy_col: "outer_strategy_source"})
    )

    audit_group_cols = [c if c != strategy_col else "outer_strategy_source" for c in group_cols]
    audit = audit.rename(columns={strategy_col: "outer_strategy_source"})
    for col in (
        "inner_gap_flagged",
        "outer_gap_flagged",
        "candidate_all_folds_inner_gap_flagged",
        "candidate_all_folds_outer_gap_flagged",
    ):
        if col not in audit.columns:
            audit[col] = False

    audit_summary = (
        audit.groupby(audit_group_cols)
        .agg(
            inner_gap_flagged_share=("inner_gap_flagged", "mean"),
            outer_gap_flagged_share=("outer_gap_flagged", "mean"),
            candidate_all_folds_inner_gap_flagged=("candidate_all_folds_inner_gap_flagged", "max"),
            candidate_all_folds_outer_gap_flagged=("candidate_all_folds_outer_gap_flagged", "max"),
        )
        .reset_index()
    )
    summary = summary.merge(audit_summary, on=audit_group_cols, how="left")

    summary["candidate_label"] = summary["branch"].astype(str) + " / " + summary["model_name"].astype(str)
    summary["rank_within_strategy_target"] = (
        summary.groupby(["outer_strategy_source", "target"])["mean_outer_test_mae"]
        .rank(method="dense")
        .astype(int)
    )
    summary = summary.sort_values(["outer_strategy_source", "target", "rank_within_strategy_target"])
    return _save_table(summary, out_dir, "outer_candidate_summary.csv")


def table_outer_year_breakdown(
    outer_results: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    """Save fold-level outer performance with per-year ranks."""
    df = _outer_metric_frame(outer_results)
    strategy_col = _strategy_col(df)
    required = {"outer_test_year", "outer_test_mae", "target", "branch", "model_name"}
    if strategy_col is None or not required.issubset(df.columns):
        return pd.DataFrame()

    out = df.copy().rename(columns={strategy_col: "outer_strategy_source"})
    out["rank_within_outer_year"] = (
        out.groupby(["outer_strategy_source", "target", "outer_test_year"])["outer_test_mae"]
        .rank(method="dense")
        .astype(int)
    )
    out = out.sort_values(["outer_strategy_source", "target", "outer_test_year", "rank_within_outer_year"])
    return _save_table(out, out_dir, "outer_year_breakdown.csv")


def table_trial_progression_summary(
    trials: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    """Aggregate best-so-far CV MAE trajectories across outer folds."""
    strategy_col = _strategy_col(trials)
    required = {"trial_number", "inner_val_mean_mae", "target", "branch", "model_name", "outer_test_year", "status"}
    if strategy_col is None or not required.issubset(trials.columns):
        return pd.DataFrame()

    df = trials[trials["status"] == "ok"].copy()
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values([strategy_col, "target", "branch", "model_name", "outer_test_year", "trial_number"])
    df["best_so_far_inner_val_mae"] = (
        df.groupby([strategy_col, "target", "branch", "model_name", "outer_test_year"])["inner_val_mean_mae"]
        .cummin()
    )

    summary = (
        df.groupby([strategy_col, "target", "branch", "model_name", "trial_number"])
        .agg(
            mean_inner_val_mae=("inner_val_mean_mae", "mean"),
            mean_best_so_far_inner_val_mae=("best_so_far_inner_val_mae", "mean"),
            mean_inner_train_mae=("inner_train_mean_mae", "mean"),
            outer_year_count=("outer_test_year", "nunique"),
        )
        .reset_index()
        .rename(columns={strategy_col: "outer_strategy_source"})
    )
    return _save_table(summary, out_dir, "trial_progression_summary.csv")


def table_winner_context(
    winners: pd.DataFrame,
    outer_candidate_summary: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    """Place final winners in the context of all candidate summaries."""
    if winners.empty or outer_candidate_summary.empty:
        return pd.DataFrame()

    summary = outer_candidate_summary.copy()
    best_by_combo = (
        summary.sort_values("mean_outer_test_mae")
        .groupby(["outer_strategy_source", "target"], as_index=False)
        .first()[["outer_strategy_source", "target", "branch", "model_name", "mean_outer_test_mae", "candidate_all_folds_outer_gap_flagged"]]
        .rename(
            columns={
                "branch": "best_candidate_branch",
                "model_name": "best_candidate_model_name",
                "mean_outer_test_mae": "best_candidate_mean_outer_test_mae",
                "candidate_all_folds_outer_gap_flagged": "best_candidate_outer_gap_flagged",
            }
        )
    )

    out = winners.rename(columns={"outer_strategy": "outer_strategy_source"}).copy()
    out = out.merge(
        summary[
            [
                "outer_strategy_source",
                "target",
                "branch",
                "model_name",
                "mean_outer_test_mae",
                "rank_within_strategy_target",
                "outer_gap_flagged_share",
                "candidate_all_folds_outer_gap_flagged",
            ]
        ].rename(
            columns={
                "mean_outer_test_mae": "winner_candidate_mean_outer_test_mae",
                "rank_within_strategy_target": "winner_rank_within_strategy_target",
                "outer_gap_flagged_share": "winner_outer_gap_flagged_share",
                "candidate_all_folds_outer_gap_flagged": "winner_candidate_outer_gap_flagged",
            }
        ),
        on=["outer_strategy_source", "target", "branch", "model_name"],
        how="left",
    )
    out = out.merge(best_by_combo, on=["outer_strategy_source", "target"], how="left")
    if "mean_outer_test_mae" not in out.columns and "winner_candidate_mean_outer_test_mae" in out.columns:
        out["mean_outer_test_mae"] = out["winner_candidate_mean_outer_test_mae"]
    out["winner_minus_best_mae"] = out["mean_outer_test_mae"] - out["best_candidate_mean_outer_test_mae"]
    out = out.sort_values(["outer_strategy_source", "target"])
    return _save_table(out, out_dir, "winner_context.csv")


def figure_outer_candidate_ranking(
    outer_candidate_summary: pd.DataFrame,
    winners: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Rank candidates by mean outer MAE for each strategy/target combination."""
    if outer_candidate_summary.empty:
        return

    winner_keys = set()
    if not winners.empty:
        winners_norm = winners.rename(columns={"outer_strategy": "outer_strategy_source"})
        winner_keys = {
            (r.outer_strategy_source, r.target, r.branch, r.model_name)
            for r in winners_norm.itertuples(index=False)
        }

    combos = list(outer_candidate_summary[["outer_strategy_source", "target"]].drop_duplicates().itertuples(index=False, name=None))
    if not combos:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = np.array(axes).flatten()
    for ax, combo in zip(axes, combos):
        strategy, target = combo
        subset = outer_candidate_summary[
            (outer_candidate_summary["outer_strategy_source"] == strategy)
            & (outer_candidate_summary["target"] == target)
        ].sort_values("mean_outer_test_mae")
        colors = []
        for row in subset.itertuples(index=False):
            key = (row.outer_strategy_source, row.target, row.branch, row.model_name)
            colors.append("#1b7837" if key in winner_keys else "#4393c3")
        ax.barh(
            subset["candidate_label"],
            subset["mean_outer_test_mae"],
            color=colors,
            edgecolor="white",
            xerr=subset["std_outer_test_mae"].fillna(0),
        )
        ax.set_title(f"{strategy_label(strategy)} / {scenario_label(target)}")
        ax.set_xlabel("Mean Outer Test MAE")
        ax.set_ylabel("")

    for ax in axes[len(combos):]:
        ax.axis("off")

    fig.suptitle("Training Candidate Ranking by Mean Outer MAE", fontsize=13)
    save_fig(out_dir / "candidate_ranking_by_outer_mae.png")
    log.info("Saved candidate ranking figure.")


def figure_trial_best_so_far_curves(
    trial_progression_summary: pd.DataFrame,
    outer_candidate_summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Show mean best-so-far CV MAE curves for the top candidates in each combo."""
    if trial_progression_summary.empty or outer_candidate_summary.empty:
        return

    combos = list(outer_candidate_summary[["outer_strategy_source", "target"]].drop_duplicates().itertuples(index=False, name=None))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = np.array(axes).flatten()

    for ax, combo in zip(axes, combos):
        strategy, target = combo
        top_candidates = (
            outer_candidate_summary[
                (outer_candidate_summary["outer_strategy_source"] == strategy)
                & (outer_candidate_summary["target"] == target)
            ]
            .sort_values("mean_outer_test_mae")
            .head(4)
        )
        if top_candidates.empty:
            ax.axis("off")
            continue

        for row in top_candidates.itertuples(index=False):
            curve = trial_progression_summary[
                (trial_progression_summary["outer_strategy_source"] == strategy)
                & (trial_progression_summary["target"] == target)
                & (trial_progression_summary["branch"] == row.branch)
                & (trial_progression_summary["model_name"] == row.model_name)
            ].sort_values("trial_number")
            if curve.empty:
                continue
            ax.plot(
                curve["trial_number"],
                curve["mean_best_so_far_inner_val_mae"],
                linewidth=1.8,
                label=f"{row.branch} / {row.model_name}",
            )

        ax.set_title(f"{strategy_label(strategy)} / {scenario_label(target)}")
        ax.set_xlabel("Trial Number")
        ax.set_ylabel("Mean Best-So-Far Inner-Val MAE")
        ax.legend(fontsize=7)

    for ax in axes[len(combos):]:
        ax.axis("off")

    fig.suptitle("Tuning Progression: Best-So-Far CV MAE", fontsize=13)
    save_fig(out_dir / "trial_best_so_far_curves.png")
    log.info("Saved trial progression figure.")


def figure_overfit_vs_performance(
    outer_candidate_summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Scatter candidate overfitting indicators against outer performance."""
    if outer_candidate_summary.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        outer_candidate_summary["mean_inner_gap_rel"],
        outer_candidate_summary["mean_outer_test_mae"],
        c=outer_candidate_summary["outer_gap_flagged_share"].fillna(0),
        cmap="viridis",
        s=80,
        alpha=0.8,
        edgecolors="white",
    )
    ax.set_xlabel("Mean Inner Train-Val Gap (rel MAE)")
    ax.set_ylabel("Mean Outer Test MAE")
    ax.set_title("Overfitting Signal vs. Outer Performance")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Outer Gap Flagged Share")
    save_fig(out_dir / "overfit_vs_performance.png")
    log.info("Saved overfit/performance scatter.")


def run_training_comparison(
    outer_results: pd.DataFrame,
    trials: pd.DataFrame,
    winners: pd.DataFrame,
    overfitting_audit: pd.DataFrame,
    out_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Build all training-stage comparison outputs."""
    apply_thesis_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: dict[str, pd.DataFrame] = {}

    if outer_results.empty:
        log.warning("training_comparison: outer_results is empty — skipping.")
        return produced

    candidate_summary = table_outer_candidate_summary(outer_results, overfitting_audit, out_dir)
    outer_year_breakdown = table_outer_year_breakdown(outer_results, out_dir)
    trial_progression = table_trial_progression_summary(trials, out_dir)
    winner_context = table_winner_context(winners, candidate_summary, out_dir)

    produced["outer_candidate_summary"] = candidate_summary
    produced["outer_year_breakdown"] = outer_year_breakdown
    produced["trial_progression_summary"] = trial_progression
    produced["winner_context"] = winner_context

    figure_outer_candidate_ranking(candidate_summary, winners, out_dir)
    figure_trial_best_so_far_curves(trial_progression, candidate_summary, out_dir)
    figure_overfit_vs_performance(candidate_summary, out_dir)
    return produced
