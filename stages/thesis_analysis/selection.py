"""
selection.py — Section B: Model selection robustness analysis.

Source: winners.csv, outer_results.csv
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
import matplotlib.patches as mpatches

from stages.post_analysis.plot_utils import apply_thesis_style, model_color_map, strategy_label, scenario_label, branch_label
from stages.post_analysis.config import FIGURE_DPI

from stages.thesis_analysis._shared import (
    save_thesis_figure,
    save_thesis_table,
    strategy_col,
    winner_combo_keys,
)

log = logging.getLogger(__name__)

_ROLE_COLORS = {"primary": "steelblue", "backup": "darkorange"}
_NOT_SELECTED_COLOR = "lightgrey"


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def table_winner_margins(winners: pd.DataFrame, outer_results: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """All candidates per (strategy, target) with MAE and gap-to-rank1."""
    try:
        if outer_results.empty:
            return pd.DataFrame()

        scol_o = strategy_col(outer_results) or "outer_strategy"
        ok = outer_results[outer_results.get("status", pd.Series(["ok"]*len(outer_results))) == "ok"].copy() \
            if "status" in outer_results.columns else outer_results.copy()

        group_id = [c for c in [scol_o, "target", "branch", "model_name"] if c in ok.columns]
        if "outer_test_mae" not in ok.columns:
            return pd.DataFrame()

        agg = (
            ok.groupby(group_id)["outer_test_mae"]
            .agg(mean_outer_test_mae="mean", std_outer_test_mae="std", n_folds="count")
            .reset_index()
        )

        # For each (strategy, target), compute rank and gap to rank-1
        strat_tgt = [c for c in [scol_o, "target"] if c in agg.columns]
        agg["candidate_rank_all"] = agg.groupby(strat_tgt)["mean_outer_test_mae"].rank(method="min")

        rank1 = agg[agg["candidate_rank_all"] == 1][strat_tgt + ["mean_outer_test_mae"]].copy()
        rank1 = rank1.rename(columns={"mean_outer_test_mae": "rank1_mae"})
        agg = agg.merge(rank1, on=strat_tgt, how="left")
        agg["mae_gap_to_rank1"] = agg["mean_outer_test_mae"] - agg["rank1_mae"]

        # Merge winner role/rank
        if not winners.empty:
            scol_w = strategy_col(winners) or "outer_strategy"
            w_sub = winners[[c for c in [scol_w, "target", "branch", "model_name",
                                          "candidate_rank", "candidate_role"] if c in winners.columns]].copy()
            w_sub = w_sub.rename(columns={scol_w: scol_o})
            agg = agg.merge(w_sub, on=[scol_o, "target", "branch", "model_name"], how="left")

        agg["in_winner_pool"] = agg["candidate_role"].notna()

        return save_thesis_table(agg, out_dir, "selection_winner_margins",
                                 caption="All candidates per (strategy, target) ranked by mean outer test MAE.")
    except Exception as exc:
        log.warning("table_winner_margins failed: %s", exc)
        return pd.DataFrame()


def table_fold_winner_stability(outer_results: pd.DataFrame, winners: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Per (strategy, target, outer_test_year): year-winner and agreement with mean-winner."""
    try:
        if outer_results.empty:
            return pd.DataFrame()

        scol = strategy_col(outer_results) or "outer_strategy"
        ok = outer_results[outer_results["status"] == "ok"].copy() if "status" in outer_results.columns else outer_results.copy()
        if "outer_test_mae" not in ok.columns:
            return pd.DataFrame()

        group_year = [c for c in [scol, "target", "outer_test_year"] if c in ok.columns]
        rows = []
        for keys, grp in ok.groupby(group_year):
            if isinstance(keys, (str, int, float)):
                keys = (keys,)
            kv = dict(zip(group_year, keys))
            best_row = grp.loc[grp["outer_test_mae"].idxmin()]
            kv["year_winner_branch"] = best_row.get("branch", "")
            kv["year_winner_model"] = best_row.get("model_name", "")
            kv["year_winner_mae"] = best_row.get("outer_test_mae", np.nan)
            rows.append(kv)

        result = pd.DataFrame(rows)

        # Compare to mean-winner (rank-1 from winners.csv)
        if not winners.empty and not result.empty:
            scol_w = strategy_col(winners) or "outer_strategy"
            rank1 = winners[winners.get("candidate_rank", pd.Series([1]*len(winners))) == 1] \
                if "candidate_rank" in winners.columns else winners.iloc[:0]
            if not rank1.empty:
                rank1 = rank1[[c for c in [scol_w, "target", "branch", "model_name"] if c in rank1.columns]].copy()
                rank1 = rank1.rename(columns={scol_w: scol, "branch": "mean_winner_branch", "model_name": "mean_winner_model"})
                merge_on = [c for c in [scol, "target"] if c in result.columns and c in rank1.columns]
                result = result.merge(rank1, on=merge_on, how="left")
                result["agrees_with_mean_winner"] = (
                    (result["year_winner_branch"] == result.get("mean_winner_branch", "")) &
                    (result["year_winner_model"] == result.get("mean_winner_model", ""))
                )

        return save_thesis_table(result, out_dir, "selection_fold_winner_stability",
                                 caption="Year-level winner stability: which model wins each outer fold vs the mean-winner.")
    except Exception as exc:
        log.warning("table_fold_winner_stability failed: %s", exc)
        return pd.DataFrame()


def table_backup_usage(winners: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Which (strategy, target) combos have backup candidates and why."""
    try:
        if winners.empty:
            return pd.DataFrame()

        scol = strategy_col(winners) or "outer_strategy"
        strat_tgt = [c for c in [scol, "target"] if c in winners.columns]

        rows = []
        for keys, grp in winners.groupby(strat_tgt):
            if isinstance(keys, (str, int, float)):
                keys = (keys,)
            kv = dict(zip(strat_tgt, keys))
            primaries = grp[grp["candidate_role"] == "primary"] if "candidate_role" in grp.columns else grp
            backups = grp[grp["candidate_role"] == "backup"] if "candidate_role" in grp.columns else grp.iloc[:0]
            flagged = primaries["candidate_all_folds_outer_gap_flagged"].any() \
                if "candidate_all_folds_outer_gap_flagged" in primaries.columns else False
            kv.update({
                "n_primaries": len(primaries),
                "n_backups": len(backups),
                "has_backup": len(backups) > 0,
                "any_primary_all_folds_flagged": bool(flagged),
            })
            if len(backups) > 0:
                kv["backup_branch"] = backups["branch"].iloc[0] if "branch" in backups.columns else ""
                kv["backup_model"] = backups["model_name"].iloc[0] if "model_name" in backups.columns else ""
                kv["backup_mean_mae"] = backups["mean_outer_test_mae"].iloc[0] if "mean_outer_test_mae" in backups.columns else np.nan
            else:
                kv.update({"backup_branch": "", "backup_model": "", "backup_mean_mae": np.nan})
            rows.append(kv)

        result = pd.DataFrame(rows)
        return save_thesis_table(result, out_dir, "selection_backup_usage",
                                 caption="Backup candidate usage: which (strategy, target) pairs triggered backup selection.")
    except Exception as exc:
        log.warning("table_backup_usage failed: %s", exc)
        return pd.DataFrame()


def table_strategy_agreement(winners: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Per target: do temporal and random CV select the same rank-1 winner?"""
    try:
        if winners.empty or "target" not in winners.columns:
            return pd.DataFrame()

        scol = strategy_col(winners) or "outer_strategy"
        rank1 = winners[winners["candidate_rank"] == 1].copy() if "candidate_rank" in winners.columns else winners.copy()

        rows = []
        for target, grp in rank1.groupby("target"):
            row = {"target": target}
            for strat in grp[scol].unique() if scol in grp.columns else []:
                sub = grp[grp[scol] == strat]
                if sub.empty:
                    continue
                prefix = strat.replace("_cv", "")
                row[f"{prefix}_branch"] = sub["branch"].iloc[0] if "branch" in sub.columns else ""
                row[f"{prefix}_model"] = sub["model_name"].iloc[0] if "model_name" in sub.columns else ""
                row[f"{prefix}_rank1_mae"] = sub["mean_outer_test_mae"].iloc[0] if "mean_outer_test_mae" in sub.columns else np.nan
            rows.append(row)

        result = pd.DataFrame(rows)
        if not result.empty and "temporal_branch" in result.columns and "random_branch" in result.columns:
            result["strategies_agree"] = (
                (result["temporal_branch"] == result["random_branch"]) &
                (result["temporal_model"] == result["random_model"])
            )

        return save_thesis_table(result, out_dir, "selection_strategy_agreement",
                                 caption="Do temporal_cv and random_cv select the same rank-1 winner per target?")
    except Exception as exc:
        log.warning("table_strategy_agreement failed: %s", exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_winner_margin_bars(margins_df: pd.DataFrame, out_dir: Path) -> None:
    """Horizontal bar chart per (strategy, target): candidates sorted by MAE, colored by role."""
    try:
        if margins_df.empty:
            return
        apply_thesis_style()

        scol = strategy_col(margins_df) or "outer_strategy"
        group_cols = [c for c in [scol, "target"] if c in margins_df.columns]

        for keys, grp in margins_df.groupby(group_cols):
            if isinstance(keys, (str, int, float)):
                keys = (keys,)
            kv = dict(zip(group_cols, keys))
            grp = grp.sort_values("mean_outer_test_mae").reset_index(drop=True)

            labels = []
            for _, row in grp.iterrows():
                br = row.get("branch", "")
                mn = row.get("model_name", "")
                labels.append(f"{br}/{mn}")

            colors = []
            for _, row in grp.iterrows():
                role = row.get("candidate_role", "")
                if pd.isna(role) or role == "":
                    colors.append(_NOT_SELECTED_COLOR)
                else:
                    colors.append(_ROLE_COLORS.get(str(role), _NOT_SELECTED_COLOR))

            fig, ax = plt.subplots(figsize=(7, max(3, len(grp) * 0.5 + 1)))
            y = np.arange(len(grp))
            xerr = grp["std_outer_test_mae"].fillna(0).values if "std_outer_test_mae" in grp.columns else None
            ax.barh(y, grp["mean_outer_test_mae"], xerr=xerr,
                    color=colors, height=0.6, edgecolor="white", capsize=3)
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel("Mean outer test MAE")
            strat = kv.get(scol, "")
            tgt = kv.get("target", "")
            ax.set_title(f"Winner margins — {strategy_label(strat)} | {scenario_label(tgt)}", fontsize=10)

            patches = [
                mpatches.Patch(color=_ROLE_COLORS["primary"], label="Primary"),
                mpatches.Patch(color=_ROLE_COLORS["backup"], label="Backup"),
                mpatches.Patch(color=_NOT_SELECTED_COLOR, label="Not selected"),
            ]
            ax.legend(handles=patches, fontsize=8, loc="lower right")
            ax.set_xlim(left=0)

            parts = [str(kv.get(c, "")) for c in group_cols]
            fname = "selection_winner_margin_bar_" + "_".join(parts) + ".png"
            caption = (
                f"Winner margin analysis for {strategy_label(strat)} | {scenario_label(tgt)}. "
                "Bars colored by selection role (primary=blue, backup=orange, not-selected=grey). "
                "Error bars = std across outer folds."
            )
            save_thesis_figure(out_dir / fname, caption)
    except Exception as exc:
        log.warning("figure_winner_margin_bars failed: %s", exc)


def figure_fold_stability_heatmap(stability_df: pd.DataFrame, out_dir: Path) -> None:
    """Heatmap: rows=targets, cols=years; cell text = branch/model; green if agrees with mean-winner."""
    try:
        if stability_df.empty:
            return
        apply_thesis_style()

        scol = strategy_col(stability_df) or "outer_strategy"
        strategies = stability_df[scol].unique() if scol in stability_df.columns else []
        targets = sorted(stability_df["target"].unique()) if "target" in stability_df.columns else []
        years = sorted(stability_df["outer_test_year"].unique()) if "outer_test_year" in stability_df.columns else []

        for strat in strategies:
            sub = stability_df[stability_df[scol] == strat] if scol in stability_df.columns else stability_df

            fig, ax = plt.subplots(figsize=(max(5, len(years) * 1.5), max(2, len(targets) * 1.0) + 1))
            ax.set_xlim(-0.5, len(years) - 0.5)
            ax.set_ylim(-0.5, len(targets) - 0.5)

            for i, tgt in enumerate(targets):
                for j, yr in enumerate(years):
                    cell = sub[(sub["target"] == tgt) & (sub["outer_test_year"] == yr)]
                    if cell.empty:
                        text = "—"
                        bg = "lightgrey"
                    else:
                        br = cell["year_winner_branch"].iloc[0]
                        mn = cell["year_winner_model"].iloc[0]
                        text = f"{br}\n{mn}"
                        agrees = cell["agrees_with_mean_winner"].iloc[0] if "agrees_with_mean_winner" in cell.columns else None
                        if agrees is True:
                            bg = "#c8e6c9"
                        elif agrees is False:
                            bg = "#ffcdd2"
                        else:
                            bg = "lightyellow"

                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                         facecolor=bg, edgecolor="white", linewidth=1.5)
                    ax.add_patch(rect)
                    ax.text(j, i, text, ha="center", va="center", fontsize=7.5)

            ax.set_xticks(range(len(years)))
            ax.set_xticklabels([str(y) for y in years])
            ax.set_yticks(range(len(targets)))
            ax.set_yticklabels([scenario_label(t) for t in targets])
            ax.set_xlabel("Outer test year")
            ax.set_ylabel("Target scenario")
            ax.set_title(f"Fold winner stability — {strategy_label(strat)}", fontsize=10)

            patches = [
                mpatches.Patch(color="#c8e6c9", label="Agrees with mean-winner"),
                mpatches.Patch(color="#ffcdd2", label="Different from mean-winner"),
            ]
            ax.legend(handles=patches, fontsize=8, loc="upper right", bbox_to_anchor=(1, -0.12))

            caption = (
                f"Fold winner stability heatmap for {strategy_label(strat)}. "
                "Each cell shows the winning model+branch for that (target, outer-year) fold. "
                "Green = agrees with the mean-winner from winners.csv rank-1; red = differs."
            )
            save_thesis_figure(out_dir / f"selection_fold_stability_heatmap_{strat}.png", caption)
    except Exception as exc:
        log.warning("figure_fold_stability_heatmap failed: %s", exc)


def figure_backup_needed(backup_df: pd.DataFrame, out_dir: Path) -> None:
    """Stacked bar per (strategy, target): no-backup vs backup-triggered."""
    try:
        if backup_df.empty:
            return
        apply_thesis_style()

        scol = strategy_col(backup_df) or "outer_strategy"
        label_cols = [c for c in [scol, "target"] if c in backup_df.columns]
        backup_df = backup_df.copy()
        backup_df["label"] = backup_df[label_cols].apply(
            lambda r: " | ".join(str(r[c]) for c in label_cols), axis=1)

        fig, ax = plt.subplots(figsize=(max(5, len(backup_df) * 0.9), 4))
        x = np.arange(len(backup_df))
        no_backup = (~backup_df["has_backup"].astype(bool)).astype(int) if "has_backup" in backup_df.columns else np.ones(len(backup_df))
        has_backup = backup_df["has_backup"].astype(int) if "has_backup" in backup_df.columns else np.zeros(len(backup_df))
        ax.bar(x, no_backup, label="No backup needed", color="steelblue", width=0.6)
        ax.bar(x, has_backup, bottom=no_backup, label="Backup added", color="darkorange", width=0.6)

        # Annotate backup model name
        for i, row in backup_df.reset_index(drop=True).iterrows():
            if row.get("has_backup"):
                model_lbl = f"{row.get('backup_branch','')}/{row.get('backup_model','')}"
                ax.text(i, 1.05, model_lbl, ha="center", va="bottom", fontsize=7, rotation=20)

        ax.set_xticks(x)
        ax.set_xticklabels(backup_df["label"].tolist(), rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Count")
        ax.set_ylim(0, 1.8)
        ax.set_title("Backup candidate usage per (strategy, target)")
        ax.legend(fontsize=8)

        caption = (
            "Backup candidate summary: for each (strategy, target) pair, shows whether backup candidates "
            "were needed (all primaries outer-gap-flagged). Annotated with backup model where applicable."
        )
        save_thesis_figure(out_dir / "selection_backup_needed_summary.png", caption)
    except Exception as exc:
        log.warning("figure_backup_needed failed: %s", exc)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_selection(winners: pd.DataFrame, outer_results: pd.DataFrame,
                  out_dir: Path, cfg=None) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}

    margins = table_winner_margins(winners, outer_results, out_dir)
    tables["selection_winner_margins"] = margins

    stability = table_fold_winner_stability(outer_results, winners, out_dir)
    tables["selection_fold_winner_stability"] = stability

    backup = table_backup_usage(winners, out_dir)
    tables["selection_backup_usage"] = backup

    agreement = table_strategy_agreement(winners, out_dir)
    tables["selection_strategy_agreement"] = agreement

    figure_winner_margin_bars(margins, out_dir)
    figure_fold_stability_heatmap(stability, out_dir)
    figure_backup_needed(backup, out_dir)

    return tables
