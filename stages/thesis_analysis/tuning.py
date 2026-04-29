"""
tuning.py — Section A: Hyperparameter tuning diagnostics.

Source: trials.csv
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
    apply_thesis_style,
    model_color_map,
    scenario_label,
    strategy_label,
    branch_label,
)
from stages.post_analysis.config import FIGURE_DPI, SCENARIO_COLORS, STRATEGY_COLORS

from stages.thesis_analysis._shared import (
    parse_params_json,
    layers_label,
    save_thesis_figure,
    save_thesis_table,
    strategy_col,
)
from stages.thesis_analysis.config import (
    MAX_TUNING_TRIALS_PER_MODEL,
    TUNING_BUDGET_SECONDS,
    PARAM_LANDSCAPE_MIN_TRIALS,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def table_trial_best_so_far(trials: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Cumulative best inner-val MAE per tuning context and trial."""
    try:
        ok = trials[trials["status"] == "ok"].copy()
        if ok.empty:
            return pd.DataFrame()

        scol = strategy_col(ok) or "outer_strategy"
        group_cols = [c for c in [scol, "target", "branch", "model_name", "outer_test_year"] if c in ok.columns]

        ok = ok.sort_values(group_cols + ["trial_number"])
        ok["best_so_far_val_mae"] = ok.groupby(group_cols)["inner_val_mean_mae"].cummin()

        keep = group_cols + ["trial_number", "inner_val_mean_mae", "best_so_far_val_mae",
                              "inner_train_mean_mae", "penalized_cv_loss", "gap_penalty"]
        keep = [c for c in keep if c in ok.columns]
        result = ok[keep].reset_index(drop=True)
        return save_thesis_table(result, out_dir, "tuning_trial_best_so_far",
                                 caption="Cumulative best inner-validation MAE per tuning context.")
    except Exception as exc:
        log.warning("table_trial_best_so_far failed: %s", exc)
        return pd.DataFrame()


def table_budget_utilization(trials: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Budget utilization: trials run vs max allowed vs time budget."""
    try:
        scol = strategy_col(trials) or "outer_strategy"
        group_cols = [c for c in [scol, "target", "branch", "model_name", "outer_test_year"] if c in trials.columns]

        rows = []
        for keys, grp in trials.groupby(group_cols):
            model = grp["model_name"].iloc[0] if "model_name" in grp.columns else "unknown"
            n_total = len(grp)
            n_ok = (grp["status"] == "ok").sum() if "status" in grp.columns else n_total
            elapsed = grp["trial_elapsed_seconds"].sum() if "trial_elapsed_seconds" in grp.columns else np.nan
            max_trials = MAX_TUNING_TRIALS_PER_MODEL.get(model, 1)
            row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
            row.update({
                "n_trials_total": n_total,
                "n_trials_ok": int(n_ok),
                "sum_elapsed_seconds": elapsed,
                "budget_fraction": elapsed / TUNING_BUDGET_SECONDS if not np.isnan(elapsed) else np.nan,
                "max_trials_for_model": max_trials,
                "trial_fraction": n_ok / max_trials if max_trials > 0 else np.nan,
            })
            rows.append(row)

        if not rows:
            return pd.DataFrame()
        result = pd.DataFrame(rows)
        return save_thesis_table(result, out_dir, "tuning_budget_utilization",
                                 caption="Hyperopt budget utilization per tuning context.")
    except Exception as exc:
        log.warning("table_budget_utilization failed: %s", exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_convergence_per_context(trial_bsf: pd.DataFrame, out_dir: Path) -> None:
    """One PNG per (strategy, target, branch, model, year) convergence curve."""
    try:
        if trial_bsf.empty:
            return
        apply_thesis_style()
        scol = strategy_col(trial_bsf) or "outer_strategy"
        group_cols = [c for c in [scol, "target", "branch", "model_name", "outer_test_year"] if c in trial_bsf.columns]

        conv_dir = out_dir / "convergence"
        conv_dir.mkdir(parents=True, exist_ok=True)

        for keys, grp in trial_bsf.groupby(group_cols):
            if isinstance(keys, (str, int, float)):
                keys = (keys,)
            kv = dict(zip(group_cols, keys))
            model = kv.get("model_name", "")
            if model == "dummy_mean":
                continue
            grp = grp.sort_values("trial_number")
            if len(grp) < 2:
                continue

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(grp["trial_number"], grp["best_so_far_val_mae"],
                    color="steelblue", linewidth=2, label="Best-so-far val MAE")
            if "inner_train_mean_mae" in grp.columns:
                ax.plot(grp["trial_number"], grp["inner_train_mean_mae"],
                        color="coral", linewidth=1, linestyle="--", alpha=0.7, label="Train MAE (each trial)")
            if not grp["best_so_far_val_mae"].empty:
                final_val = grp["best_so_far_val_mae"].iloc[-1]
                ax.axhline(final_val, color="steelblue", linestyle=":", linewidth=1, alpha=0.6)
            ax.set_xlabel("Trial number")
            ax.set_ylabel("MAE")

            strat = kv.get(scol, "")
            tgt = kv.get("target", "")
            br = kv.get("branch", "")
            yr = kv.get("outer_test_year", "")
            title = f"{strategy_label(strat)} | {scenario_label(tgt)} | {branch_label(br)} | {model} | year {yr}"
            ax.set_title(title, fontsize=9)
            ax.legend(fontsize=8)
            ax.set_ylim(bottom=0)

            parts = [str(kv.get(c, "")) for c in group_cols]
            fname = "tuning_convergence_" + "_".join(parts) + ".png"
            caption = (
                f"Hyperopt convergence for {model} | {strategy_label(strat)} | "
                f"{scenario_label(tgt)} | {branch_label(br)} | outer-year {yr}. "
                "Solid = best-so-far inner validation MAE; dashed = inner training MAE at each trial."
            )
            save_thesis_figure(conv_dir / fname, caption)
    except Exception as exc:
        log.warning("figure_convergence_per_context failed: %s", exc)


def figure_budget_utilization(budget_df: pd.DataFrame, out_dir: Path) -> None:
    """Grouped bar chart of mean trial_fraction per model."""
    try:
        if budget_df.empty or "trial_fraction" not in budget_df.columns:
            return
        apply_thesis_style()

        mean_frac = (
            budget_df[budget_df["model_name"] != "dummy_mean"]
            .groupby("model_name")["trial_fraction"]
            .agg(mean="mean", std="std")
            .reset_index()
            .sort_values("mean", ascending=False)
        )
        if mean_frac.empty:
            return

        colors = model_color_map(mean_frac["model_name"].tolist())
        fig, ax = plt.subplots(figsize=(max(6, len(mean_frac) * 0.8), 4))
        x = np.arange(len(mean_frac))
        bars = ax.bar(x, mean_frac["mean"], yerr=mean_frac["std"].fillna(0),
                      color=[colors[m] for m in mean_frac["model_name"]],
                      capsize=4, width=0.6, edgecolor="white")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Budget exhausted (fraction = 1.0)")
        ax.set_xticks(x)
        ax.set_xticklabels(mean_frac["model_name"], rotation=30, ha="right")
        ax.set_ylabel("Mean trial fraction (trials run / max trials)")
        ax.set_ylim(0, min(1.2, mean_frac["mean"].max() * 1.3 + 0.1))
        ax.set_title("Hyperopt budget utilization per model")
        ax.legend(fontsize=8)

        caption = (
            "Hyperopt budget utilization: fraction of maximum allowed trials reached per model, "
            "averaged across all (strategy, target, branch, outer-year) contexts. "
            "Error bars = std. Dashed line = budget fully exhausted."
        )
        save_thesis_figure(out_dir / "tuning_budget_utilization.png", caption)
    except Exception as exc:
        log.warning("figure_budget_utilization failed: %s", exc)


def figure_gap_penalty_scatter(trials: pd.DataFrame, out_dir: Path) -> None:
    """Scatter of inner_val_mean_mae vs penalized_cv_loss colored by gap_penalty — one per model."""
    try:
        if trials.empty:
            return
        apply_thesis_style()

        ok = trials[(trials["status"] == "ok") & (trials["model_name"] != "dummy_mean")].copy()
        if ok.empty:
            return

        # Only models with >1 successful trial
        counts = ok.groupby("model_name").size()
        eligible = counts[counts > 1].index.tolist()

        for model in sorted(eligible):
            sub = ok[ok["model_name"] == model].copy()
            if "inner_val_mean_mae" not in sub.columns or "penalized_cv_loss" not in sub.columns:
                continue

            sub = sub.dropna(subset=["inner_val_mean_mae", "penalized_cv_loss"])
            if len(sub) < 2:
                continue

            gap = sub["gap_penalty"].fillna(0) if "gap_penalty" in sub.columns else pd.Series(0, index=sub.index)

            fig, ax = plt.subplots(figsize=(6, 5))
            sc = ax.scatter(sub["inner_val_mean_mae"], sub["penalized_cv_loss"],
                            c=gap, cmap="viridis", alpha=0.6, s=25, edgecolors="none")
            plt.colorbar(sc, ax=ax, label="gap_penalty")

            lo = min(sub["inner_val_mean_mae"].min(), sub["penalized_cv_loss"].min())
            hi = max(sub["inner_val_mean_mae"].max(), sub["penalized_cv_loss"].max())
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x (no penalty)")

            ax.set_xlabel("Inner val MAE")
            ax.set_ylabel("Penalized CV loss")
            ax.set_title(f"Gap penalty effect — {model}")
            ax.legend(fontsize=8)

            caption = (
                f"Gap penalty effect for {model}: scatter of inner validation MAE vs penalized CV loss. "
                "Points above the y=x line had non-zero gap penalty. Color = gap_penalty magnitude."
            )
            save_thesis_figure(out_dir / f"tuning_gap_penalty_scatter_{model}.png", caption)
    except Exception as exc:
        log.warning("figure_gap_penalty_scatter failed: %s", exc)


def _select_top2_numerical_params(param_dicts: list[dict]) -> list[str]:
    """
    From a list of parsed param dicts, identify the top-2 most-varied numerical params.

    Hidden-layer-sizes-style list values are excluded (handled as categorical).
    Returns a list of 0, 1, or 2 param names.
    """
    from collections import defaultdict
    values_by_key: dict[str, list] = defaultdict(list)
    for d in param_dicts:
        for k, v in d.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                values_by_key[k].append(v)
            # list values (e.g. hidden_layer_sizes) are excluded here

    iqr_by_key = {}
    for k, vals in values_by_key.items():
        if len(vals) < 3:
            continue
        arr = np.array(vals, dtype=float)
        q75, q25 = np.percentile(arr, 75), np.percentile(arr, 25)
        iqr_by_key[k] = q75 - q25

    if not iqr_by_key:
        return []
    top = sorted(iqr_by_key, key=lambda k: iqr_by_key[k], reverse=True)
    return top[:2]


def _select_list_params(param_dicts: list[dict]) -> list[str]:
    """Return param names whose values are lists (e.g. hidden_layer_sizes)."""
    list_keys = set()
    for d in param_dicts:
        for k, v in d.items():
            if isinstance(v, list):
                list_keys.add(k)
    return sorted(list_keys)


def figure_param_landscapes(trials: pd.DataFrame, out_dir: Path) -> None:
    """
    Model-agnostic param landscape: one PNG per (model, strategy, target, branch).

    Axes = top-2 most-varied numerical params; color = inner_val_mean_mae.
    If a model has list-valued params (e.g. hidden_layer_sizes), one of those
    is used as the categorical Y axis and the top-1 numerical param as X.
    Skips dummy_mean and any model with < PARAM_LANDSCAPE_MIN_TRIALS ok trials.
    """
    try:
        if trials.empty:
            return
        apply_thesis_style()

        ok = trials[(trials["status"] == "ok") & (trials["model_name"] != "dummy_mean")].copy()
        if ok.empty:
            return
        scol = strategy_col(ok) or "outer_strategy"
        group_cols = [c for c in [scol, "target", "branch", "model_name"] if c in ok.columns]

        for keys, grp in ok.groupby(group_cols):
            if isinstance(keys, (str, int, float)):
                keys = (keys,)
            kv = dict(zip(group_cols, keys))
            model = kv.get("model_name", "")
            if len(grp) < PARAM_LANDSCAPE_MIN_TRIALS:
                continue
            if "params_json" not in grp.columns or "inner_val_mean_mae" not in grp.columns:
                continue

            param_dicts = [parse_params_json(s) for s in grp["params_json"].fillna("{}")]
            num_params = _select_top2_numerical_params(param_dicts)
            list_params = _select_list_params(param_dicts)

            # Decide axis configuration
            if len(num_params) >= 2:
                xparam, yparam = num_params[0], num_params[1]
                y_categorical = False
            elif len(num_params) == 1 and list_params:
                xparam = num_params[0]
                yparam = list_params[0]
                y_categorical = True
            elif list_params and len(list_params) >= 2:
                xparam, yparam = list_params[0], list_params[1]
                y_categorical = True
            else:
                log.debug("Skipping param landscape for %s (<%d varied params)", model, 2)
                continue

            # Extract values
            x_vals, y_vals, mae_vals, gap_vals = [], [], [], []
            for d, mae, gap in zip(
                param_dicts,
                grp["inner_val_mean_mae"].values,
                grp["gap_penalty"].fillna(0).values if "gap_penalty" in grp.columns else [0]*len(grp),
            ):
                if np.isnan(mae):
                    continue
                xv = d.get(xparam)
                yv = d.get(yparam)
                if xv is None or yv is None:
                    continue
                if not y_categorical and not isinstance(yv, (int, float)):
                    continue
                x_vals.append(xv)
                y_vals.append(yv)
                mae_vals.append(mae)
                gap_vals.append(gap)

            if len(x_vals) < 3:
                continue

            fig, ax = plt.subplots(figsize=(6, 5))

            x_arr = np.array(x_vals, dtype=float)

            # Log-scale X if range > 2 orders of magnitude
            x_range = x_arr.max() - x_arr.min()
            x_log = x_arr.min() > 0 and (x_arr.max() / max(x_arr.min(), 1e-12)) > 100

            if x_log:
                x_plot = np.log10(np.clip(x_arr, 1e-12, None))
                xlabel = f"log₁₀({xparam})"
            else:
                x_plot = x_arr
                xlabel = xparam

            if y_categorical:
                y_labels = [layers_label(v) for v in y_vals]
                unique_labels = sorted(set(y_labels))
                y_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
                y_plot = np.array([y_idx[lbl] for lbl in y_labels], dtype=float)
                y_plot += np.random.default_rng(42).uniform(-0.15, 0.15, size=len(y_plot))
                ax.set_yticks(list(y_idx.values()))
                ax.set_yticklabels(list(y_idx.keys()), fontsize=8)
                ylabel = yparam
            else:
                y_arr = np.array(y_vals, dtype=float)
                y_log = y_arr.min() > 0 and (y_arr.max() / max(y_arr.min(), 1e-12)) > 100
                if y_log:
                    y_plot = np.log10(np.clip(y_arr, 1e-12, None))
                    ylabel = f"log₁₀({yparam})"
                else:
                    y_plot = y_arr
                    ylabel = yparam

            sc = ax.scatter(x_plot, y_plot,
                            c=mae_vals, cmap="viridis_r",
                            s=np.clip(np.array(gap_vals) * 80 + 20, 10, 120),
                            alpha=0.7, edgecolors="none")
            plt.colorbar(sc, ax=ax, label="Inner val MAE")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            strat = kv.get(scol, "")
            tgt = kv.get("target", "")
            br = kv.get("branch", "")
            ax.set_title(f"Param landscape — {model} | {strategy_label(strat)} | {scenario_label(tgt)} | {branch_label(br)}",
                         fontsize=9)

            parts = [str(kv.get(c, "")) for c in group_cols]
            fname = "tuning_param_landscape_" + "_".join(parts) + ".png"
            caption = (
                f"Hyperparameter landscape for {model} | {strategy_label(strat)} | {scenario_label(tgt)} | {branch_label(br)}. "
                f"X-axis = {xlabel} (top-varied param); Y-axis = {ylabel}. "
                "Color = inner validation MAE (darker = lower = better). Point size ∝ gap_penalty."
            )
            save_thesis_figure(out_dir / fname, caption)
    except Exception as exc:
        log.warning("figure_param_landscapes failed: %s", exc)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_tuning(trials: pd.DataFrame, out_dir: Path, cfg=None) -> dict[str, pd.DataFrame]:
    """Run all Section A outputs. Returns dict of produced tables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}

    if trials.empty:
        log.info("Section A: trials.csv is empty — skipping.")
        return tables

    bsf = table_trial_best_so_far(trials, out_dir)
    tables["tuning_trial_best_so_far"] = bsf

    budget_df = table_budget_utilization(trials, out_dir)
    tables["tuning_budget_utilization"] = budget_df

    figure_convergence_per_context(bsf, out_dir)
    figure_budget_utilization(budget_df, out_dir)
    figure_gap_penalty_scatter(trials, out_dir)
    figure_param_landscapes(trials, out_dir)

    return tables
