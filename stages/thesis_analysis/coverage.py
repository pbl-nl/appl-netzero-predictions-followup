"""
coverage.py — Section G: Strict vs lenient test-set coverage analysis.

Source: final_test_missingness.csv, final_prediction_summary.csv, final_results.csv
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from stages.post_analysis.config import NATURALEARTH_DOWNLOAD_URL
from stages.post_analysis.matplotlib_setup import configure_matplotlib_environment
configure_matplotlib_environment()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from stages.post_analysis.plot_utils import (
    apply_thesis_style, model_color_map, strategy_label,
    branch_label, add_diagonal, annotate_n,
)
from stages.post_analysis.config import FIGURE_DPI
from stages.post_analysis.benchmark import _try_load_naturalearth

from stages.thesis_analysis._shared import save_thesis_figure, save_thesis_table, strategy_col

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def table_set_composition(test_missingness: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Per branch: country counts in strict / lenient / all prediction sets."""
    try:
        if test_missingness.empty:
            return pd.DataFrame()

        rows = []
        for br, grp in test_missingness.groupby("branch") if "branch" in test_missingness.columns else [("all", test_missingness)]:
            n_strict = grp["strict_included"].sum() if "strict_included" in grp.columns else np.nan
            n_lenient = grp["lenient_included"].sum() if "lenient_included" in grp.columns else np.nan
            n_all = grp["all_included"].sum() if "all_included" in grp.columns else np.nan

            strict_col = grp["strict_included"].astype(bool) if "strict_included" in grp.columns else pd.Series(False, index=grp.index)
            lenient_col = grp["lenient_included"].astype(bool) if "lenient_included" in grp.columns else pd.Series(False, index=grp.index)
            all_col = grp["all_included"].astype(bool) if "all_included" in grp.columns else pd.Series(True, index=grp.index)

            n_all_only = (all_col & ~lenient_col).sum()
            n_lenient_only = (lenient_col & ~strict_col).sum()

            rows.append({
                "branch": br,
                "n_strict": int(n_strict),
                "n_lenient": int(n_lenient),
                "n_all": int(n_all),
                "n_lenient_only": int(n_lenient_only),
                "n_all_only": int(n_all_only),
                "pct_strict_of_all": round(100 * n_strict / n_all, 1) if n_all > 0 else np.nan,
            })

        result = pd.DataFrame(rows)
        return save_thesis_table(result, out_dir, "coverage_set_composition",
                                 caption="Test set composition by imputation branch: how many countries survive strict vs lenient row-dropping.")
    except Exception as exc:
        log.warning("table_set_composition failed: %s", exc)
        return pd.DataFrame()


def table_strict_vs_lenient_mae(final_results: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Per winner: strict vs lenient final test MAE with ratio."""
    try:
        if final_results.empty:
            return pd.DataFrame()

        cols = [c for c in [
            "target", "branch", "model_name", "candidate_rank", "candidate_role",
            "final_model_test_mae", "final_model_test_mae_lenient",
            "final_test_rows", "final_test_rows_lenient",
        ] if c in final_results.columns]
        scol = strategy_col(final_results)
        if scol:
            cols = [scol] + cols

        result = final_results[cols].copy()
        if "final_model_test_mae" in result.columns and "final_model_test_mae_lenient" in result.columns:
            result["mae_ratio_lenient_over_strict"] = (
                result["final_model_test_mae_lenient"] / result["final_model_test_mae"].replace(0, np.nan)
            )

        return save_thesis_table(result, out_dir, "coverage_strict_vs_lenient_mae",
                                 caption="Strict vs lenient final holdout MAE per winner — shows impact of test-set composition on evaluation.")
    except Exception as exc:
        log.warning("table_strict_vs_lenient_mae failed: %s", exc)
        return pd.DataFrame()


def table_country_inclusion_detail(test_missingness: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Per country: missingness fraction and inclusion status across branches side by side."""
    try:
        if test_missingness.empty or "country" not in test_missingness.columns:
            return pd.DataFrame()

        branches = test_missingness["branch"].unique() if "branch" in test_missingness.columns else []
        country_col = "country"

        dfs = []
        for br in branches:
            sub = test_missingness[test_missingness["branch"] == br].copy() if "branch" in test_missingness.columns else test_missingness
            rename = {}
            for col in ["row_missingness_fraction", "strict_included", "lenient_included", "all_included"]:
                if col in sub.columns:
                    rename[col] = f"{br}_{col}"
            dfs.append(sub[[country_col] + list(rename.keys())].rename(columns=rename))

        if not dfs:
            return pd.DataFrame()

        result = dfs[0]
        for d in dfs[1:]:
            result = result.merge(d, on=country_col, how="outer")

        # Derived flags
        strict_cols = [c for c in result.columns if c.endswith("_strict_included")]
        lenient_cols = [c for c in result.columns if c.endswith("_lenient_included")]
        all_cols = [c for c in result.columns if c.endswith("_all_included")]

        if strict_cols:
            result["strict_in_any_branch"] = result[strict_cols].any(axis=1)
        if lenient_cols:
            result["lenient_in_any_branch"] = result[lenient_cols].any(axis=1)
        if strict_cols and all_cols:
            result["all_only"] = (~result[strict_cols].any(axis=1)) & result[all_cols].any(axis=1)

        return save_thesis_table(result, out_dir, "coverage_country_inclusion_detail",
                                 caption="Per-country 2022 test set inclusion status across imputation branches.")
    except Exception as exc:
        log.warning("table_country_inclusion_detail failed: %s", exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def figure_missingness_histogram(test_missingness: pd.DataFrame, out_dir: Path) -> None:
    """Histogram of row_missingness_fraction per branch, colored by inclusion status."""
    try:
        if test_missingness.empty or "row_missingness_fraction" not in test_missingness.columns:
            return
        apply_thesis_style()

        for br, grp in test_missingness.groupby("branch") if "branch" in test_missingness.columns else [("all", test_missingness)]:
            fig, ax = plt.subplots(figsize=(7, 4))

            frac = grp["row_missingness_fraction"].dropna()
            strict = grp["strict_included"].astype(bool) if "strict_included" in grp.columns else pd.Series(True, index=grp.index)
            lenient = grp["lenient_included"].astype(bool) if "lenient_included" in grp.columns else pd.Series(True, index=grp.index)
            all_inc = grp["all_included"].astype(bool) if "all_included" in grp.columns else pd.Series(True, index=grp.index)

            bins = np.linspace(0, 1, 25)

            # Layered histogram by inclusion category
            ax.hist(frac[all_inc & ~lenient], bins=bins, color="lightgrey", alpha=0.9,
                    label="All only (prediction-only)", edgecolor="white")
            ax.hist(frac[lenient & ~strict], bins=bins, color="darkorange", alpha=0.8,
                    label="Lenient only", edgecolor="white")
            ax.hist(frac[strict], bins=bins, color="steelblue", alpha=0.8,
                    label="Strict included", edgecolor="white")

            ax.set_xlabel("Row missingness fraction")
            ax.set_ylabel("Number of countries")
            ax.set_title(f"2022 row missingness distribution — {branch_label(br)}", fontsize=10)
            ax.legend(fontsize=8)
            ax.set_xlim(0, 1)

            caption = (
                f"Distribution of 2022 pre-imputation row missingness fraction for {branch_label(br)} imputation. "
                "Blue = in strict test set (drop_rows=True); orange = lenient only (drop_rows='targets_only'); "
                "grey = prediction-only (all_included, no actual available for scoring)."
            )
            save_thesis_figure(out_dir / f"coverage_missingness_histogram_{br}.png", caption)
    except Exception as exc:
        log.warning("figure_missingness_histogram failed: %s", exc)


def figure_strict_vs_lenient_mae_scatter(final_results: pd.DataFrame, out_dir: Path) -> None:
    """Scatter of strict vs lenient final test MAE per winner."""
    try:
        if final_results.empty:
            return
        xcol = "final_model_test_mae"
        ycol = "final_model_test_mae_lenient"
        if xcol not in final_results.columns or ycol not in final_results.columns:
            return
        apply_thesis_style()

        scol = strategy_col(final_results) or "outer_strategy_source"
        models = final_results["model_name"].unique().tolist() if "model_name" in final_results.columns else []
        colors = model_color_map(models)

        for strat in (final_results[scol].unique() if scol in final_results.columns else ["all"]):
            sub = final_results[final_results[scol] == strat] if scol in final_results.columns else final_results
            sub = sub.dropna(subset=[xcol, ycol])
            if sub.empty:
                continue

            fig, ax = plt.subplots(figsize=(6, 5))
            for model, mgrp in sub.groupby("model_name") if "model_name" in sub.columns else [("all", sub)]:
                col = colors.get(model, "steelblue")
                ax.scatter(mgrp[xcol], mgrp[ycol], color=col, alpha=0.8, s=50, label=model)
                for _, row in mgrp.iterrows():
                    rank = row.get("candidate_rank", "")
                    ax.annotate(f"r{rank}", (row[xcol], row[ycol]),
                                textcoords="offset points", xytext=(4, 3), fontsize=7)

            add_diagonal(ax)
            ax.set_xlabel("Final test MAE (strict, drop_rows=True)")
            ax.set_ylabel("Final test MAE (lenient, drop_rows='targets_only')")
            ax.set_title(f"Strict vs lenient final test MAE — {strategy_label(strat)}", fontsize=10)
            ax.legend(fontsize=8)
            annotate_n(ax, len(sub))

            caption = (
                f"Strict vs lenient final holdout MAE for {strategy_label(strat)}. "
                "X = test MAE on strict set (rows with >30% missingness dropped); "
                "Y = test MAE on lenient set (only target-missing rows dropped). "
                "Points above y=x: lenient is harder (wider test set includes more difficult countries). "
                "Answers: does the test-set composition decision change conclusions?"
            )
            save_thesis_figure(out_dir / f"coverage_strict_vs_lenient_mae_scatter_{strat}.png", caption)
    except Exception as exc:
        log.warning("figure_strict_vs_lenient_mae_scatter failed: %s", exc)


def figure_country_coverage_map(test_missingness: pd.DataFrame, out_dir: Path) -> None:
    """Choropleth world map of inclusion status per country (requires geopandas)."""
    try:
        import geopandas as gpd
    except ImportError:
        log.info("geopandas not available — skipping coverage_country_coverage_map.png")
        return

    try:
        if test_missingness.empty or "country" not in test_missingness.columns:
            return
        apply_thesis_style()

        # Determine strictest inclusion status per country across all branches
        strict_cols = [c for c in test_missingness.columns if c.endswith("_strict_included") or c == "strict_included"]
        lenient_cols = [c for c in test_missingness.columns if c.endswith("_lenient_included") or c == "lenient_included"]
        all_cols = [c for c in test_missingness.columns if c.endswith("_all_included") or c == "all_included"]

        country_status: dict[str, str] = {}
        for country, grp in test_missingness.groupby("country"):
            strict_any = any(grp[c].astype(bool).any() for c in strict_cols if c in grp.columns)
            lenient_any = any(grp[c].astype(bool).any() for c in lenient_cols if c in grp.columns)
            all_any = any(grp[c].astype(bool).any() for c in all_cols if c in grp.columns)
            if strict_any:
                country_status[country] = "strict"
            elif lenient_any:
                country_status[country] = "lenient_only"
            elif all_any:
                country_status[country] = "all_only"
            else:
                country_status[country] = "absent"

        status_df = pd.DataFrame(list(country_status.items()), columns=["iso3", "status"])

        world = _try_load_naturalearth()
        if world is None:
            log.info("naturalearth shapefile not found — skipping coverage map.")
            return
        world = world.merge(status_df, left_on="iso_a3", right_on="iso3", how="left")
        world["status"] = world["status"].fillna("not_in_dataset")

        color_map = {
            "strict": "#1565c0",
            "lenient_only": "#fb8c00",
            "all_only": "#b0bec5",
            "absent": "#e0e0e0",
            "not_in_dataset": "white",
        }
        world["color"] = world["status"].map(color_map)

        fig, ax = plt.subplots(figsize=(14, 7))
        world.boundary.plot(ax=ax, linewidth=0.3, color="grey")
        for status, color in color_map.items():
            sub = world[world["status"] == status]
            if not sub.empty:
                sub.plot(ax=ax, color=color)

        patches = [
            mpatches.Patch(color="#1565c0", label="Strict included (both branches)"),
            mpatches.Patch(color="#fb8c00", label="Lenient only"),
            mpatches.Patch(color="#b0bec5", label="Prediction only (all_included)"),
            mpatches.Patch(color="#e0e0e0", label="Absent from test data"),
        ]
        ax.legend(handles=patches, fontsize=8, loc="lower left")
        ax.set_title("2022 test set coverage — per-country strictest inclusion status", fontsize=11)
        ax.axis("off")

        caption = (
            "World map of 2022 test set inclusion status per country (strictest across both imputation branches). "
            "Dark blue = in strict test set; orange = lenient only; grey = prediction-only (no ground-truth scoring); "
            "light grey = absent from the panel data."
        )
        save_thesis_figure(out_dir / "coverage_country_coverage_map.png", caption)
    except Exception as exc:
        log.warning("figure_country_coverage_map failed: %s", exc)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_coverage(test_missingness: pd.DataFrame, prediction_summary: pd.DataFrame,
                 final_results: pd.DataFrame, out_dir: Path, cfg=None) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}

    comp = table_set_composition(test_missingness, out_dir)
    tables["coverage_set_composition"] = comp

    sl_mae = table_strict_vs_lenient_mae(final_results, out_dir)
    tables["coverage_strict_vs_lenient_mae"] = sl_mae

    detail = table_country_inclusion_detail(test_missingness, out_dir)
    tables["coverage_country_inclusion_detail"] = detail

    figure_missingness_histogram(test_missingness, out_dir)
    figure_strict_vs_lenient_mae_scatter(final_results, out_dir)
    figure_country_coverage_map(test_missingness, out_dir)

    return tables
