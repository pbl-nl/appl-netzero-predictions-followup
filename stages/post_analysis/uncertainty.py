"""
uncertainty.py – Bootstrap confidence interval and uncertainty analysis.

Uses the 200-iteration bootstrap predictions from final_bootstrap_predictions.csv
and the aggregated summary from final_prediction_summary.csv to characterise
prediction uncertainty per country, scenario, and model.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from stages.post_analysis.config import (
    SCENARIO_COLORS,
    UNCERTAINTY_BOOTSTRAP_TOP_N,
    UNCERTAINTY_HIGH_PERCENTILE,
    UNCERTAINTY_LOW_PERCENTILE,
)
from stages.post_analysis.plot_utils import apply_thesis_style, save_fig, scenario_label

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CI width computation
# ---------------------------------------------------------------------------


def compute_ci_width(prediction_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``ci_width = prediction_p95 − prediction_p05`` to *prediction_summary*.

    Returns the enriched DataFrame (a copy).
    """
    df = prediction_summary.copy()
    if "prediction_p05" in df.columns and "prediction_p95" in df.columns:
        df["ci_width"] = (
            pd.to_numeric(df["prediction_p95"], errors="coerce")
            - pd.to_numeric(df["prediction_p05"], errors="coerce")
        )
    else:
        log.warning("uncertainty: prediction_p05/p95 columns missing — ci_width not computed.")
        df["ci_width"] = float("nan")
    return df


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def table_country_uncertainty_ranking(
    ci_df: pd.DataFrame,
    out_dir: Path,
    high_pct: int = UNCERTAINTY_HIGH_PERCENTILE,
    low_pct: int  = UNCERTAINTY_LOW_PERCENTILE,
) -> pd.DataFrame:
    """
    Rank countries by mean ci_width (across all models/strategies/branches).

    Flags top *high_pct* percentile as ``high_uncertainty`` and bottom
    *low_pct* percentile as ``low_uncertainty``.

    Saves and returns ``country_uncertainty_ranking.csv``.
    """
    if "ci_width" not in ci_df.columns or "country" not in ci_df.columns:
        log.warning("uncertainty: required columns missing for ranking.")
        return pd.DataFrame()

    group_cols = ["country"]
    if "target" in ci_df.columns:
        group_cols.append("target")

    agg = (
        ci_df.groupby(group_cols)["ci_width"]
        .agg(mean_ci_width="mean", max_ci_width="max", std_ci_width="std")
        .reset_index()
        .sort_values("mean_ci_width", ascending=False)
    )

    hi_thresh = np.nanpercentile(agg["mean_ci_width"], high_pct)
    lo_thresh = np.nanpercentile(agg["mean_ci_width"], low_pct)
    agg["uncertainty_tier"] = "medium"
    agg.loc[agg["mean_ci_width"] >= hi_thresh, "uncertainty_tier"] = "high"
    agg.loc[agg["mean_ci_width"] <= lo_thresh, "uncertainty_tier"] = "low"

    path = out_dir / "country_uncertainty_ranking.csv"
    agg.to_csv(path, index=False)
    log.info("Saved: %s", path)
    return agg


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def figure_ci_width_histogram(ci_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Histogram of ci_width per target scenario (3 panels).
    """
    if "ci_width" not in ci_df.columns:
        return

    targets = (
        sorted(ci_df["target"].unique()) if "target" in ci_df.columns else [None]
    )
    n = len(targets)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, target in zip(axes, targets):
        subset = ci_df[ci_df["target"] == target]["ci_width"].dropna() if target else ci_df["ci_width"].dropna()
        color = SCENARIO_COLORS.get(str(target), "#4393c3") if target else "#4393c3"
        ax.hist(subset, bins=30, color=color, edgecolor="white", alpha=0.85)
        ax.set_xlabel("CI Width (p95 − p05, log scale)")
        ax.set_ylabel("Count")
        title = scenario_label(str(target)) if target else "All scenarios"
        ax.set_title(f"{title}\nmedian = {subset.median():.3f}")

    fig.suptitle("Bootstrap Prediction Uncertainty (CI Width)", fontsize=13)
    save_fig(out_dir / "ci_width_histogram.png")
    log.info("Saved CI width histogram.")


def figure_ci_width_map(
    ci_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Choropleth of mean ci_width per country, one map per target scenario.
    """
    if "ci_width" not in ci_df.columns or "country" not in ci_df.columns:
        return

    try:
        import geopandas as gpd
        from stages.post_analysis.benchmark import _try_load_naturalearth, _ISO_CORRECTIONS
    except ImportError:
        log.warning("geopandas not available — skipping CI width maps.")
        return

    world = _try_load_naturalearth()
    if world is None:
        return

    iso_col = next((c for c in ["iso_a3", "ISO_A3", "ADM0_A3"] if c in world.columns), None)
    if iso_col is None:
        return

    targets = sorted(ci_df["target"].unique()) if "target" in ci_df.columns else [None]

    for target in targets:
        subset = ci_df[ci_df["target"] == target] if target else ci_df
        country_ci = (
            subset.groupby("country")["ci_width"]
            .mean()
            .reset_index()
            .rename(columns={"country": iso_col, "ci_width": "_ci"})
        )
        country_ci[iso_col] = country_ci[iso_col].replace(_ISO_CORRECTIONS)

        merged_geo = world.merge(country_ci, on=iso_col, how="left")

        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        world.boundary.plot(ax=ax, linewidth=0.3, color="grey")
        merged_geo.dropna(subset=["_ci"]).plot(
            column="_ci", ax=ax, cmap="YlOrRd",
            legend=True,
            legend_kwds={"label": "Mean CI Width (log scale)", "shrink": 0.6},
            missing_kwds={"color": "#e0e0e0"},
        )
        merged_geo[merged_geo["_ci"].isna()].plot(ax=ax, color="#e0e0e0")

        title = f"Prediction Uncertainty (CI Width) — {scenario_label(str(target))}" if target else "Prediction Uncertainty (CI Width)"
        ax.set_title(title)
        ax.axis("off")

        fname = f"ci_width_map_{target}.png" if target else "ci_width_map.png"
        save_fig(out_dir / fname)
        log.info("Saved CI width map: %s", fname)


def figure_bootstrap_distribution(
    bootstrap_predictions: pd.DataFrame,
    out_dir: Path,
    top_n: int = UNCERTAINTY_BOOTSTRAP_TOP_N,
) -> None:
    """
    KDE plot of the 200 bootstrap prediction values for the top-N highest-
    uncertainty countries per target scenario.
    """
    if bootstrap_predictions.empty or "prediction" not in bootstrap_predictions.columns:
        return
    if "country" not in bootstrap_predictions.columns:
        return

    targets = (
        sorted(bootstrap_predictions["target"].unique())
        if "target" in bootstrap_predictions.columns
        else [None]
    )

    for target in targets:
        subset = (
            bootstrap_predictions[bootstrap_predictions["target"] == target]
            if target
            else bootstrap_predictions
        )

        # Find top-N countries by std of bootstrap predictions
        country_std = (
            subset.groupby("country")["prediction"]
            .std()
            .nlargest(top_n)
        )
        top_countries = country_std.index.tolist()
        if not top_countries:
            continue

        n = len(top_countries)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
        axes = np.array(axes).flatten()

        for ax, country in zip(axes, top_countries):
            vals = subset[subset["country"] == country]["prediction"].dropna()
            color = SCENARIO_COLORS.get(str(target), "#4393c3") if target else "#4393c3"
            ax.hist(vals, bins=20, color=color, edgecolor="white", alpha=0.8, density=True)
            try:
                import scipy.stats as stats
                kde = stats.gaussian_kde(vals)
                xs = np.linspace(vals.min(), vals.max(), 200)
                ax.plot(xs, kde(xs), color="black", linewidth=1.2)
            except Exception:
                pass
            ax.set_title(f"{country}\nσ = {vals.std():.3f}", fontsize=9)
            ax.set_xlabel("Bootstrap prediction")
            ax.set_ylabel("")

        # Hide unused subplots
        for ax in axes[n:]:
            ax.set_visible(False)

        scenario_str = scenario_label(str(target)) if target else "All scenarios"
        fig.suptitle(f"Bootstrap Distribution — Top-{n} Uncertain Countries\n{scenario_str}", fontsize=12)

        fname = f"bootstrap_kde_{target}.png" if target else "bootstrap_kde.png"
        save_fig(out_dir / fname)
        log.info("Saved bootstrap KDE: %s", fname)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_uncertainty(
    bootstrap_predictions: pd.DataFrame,
    prediction_summary: pd.DataFrame,
    out_dir: Path,
    high_pct: int = UNCERTAINTY_HIGH_PERCENTILE,
    low_pct: int  = UNCERTAINTY_LOW_PERCENTILE,
) -> dict[str, pd.DataFrame]:
    """
    Main entry point for uncertainty analysis.

    Returns dict of produced DataFrames.
    """
    apply_thesis_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: dict[str, pd.DataFrame] = {}

    if prediction_summary.empty:
        log.warning("uncertainty: prediction_summary is empty — skipping.")
        return produced

    ci_df = compute_ci_width(prediction_summary)
    ranking_df = table_country_uncertainty_ranking(ci_df, out_dir, high_pct, low_pct)
    produced["country_uncertainty_ranking"] = ranking_df

    figure_ci_width_histogram(ci_df, out_dir)
    figure_ci_width_map(ci_df, out_dir)

    if not bootstrap_predictions.empty:
        figure_bootstrap_distribution(bootstrap_predictions, out_dir)

    return produced
