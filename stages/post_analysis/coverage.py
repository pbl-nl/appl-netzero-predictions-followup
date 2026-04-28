"""
coverage.py – Missingness-stratified performance and abstention-curve analysis.

Uses final_test_missingness.csv (pre-imputation row missingness fraction for all
2022 countries by imputation branch) joined to final_prediction_summary.csv to produce:

  - missingness_stratified_mae.csv: MAE and row count per missingness band
  - abstention_curve.csv: candidate-level MAE vs coverage as the missingness
    threshold varies from 0 to 1
  - abstention_curve_primary_summary.csv / abstention_curve.png: compact
    target-level view using primary rank-1 candidates
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stages.post_analysis.plot_utils import apply_thesis_style, save_fig

log = logging.getLogger(__name__)

_MISS_BINS = [0.0, 0.10, 0.30, 0.50, 1.01]
_MISS_LABELS = ["0–10%", "10–30%", "30–50%", ">50%"]
_ABSTENTION_THRESHOLDS = np.round(np.arange(0.0, 1.05, 0.05), 2).tolist()


def _join_missingness(
    prediction_summary: pd.DataFrame,
    test_missingness: pd.DataFrame,
) -> pd.DataFrame | None:
    if test_missingness.empty:
        log.warning("coverage: test_missingness is empty — skipping.")
        return None
    if prediction_summary.empty:
        log.warning("coverage: prediction_summary is empty — skipping.")
        return None

    required = {"country", "year", "row_missingness_fraction"}
    if not required.issubset(test_missingness.columns):
        log.warning(
            "coverage: test_missingness missing columns %s — skipping.",
            required - set(test_missingness.columns),
        )
        return None

    key_cols = (
        ["branch", "country", "year"]
        if "branch" in prediction_summary.columns and "branch" in test_missingness.columns
        else ["country", "year"]
    )
    duplicated = test_missingness.duplicated(key_cols, keep=False)
    if duplicated.any():
        log.warning(
            "coverage: test_missingness has %d duplicate rows for join keys %s — skipping.",
            int(duplicated.sum()),
            key_cols,
        )
        return None

    miss_cols = key_cols + ["row_missingness_fraction"]
    for col in ["strict_included", "lenient_included", "all_included"]:
        if col in test_missingness.columns:
            miss_cols.append(col)

    before_rows = len(prediction_summary)
    joined = prediction_summary.merge(
        test_missingness[miss_cols],
        on=key_cols,
        how="left",
        validate="many_to_one",
    )
    if len(joined) != before_rows:
        log.warning(
            "coverage: missingness join changed row count from %d to %d — skipping.",
            before_rows,
            len(joined),
        )
        return None
    n_missing = joined["row_missingness_fraction"].isna().sum()
    if n_missing > 0:
        log.warning(
            "coverage: %d prediction rows could not be joined to missingness data.", n_missing
        )
    return joined


def run_missingness_stratified_table(
    prediction_summary: pd.DataFrame,
    test_missingness: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    """
    Compute MAE per missingness band per (target, model, strategy).

    MAE = mean(|actual - prediction_mean|) over rows in that band.
    Only rows with a non-null actual value contribute to MAE.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    joined = _join_missingness(prediction_summary, test_missingness)
    if joined is None:
        return pd.DataFrame()

    has_actual = joined["actual"].notna()
    rows = joined[has_actual].copy()
    if rows.empty:
        log.warning("coverage: no rows with actual values — skipping stratified table.")
        return pd.DataFrame()

    rows["abs_error"] = (rows["actual"] - rows["prediction_mean"]).abs()
    rows["missingness_band"] = pd.cut(
        rows["row_missingness_fraction"],
        bins=_MISS_BINS,
        labels=_MISS_LABELS,
        right=False,
        include_lowest=True,
    )

    group_cols = [
        c for c in [
            "target",
            "outer_strategy_source",
            "branch",
            "candidate_rank",
            "candidate_role",
            "model_name",
            "missingness_band",
        ]
        if c in rows.columns
    ]
    agg = (
        rows.groupby(group_cols, observed=True)
        .agg(mae=("abs_error", "mean"), n_rows=("abs_error", "count"))
        .reset_index()
    )

    path = out_dir / "missingness_stratified_mae.csv"
    agg.to_csv(path, index=False)
    log.info("Saved: %s (%d rows)", path, len(agg))
    return agg


def run_abstention_curve(
    prediction_summary: pd.DataFrame,
    test_missingness: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    """
    For each missingness threshold t, compute candidate-level MAE and coverage
    fraction on rows with row_missingness_fraction <= t.

    The detailed CSV is candidate-granular. The PNG uses a compact target-level
    summary of primary rank-1 candidates.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    joined = _join_missingness(prediction_summary, test_missingness)
    if joined is None:
        return pd.DataFrame()

    has_actual = joined["actual"].notna() & joined["row_missingness_fraction"].notna()
    rows = joined[has_actual].copy()
    if rows.empty:
        log.warning("coverage: no rows with actual + missingness — skipping abstention curve.")
        return pd.DataFrame()

    rows["abs_error"] = (rows["actual"] - rows["prediction_mean"]).abs()
    group_cols = [
        c for c in [
            "target",
            "outer_strategy_source",
            "branch",
            "candidate_rank",
            "candidate_role",
            "model_name",
        ]
        if c in rows.columns
    ]

    records: list[dict] = []
    grouped = rows.groupby(group_cols, dropna=False) if group_cols else [((), rows)]
    for key_values, grp in grouped:
        if group_cols:
            if len(group_cols) == 1:
                key_values = (key_values,)
            base_record = dict(zip(group_cols, key_values))
        else:
            base_record = {}
        n_total = len(grp)
        for t in _ABSTENTION_THRESHOLDS:
            subset = grp[grp["row_missingness_fraction"] <= t]
            coverage = len(subset) / n_total if n_total > 0 else float("nan")
            mae = float(subset["abs_error"].mean()) if not subset.empty else float("nan")
            records.append({
                **base_record,
                "threshold": t,
                "coverage": coverage,
                "mae": mae,
                "n_rows": int(len(subset)),
                "n_total": int(n_total),
            })

    curve_df = pd.DataFrame(records)
    path = out_dir / "abstention_curve.csv"
    curve_df.to_csv(path, index=False)
    log.info("Saved: %s", path)

    primary_summary = _primary_abstention_summary(curve_df)
    summary_path = out_dir / "abstention_curve_primary_summary.csv"
    primary_summary.to_csv(summary_path, index=False)
    log.info("Saved: %s", summary_path)

    _plot_abstention_curve(primary_summary, out_dir)
    return curve_df


def _primary_abstention_summary(curve_df: pd.DataFrame) -> pd.DataFrame:
    if curve_df.empty:
        return curve_df

    primary = curve_df.copy()
    if "candidate_role" in primary.columns:
        role_filtered = primary[primary["candidate_role"] == "primary"]
        if not role_filtered.empty:
            primary = role_filtered
    if "candidate_rank" in primary.columns:
        rank = pd.to_numeric(primary["candidate_rank"], errors="coerce")
        rank_filtered = primary[rank == 1]
        if not rank_filtered.empty:
            primary = rank_filtered

    group_cols = [c for c in ["threshold", "target"] if c in primary.columns]
    if not group_cols:
        return primary
    return (
        primary.groupby(group_cols, dropna=False)
        .agg(
            coverage=("coverage", "mean"),
            mae=("mae", "mean"),
            n_candidates=("mae", "count"),
        )
        .reset_index()
    )


def _plot_abstention_curve(curve_df: pd.DataFrame, out_dir: Path) -> None:
    apply_thesis_style()
    targets = sorted(curve_df["target"].dropna().unique()) if "target" in curve_df.columns else []
    if not targets:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.Set2(np.linspace(0, 0.8, len(targets)))

    for tgt, color in zip(targets, colors):
        sub = curve_df[curve_df["target"] == tgt].sort_values("coverage")
        ax.plot(sub["coverage"], sub["mae"], marker="o", markersize=3, label=tgt, color=color)

    ax.axvline(x=curve_df.loc[
        (curve_df["threshold"] - 0.30).abs() < 0.01, "coverage"
    ].mean(), color="grey", linestyle="--", linewidth=0.8, label="threshold=0.30")

    ax.set_xlabel("Coverage (fraction of target-valid 2022 countries included)")
    ax.set_ylabel("MAE")
    ax.set_title("Abstention Curve: MAE vs Coverage by Missingness Threshold")
    ax.legend(title="Target scenario")
    save_fig(out_dir / "abstention_curve.png")
    plt.close(fig)
