"""Evaluation utilities for modelTraining."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from modelTraining.config.base import ModelTrainingConfig
from modelTraining.utils import confidence_interval


METRIC_COLUMNS = ["RMSE_log", "MAE_log", "R2", "Within_RMSE", "NZ_MAE"]


def summarize_run_metrics(
    run_metrics_df: pd.DataFrame,
    confidence_level: float,
) -> dict[str, float]:
    """Aggregate run-level metrics into mean/std/CI fields."""
    summary: dict[str, float] = {}

    for metric in METRIC_COLUMNS:
        if metric not in run_metrics_df.columns:
            values = np.array([], dtype=float)
        else:
            values = pd.to_numeric(run_metrics_df[metric], errors="coerce").to_numpy(dtype=float)

        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            summary[f"{metric}_mean"] = float("nan")
            summary[f"{metric}_std"] = float("nan")
            summary[f"{metric}_ci_low"] = float("nan")
            summary[f"{metric}_ci_high"] = float("nan")
            continue

        summary[f"{metric}_mean"] = float(np.mean(finite_values))
        summary[f"{metric}_std"] = float(np.std(finite_values, ddof=0))
        ci_low, ci_high = confidence_interval(finite_values, confidence_level)
        summary[f"{metric}_ci_low"] = ci_low
        summary[f"{metric}_ci_high"] = ci_high

    return summary


def add_nz_mae_to_runs(
    run_metrics_df: pd.DataFrame,
    inversion_per_run_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach NZ_MAE to run metrics using per-run inversion outputs."""
    out = run_metrics_df.copy()
    if inversion_per_run_df.empty:
        out["NZ_MAE"] = float("nan")
        return out

    required = {"run_id", "nz_year_pred", "pledge_year"}
    if not required.issubset(set(inversion_per_run_df.columns)):
        out["NZ_MAE"] = float("nan")
        return out

    per_run_mae = (
        inversion_per_run_df.assign(
            abs_err=lambda d: np.abs(d["nz_year_pred"] - d["pledge_year"])
        )
        .groupby("run_id", as_index=False)["abs_err"]
        .mean()
        .rename(columns={"abs_err": "NZ_MAE"})
    )

    out = out.merge(per_run_mae, on="run_id", how="left")
    return out


def build_evaluation_row(
    algorithm: str,
    split: str,
    scenario: str,
    run_metrics_df: pd.DataFrame,
    config: ModelTrainingConfig,
) -> pd.DataFrame:
    """Create one summary row for algorithm/split/scenario."""
    summary = summarize_run_metrics(
        run_metrics_df=run_metrics_df,
        confidence_level=config.confidence_level,
    )

    row = {
        "Algorithm": algorithm,
        "Split": split,
        "Scenario": scenario,
        **summary,
    }
    return pd.DataFrame([row])


def build_algorithm_comparison(
    eval_rows: pd.DataFrame,
    ridge_baselines: pd.DataFrame,
) -> pd.DataFrame:
    """Build cross-algorithm comparison matrix."""
    if eval_rows.empty:
        return pd.DataFrame(
            columns=[
                "Algorithm",
                "Split",
                "Scenario",
                "RMSE_mean",
                "RMSE_std",
                "Within_RMSE",
                "NZ_MAE",
                "vs_Ridge",
            ]
        )

    ridge_lookup = (
        ridge_baselines[ridge_baselines["Algorithm"] == "ridge"]
        .set_index(["Split", "Scenario"])["RMSE_log"]
        .to_dict()
    )

    rows = []
    for row in eval_rows.itertuples(index=False):
        ridge_rmse = ridge_lookup.get((row.Split, row.Scenario))
        if ridge_rmse is None or not np.isfinite(ridge_rmse):
            vs_ridge = float("nan")
        else:
            vs_ridge = float((row.RMSE_log_mean - ridge_rmse) / ridge_rmse)

        rows.append(
            {
                "Algorithm": row.Algorithm,
                "Split": row.Split,
                "Scenario": row.Scenario,
                "RMSE_mean": row.RMSE_log_mean,
                "RMSE_std": row.RMSE_log_std,
                "Within_RMSE": row.Within_RMSE_mean,
                "NZ_MAE": row.NZ_MAE_mean,
                "vs_Ridge": vs_ridge,
            }
        )

    return pd.DataFrame(rows).sort_values(["Split", "Scenario", "Algorithm"]).reset_index(drop=True)


def build_cv_comparison(cv_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize outer CV results per algorithm/split/scenario.

    Input has one row per fold.  Output has one row per algorithm/split/scenario
    with mean/std RMSE_log across folds.
    """
    if cv_results.empty:
        return pd.DataFrame()

    grouped = cv_results.groupby(["Algorithm", "Split", "Scenario"], as_index=False)
    summary = grouped.agg(
        RMSE_log_mean=("RMSE_log", "mean"),
        RMSE_log_std=("RMSE_log", "std"),
        MAE_log_mean=("MAE_log", "mean"),
        R2_mean=("R2", "mean"),
        n_folds=("fold", "count"),
    )
    summary["RMSE_log_std"] = summary["RMSE_log_std"].fillna(0.0)
    return summary.sort_values(["Split", "Scenario", "RMSE_log_mean"]).reset_index(drop=True)


def feature_importance_consistency(
    importances: Iterable[pd.Series],
    top_k: int = 10,
) -> float:
    """Estimate consistency of top-K feature rankings across runs.

    Returns average Jaccard overlap in [0, 1].
    """
    sets = []
    for series in importances:
        if series is None or series.empty:
            continue
        top = set(series.sort_values(ascending=False).head(top_k).index.tolist())
        sets.append(top)

    if len(sets) < 2:
        return float("nan")

    overlaps = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            if not union:
                continue
            overlaps.append(len(sets[i] & sets[j]) / len(union))

    if not overlaps:
        return float("nan")
    return float(np.mean(overlaps))
