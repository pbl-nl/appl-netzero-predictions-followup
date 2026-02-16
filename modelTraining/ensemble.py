"""Stacking ensemble across algorithms using Ridge meta-learner."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

logger = logging.getLogger(__name__)


def build_stacking_ensemble(
    valid_predictions: dict[str, pd.DataFrame],
    inference_predictions: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Build a stacking ensemble from per-algorithm predictions.

    Parameters
    ----------
    valid_predictions : dict[str, DataFrame]
        ``{algorithm: valid_predictions_df}`` — each df has columns
        ``run_id, country, year, y_pred_rce, y_true_rce``.
    inference_predictions : dict[str, DataFrame]
        ``{algorithm: inference_predictions_df}`` — each df has columns
        ``run_id, country, (year), rce_pred``.

    Returns
    -------
    (ensemble_valid_df, ensemble_inference_df) or None if stacking
    is not possible (< 2 algorithms).
    """
    algorithms = sorted(valid_predictions.keys())
    if len(algorithms) < 2:
        logger.info("Stacking requires >= 2 algorithms, got %d. Skipping.", len(algorithms))
        return None

    # Step 1: Collect median-across-runs validation prediction per country for
    # each algorithm.
    valid_medians: dict[str, pd.DataFrame] = {}
    for algo in algorithms:
        vdf = valid_predictions[algo]
        if vdf.empty or "y_pred_rce" not in vdf.columns:
            continue

        group_cols = ["country"]
        if "year" in vdf.columns:
            group_cols.append("year")

        median_df = (
            vdf.groupby(group_cols, as_index=False)
            .agg(
                pred_rce=("y_pred_rce", "median"),
                actual_rce=("y_true_rce", "first"),
            )
        )
        valid_medians[algo] = median_df

    usable_algos = sorted(valid_medians.keys())
    if len(usable_algos) < 2:
        logger.info("Not enough usable algorithms for stacking (%d). Skipping.", len(usable_algos))
        return None

    # Build the validation matrix — rows are country(+year), columns are algorithm predictions.
    # Use the first algorithm's frame as the key index.
    ref = valid_medians[usable_algos[0]]
    group_cols = ["country"]
    if "year" in ref.columns:
        group_cols.append("year")

    matrix = ref[group_cols + ["actual_rce"]].copy()
    matrix = matrix.rename(columns={"pred_rce": usable_algos[0]})

    for algo in usable_algos:
        algo_df = valid_medians[algo][group_cols + ["pred_rce"]].rename(
            columns={"pred_rce": algo}
        )
        matrix = matrix.merge(algo_df, on=group_cols, how="inner")

    if len(matrix) < 3:
        logger.warning("Stacking matrix has only %d rows, too few for LOO-CV. Skipping.", len(matrix))
        return None

    X_stack = matrix[usable_algos].to_numpy(dtype=float)
    y_actual = matrix["actual_rce"].to_numpy(dtype=float)

    # Step 2: LOO-CV stacking predictions (for diagnostics / unbiased eval)
    loo_preds = np.full(len(y_actual), np.nan, dtype=float)
    for i in range(len(y_actual)):
        mask = np.ones(len(y_actual), dtype=bool)
        mask[i] = False
        ridge_loo = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        ridge_loo.fit(X_stack[mask], y_actual[mask])
        loo_preds[i] = float(ridge_loo.predict(X_stack[i : i + 1])[0])

    # Step 3: Train final Ridge on all validation countries.
    final_ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    final_ridge.fit(X_stack, y_actual)
    logger.info(
        "Stacking Ridge alpha=%.4f, weights=%s",
        final_ridge.alpha_,
        dict(zip(usable_algos, final_ridge.coef_.tolist())),
    )

    # Build ensemble validation DataFrame (with LOO predictions).
    ensemble_valid = matrix[group_cols + ["actual_rce"]].copy()
    ensemble_valid["y_pred_rce"] = loo_preds
    ensemble_valid["y_true_rce"] = ensemble_valid["actual_rce"]
    # Add a dummy run_id=0 so it integrates with existing pipeline expectations.
    ensemble_valid["run_id"] = 0

    # Step 4: Apply learned weights to inference predictions.
    infer_medians: dict[str, pd.DataFrame] = {}
    for algo in usable_algos:
        idf = inference_predictions.get(algo)
        if idf is None or idf.empty:
            continue

        group_cols_inf = ["country"]
        if "year" in idf.columns:
            group_cols_inf.append("year")

        median_df = (
            idf.groupby(group_cols_inf, as_index=False)
            .agg(pred_rce=("rce_pred", "median"))
        )
        infer_medians[algo] = median_df

    if len(infer_medians) < 2:
        logger.warning("Not enough inference predictions for stacking. Skipping inference.")
        return None

    ref_inf = infer_medians[usable_algos[0]]
    group_cols_inf = ["country"]
    if "year" in ref_inf.columns:
        group_cols_inf.append("year")

    infer_matrix = ref_inf[group_cols_inf].copy()
    for algo in usable_algos:
        if algo not in infer_medians:
            logger.warning("Algorithm %s missing from inference predictions. Skipping stacking.", algo)
            return None
        algo_df = infer_medians[algo][group_cols_inf + ["pred_rce"]].rename(
            columns={"pred_rce": algo}
        )
        infer_matrix = infer_matrix.merge(algo_df, on=group_cols_inf, how="inner")

    X_infer = infer_matrix[usable_algos].to_numpy(dtype=float)
    ensemble_rce = final_ridge.predict(X_infer)

    ensemble_inference = infer_matrix[group_cols_inf].copy()
    ensemble_inference["rce_pred"] = ensemble_rce
    ensemble_inference["run_id"] = 0

    return ensemble_valid, ensemble_inference
