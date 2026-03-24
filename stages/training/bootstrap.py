from __future__ import annotations

import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from stages.training.config import (
    COUNTRY_COL,
    DEFAULT_CPU_WORKERS,
    FINAL_BOOTSTRAP_ITERATIONS,
    FINAL_BOOTSTRAP_MAX_WORKERS,
    FINAL_PREDICTION_SUMMARY_COLUMNS,
    MODEL_RANDOM_STATE,
    PIPELINE_RANDOM_STATE,
    YEAR_COL,
)
from stages.training.helpers import (
    build_model,
    evaluate_metrics,
    fit_model,
)

logger = logging.getLogger(__name__)

_WORKER_STATE: dict[str, Any] | None = None


def build_prediction_summary(predictions_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "outer_strategy_source", "target", "branch", "model_name",
        COUNTRY_COL, YEAR_COL, "actual",
    ]
    summary_df = predictions_df.groupby(group_cols, as_index=False, dropna=False).agg(
        prediction_mean=("prediction", "mean"),
        prediction_std=("prediction", "std"),
        prediction_median=("prediction", "median"),
    )
    quantiles_df = (
        predictions_df.groupby(group_cols, dropna=False)["prediction"]
        .quantile([0.05, 0.95])
        .unstack()
        .reset_index()
        .rename(columns={0.05: "prediction_p05", 0.95: "prediction_p95"})
    )
    summary_df = summary_df.merge(quantiles_df, on=group_cols, how="left")
    summary_df["prediction_std"] = summary_df["prediction_std"].fillna(0.0)
    return summary_df[FINAL_PREDICTION_SUMMARY_COLUMNS].copy()


def _run_one_iteration(
    model_name: str,
    best_params: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test_scored: pd.DataFrame,
    y_test_scored: pd.Series,
    x_test_all: pd.DataFrame,
    iteration_idx: int,
) -> tuple[int, dict[str, float], dict[str, float], np.ndarray]:
    bootstrap_seed = PIPELINE_RANDOM_STATE + iteration_idx
    x_boot = x_train.sample(frac=1.0, replace=True, random_state=bootstrap_seed)
    y_boot = y_train.loc[x_boot.index]

    model = fit_model(
        build_model(model_name, best_params, random_state=MODEL_RANDOM_STATE),
        x_boot, y_boot,
    )

    train_preds = model.predict(x_train)
    train_metrics = evaluate_metrics(y_train, train_preds)

    scored_preds = model.predict(x_test_scored)
    scored_metrics = evaluate_metrics(y_test_scored, scored_preds)

    all_preds = model.predict(x_test_all)
    return iteration_idx, train_metrics, scored_metrics, all_preds


def _init_worker(
    model_name: str,
    best_params: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test_scored: pd.DataFrame,
    y_test_scored: pd.Series,
    x_test_all: pd.DataFrame,
) -> None:
    global _WORKER_STATE
    _WORKER_STATE = {
        "model_name": model_name,
        "best_params": best_params,
        "x_train": x_train,
        "y_train": y_train,
        "x_test_scored": x_test_scored,
        "y_test_scored": y_test_scored,
        "x_test_all": x_test_all,
    }
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message=".*Early stopping criterion not reached.*")


def _run_iteration_from_state(
    iteration_idx: int,
) -> tuple[int, dict[str, float], dict[str, float], np.ndarray]:
    state = _WORKER_STATE
    if state is None:
        raise RuntimeError("Worker state not initialized.")
    return _run_one_iteration(
        model_name=str(state["model_name"]),
        best_params=dict(state["best_params"]),
        x_train=state["x_train"],
        y_train=state["y_train"],
        x_test_scored=state["x_test_scored"],
        y_test_scored=state["y_test_scored"],
        x_test_all=state["x_test_all"],
        iteration_idx=iteration_idx,
    )


def run_final_bootstrap_iterations(
    target: str,
    model_name: str,
    best_params: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test_scored: pd.DataFrame,
    y_test_scored: pd.Series,
    x_test_all: pd.DataFrame,
) -> list[tuple[int, dict[str, float], dict[str, float], np.ndarray]]:
    iteration_indices = list(range(FINAL_BOOTSTRAP_ITERATIONS))
    max_workers = min(
        FINAL_BOOTSTRAP_ITERATIONS,
        max(1, FINAL_BOOTSTRAP_MAX_WORKERS),
        DEFAULT_CPU_WORKERS,
    )

    if FINAL_BOOTSTRAP_ITERATIONS > 1 and max_workers > 1:
        logger.info(
            f"Final bootstrap: target={target} workers={max_workers} "
            f"iterations={FINAL_BOOTSTRAP_ITERATIONS}"
        )
        try:
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_init_worker,
                initargs=(
                    model_name, best_params,
                    x_train, y_train,
                    x_test_scored, y_test_scored,
                    x_test_all,
                ),
            ) as executor:
                futures = [
                    executor.submit(_run_iteration_from_state, idx)
                    for idx in iteration_indices
                ]
                return sorted(
                    [f.result() for f in as_completed(futures)],
                    key=lambda item: item[0],
                )
        except PermissionError as exc:
            logger.warning(
                f"Process pool unavailable for target={target} ({exc}). "
                f"Falling back to sequential."
            )

    logger.info(
        f"Final bootstrap sequential: target={target} iterations={FINAL_BOOTSTRAP_ITERATIONS}"
    )
    return [
        _run_one_iteration(
            model_name=model_name,
            best_params=best_params,
            x_train=x_train,
            y_train=y_train,
            x_test_scored=x_test_scored,
            y_test_scored=y_test_scored,
            x_test_all=x_test_all,
            iteration_idx=idx,
        )
        for idx in iteration_indices
    ]
