"""Outer cross-validation for algorithm comparison (opt-in via --outer-cv)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from modelPreparation.config.base import ModelPreparationConfig
from modelPreparation.pipeline import ModelPreparationPipeline
from modelTraining.config.base import ModelTrainingConfig, PANEL_SPLITS
from modelTraining.core.algorithms import build_model
from modelTraining.tuning import tune_hyperparams
from modelTraining.utils import (
    apply_partition_masks,
    compute_regression_metrics,
    drop_missing_target,
    safe_log_target_with_shift,
    split_X_y,
)

logger = logging.getLogger(__name__)


def _panel_outer_folds(
    split_df: pd.DataFrame,
    partitions: dict[str, list[list[object]]],
    n_folds: int,
) -> list[tuple[pd.Series, pd.Series]]:
    """Build expanding-window outer folds for panel splits.

    Returns list of (train_mask, valid_mask) pairs over the combined
    train+valid partition.
    """
    masks = apply_partition_masks(split_df, partitions)
    observed_mask = masks["train"] | masks["valid"]
    observed_df = split_df.loc[observed_mask]

    years = sorted(observed_df["year"].astype(int).unique().tolist())
    if len(years) < 2:
        return []

    n_folds = min(n_folds, len(years) - 1)
    folds: list[tuple[pd.Series, pd.Series]] = []
    for valid_year in years[-n_folds:]:
        train_mask = observed_mask & (split_df["year"].astype(int) < valid_year)
        valid_mask = observed_mask & (split_df["year"].astype(int) == valid_year)
        if train_mask.sum() == 0 or valid_mask.sum() == 0:
            continue
        folds.append((train_mask, valid_mask))

    return folds


def _snapshot_outer_folds(
    split_df: pd.DataFrame,
    partitions: dict[str, list[list[object]]],
    n_folds: int,
    base_seed: int,
) -> list[tuple[pd.Series, pd.Series]]:
    """Build repeated random country-split folds for snapshot splits."""
    masks = apply_partition_masks(split_df, partitions)
    observed_mask = masks["train"] | masks["valid"]
    observed_countries = split_df.loc[observed_mask, "country"].unique().tolist()

    if len(observed_countries) < 3:
        return []

    folds: list[tuple[pd.Series, pd.Series]] = []
    for fold_id in range(n_folds):
        rng = np.random.default_rng(base_seed + fold_id)
        rng.shuffle(observed_countries)
        split_point = max(1, int(0.7 * len(observed_countries)))
        train_countries = set(observed_countries[:split_point])

        train_mask = observed_mask & split_df["country"].isin(train_countries)
        valid_mask = observed_mask & ~split_df["country"].isin(train_countries)

        if train_mask.sum() == 0 or valid_mask.sum() == 0:
            continue
        folds.append((train_mask, valid_mask))

    return folds


def run_outer_cv_comparison(
    split: str,
    scenario: str,
    split_df: pd.DataFrame,
    partitions: dict[str, list[list[object]]],
    feature_columns: list[str],
    config: ModelTrainingConfig,
) -> pd.DataFrame:
    """Run outer CV comparison across algorithms for one split/scenario.

    Returns a DataFrame with columns:
        Algorithm, Split, Scenario, fold, RMSE_log, MAE_log, R2
    """
    is_panel = split in PANEL_SPLITS

    if is_panel:
        folds = _panel_outer_folds(split_df, partitions, config.n_outer_cv_folds)
    else:
        folds = _snapshot_outer_folds(
            split_df, partitions, config.n_outer_cv_folds, config.random_seed,
        )

    if not folds:
        logger.warning("No outer CV folds for split=%s scenario=%s", split, scenario)
        return pd.DataFrame()

    # Only run classical algorithms in outer CV (skip neural for speed).
    cv_algorithms = [
        a for a in config.algorithms
        if a in {"LinearRegression", "RandomForest", "XGBoost", "SVM"}
    ]
    if not cv_algorithms:
        logger.warning("No classical algorithms selected for outer CV")
        return pd.DataFrame()

    rows: list[dict[str, object]] = []

    for fold_idx, (train_mask, valid_mask) in enumerate(folds):
        X_train, y_train = split_X_y(split_df, feature_columns, scenario, train_mask)
        X_valid, y_valid = split_X_y(split_df, feature_columns, scenario, valid_mask)
        X_train, y_train = drop_missing_target(X_train, y_train)
        X_valid, y_valid = drop_missing_target(X_valid, y_valid)

        if len(X_train) == 0 or len(X_valid) == 0:
            continue

        for algorithm in cv_algorithms:
            if split == "2024" and algorithm == "LSTM":
                continue

            try:
                scaling_mode = config.scaling_mode_by_algorithm[algorithm]
                pipeline = ModelPreparationPipeline(
                    ModelPreparationConfig(scaling_mode=scaling_mode)
                )
                X_train_proc, y_train_proc = pipeline.fit_transform(
                    X_train, y_train.to_frame(name=scenario),
                )
                X_valid_proc, _ = pipeline.transform(X_valid)

                y_train_arr = y_train_proc[scenario].to_numpy(dtype=float)
                y_valid_arr, _ = safe_log_target_with_shift(
                    y=y_valid,
                    shift_constant=pipeline.target_shift_constants_[scenario],
                )

                best_params = tune_hyperparams(
                    algorithm=algorithm,
                    X_proc=X_train_proc,
                    y_proc=y_train_proc[scenario],
                    X_context=X_train,
                    split=split,
                    seed=config.random_seed + fold_idx,
                    param_grid=config.param_grids.get(algorithm, {}),
                    model_threads=config.model_threads,
                    n_inner_cv_folds=config.n_inner_cv_folds,
                )
                if algorithm in {"RandomForest", "XGBoost"}:
                    best_params.setdefault("n_jobs", config.model_threads)

                model = build_model(algorithm, best_params, seed=config.random_seed + fold_idx)
                model.fit(X_train_proc, y_train_arr)
                pred_valid_log = np.asarray(model.predict(X_valid_proc), dtype=float)

                metrics = compute_regression_metrics(
                    y_true_log=y_valid_arr,
                    y_pred_log=pred_valid_log,
                    countries=X_valid["country"],
                )

                rows.append({
                    "Algorithm": algorithm,
                    "Split": split,
                    "Scenario": scenario,
                    "fold": fold_idx,
                    **metrics,
                })

            except Exception as exc:
                logger.warning(
                    "Outer CV fold %d failed for %s/%s/%s: %s",
                    fold_idx, algorithm, split, scenario, exc,
                )

    return pd.DataFrame(rows)
