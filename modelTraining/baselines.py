"""Baseline models for modelTraining."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import KFold

from modelPreparation.config.base import ModelPreparationConfig
from modelPreparation.pipeline import ModelPreparationPipeline
from modelTraining.config.base import ModelTrainingConfig
from modelTraining.utils import (
    apply_partition_masks,
    compute_regression_metrics,
    drop_missing_target,
    safe_log_target_with_shift,
    split_X_y,
)


@dataclass
class BaselineRunOutput:
    """Container for baseline metrics."""

    metrics: pd.DataFrame


def _panel_inner_split(X_train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
    years = sorted(X_train["year"].astype(int).unique().tolist())
    if len(years) < 2:
        return None
    valid_year = years[-1]
    train_idx = np.flatnonzero(X_train["year"].astype(int).to_numpy() < valid_year)
    valid_idx = np.flatnonzero(X_train["year"].astype(int).to_numpy() == valid_year)
    if train_idx.size == 0 or valid_idx.size == 0:
        return None
    return train_idx, valid_idx


def _tune_alpha(
    model_name: str,
    alpha_grid: list[float],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    split: str,
    scenario: str,
    random_seed: int,
) -> float:
    """Tune regularization alpha on training partition only."""
    y_name = scenario
    is_panel = split in {"1990_2024", "2010_2024"}

    if len(alpha_grid) == 1:
        return alpha_grid[0]

    candidate_scores: dict[float, list[float]] = {alpha: [] for alpha in alpha_grid}

    if is_panel:
        fold = _panel_inner_split(X_train)
        if fold is None:
            return alpha_grid[0]
        folds = [fold]
    else:
        n_splits = min(5, len(X_train))
        if n_splits < 2:
            return alpha_grid[0]
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        folds = list(kfold.split(X_train))

    for train_idx, valid_idx in folds:
        X_fold_train = X_train.iloc[train_idx].reset_index(drop=True)
        y_fold_train = y_train.iloc[train_idx].reset_index(drop=True)
        X_fold_valid = X_train.iloc[valid_idx].reset_index(drop=True)
        y_fold_valid = y_train.iloc[valid_idx].reset_index(drop=True)

        pipeline = ModelPreparationPipeline(ModelPreparationConfig(scaling_mode="standard"))
        X_fold_train_proc, y_fold_train_proc = pipeline.fit_transform(
            X_fold_train,
            y_fold_train.to_frame(name=y_name),
        )
        X_fold_valid_proc, _ = pipeline.transform(X_fold_valid)
        y_fold_valid_log, n_clipped = safe_log_target_with_shift(
            y=y_fold_valid,
            shift_constant=pipeline.target_shift_constants_[y_name],
        )
        if n_clipped > 0:
            logger.warning(
                "Baseline alpha tuning (%s/%s): %d fold-valid targets clipped to log(1e-12)",
                split, scenario, n_clipped,
            )

        for alpha in alpha_grid:
            if model_name == "ridge":
                model = Ridge(alpha=alpha)
            else:
                model = Lasso(alpha=alpha, max_iter=10_000)

            model.fit(X_fold_train_proc, y_fold_train_proc[y_name].to_numpy())
            pred = model.predict(X_fold_valid_proc)
            rmse = float(
                np.sqrt(
                    np.mean(
                        np.square(pred - y_fold_valid_log)
                    )
                )
            )
            candidate_scores[alpha].append(rmse)

    best_alpha = min(
        alpha_grid,
        key=lambda a: np.mean(candidate_scores[a]) if candidate_scores[a] else float("inf"),
    )
    return float(best_alpha)


def run_baselines(
    split_df: pd.DataFrame,
    partitions: dict[str, list[list[object]]],
    feature_columns: list[str],
    split: str,
    scenario: str,
    config: ModelTrainingConfig,
) -> BaselineRunOutput:
    """Run baseline models for one split/scenario."""
    masks = apply_partition_masks(split_df, partitions)

    X_train, y_train = split_X_y(split_df, feature_columns, scenario, masks["train"])
    X_valid, y_valid = split_X_y(split_df, feature_columns, scenario, masks["valid"])
    X_train, y_train = drop_missing_target(X_train, y_train)
    X_valid, y_valid = drop_missing_target(X_valid, y_valid)

    if len(X_train) == 0:
        raise ValueError(f"Split {split} has empty train partition")
    if len(X_valid) == 0:
        raise ValueError(f"Split {split} has empty valid partition")

    y_name = scenario
    rows: list[dict[str, object]] = []

    # Country mean and global mean share one fit/transform flow.
    mean_pipeline = ModelPreparationPipeline(ModelPreparationConfig(scaling_mode="none"))
    _, y_train_proc = mean_pipeline.fit_transform(
        X_train,
        y_train.to_frame(name=y_name),
    )
    _, _ = mean_pipeline.transform(
        X_valid,
    )
    y_valid_log, n_clipped = safe_log_target_with_shift(
        y=y_valid,
        shift_constant=mean_pipeline.target_shift_constants_[y_name],
    )
    if n_clipped > 0:
        logger.warning(
            "Baselines mean (%s/%s): %d valid targets clipped to log(1e-12)",
            split, scenario, n_clipped,
        )

    train_country = X_train["country"].astype(str).reset_index(drop=True)
    valid_country = X_valid["country"].astype(str).reset_index(drop=True)

    country_mean_map = (
        pd.DataFrame({"country": train_country, "y": y_train_proc[y_name].to_numpy()})
        .groupby("country")["y"]
        .mean()
        .to_dict()
    )
    global_mean = float(y_train_proc[y_name].mean())

    country_pred = valid_country.map(country_mean_map).fillna(global_mean).to_numpy()
    metrics = compute_regression_metrics(
        y_valid_log,
        country_pred,
        countries=valid_country,
    )
    rows.append({
        "Algorithm": "country_mean",
        "Split": split,
        "Scenario": scenario,
        **metrics,
    })

    global_pred = np.full(len(X_valid), global_mean, dtype=float)
    metrics = compute_regression_metrics(
        y_valid_log,
        global_pred,
        countries=valid_country,
    )
    rows.append({
        "Algorithm": "global_mean",
        "Split": split,
        "Scenario": scenario,
        **metrics,
    })

    for model_name in ["ridge", "lasso"]:
        best_alpha = _tune_alpha(
            model_name=model_name,
            alpha_grid=config.alpha_grid,
            X_train=X_train,
            y_train=y_train,
            split=split,
            scenario=scenario,
            random_seed=config.random_seed,
        )

        pipeline = ModelPreparationPipeline(ModelPreparationConfig(scaling_mode="standard"))
        X_train_proc, y_train_proc = pipeline.fit_transform(
            X_train,
            y_train.to_frame(name=y_name),
        )
        X_valid_proc, _ = pipeline.transform(X_valid)
        y_valid_log, n_clipped = safe_log_target_with_shift(
            y=y_valid,
            shift_constant=pipeline.target_shift_constants_[y_name],
        )
        if n_clipped > 0:
            logger.warning(
                "Baseline %s (%s/%s): %d valid targets clipped to log(1e-12)",
                model_name, split, scenario, n_clipped,
            )

        if model_name == "ridge":
            model = Ridge(alpha=best_alpha)
        else:
            model = Lasso(alpha=best_alpha, max_iter=10_000)

        model.fit(X_train_proc, y_train_proc[y_name].to_numpy())
        pred_valid = model.predict(X_valid_proc)

        metrics = compute_regression_metrics(
            y_valid_log,
            pred_valid,
            countries=valid_country,
        )
        rows.append({
            "Algorithm": model_name,
            "Split": split,
            "Scenario": scenario,
            "best_alpha": best_alpha,
            **metrics,
        })

    metrics_df = pd.DataFrame(rows)
    return BaselineRunOutput(metrics=metrics_df)
