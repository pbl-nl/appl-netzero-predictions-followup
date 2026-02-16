"""Hyperparameter tuning utilities for modelTraining."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from modelTraining.core.algorithms import build_model
from modelTraining.utils import iter_grid


def get_inner_split(
    split: str,
    X_context: pd.DataFrame,
    seed: int,
    n_inner_cv_folds: int = 3,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build inner-CV splits based on split type.

    Panel splits use expanding-window temporal CV (last *n_inner_cv_folds*
    years as successive validation sets). Snapshot uses KFold.
    """
    if split in {"1990_2024", "2010_2024"}:
        years = sorted(X_context["year"].astype(int).unique().tolist())
        if len(years) < 2:
            return []

        year_arr = X_context["year"].astype(int).to_numpy()

        # Build expanding-window folds: for the last n_inner_cv_folds years,
        # train on all years before the validation year, validate on that year.
        n_folds = min(n_inner_cv_folds, len(years) - 1)
        folds: list[tuple[np.ndarray, np.ndarray]] = []
        for valid_year in years[-n_folds:]:
            train_idx = np.flatnonzero(year_arr < valid_year)
            valid_idx = np.flatnonzero(year_arr == valid_year)
            if train_idx.size == 0 or valid_idx.size == 0:
                continue
            folds.append((train_idx, valid_idx))

        return folds

    n_splits = min(5, len(X_context))
    if n_splits < 2:
        return []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(X_context))


def tune_hyperparams(
    algorithm: str,
    X_proc: pd.DataFrame,
    y_proc: pd.Series,
    X_context: pd.DataFrame,
    split: str,
    seed: int,
    param_grid: dict[str, list[Any]],
    model_threads: int = 1,
    n_inner_cv_folds: int = 3,
) -> dict[str, Any]:
    """Exhaustive grid search for classical ML algorithms."""
    param_candidates = iter_grid(param_grid)
    if not param_candidates:
        return {}

    folds = get_inner_split(split=split, X_context=X_context, seed=seed, n_inner_cv_folds=n_inner_cv_folds)
    if not folds:
        return param_candidates[0]

    best_params = param_candidates[0]
    best_score = float("inf")

    y_values = y_proc.to_numpy(dtype=float)

    for params in param_candidates:
        scores: list[float] = []

        for train_idx, valid_idx in folds:
            X_train = X_proc.iloc[train_idx]
            X_valid = X_proc.iloc[valid_idx]
            y_train = y_values[train_idx]
            y_valid = y_values[valid_idx]

            model_params = dict(params)
            if algorithm in {"RandomForest", "XGBoost"}:
                model_params.setdefault("n_jobs", model_threads)

            model = build_model(algorithm, model_params, seed=seed)

            if algorithm == "XGBoost":
                fit_kwargs: dict[str, Any] = {}
                # Keep early stopping best-effort; older versions may not support all args.
                fit_kwargs["eval_set"] = [(X_valid, y_valid)]
                fit_kwargs["verbose"] = False
                fit_kwargs["early_stopping_rounds"] = 20
                try:
                    model.fit(X_train, y_train, **fit_kwargs)
                except TypeError:
                    model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)

            pred = model.predict(X_valid)
            rmse = float(np.sqrt(np.mean(np.square(pred - y_valid))))
            scores.append(rmse)

        mean_score = float(np.mean(scores)) if scores else float("inf")
        if mean_score < best_score:
            best_score = mean_score
            best_params = params

    return dict(best_params)


def sample_random_params(
    param_grid: dict[str, list[Any]],
    n_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Sample random hyperparameter configs without replacement when possible."""
    all_candidates = iter_grid(param_grid)
    if not all_candidates:
        return [{}]

    if n_samples >= len(all_candidates):
        return all_candidates

    rng = np.random.default_rng(seed)
    idx = rng.choice(np.arange(len(all_candidates)), size=n_samples, replace=False)
    return [all_candidates[int(i)] for i in idx]


def tune_neural_random_search(
    evaluate_fn: Callable[[dict[str, Any]], float],
    param_grid: dict[str, list[Any]],
    n_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Random-search wrapper for neural models.

    The callback should train/evaluate one config and return validation RMSE.
    """
    candidates = sample_random_params(param_grid, n_samples=n_samples, seed=seed)

    best_params = candidates[0]
    best_score = float("inf")

    for params in candidates:
        score = float(evaluate_fn(params))
        if score < best_score:
            best_score = score
            best_params = params

    return dict(best_params)
