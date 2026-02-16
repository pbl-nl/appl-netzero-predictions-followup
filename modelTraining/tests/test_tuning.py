"""Tests for tuning utilities."""

from __future__ import annotations

import pandas as pd

from modelTraining.tuning import get_inner_split, sample_random_params, tune_hyperparams


def test_panel_inner_split_uses_expanding_window():
    X_context = pd.DataFrame(
        {
            "country": ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB"],
            "year": [2018, 2019, 2020, 2018, 2019, 2020],
        }
    )
    folds = get_inner_split(split="1990_2024", X_context=X_context, seed=42, n_inner_cv_folds=3)
    # 3 years, n_inner_cv_folds=3 → min(3, 3-1)=2 folds
    assert len(folds) == 2

    # First fold: train on 2018, validate on 2019
    train_idx0, valid_idx0 = folds[0]
    assert set(X_context.iloc[train_idx0]["year"].tolist()) == {2018}
    assert set(X_context.iloc[valid_idx0]["year"].tolist()) == {2019}

    # Second fold: train on 2018+2019, validate on 2020
    train_idx1, valid_idx1 = folds[1]
    assert set(X_context.iloc[train_idx1]["year"].tolist()) == {2018, 2019}
    assert set(X_context.iloc[valid_idx1]["year"].tolist()) == {2020}


def test_panel_inner_split_with_two_years_gives_one_fold():
    X_context = pd.DataFrame(
        {
            "country": ["AAA", "AAA", "BBB", "BBB"],
            "year": [2019, 2020, 2019, 2020],
        }
    )
    folds = get_inner_split(split="1990_2024", X_context=X_context, seed=42)
    assert len(folds) == 1
    train_idx, valid_idx = folds[0]
    assert set(X_context.iloc[train_idx]["year"].tolist()) == {2019}
    assert set(X_context.iloc[valid_idx]["year"].tolist()) == {2020}


def test_snapshot_inner_split_uses_kfold():
    X_context = pd.DataFrame(
        {
            "country": ["AAA", "BBB", "CCC", "DDD", "EEE"],
            "year": [2024, 2024, 2024, 2024, 2024],
        }
    )
    folds = get_inner_split(split="2024", X_context=X_context, seed=42)
    assert len(folds) == 5


def test_sample_random_params_returns_requested_count():
    grid = {"a": [1, 2, 3], "b": [10, 20]}
    sampled = sample_random_params(grid, n_samples=3, seed=123)
    assert len(sampled) == 3


def test_tune_hyperparams_returns_valid_grid_point():
    X_proc = pd.DataFrame({"x1": [0.0, 1.0, 2.0, 3.0], "x2": [1.0, 1.0, 2.0, 2.0]})
    y_proc = pd.Series([0.0, 1.0, 2.0, 3.0])
    X_context = pd.DataFrame(
        {
            "country": ["A", "A", "B", "B"],
            "year": [2019, 2020, 2019, 2020],
        }
    )

    grid = {"n_estimators": [10, 20], "max_depth": [2, None], "min_samples_leaf": [1, 2]}
    best = tune_hyperparams(
        algorithm="RandomForest",
        X_proc=X_proc,
        y_proc=y_proc,
        X_context=X_context,
        split="1990_2024",
        seed=42,
        param_grid=grid,
        model_threads=1,
    )

    assert best["n_estimators"] in grid["n_estimators"]
    assert best["max_depth"] in grid["max_depth"]
    assert best["min_samples_leaf"] in grid["min_samples_leaf"]
