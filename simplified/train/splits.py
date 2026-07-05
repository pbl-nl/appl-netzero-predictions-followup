"""Outer and inner cross-validation fold generators."""
import pandas as pd
from sklearn.model_selection import KFold

from .config import OUTER_TEST_YEARS, N_INNER_FOLDS


def outer_folds(panel):
    """
    Yield (outer_train, outer_test, test_year) for each outer test year.
    train = all rows with year < test_year
    test  = all rows with year == test_year
    """
    for test_year in OUTER_TEST_YEARS:
        outer_train = panel[panel["year"] < test_year].copy()
        outer_test = panel[panel["year"] == test_year].copy()
        yield outer_train, outer_test, test_year


def inner_folds_temporal(outer_train, outer_test_year):
    """
    Yield (inner_train, inner_val, val_year) for the N_INNER_FOLDS most recent
    years before outer_test_year, using an expanding-window scheme.
    """
    years = sorted(outer_train["year"].unique())
    # Take the last N_INNER_FOLDS years as validation years
    val_years = years[-N_INNER_FOLDS:]
    for val_year in val_years:
        inner_train = outer_train[outer_train["year"] < val_year].copy()
        inner_val = outer_train[outer_train["year"] == val_year].copy()
        yield inner_train, inner_val, val_year


def inner_folds_random(outer_train, seed=42):
    """
    Yield (inner_train, inner_val, fold_idx) using standard 5-fold KFold,
    ignoring temporal ordering.
    """
    kf = KFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=seed)
    idx = outer_train.index.tolist()
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(idx)):
        inner_train = outer_train.iloc[train_idx].copy()
        inner_val = outer_train.iloc[val_idx].copy()
        yield inner_train, inner_val, fold_idx
