"""
config.py — Thesis-analysis-specific constants.

Does not re-define constants already in stages.post_analysis.config.
Those are imported directly where needed.
"""
from __future__ import annotations

SUBDIR_TUNING        = "tuning"
SUBDIR_SELECTION     = "selection"
SUBDIR_OVERFITTING   = "overfitting"
SUBDIR_GENERALIZATION = "generalization"
SUBDIR_BOOTSTRAP     = "bootstrap"
SUBDIR_SCENARIOS     = "scenarios"
SUBDIR_COVERAGE      = "coverage"
SUBDIR_SUMMARY       = "summary"

# Full registry of 13 models (source-of-truth: stages/training/config.py).
# Copied here to avoid importing training config (which has side effects).
MAX_TUNING_TRIALS_PER_MODEL: dict[str, int] = {
    "dummy_mean": 1,
    "ridge": 25,
    "elasticnet": 40,
    "xgboost": 50,
    "svr": 60,
    "random_forest": 40,
    "mlp": 40,
    "extra_trees": 40,
    "hist_gradient_boosting": 45,
    "adaboost": 35,
    "kernel_ridge": 35,
    "lightgbm": 50,
    "catboost": 45,
}

TUNING_BUDGET_SECONDS: float = 240.0

# Overfitting thresholds (mirrors training/config) — used for reference lines.
OUTER_GAP_ABS_THRESHOLD: float = 0.1
OVERFIT_INNER_GAP_RATIO_THRESHOLD: float = 0.50

# Param landscape: min successful trials required to produce a figure for a model.
PARAM_LANDSCAPE_MIN_TRIALS: int = 5

# How many top countries to show in CI-width bar chart.
BOOTSTRAP_CI_TOP_N: int = 20
