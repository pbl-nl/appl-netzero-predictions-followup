from __future__ import annotations

import datetime
import logging
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from stages.preprocessing import (
    COUNTRY_COL, ID_COLS, RANDOM_STATE as PIPELINE_RANDOM_STATE,
    TARGET_COLS, TEST_YEAR, YEAR_COL, Preprocessor,
    load_panel_data, split_raw_panel,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_CPU_WORKERS = max(1, (os.cpu_count() or 2) - 1)
FINAL_BOOTSTRAP_ITERATIONS = 200
FINAL_BOOTSTRAP_MAX_WORKERS = DEFAULT_CPU_WORKERS
TUNING_BUDGET_SECONDS = 240.0
MIN_TUNING_TRIALS = 3
SVR_MAX_ITER = 5000

ACTIVE_MODELS = ["ridge", "elasticnet", "xgboost", "svr", "random_forest", "mlp"]
# ACTIVE_MODELS = ["huber", "bayesian_ridge", "extra_trees", "gradient_boosting", "hist_gradient_boosting", "adaboost"]
ACTIVE_TARGETS = ["linear", "optimistic", "pessimistic"]
ACTIVE_BRANCHES = ["simple", "mice"]
ACTIVE_OUTER_STRATEGIES = ["temporal_cv", "random_cv"]

# Nested temporal CV: outer test years for model selection (2022 is final holdout)
OUTER_TEST_YEARS = [2018, 2019, 2020, 2021]

# Inner expanding-window temporal CV
N_TEMPORAL_CV_FOLDS = 5
MIN_VALID_FOLDS = 2

# Training year range: only use data from this year onwards (applied after feature engineering).
# Set to None to use all available years.
TRAIN_START_YEAR: int | None = None

# Maximum allowed TRAIN_START_YEAR — guarantees enough years for N_TEMPORAL_CV_FOLDS
# expanding-window folds in the most restrictive outer test year.
# Formula: min(OUTER_TEST_YEARS) - N_TEMPORAL_CV_FOLDS - 1
# NEVER CHANGE THIS VALUE — it is derived from OUTER_TEST_YEARS and N_TEMPORAL_CV_FOLDS
MAX_TRAIN_START_YEAR = min(OUTER_TEST_YEARS) - N_TEMPORAL_CV_FOLDS - 1  # = 2012

# Overfitting-control selection rule for winner selection
# Models within SELECTION_TOLERANCE of the best mean outer test MAE are treated as
# equivalent; among them the simplest model is preferred.
SELECTION_TOLERANCE = 0.01  # 1% relative tolerance
MODEL_COMPLEXITY_RANK: dict[str, int] = {
    "ridge": 1,
    "elasticnet": 2,
    "bayesian_ridge": 3,
    "huber": 4,
    "svr": 5,
    "adaboost": 6,
    "random_forest": 7,
    "extra_trees": 8,
    "gradient_boosting": 9,
    "hist_gradient_boosting": 10,
    "xgboost": 11,
    "mlp": 12,
}
MAX_LOSS = 1e12
MODEL_RANDOM_STATE = PIPELINE_RANDOM_STATE
MODEL_N_JOBS = 1

MAX_TUNING_TRIALS_PER_MODEL: dict[str, int] = {
    "ridge": 20,
    "elasticnet": 30,
    "xgboost": 40,
    "svr": 20,
    "random_forest": 30,
    "mlp": 30,
    "huber": 20,
    "bayesian_ridge": 15,
    "extra_trees": 30,
    "gradient_boosting": 35,
    "hist_gradient_boosting": 35,
    "adaboost": 25,
}

# Penalization: penalized_loss = cv_mean_mae * (1 + gap_penalty)
# where gap_penalty = clamp(cv_mae_gap_ratio, 0, GAP_RATIO_CAP) * GAP_PENALTY_WEIGHT
GAP_PENALTY_WEIGHT = 0.5
GAP_RATIO_CAP = 3.0

# Overfitting: exclude if cv_mae_gap_ratio >= threshold
OVERFIT_GAP_RATIO_THRESHOLD = 0.50

# Feature selection
N_FEATURES_TO_SELECT = 30

# =============================================================================
# Output paths
# =============================================================================

def _build_run_tag() -> str:
    tag = os.environ.get("RUN_TAG", "").strip()
    if tag:
        return tag
    tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.environ["RUN_TAG"] = tag  # freeze for spawn-ed worker processes
    return tag


RUN_TAG = _build_run_tag()
OUTPUT_DIR = Path(f"results/training_{RUN_TAG}")
TRIALS_PATH = OUTPUT_DIR / "trials.csv"
OUTER_RESULTS_PATH = OUTPUT_DIR / "outer_results.csv"
WINNERS_PATH = OUTPUT_DIR / "winners.csv"
FINAL_RESULTS_PATH = OUTPUT_DIR / "final_results.csv"
FINAL_BOOTSTRAP_PREDICTIONS_PATH = OUTPUT_DIR / "final_bootstrap_predictions.csv"
FINAL_PREDICTION_SUMMARY_PATH = OUTPUT_DIR / "final_prediction_summary.csv"
OVERFITTING_AUDIT_PATH = OUTPUT_DIR / "overfitting_audit.csv"
OUTER_ARTIFACTS_DIR = OUTPUT_DIR / "artifacts" / "outer_winners"
FINAL_ARTIFACTS_DIR = OUTPUT_DIR / "artifacts" / "final"

FINAL_RESULTS_COLUMNS = [
    "target",
    "outer_strategy_source",
    "branch",
    "model_name",
    "best_params_json",
    "final_train_rows",
    "final_test_rows",
    "feature_count",
    "bootstrap_iterations",
    "final_train_mae",
    "final_train_rmse",
    "final_train_r2",
    "test_mae",
    "test_rmse",
    "test_r2",
    "artifact_path",
]
FINAL_PREDICTIONS_COLUMNS = [
    COUNTRY_COL,
    YEAR_COL,
    "actual",
    "outer_strategy_source",
    "target",
    "branch",
    "model_name",
    "iteration",
    "prediction",
]
FINAL_PREDICTION_SUMMARY_COLUMNS = [
    "outer_strategy_source",
    "target",
    "branch",
    "model_name",
    COUNTRY_COL,
    YEAR_COL,
    "actual",
    "prediction_mean",
    "prediction_std",
    "prediction_median",
    "prediction_p05",
    "prediction_p95",
]
