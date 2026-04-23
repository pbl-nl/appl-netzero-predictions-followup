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
    COUNTRY_COL,
    RANDOM_STATE as PIPELINE_RANDOM_STATE,
    TEST_YEAR,
    YEAR_COL,
)

logger = logging.getLogger(__name__)


# --- Constants ---

DEFAULT_CPU_WORKERS = max(1, os.cpu_count())
FINAL_BOOTSTRAP_ITERATIONS = 200
FINAL_BOOTSTRAP_MAX_WORKERS = DEFAULT_CPU_WORKERS
TUNING_BUDGET_SECONDS = 240.0
MIN_TUNING_TRIALS = 3
SVR_MAX_ITER = 5000

# ACTIVE_MODELS = ["mlp", "hist_gradient_boosting", "kernel_ridge"]
ACTIVE_MODELS = ["catboost", "ridge", "adaboost", "lightgbm", "random_forest", "elasticnet", "xgboost", "svr", "mlp", "extra_trees", "hist_gradient_boosting", "kernel_ridge"] #
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

# Candidate selection rule
# For each (strategy, target) pair, rank all (branch, model_name) combos by mean
# outer_test_mae across outer folds. Take top CANDIDATE_PRIMARY_COUNT as primary.
# For any primary with candidate_all_folds_outer_gap_flagged=True, append the
# next-best non-flagged candidate as backup. Result: between 3 and 6 per pair.
# Note: outer_test_mae refers to the STRICT test set (drop_rows=True), which is
# the primary evaluation surface so train and test share the same completeness
# floor. The lenient variant (drop_rows="targets_only") is carried as the
# outer_test_{metric}_lenient / final_test_{metric}_lenient secondary audit.
CANDIDATE_PRIMARY_COUNT = 3
SELECTION_TOLERANCE = 0.01  # legacy knob, kept for reference tables
MODEL_COMPLEXITY_RANK: dict[str, int] = {
    "ridge": 1,
    "elasticnet": 2,
    "svr": 5,
    "kernel_ridge": 5,
    "adaboost": 6,
    "random_forest": 7,
    "extra_trees": 8,
    "hist_gradient_boosting": 10,
    "xgboost": 11,
    "lightgbm": 11,
    "catboost": 12,
    "mlp": 12,
}
MAX_LOSS = 1e12
MODEL_RANDOM_STATE = PIPELINE_RANDOM_STATE
MODEL_N_JOBS = 1

MAX_TUNING_TRIALS_PER_MODEL: dict[str, int] = {
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

# Penalization: penalized_loss = inner_val_mean_mae * (1 + gap_penalty)
# where gap_penalty = clamp(inner_train_val_gap_rel_mae, 0, GAP_RATIO_CAP) * GAP_PENALTY_WEIGHT
GAP_PENALTY_WEIGHT = 0.5
GAP_RATIO_CAP = 3.0

# Stabilizer for relative gap denominators. Used wherever a gap_rel metric is
# computed to avoid blowing up when the denominator MAE is very small.
REL_GAP_FLOOR = 0.05

# Inner train-vs-validation gap flag (audit-only): per-fold flag set when the
# relative MAE gap between inner training and inner validation exceeds the
# threshold. This signal does NOT gate candidate selection; it is reported for
# transparency. Tuning-time gap_penalty biases Hyperopt against pathological fits.
OVERFIT_INNER_GAP_RATIO_THRESHOLD = 0.50

# Outer generalization gap flag (primary audit): fires when the held-out outer
# test error both exceeds the inner validation error by an absolute margin AND
# the relative gap exceeds the relative threshold. Both must be true — absolute
# is primary (keeps us from flagging trivially small gaps), relative is the
# stabilizer.
OUTER_GAP_ABS_THRESHOLD = 0.1
OUTER_GAP_REL_THRESHOLD = 0.5

# Feature selection
N_FEATURES_TO_SELECT = 50

# --- Output paths ---

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

# Five core metrics tracked at every evaluation stage.
CORE_METRICS = ("mae", "mse", "rmse", "r2", "explained_variance")

FINAL_RESULTS_COLUMNS = [
    "target",
    "outer_strategy_source",
    "branch",
    "model_name",
    "candidate_rank",
    "candidate_role",
    "best_params_json",
    "final_train_rows",
    "final_test_rows",
    "final_test_rows_lenient",
    "feature_count",
    "bootstrap_iterations",
    # final_train_{metric} for each of the 5 metrics
    "final_train_mae", "final_train_mse", "final_train_rmse",
    "final_train_r2", "final_train_explained_variance",
    # final_test_{metric} (primary — strict drop_rows=True) for each of the 5 metrics
    "final_test_mae", "final_test_mse", "final_test_rmse",
    "final_test_r2", "final_test_explained_variance",
    # final_test_{metric}_lenient (secondary audit — drop_rows="targets_only")
    "final_test_mae_lenient", "final_test_mse_lenient", "final_test_rmse_lenient",
    "final_test_r2_lenient", "final_test_explained_variance_lenient",
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
    "candidate_rank",
    "candidate_role",
    "iteration",
    "prediction",
]
FINAL_PREDICTION_SUMMARY_COLUMNS = [
    "outer_strategy_source",
    "target",
    "branch",
    "model_name",
    "candidate_rank",
    "candidate_role",
    COUNTRY_COL,
    YEAR_COL,
    "actual",
    "prediction_mean",
    "prediction_std",
    "prediction_median",
    "prediction_p05",
    "prediction_p95",
]
