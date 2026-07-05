"""
Model registry, TPE hyperparameter search spaces, and training constants.
Search spaces are defined as callables to allow fresh hp objects per fmin call.
"""
import numpy as np
from hyperopt import hp
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# --- Training constants ---
OUTER_TEST_YEARS = [2018, 2019, 2020, 2021]
FINAL_HOLDOUT_YEAR = 2022
N_INNER_FOLDS = 5
BOOTSTRAP_ITERS = 200
TARGETS = ["target_linear", "target_optimistic", "target_pessimistic"]
TUNE_BUDGET_SECONDS = 240

# --- Model registry ---
# Each entry: model_name -> (class, fixed_params)
# Fixed params are NOT tuned; they are always applied.
MODELS = {
    "ridge": (
        Ridge,
        {"fit_intercept": True},
    ),
    "elasticnet": (
        ElasticNet,
        {"max_iter": 50000, "random_state": 42},
    ),
    "kernel_ridge": (
        KernelRidge,
        {},
    ),
    "svr": (
        SVR,
        {"max_iter": 10_000_000},
    ),
    "adaboost": (
        AdaBoostRegressor,
        {"estimator": DecisionTreeRegressor(), "random_state": 42},
    ),
    "random_forest": (
        RandomForestRegressor,
        {"random_state": 42, "n_jobs": -1},
    ),
    "extra_trees": (
        ExtraTreesRegressor,
        {"random_state": 42, "n_jobs": -1},
    ),
    "hist_gradient_boosting": (
        HistGradientBoostingRegressor,
        {"early_stopping": False, "random_state": 42},
    ),
    "xgboost": (
        XGBRegressor,
        {"objective": "reg:squarederror", "tree_method": "hist", "seed": 42, "verbosity": 0},
    ),
    "lightgbm": (
        LGBMRegressor,
        {"objective": "regression_l1", "random_state": 42, "verbosity": -1},
    ),
    "catboost": (
        CatBoostRegressor,
        {"loss_function": "MAE", "random_seed": 42, "verbose": 0, "allow_writing_files": False},
    ),
    "mlp": (
        MLPRegressor,
        {
            "activation": "tanh",
            "early_stopping": True,
            "n_iter_no_change": 20,
            "max_iter": 2000,
            "random_state": 42,
        },
    ),
    "dummy_mean": (
        DummyRegressor,
        {"strategy": "mean"},
    ),
}

# Complexity rank (lower = simpler). Used as tiebreaker in winner selection.
COMPLEXITY_RANK = {
    "ridge": 1,
    "elasticnet": 2,
    "kernel_ridge": 3,
    "svr": 4,
    "adaboost": 5,
    "random_forest": 6,
    "extra_trees": 7,
    "hist_gradient_boosting": 8,
    "xgboost": 9,
    "lightgbm": 10,
    "catboost": 11,
    "mlp": 12,
    "dummy_mean": 13,
}

# Maximum number of TPE trials per model
MAX_TRIALS = {
    "ridge": 25,
    "elasticnet": 40,
    "kernel_ridge": 35,
    "svr": 60,
    "adaboost": 35,
    "random_forest": 40,
    "extra_trees": 40,
    "hist_gradient_boosting": 45,
    "xgboost": 50,
    "lightgbm": 50,
    "catboost": 45,
    "mlp": 40,
    "dummy_mean": 1,
}

# --- Hyperparameter search spaces (callables, returns fresh hp objects) ---
_MLP_ARCHITECTURES = [(32,), (64,), (128,), (8, 16), (16, 8), (32, 16), (64, 32)]


def _space_ridge():
    return {"alpha": hp.loguniform("alpha", np.log(1e-5), np.log(1e4))}


def _space_elasticnet():
    return {
        "alpha": hp.loguniform("alpha", np.log(1e-5), np.log(1e2)),
        "l1_ratio": hp.uniform("l1_ratio", 0.0, 1.0),
    }


def _space_kernel_ridge():
    return {
        "alpha": hp.loguniform("alpha", np.log(1.0), np.log(1e3)),
        "kernel": hp.choice("kernel", ["rbf", "laplacian"]),
        "gamma": hp.loguniform("gamma", np.log(0.001), np.log(0.1)),
    }


def _space_svr():
    return {
        "kernel": hp.choice("kernel", ["rbf"]),
        "C": hp.loguniform("C", np.log(0.01), np.log(50.0)),
        "epsilon": hp.loguniform("epsilon", np.log(0.005), np.log(1.0)),
        "gamma": hp.loguniform("gamma", np.log(1e-4), np.log(0.25)),
    }


def _space_adaboost():
    return {
        "n_estimators": hp.quniform("n_estimators", 25, 400, 5),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.005), np.log(2.0)),
        "loss": hp.choice("loss", ["square", "exponential"]),
        "estimator__max_depth": hp.quniform("estimator__max_depth", 1, 3, 1),
    }


def _space_random_forest():
    return {
        "n_estimators": hp.quniform("n_estimators", 100, 500, 25),
        "max_depth": hp.choice("max_depth", [5, 10, 12, 15]),
        "min_samples_split": hp.quniform("min_samples_split", 10, 100, 2),
        "min_samples_leaf": hp.quniform("min_samples_leaf", 15, 60, 1),
        "max_features": hp.choice("max_features", [0.5, 0.7]),
    }


def _space_extra_trees():
    return {
        "n_estimators": hp.quniform("n_estimators", 100, 500, 25),
        "max_depth": hp.choice("max_depth", [5, 10, 12, 15]),
        "min_samples_split": hp.quniform("min_samples_split", 10, 100, 2),
        "min_samples_leaf": hp.quniform("min_samples_leaf", 10, 60, 1),
        "max_features": hp.choice("max_features", [0.5, 0.7]),
    }


def _space_hist_gradient_boosting():
    return {
        "max_iter": hp.quniform("max_iter", 100, 400, 10),
        "max_depth": hp.choice("max_depth", [None, 4, 6]),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.005), np.log(0.15)),
        "min_samples_leaf": hp.quniform("min_samples_leaf", 20, 80, 5),
        "max_leaf_nodes": hp.quniform("max_leaf_nodes", 15, 50, 1),
        "l2_regularization": hp.loguniform("l2_regularization", np.log(0.1), np.log(100.0)),
    }


def _space_xgboost():
    return {
        "n_estimators": hp.quniform("n_estimators", 50, 500, 10),
        "max_depth": hp.quniform("max_depth", 2, 5, 1),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.005), np.log(0.3)),
        "subsample": hp.uniform("subsample", 0.5, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 0.9),
        "reg_lambda": hp.loguniform("reg_lambda", np.log(0.01), np.log(100.0)),
        "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-4), np.log(50.0)),
        "min_child_weight": hp.quniform("min_child_weight", 5, 50, 1),
        "gamma": hp.uniform("gamma", 0.0, 1.0),
    }


def _space_lightgbm():
    return {
        "boosting_type": hp.choice("boosting_type", ["gbdt", "dart"]),
        "n_estimators": hp.quniform("n_estimators", 50, 400, 10),
        "max_depth": hp.quniform("max_depth", 2, 6, 1),
        "num_leaves": hp.quniform("num_leaves", 8, 32, 4),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.005), np.log(0.2)),
        "min_child_samples": hp.quniform("min_child_samples", 20, 80, 5),
        "subsample": hp.uniform("subsample", 0.5, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.4, 1.0),
        "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-3), np.log(50.0)),
        "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-3), np.log(100.0)),
    }


def _space_catboost():
    return {
        "iterations": hp.quniform("iterations", 150, 400, 25),
        "depth": hp.quniform("depth", 5, 7, 1),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.005), np.log(0.3)),
        "l2_leaf_reg": hp.loguniform("l2_leaf_reg", np.log(0.01), np.log(50.0)),
        "random_strength": hp.uniform("random_strength", 0.01, 50.0),
        "bagging_temperature": hp.uniform("bagging_temperature", 0.0, 1.0),
    }


def _space_mlp():
    return {
        "hidden_layer_sizes": hp.choice("hidden_layer_sizes", _MLP_ARCHITECTURES),
        "alpha": hp.loguniform("alpha", np.log(1e-3), np.log(1e-1)),
        "learning_rate_init": hp.loguniform("learning_rate_init", np.log(1e-4), np.log(1e-1)),
        "batch_size": hp.choice("batch_size", [128, 256]),
    }


def _space_dummy():
    return {}


SEARCH_SPACES = {
    "ridge": _space_ridge,
    "elasticnet": _space_elasticnet,
    "kernel_ridge": _space_kernel_ridge,
    "svr": _space_svr,
    "adaboost": _space_adaboost,
    "random_forest": _space_random_forest,
    "extra_trees": _space_extra_trees,
    "hist_gradient_boosting": _space_hist_gradient_boosting,
    "xgboost": _space_xgboost,
    "lightgbm": _space_lightgbm,
    "catboost": _space_catboost,
    "mlp": _space_mlp,
    "dummy_mean": _space_dummy,
}
