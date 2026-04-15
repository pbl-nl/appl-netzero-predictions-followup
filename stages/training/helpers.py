from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from hyperopt import hp
from hyperopt.pyll.base import scope
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from stages.training.config import (
    CORE_METRICS,
    FINAL_PREDICTION_SUMMARY_COLUMNS,
    FINAL_PREDICTIONS_COLUMNS,
    FINAL_RESULTS_COLUMNS,
    GAP_PENALTY_WEIGHT,
    GAP_RATIO_CAP,
    MODEL_N_JOBS,
    OUTER_RESULTS_PATH,
    OUTPUT_DIR,
    OUTER_ARTIFACTS_DIR,
    FINAL_ARTIFACTS_DIR,
    REL_GAP_FLOOR,
    SVR_MAX_ITER,
    TRIALS_PATH,
)


# --- Output directories ---


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTER_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# --- Serialization helpers ---


def to_native(value: Any, json_compat: bool = False) -> Any:
    if isinstance(value, dict):
        return {key: to_native(inner, json_compat) for key, inner in value.items()}
    if isinstance(value, tuple):
        items = [to_native(inner, json_compat) for inner in value]
        return items if json_compat else tuple(items)
    if isinstance(value, list):
        return [to_native(inner, json_compat) for inner in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def serialize_params(params: dict[str, Any]) -> str:
    return json.dumps(to_native(params, json_compat=True), sort_keys=True)


# --- Metrics ---


def evaluate_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the five core metrics: mae, mse, rmse, r2, explained_variance."""
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    if len(y_true) > 1:
        r2 = float(r2_score(y_true, y_pred))
        ev = float(explained_variance_score(y_true, y_pred))
    else:
        r2 = float("nan")
        ev = float("nan")
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "explained_variance": ev,
    }


def _relative_gap_floored(numerator: float, denominator: float, floor: float = REL_GAP_FLOOR) -> float:
    """Relative gap with a denominator floor to stabilize small-magnitude ratios."""
    return float(numerator / max(abs(denominator), floor))


def summarize_cv_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate inner-CV fold metrics into inner_train/inner_val mean/std plus
    the audit-only inner train-vs-validation gap.

    Each fold_metrics entry has keys: fold_id, train_{metric}, val_{metric} for
    every metric in CORE_METRICS.
    """
    df = pd.DataFrame(fold_metrics)
    train_mae = float(df["train_mae"].mean())
    val_mae = float(df["val_mae"].mean())
    positive_gap_share = float((df["val_mae"] > df["train_mae"]).mean())

    summary: dict[str, float] = {
        "inner_positive_gap_fold_share": positive_gap_share,
        "folds_used": int(len(fold_metrics)),
    }
    for metric in CORE_METRICS:
        train_col = f"train_{metric}"
        val_col = f"val_{metric}"
        if train_col not in df.columns or val_col not in df.columns:
            continue
        train_mean = float(df[train_col].mean())
        val_mean = float(df[val_col].mean())
        val_std = float(df[val_col].std(ddof=0))
        summary[f"inner_train_mean_{metric}"] = train_mean
        summary[f"inner_val_mean_{metric}"] = val_mean
        summary[f"inner_val_std_{metric}"] = val_std
        summary[f"inner_train_val_gap_abs_{metric}"] = float(val_mean - train_mean)
        summary[f"inner_train_val_gap_rel_{metric}"] = _relative_gap_floored(
            val_mean - train_mean, train_mean
        )
    return summary


# --- Penalized objective ---


def compute_penalized_cv_objective(cv_summary: dict[str, Any]) -> dict[str, float]:
    gap_ratio = max(0.0, cv_summary.get("inner_train_val_gap_rel_mae", 0.0))
    gap_penalty = min(gap_ratio, GAP_RATIO_CAP) * GAP_PENALTY_WEIGHT
    penalized_loss = cv_summary["inner_val_mean_mae"] * (1.0 + gap_penalty)
    return {
        "penalized_cv_loss": penalized_loss,
        "gap_penalty": gap_penalty,
    }


# --- Model building / fitting / evaluation ---


def fit_predict_evaluate(
    model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_eval: pd.DataFrame,
    y_eval: pd.Series,
) -> tuple[dict[str, float], np.ndarray, Any]:
    model.fit(x_train, y_train)
    predictions = model.predict(x_eval)
    metrics = evaluate_metrics(y_eval, predictions)
    return metrics, predictions, model


def fit_model(model: Any, x_train: pd.DataFrame, y_train: pd.Series) -> Any:
    model.fit(x_train, y_train)
    return model


def predict_evaluate(
    model: Any,
    x_eval: pd.DataFrame,
    y_eval: pd.Series,
) -> tuple[dict[str, float], np.ndarray]:
    predictions = model.predict(x_eval)
    metrics = evaluate_metrics(y_eval, predictions)
    return metrics, predictions


# --- Partial output persistence ---


def persist_partial_outputs(
    all_trial_rows: list[dict[str, Any]],
    all_outer_result_rows: list[dict[str, Any]],
) -> None:
    if all_trial_rows:
        pd.DataFrame(all_trial_rows).to_csv(TRIALS_PATH, index=False)
    if all_outer_result_rows:
        pd.DataFrame(all_outer_result_rows).to_csv(OUTER_RESULTS_PATH, index=False)


# --- Empty DataFrame constructors ---


def empty_final_results_df() -> pd.DataFrame:
    return pd.DataFrame(columns=FINAL_RESULTS_COLUMNS)


def empty_final_predictions_df() -> pd.DataFrame:
    return pd.DataFrame(columns=FINAL_PREDICTIONS_COLUMNS)


def empty_final_prediction_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=FINAL_PREDICTION_SUMMARY_COLUMNS)


# --- Hyperopt search spaces and model construction ---


def get_hyperopt_space(model_name: str) -> dict[str, Any]:
    if model_name == "ridge":
        return {"alpha": hp.loguniform("alpha", np.log(1e-5), np.log(1e3))}
    if model_name == "elasticnet":
        return {
            "alpha": hp.loguniform("alpha", np.log(1e-5), np.log(1e2)),
            "l1_ratio": hp.uniform("l1_ratio", 0.0, 1.0),
        }
    if model_name == "xgboost":
        return {
            "n_estimators": scope.int(hp.quniform("n_estimators", 50, 500, 10)),
            "max_depth": scope.int(hp.quniform("max_depth", 2, 10, 1)),
            "learning_rate": hp.loguniform("learning_rate", np.log(0.005), np.log(0.3)),
            "subsample": hp.uniform("subsample", 0.5, 1.0),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.4, 1.0),
            "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-4), np.log(100.0)),
            "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-4), np.log(50.0)),
            "min_child_weight": hp.quniform("min_child_weight", 1.0, 50.0, 1.0),
            "gamma": hp.uniform("gamma", 0.0, 10.0),
        }
    if model_name == "svr":
        return {
            "kernel": hp.choice("kernel", ["linear", "rbf", "poly", "sigmoid"]),
            "C": hp.loguniform("C", np.log(1e-3), np.log(1e3)),
            "epsilon": hp.loguniform("epsilon", np.log(1e-4), np.log(1.0)),
            "gamma": hp.loguniform("gamma", np.log(1e-6), np.log(10.0)),
        }
    if model_name == "random_forest":
        return {
            "n_estimators": scope.int(hp.quniform("n_estimators", 100, 500, 25)),
            "max_depth": hp.choice("max_depth", [None, 5, 10, 15, 20]),
            "min_samples_split": scope.int(hp.quniform("min_samples_split", 2, 100, 2)),
            "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 1, 60, 1)),
            "max_features": hp.choice("max_features", [0.2, 0.3, 0.5, 0.7, "sqrt", "log2"]),
        }
    if model_name == "mlp":
        return {
            "hidden_layer_sizes": hp.choice(
                "hidden_layer_sizes",
                [(16,), (32,), (64,), (128,),
                 (32, 16), (64, 32), (100, 100), (128, 64),
                 (64, 64, 64), (128, 64, 32)],
            ),
            "activation": hp.choice("activation", ["relu", "tanh"]),
            "alpha": hp.loguniform("alpha", np.log(1e-6), np.log(1e-1)),
            "learning_rate_init": hp.loguniform(
                "learning_rate_init", np.log(1e-5), np.log(1e-1),
            ),
            "batch_size": hp.choice("batch_size", [32, 64, 128, 256]),
        }
    if model_name == "extra_trees":
        return {
            "n_estimators": scope.int(hp.quniform("n_estimators", 100, 500, 25)),
            "max_depth": hp.choice("max_depth", [None, 5, 10, 15, 20]),
            "min_samples_split": scope.int(hp.quniform("min_samples_split", 2, 100, 2)),
            "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 1, 60, 1)),
            "max_features": hp.choice("max_features", [0.2, 0.3, 0.5, 0.7, "sqrt", "log2"]),
        }
    if model_name == "hist_gradient_boosting":
        return {
            "max_iter": scope.int(hp.quniform("max_iter", 100, 500, 10)),
            "max_depth": hp.choice("max_depth", [None, 4, 6, 8, 10]),
            "learning_rate": hp.loguniform("learning_rate", np.log(0.005), np.log(0.3)),
            "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 5, 80, 5)),
            "max_leaf_nodes": scope.int(hp.quniform("max_leaf_nodes", 8, 127, 1)),
            "l2_regularization": hp.loguniform(
                "l2_regularization", np.log(1e-3), np.log(100.0),
            ),
        }
    if model_name == "adaboost":
        return {
            "n_estimators": scope.int(hp.quniform("n_estimators", 25, 400, 5)),
            "learning_rate": hp.loguniform("learning_rate", np.log(0.005), np.log(2.0)),
            "loss": hp.choice("loss", ["linear", "square", "exponential"]),
            "base_max_depth": scope.int(hp.quniform("base_max_depth", 1, 6, 1)),
        }
    if model_name == "lightgbm":
        return {
            "boosting_type": hp.choice("boosting_type", ["gbdt", "dart"]),
            "n_estimators": scope.int(hp.quniform("n_estimators", 50, 500, 10)),
            "max_depth": scope.int(hp.quniform("max_depth", 2, 8, 1)),
            "num_leaves": scope.int(hp.quniform("num_leaves", 8, 128, 4)),
            "learning_rate": hp.loguniform("learning_rate", np.log(0.005), np.log(0.2)),
            "min_child_samples": scope.int(hp.quniform("min_child_samples", 5, 80, 5)),
            "subsample": hp.uniform("subsample", 0.5, 1.0),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.4, 1.0),
            "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-4), np.log(50.0)),
            "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-4), np.log(100.0)),
        }
    if model_name == "catboost":
        return {
            "iterations": scope.int(hp.quniform("iterations", 100, 800, 25)),
            "depth": scope.int(hp.quniform("depth", 3, 10, 1)),
            "learning_rate": hp.loguniform("learning_rate", np.log(0.005), np.log(0.3)),
            "l2_leaf_reg": hp.loguniform("l2_leaf_reg", np.log(0.1), np.log(100.0)),
            "random_strength": hp.loguniform("random_strength", np.log(0.01), np.log(50.0)),
            "bagging_temperature": hp.uniform("bagging_temperature", 0.0, 1.0),
        }
    if model_name == "kernel_ridge":
        return {
            "alpha": hp.loguniform("alpha", np.log(1e-5), np.log(1e3)),
            "kernel": hp.choice("kernel", ["rbf", "laplacian", "linear", "polynomial"]),
            "gamma": hp.loguniform("gamma", np.log(1e-6), np.log(10.0)),
        }
    raise ValueError(f"Unsupported model name: {model_name}")


_INT_KEYS_BY_MODEL: dict[str, list[str]] = {
    "random_forest": ["n_estimators", "min_samples_split", "min_samples_leaf"],
    "extra_trees": ["n_estimators", "min_samples_split", "min_samples_leaf"],
    "hist_gradient_boosting": ["max_iter", "min_samples_leaf", "max_leaf_nodes"],
    "adaboost": ["n_estimators", "base_max_depth"],
    "lightgbm": ["n_estimators", "max_depth", "num_leaves", "min_child_samples"],
    "catboost": ["iterations", "depth"],
    "xgboost": ["n_estimators", "max_depth"],
}


def normalize_hyperopt_params(model_name: str, params: dict[str, Any]) -> dict[str, Any]:
    normalized = {key: to_native(value) for key, value in params.items()}
    if model_name == "mlp":
        normalized["hidden_layer_sizes"] = tuple(normalized["hidden_layer_sizes"])
    for key in _INT_KEYS_BY_MODEL.get(model_name, []):
        normalized[key] = int(normalized[key])
    return normalized


def build_model(model_name: str, params: dict[str, Any], random_state: int) -> Any:
    params = normalize_hyperopt_params(model_name, params)
    if model_name == "ridge":
        return Ridge(alpha=float(params["alpha"]))
    if model_name == "elasticnet":
        return ElasticNet(
            alpha=float(params["alpha"]),
            l1_ratio=float(params["l1_ratio"]),
            max_iter=20000,
            random_state=random_state,
        )
    if model_name == "xgboost":
        return XGBRegressor(
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=MODEL_N_JOBS,
            verbosity=0,
            tree_method="hist",
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_lambda=float(params["reg_lambda"]),
            reg_alpha=float(params["reg_alpha"]),
            min_child_weight=float(params["min_child_weight"]),
            gamma=float(params["gamma"]),
        )
    if model_name == "svr":
        return SVR(
            kernel=str(params["kernel"]),
            C=float(params["C"]),
            epsilon=float(params["epsilon"]),
            gamma=float(params["gamma"]),
            max_iter=SVR_MAX_ITER,
        )
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=params["max_depth"],
            min_samples_split=int(params["min_samples_split"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            random_state=random_state,
            n_jobs=MODEL_N_JOBS,
        )
    if model_name == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
            activation=str(params["activation"]),
            alpha=float(params["alpha"]),
            learning_rate_init=float(params["learning_rate_init"]),
            batch_size=int(params["batch_size"]),
            early_stopping=True,
            n_iter_no_change=20,
            max_iter=1000,
            random_state=random_state,
        )
    if model_name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=params["max_depth"],
            min_samples_split=int(params["min_samples_split"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            random_state=random_state,
            n_jobs=MODEL_N_JOBS,
        )
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=int(params["max_iter"]),
            max_depth=params["max_depth"],
            learning_rate=float(params["learning_rate"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_leaf_nodes=int(params["max_leaf_nodes"]),
            l2_regularization=float(params["l2_regularization"]),
            early_stopping=False,
            random_state=random_state,
        )
    if model_name == "adaboost":
        base = DecisionTreeRegressor(
            max_depth=int(params["base_max_depth"]),
            random_state=random_state,
        )
        return AdaBoostRegressor(
            estimator=base,
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            loss=str(params["loss"]),
            random_state=random_state,
        )
    if model_name == "lightgbm":
        return LGBMRegressor(
            objective="regression_l1",
            random_state=random_state,
            n_jobs=MODEL_N_JOBS,
            verbose=-1,
            boosting_type=str(params["boosting_type"]),
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            num_leaves=int(params["num_leaves"]),
            learning_rate=float(params["learning_rate"]),
            min_child_samples=int(params["min_child_samples"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]),
        )
    if model_name == "catboost":
        return CatBoostRegressor(
            iterations=int(params["iterations"]),
            depth=int(params["depth"]),
            learning_rate=float(params["learning_rate"]),
            l2_leaf_reg=float(params["l2_leaf_reg"]),
            random_strength=float(params["random_strength"]),
            bagging_temperature=float(params["bagging_temperature"]),
            loss_function="MAE",
            random_seed=random_state,
            thread_count=MODEL_N_JOBS,
            verbose=False,
            allow_writing_files=False,
        )
    if model_name == "kernel_ridge":
        return KernelRidge(
            alpha=float(params["alpha"]),
            kernel=str(params["kernel"]),
            gamma=float(params["gamma"]),
        )
    raise ValueError(f"Unsupported model name: {model_name}")
